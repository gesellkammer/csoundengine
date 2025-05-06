from __future__ import annotations

import dataclasses
import io as _io
import logging as _logging
import os as _os
import re as _re
import shutil as _shutil
import tempfile as _tempfile
import textwrap as _textwrap
from pathlib import Path as _Path
from typing import TYPE_CHECKING

import emlib.mathlib
import emlib.misc
import emlib.textlib
import numpy as np

from csoundengine.config import config

from . import csoundlib, csoundparse
from .renderjob import RenderJob

if TYPE_CHECKING:
    from typing import Sequence, Set


__all__ = (
    'Csd',
)


logger = _logging.getLogger("csoundengine")


@dataclasses.dataclass
class _InstrDef:
    p1: int | str
    body: str
    samelineComment: str = ''
    preComment: str = ''
    postComment: str = ''
    extranames: list[int | str] | None = None


@dataclasses.dataclass
class _OpcodeDef:
    name: str
    outargs: str
    inargs: str
    body: str


@dataclasses.dataclass
class _TableDataFile:
    """
    A table holding either the data or a file to the data

    Attributes:
        tabnum: the f-table number
        data: the data itself or a path to a file
        fmt: the format of the file
        start: start time to define the table
        size: the size of the table
    """
    tabnum: int
    """The assigned table number"""

    data: Sequence[float] | np.ndarray | str
    """the data itself or a path to a file"""

    fmt: str   # One of 'wav', 'flac', 'gen23', etc
    """The format of the data, one of 'wav', flac', 'gen23', etc"""

    start: float = 0
    """Allocation time of the table (p2)"""

    size: int = 0
    """Size of the data"""

    chan: int = 0
    """Which channel to read, if applicable. 0=all"""

    def __post_init__(self):
        assert self.fmt in {'gen23', 'wav', 'aif', 'aiff', 'flac'}, \
            f"Format not supported: {self.fmt}"
        if self.fmt == 'gen23' and isinstance(self.data, np.ndarray):
            assert len(self.data.shape) == 1 or self.data.shape[1] == 1

    def write(self, outfile: str) -> None:
        if isinstance(self.data, str):
            # just copy the file
            assert _os.path.exists(self.data)
            _shutil.copy(self.data, outfile)
            return

        base, ext = _os.path.splitext(outfile)
        if self.fmt == 'gen23':
            if ext != '.gen23':
                raise ValueError(f"Wrong extension: it should be .gen23, got {outfile}")
            csoundlib.saveAsGen23(self.data, outfile=outfile)
        elif self.fmt in ('wav', 'aif', 'aiff', 'flac'):
            import sndfileio
            dataarr = np.asarray(self.data, dtype=float)
            sndfileio.sndwrite(outfile, dataarr, sr=44100,
                               metadata={'comment': 'Datafile'})

    def scoreLine(self, outfile: str) -> str:
        if self.fmt == 'gen23':
            return f'f {self.tabnum} {self.start} {self.size} -23 "{outfile}"'
        elif self.fmt == 'wav':
            # time  size  1  filcod  skiptime  format  channel
            return f'f {self.tabnum} {self.start} {self.size} -1 "{outfile}" 0 0 0'
        raise ValueError(f"Unknown format {self.fmt}")

    def orchestraLine(self, outfile: str) -> str:
        if self.fmt == 'gen23':
            return f'ftgen {self.tabnum}, {self.start}, {self.size}, -23, "{outfile}"'
        elif self.fmt in ('wav', 'aif', 'aiff', 'flac'):
            return f'ftgen {self.tabnum}, {self.start}, {self.size}, -1, "{outfile}", 0, 0, 0'
        raise ValueError(f"Unknown format {self.fmt}")


@dataclasses.dataclass
class ScoreLine:
    """
    An event line in the score (an instrument, a table declaration, etc.)

    Attributes:
        kind: 'i' for instrument event, 'f' for table definition
        p1: the p1 of the event
        start: the start time of the event
        dur: the duration of the event
        args: any other args of the event (starting with p4)
    """
    kind: str
    pfields: list[float | str]
    comment: str = ''

    @property
    def p1(self) -> float | str:
        return self.pfields[0]

    @property
    def start(self) -> float:
        if self.kind == 'i' or self.kind == 'f':
            start = self.pfields[1]
            assert isinstance(start, (int, float))
            return start
        elif self.kind == 'e':
            end = self.pfields[0]
            assert isinstance(end, (int, float))
            return end
        elif self.kind == 'C':
            return 0.
        else:
            return 0

    @property
    def dur(self) -> float:
        if self.kind in 'i':
            dur = self.pfields[2]
            assert isinstance(dur, (int, float))
            return dur
        else:
            logger.debug(f"Score line of type '{self.kind}' does not have a duration")
            return 0.

    @property
    def end(self) -> float:
        return self.start + self.dur

    def asline(self) -> str:
        parts = [self.kind]
        for pfield in self.pfields:
            if isinstance(pfield, (int, float)):
                parts.append(str(pfield))
            elif isinstance(pfield, str):
                if pfield.startswith('"'):
                    parts.append(pfield)
                else:
                    parts.append(f'"{pfield}"')
            elif hasattr(pfield, '__float__') or hasattr(pfield, '__int__'):
                parts.append(str(float(pfield)))
            else:
                raise TypeError(f"Invalid pfield: {pfield}, {type(pfield)=}, {self.pfields=}")
        if self.comment:
            parts.append(f'    ; {self.comment}')
        return ' '.join(parts)

    def asrow(self) -> list:
        return [self.kind] + self.pfields

    def copy(self) -> ScoreLine:
        return ScoreLine(kind=self.kind, pfields=self.pfields.copy())


_builtinInstrs = {
    '_playgen1': r'''
      kgain  = p4
      kspeed = p5
      ; 6      7      8      9
      itabnum, ichan, ifade, ioffset passign 6
      ifade = max(ifade, 0.005)
      ksampsplayed = 0
      inumsamples = nsamp(itabnum)
      itabsr = ftsr(itabnum)
      istartframe = ioffset * itabsr
      ksampsplayed += ksmps * kspeed
      aouts[] loscilx kgain, kspeed, itabnum, 4, 1, istartframe
      aenv linsegr 0, ifade, 1, ifade, 0
      aouts = aouts * aenv
      inumouts = lenarray(aouts)
      kchan = 0
      while kchan < inumouts do
        outch kchan+ichan, aouts[kchan]
        kchan += 1
      od
      if ksampsplayed >= inumsamples then
        turnoff
      endif
    ''',
    '_ftnew': r'''
      itabnum = p4
      isize = p5
      isr = p6
      inumchannels = p7
      ift ftgen itabnum, 0, -isize, -2, 0
      if isr > 0 || inumchannels > 0 then
        ftsetparams itabnum, isr, inumchannels
      endif
    ''',
    '_ftfree': r'''
        itabnum = p4
        ftfree itabnum, 0
        turnoff
    ''',
    '_setp': r'''
      itype = p4   ; 1 if instr is a string
      if itype == 1 then
          Sp1 = p5
          ip1 = namedinstrtofrac(Sp1)
      else
          ip1 = p5
      endif
      pwrite ip1, p7, p8
    '''
}


class Csd:
    """
    Build a csound script by adding global code, instruments, score events, etc.

    Args:
        sr: the sample rate of the generated audio
        ksmps: the samples per cycle to use
        nchnls: the number of output channels
        nchnls_i: if given, the number of input channels
        a4: the reference frequency
        options: any number of command-line options passed to csound
        nodisplay: if True, avoid outputting debug information
        carry: should carry be enabled in the score?
        reservedTables: when creating tables, table numbers are autoassigned from
            python. There can be conflicts of any code uses ``ftgen``

    Example
    ~~~~~~~

    .. code::

        >>> from csoundengine.csd import Csd
        >>> csd = Csd(ksmps=32, nchnls=4)
        >>> csd.addInstr('sine', r'''
        ...   ifreq = p4
        ...   outch 1, oscili:a(0.1, ifreq)
        ... ''')
        >>> source = csd.addSndfile("sounds/sound1.wav")
        >>> csd.playTable(source)
        >>> csd.addEvent('sine', 0, 2, [1000])
        >>> csd.write('out.csd')
    """

    def __init__(self,
                 sr: int = 44100,
                 ksmps: int = 64,
                 nchnls: int = 2,
                 a4: float = 442.,
                 options: list[str] | None = None,
                 nodisplay=False,
                 carry=False,
                 nchnls_i: int | None = None,
                 numthreads: int = 0,
                 reservedTables: int = 0):
        self.score: list[ScoreLine] = []
        """The score, a list of ScoreLine"""

        self.instrs: dict[str | int, _InstrDef] = {}
        """The orchestra"""

        self.opcodes: dict[str, _OpcodeDef] = {}
        """User defined opcodes"""

        self.globalcodes: list[str] = []
        """Code to evaluate at the instr0 level"""

        self.options: list[str] = []
        """Command line options"""

        self._sr = sr
        """Samplerate"""

        self.ksmps = ksmps
        """Samples per cycle"""

        self.nchnls = nchnls
        """Number of output channels"""

        self.nchnls_i = nchnls_i
        """Number of input channels"""

        self.a4 = a4
        """Reference frequency"""

        self.nodisplay = nodisplay
        """Disable display opcodes"""

        self.enableCarry = carry
        """Enable carry in the score"""

        self.numthreads = numthreads
        """Number of threads used for rendering"""

        self.datafiles: dict[int, _TableDataFile] = {}
        """Maps assigned table numbers to their metadata"""

        self._datafileIndex: dict[str, _TableDataFile] = {}
        """Maps soundfiles read to their assigned table number"""

        self._strLastIndex = 20
        self._str2index: dict[str, int] = {}

        if options:
            self.addOptions(*options)

        self._outfileFormat = ''
        self._outfileEncoding = ''
        self._compressionQuality = ''

        self._definedTables: Set[int] = set()
        self._minTableIndex = 1
        self._endMarker: float = 0
        self._numReservedTables = reservedTables
        self._maxTableNumber = reservedTables
        self.score.append(ScoreLine(kind='C', pfields=[0.], comment='Disable carry'))

    @property
    def sr(self) -> int:
        """Samplerate"""
        return self._sr

    @sr.setter
    def sr(self, value: int):
        if not isinstance(value, int):
            raise TypeError(f"Samplerate must be an int, got {value}")
        self._sr = value

    def copy(self) -> Csd:
        """
        Copy this csd
        """
        out = Csd(sr=self.sr,
                  ksmps=self.ksmps,
                  nchnls=self.nchnls,
                  a4=self.a4,
                  options=self.options.copy(),
                  nodisplay=self.nodisplay,
                  carry=self.enableCarry,
                  nchnls_i=self.nchnls_i,
                  numthreads=self.numthreads)

        out.instrs = self.instrs.copy()
        out.score = self.score.copy()
        out._str2index = self._str2index.copy()
        out._strLastIndex = self._strLastIndex
        if self.globalcodes:
            for code in self.globalcodes:
                out.addGlobalCode(code)

        out._outfileEncoding = self._outfileEncoding
        out._outfileFormat = self._outfileFormat
        out._compressionQuality = self._compressionQuality

        out._definedTables = self._definedTables
        out._minTableIndex = self._minTableIndex
        out._maxTableNumber = self._maxTableNumber

        if self.datafiles:
            out.datafiles = self.datafiles.copy()

        if self._outfileEncoding:
            out.setSampleEncoding(self._outfileEncoding)

        return out

    def cropScore(self, start=0., end=0.) -> None:
        """
        Crop the score at the given boundaries

        Any event starting earlier or ending after the given times will
        be cropped, any event ending before start or starting before
        end will be removed
        """
        score = _cropScore(self.score, start, end)
        self.score = score

    def dumpScore(self) -> None:
        """
        Show the score as a table
        """
        maxp = max(len(event.pfields) for event in self.score)
        headers = ["#"] + [f'p{n}' for n in range(maxp)]
        rows = [scoreline.asrow() for scoreline in self.score]
        emlib.misc.print_table(rows, headers=headers, floatfmt=".3f")

    def addScoreLine(self, line: str | list[int | float | str], comment='') -> None:
        """
        Add a score line verbatim

        Args:
            line: the line to add
            comment: add a comment to the score line, when written
        """
        if isinstance(line, str):
            try:
                parts = csoundparse.splitScoreLine(line)
            except ValueError as e:
                raise ValueError(f"Could not parse line '{line}', error: {e}")
        else:
            parts = line

        if not comment:
            last = parts[-1]
            if isinstance(last, str) and last.lstrip().startswith(';'):
                parts = parts[:-1]
                comment = last.split(';')[-1]

        kind = parts[0]
        assert isinstance(kind, str) and kind in 'ifCed', f"Invalid score statement: {line}"
        self.score.append(ScoreLine(kind=kind, pfields=parts[1:], comment=comment))

    def addEvent(self,
                 instr: int | float | str,
                 start: float,
                 dur: float,
                 args: Sequence[float | str] | None = None,
                 comment='',
                 numdigits=8) -> None:
        """
        Add an instrument ("i") event to the score

        Args:
            instr: the instr number or name, as passed to addInstr
            start: the start time
            dur: the duration of the event
            args: pargs beginning at p4
            numdigits: if given, round start and duration to this number of digits
            comment: if given, the text is attached as a comment to the event
                line in the score
        """
        if numdigits:
            start = round(start, numdigits)
            dur = round(dur, numdigits)
        pfields = [instr, start, dur]

        if args:
            pfields.extend(args)

        self.score.append(ScoreLine(kind='i', pfields=pfields, comment=comment))

    def strset(self, s: str, index: int | None) -> int:
        """
        Add a strset to this csd

        If ``s`` has already been passed, the same index is returned
        """
        if s in self._str2index:
            if index is not None and index != self._str2index[s]:
                raise KeyError(f"String '{s}' already set with different index "
                               f"(old: {self._str2index[s]}, new: {index})")
            return self._str2index[s]

        if index is None:
            index = self._strLastIndex
        else:
            self._strLastIndex = max(self._strLastIndex, index)
        self._strLastIndex += 1
        self._str2index[s] = index
        return index

    def _assignTableIndex(self, tabnum=0) -> int:
        if tabnum == 0:
            tabnum = self._maxTableNumber + 1
        else:
            if tabnum in self._definedTables:
                raise ValueError(f"ftable {tabnum} already defined")
        if tabnum > self._maxTableNumber:
            self._maxTableNumber = tabnum
        self._definedTables.add(tabnum)
        assert tabnum > 0
        return tabnum

    def _addTable(self, pargs: Sequence[float | int | str], comment='') -> int:
        """
        Adds a ftable to the score

        Args:
            pargs: as passed to csound (without the "f")
                p1 can be 0, in which case a table number
                is assigned

        Returns:
            The index of the new ftable
        """
        tabnum = pargs[0]
        if tabnum == 0:
            tabnum = self._assignTableIndex()
        else:
            assert tabnum in self._definedTables, f"Table {tabnum} not known, defined tables: {self._definedTables}"

        pfields = [tabnum, *pargs[1:]]
        self.score.append(ScoreLine(kind='f', pfields=pfields, comment=comment))
        return int(tabnum)


    def addTableFromData(self,
                         data: Sequence[float] | np.ndarray,
                         tabnum: int = 0,
                         start=0.,
                         filefmt='',
                         sr: int = 0,
                         ) -> int:
        """
        Add a table definition with the data

        Args:
            data: a sequence of floats to fill the table. The size of the
                table is determined by the size of the seq.
            tabnum: 0 to auto-assign an index
            start: allocation time of the table
            filefmt: format to use when saving the table as a datafile. If not given,
                the default is used. Possible values: 'gen23', 'wav'
            sr: if given and data is a numpy array, it is saved as a soundfile
                and loaded via gen1

        Returns:
            the table number

        .. note::

            The data is either included in the table definition (if it is
            small enough) or saved as an external file. All external files are
            saved relative to the generated .csd file when writing. Table data
            is saved as 32 bit floats, so it might loose some precission from
            the original.
        """
        sizeThreshold = config['offline_score_table_size_limit']

        if isinstance(data, np.ndarray) and sr:
            sndfile = _tempfile.mktemp(suffix=".wav")
            import sndfileio
            sndfileio.sndwrite(sndfile, samples=data, sr=sr, encoding='float32')
            tabnum = self.addSndfile(sndfile, tabnum=tabnum, asProjectFile=True,
                                     start=start)
        else:
            if not filefmt:
                filefmt = config['datafile_format']

            tabnum = self._assignTableIndex(tabnum)

            if len(data) > sizeThreshold:
                # If the data is big, we save the data. We will write
                # it to a file when rendering
                datafile = _TableDataFile(tabnum, data, start=start, fmt=filefmt)
                self._addProjectFile(datafile)
            else:
                pargs = [tabnum, start, -len(data), -2]
                pargs.extend(data)
                tabnum = self._addTable(pargs)

        assert tabnum > 0
        return tabnum

    def _addProjectFile(self, datafile: _TableDataFile) -> None:
        self.datafiles[datafile.tabnum] = datafile
        if isinstance(datafile.data, str):
            self._datafileIndex[datafile.data] = datafile
        assert datafile.tabnum in self._definedTables

    def addEmptyTable(self, size: int, tabnum: int = 0, sr: int = 0,
                      numchannels=1, time=0.
                      ) -> int:
        """
        Add an empty table to this Csd

        A table remains valid until the end of the csound process or until
        the table is explicitely freed (see :meth:`~Csd.freeTable`)

        Args:
            tabnum: use 0 to autoassign an index
            size: the size of the empty table
            sr: if given, set the sr of the empty table to the given sr
            numchannels: the number of channels in the table
            time: when to do the allocation.

        Returns:
            The index of the created table
        """
        if sr == 0:
            pargs = (tabnum, 0, -size, -2, 0)
            return self._addTable(pargs)
        else:
            tabnum = self._assignTableIndex(tabnum)
            self._ensureBuiltinInstr('_ftnew')
            args = [tabnum, size, sr, numchannels]
            self.addEvent('_ftnew', start=time, dur=0, args=args)
            return tabnum

    def freeTable(self, tabnum: int, time: float):
        """
        Free a table

        Args:
            tabnum: the table number
            time: when to free it
        """
        self._ensureBuiltinInstr('_ftfree')
        self.addEvent('_ftfree', start=time, dur=0, args=[tabnum])

    def _ensureBuiltinInstr(self, name: str):
        if self.instrs.get(name) is None:
            self.addInstr(name, _builtinInstrs[name])

    def addSndfile(self, sndfile: str, tabnum=0, start=0., skiptime=0., chan=0,
                   asProjectFile=False) -> int:
        """
        Add a table which will load this sndfile

        Args:
            sndfile: the output to load
            tabnum: fix the table number or use 0 to generate a unique table number
            start: when to load this output (normally this should be left 0)
            skiptime: begin reading at `skiptime` seconds into the file.
            chan: channel number to read. 0 denotes read all channels.
            asProjectFile: if True, the sndfile is included as a project file and
                copied to a path relative to the .csd when writing

        Returns:
            the table number
        """
        sndfmt = _os.path.splitext(sndfile)[1][1:].lower()
        supportedFormats = ('wav', 'aif', 'aiff', 'flac')
        if sndfmt not in supportedFormats:
            raise ValueError(f"Format '{sndfmt}' not supported, "
                             f"supported formats: {supportedFormats}")

        if datafile := self._datafileIndex.get(sndfile):
            return datafile.tabnum

        tabnum = self._assignTableIndex(tabnum)
        datafile = _TableDataFile(tabnum, data=sndfile, start=start, fmt=sndfmt)

        if not asProjectFile:
            pargs = [tabnum, start, 0, -1, sndfile, skiptime, 0, chan]
            self._datafileIndex[sndfile] = datafile
            self._addTable(pargs)
        else:
            self._addProjectFile(datafile)
        return tabnum

    def destroyTable(self, tabnum: int, time: float) -> None:
        """
        Schedule ftable with index `source` to be destroyed at time `time`

        Args:
            tabnum: the index of the table to be destroyed
            time: the time to destroy it
        """
        self.score.append(ScoreLine('f', [-tabnum, time]))

    def setEndMarker(self, time: float) -> None:
        """
        Add an end marker to the score

        This is needed if, for example, all events are endless
        events (with dur == -1).

        If an end marker has been already set, setting it later will remove
        the previous endmarker (there can be only one)
        """
        if time == 0 or self._endMarker > 0:
            self.removeEndMarker()
        self._endMarker = time
        # We don't add the marker to the score because this needs to go at the end
        # of the score. Any score line after the end marker will not be read

    def removeEndMarker(self) -> None:
        """
        Remove the end-of-score marker
        """
        self._endMarker = 0

    def setComment(self, comment: str) -> None:
        """ Add a comment to the renderer output soundfile"""
        self.addOptions(f'-+id_comment="{comment}"')

    def setOutfileFormat(self, fmt: str) -> None:
        """
        Sets the format for the output soundfile

        If this is not explicitely set it will be induced from
        the output soundfile set when running the csd

        Args:
            fmt: the format to use ('wav', 'aif', 'flac', etc)
        """
        assert fmt in {'wav', 'aif', 'aiff', 'flac', 'ogg'}
        self._outfileFormat = fmt

    def setSampleEncoding(self, encoding: str) -> None:
        """
        Set the sample encoding for recording

        If not set, csound's own default for encoding will be used

        Args:
            encoding: one of 'pcm16', 'pcm24', 'pcm32', 'float32', 'float64'

        """
        assert encoding in {'pcm16', 'pcm24', 'pcm32', 'float32', 'float64', 'vorbis'}
        self._outfileEncoding = encoding

    def setCompressionQuality(self, quality=0.4) -> None:
        """
        Set the compression quality

        Args:
            quality: a value between 0 and 1
        """
        self._compressionQuality = quality

    def setCompressionBitrate(self, bitrate=128, format='ogg') -> None:
        """
        Set the compression quality by defining a bitrate

        Args:
            bitrate: the bitrate in kB/s
            format: the format used (only 'ogg' at the moment)
        """
        from . import csoundlib
        self.setCompressionQuality(csoundlib.compressionBitrateToQuality(bitrate, format))

    def _writeScore(self, stream, datadir='.', dataprefix='') -> None:
        """
        Write the score to `stream`

        Args:
            stream (file-like): the open stream to write to
            datadir: the folder to save data files
        """
        self.score.sort(key=lambda ev: ev.start)
        for event in self.score:
            stream.write(event.asline())
            stream.write("\n")
        for tabnum, datafile in self.datafiles.items():
            assert tabnum > 0
            outfilebase = f'table-{tabnum:04d}.{datafile.fmt}'
            if dataprefix:
                outfilebase = f'{dataprefix}-{outfilebase}'
            datadirpath = _Path(datadir)
            outfile = datadirpath / outfilebase
            datafile.write(outfile.as_posix())
            relpath = outfile.relative_to(datadirpath.parent)
            stream.write(datafile.scoreLine(relpath.as_posix()))
            stream.write('\n')
        if self._endMarker:
            stream.write(f'e {self._endMarker}    ; end marker')

    def scoreDuration(self) -> float:
        """
        Returns the duration of this score

        If the end marker was set, this will determine the returned duration
        """
        if self._endMarker:
            return self._endMarker

        endtime = 0.
        for ev in self.score:
            end = ev.end
            if end == float('inf'):
                return float('inf')
            elif end is not None and end > endtime:
                endtime = end
        return endtime

    def addInstr(self, instr: int | str, body: str, instrComment='',
                 extranames: list[int | str] | None = None
                 ) -> None:
        """
        Add an instrument definition to this csd

        Args:
            instr: the instrument number of name
            body: the body of the instrument (the part between 'instr' / 'endin')
            instrComment: if given, it will be added at the end of the 'instr' line
            extranames: an instr can have multiple names/numbers assigned

        """
        if _re.search(r"^\s*instr", body):
            raise ValueError(f"The body should only include the instrument definition, "
                             f"the part between 'instr' / 'endin', got: {body}")

        instrdef = _InstrDef(p1=instr, body=body, samelineComment=instrComment, extranames=extranames)
        self.instrs[instr] = instrdef

    def addOpcode(self, name: str, outargs: str, inargs: str, body: str) -> None:
        """
        Add an opcode to this csd

        Args:
            name: the opcode name
            outargs: the output arguments
            inargs: the input arguments
            body: the body of the opcode

        Example
        ~~~~~~~

        .. code::

            csd.addOpcode("gain", "a", "ak", r'''
              asig, kgain xin
              asig *=kgain
              xout asig
            ''')

        """
        self.opcodes[name] = _OpcodeDef(name, outargs=outargs, inargs=inargs, body=body)

    def addGlobalCode(self, code: str, acceptDuplicates=True) -> None:
        """
        Add code to the instr 0

        Args:
            code: code to add
            acceptDuplicates: add copies of the same code even if the same csound
                code has already been added
        """
        if not acceptDuplicates and code in self.globalcodes:
            return
        self.globalcodes.append(code)

    def addOptions(self, *options: str) -> None:
        """
        Adds options to this csd

        Options are any command-line options passed to csound itself or which could
        be used within a <CsOptions> tag. They are not checked for correctness
        """
        self.options.extend(options)

    def dump(self) -> str:
        """ Returns a string with the .csd """
        stream = _io.StringIO()
        self._writeCsd(stream)
        return stream.getvalue()

    def playTable(self, tabnum: int, start: float, dur: float = -1,
                  gain=1., speed=1., chan=1, fade=0.05,
                  skip=0.) -> None:
        """
        Add an event to play the given table

        Args:
            tabnum: the table number to play
            start: schedule time (p2)
            dur: duration of the event (leave -1 to play until the end)
            gain: a gain factor applied to the table samples
            chan: ??
            fade: fade time (both fade-in and fade-out
            skip: time to skip from playback (enables playback to crop a fragment at the beginning)

        Example
        ~~~~~~~

            >>> csd = Csd()
            >>> source = csd.addSndfile("stereo.wav")
            >>> csd.playTable(source, source, start=1, fade=0.1, speed=0.5)
            >>> csd.write("out.csd")
        """
        if self.instrs.get('_playgen1') is None:
            self.addInstr('_playgen1', _builtinInstrs['_playgen1'])
        assert tabnum > 0
        args = [gain, speed, tabnum, chan, fade, skip]
        self.addEvent('_playgen1', start=start, dur=dur, args=args)

    def write(self, csdfile: str) -> None:
        """
        Write this as a .csd

        Any data files added are written to a folder <csdfile>.assets besides the
        generated .csd file.

        Args:
            csdfile: the path to save to


        Example
        ~~~~~~~

            >>> from csoundengine.csd import Csd
            >>> csd = Csd(...)
            >>> csd.write("myscript.csd")

        This will generate a ``myscript.csd`` file and a folder ``myscript.assets`` holding
        any data file needed. If no data files are used, no ``.assets`` folder is created

        """
        csdfile = _os.path.expanduser(csdfile)
        base = _os.path.splitext(csdfile)[0]
        stream = open(csdfile, "w")
        if self.datafiles:
            datadir = base + ".assets"
            _os.makedirs(datadir, exist_ok=True)
        else:
            datadir = ''
        self._writeCsd(stream, datadir=datadir)

    def _writeCsd(self, stream, datadir='') -> None:
        """
        Write this as a csd

        Args:
            stream: the stream to write to. Either an open file or
                a io.StringIO
            datadir: the folder where all datafiles are written. Datafiles are
                used whenever the user defines tables with data too large to
                include 'inline' (as gen2) or when adding soundfiles.
        """
        write = stream.write
        write("<CsoundSynthesizer>\n<CsOptions>\n")
        options = self.options.copy()
        if self.nodisplay:
            options.append("-m0")

        if self.numthreads > 1:
            options.append(f"-j {self.numthreads}")

        if self._outfileFormat:
            options.extend(csoundlib.csoundOptionsForOutputFormat(self._outfileFormat, self._outfileEncoding))
        elif self._outfileEncoding:
            options.append(csoundlib.csoundOptionForSampleEncoding(self._outfileEncoding))

        for option in options:
            write(option)
            write("\n")
        write("</CsOptions>\n")

        srstr = f"sr     = {self.sr}" if self.sr is not None else ""

        txt = rf"""
            <CsInstruments>

            {srstr}
            ksmps  = {self.ksmps}
            0dbfs  = 1
            A4     = {self.a4}
            nchnls = {self.nchnls}
            """
        txt = _textwrap.dedent(txt)
        write(txt)
        if self.nchnls_i is not None:
            write(f'nchnls_i = {self.nchnls_i}\n')
        tab = "  "

        if self._str2index:
            for s, idx in self._str2index.items():
                write(f'strset {idx}, "{s}"\n')
            write("\n")

        if self.globalcodes:
            write("; ----- global code\n")
            for globalcode in self.globalcodes:
                write(globalcode)
                write("\n")
            write("; ----- end global code\n\n")

        for name, opcodedef in self.opcodes.items():
            write(f"opcode {name}, {opcodedef.outargs}, {opcodedef.inargs}")
            body = _textwrap.dedent(opcodedef.body)
            write(_textwrap.indent(body, tab))
            write("endop\n")

        for instr, instrdef in self.instrs.items():
            if instrdef.preComment:
                for line in instrdef.preComment.splitlines():
                    write(f";;  {line}\n")
            if instrdef.extranames:
                extranames = ', '.join(str(n) for n in instrdef.extranames)
                instrline = f"instr {instrdef.p1}, {extranames}"
            else:
                instrline = f"instr {instr}"
            if instrdef.samelineComment:
                instrline += f"  ; {instrdef.samelineComment}\n"
            else:
                instrline += "\n"
            write(instrline)
            if instrdef.postComment:
                if instrdef.preComment:
                    for line in instrdef.preComment.splitlines():
                        write(f"{tab};;  {line}\n")
            body = _textwrap.dedent(instrdef.body)
            body = _textwrap.indent(body, tab)
            write(body)
            write("\nendin\n")

        write("\n</CsInstruments>\n")
        write("\n<CsScore>\n\n")

        self._writeScore(stream, datadir=datadir)

        write("\n</CsScore>\n")
        write("</CsoundSynthesizer>")

    def render(self, outfile='', verbose=False) -> RenderJob:
        """
        Render this csd offline

        Args:
            outfile: the soundfile to generate, if not given a tempfile is used
            verbose: output rendering information. If False, stdout and stderr can still
                be read through the Popen object

        Returns:
            a RenderJob object
        """
        if not outfile:
            if self._outfileFormat:
                suffix = '.' + self._outfileFormat
            else:
                suffix = '.wav'
            outfile = _tempfile.mktemp(prefix='csoundengine-', suffix=suffix)
        return self.run(output=outfile, piped=not verbose, nomessages=not verbose)

    def run(self,
            output: str,
            csdfile='',
            inputdev='',
            backend='',
            suppressdisplay=True,
            nomessages=False,
            piped=False,
            extraOptions: list[str] | None = None
            ) -> RenderJob:
        """
        Run this csd.

        Args:
            output: the output of the csd. This will be passed
                as the -o argument to csound. If an empty string or None is given,
                no sound is produced (adds the '--nosound' flag).
            inputdev: the input device to use when running in realtime
            csdfile: if given, the csd file will be saved to this path and run
                from it. Otherwise a temp file is created and run.
            backend: the backend to use
            suppressdisplay: if True, display (table plots, etc.) is supressed
            nomessages: if True, debugging scheduling information is suppressed
            piped: if True, stdout and stderr are piped through
                the Popen object, accessible through .stdout and .stderr
                streams
            extraOptions: any extra args passed to the csound binary

        Returns:
            a RenderJob holding a subprocess.Popen object

        """
        options = self.options.copy()
        outfileFormat = ''
        outfileEncoding = ''
        offline = True
        if not output:
            options.append('--nosound')
        elif not output.startswith('dac'):
            outfileFormat = self._outfileFormat or _os.path.splitext(output)[1][1:]
            outfileEncoding = self._outfileEncoding or csoundlib.bestSampleEncodingForExtension(outfileFormat)
            if self._compressionQuality:
                options.append(f'--vbr-quality={self._compressionQuality}')
        else:
            offline = False

        if not csdfile:
            csdfile = _tempfile.mktemp(suffix=".csd")
            logger.debug(f"Runnings Csd from tempfile {csdfile}")

        if outfileFormat:
            options.extend(csoundlib.csoundOptionsForOutputFormat(outfileFormat, outfileEncoding))

        if extraOptions:
            options.extend(extraOptions)

        options = emlib.misc.remove_duplicates(options)

        self.write(csdfile)
        proc = csoundlib.runCsd(csdfile, outdev=output, indev=inputdev,
                                backend=backend, nodisplay=suppressdisplay,
                                nomessages=nomessages,
                                piped=piped, extra=options)
        return RenderJob(outfile=output if offline else '',
                         samplerate=self.sr,
                         encoding=outfileEncoding,
                         process=proc)

    def setPfield(self, p1: int | float | str, pindex: int, value: float, start: float) -> None:
        """
        Set the value of a pfield for a scheduled event

        Args:
            p1: the instr number/name of the event
            pindex: the index of the pfield, 4=p4, 5=p5, etc
            value: the new value of the pfield
            start: when to set the pfield (absolute time)
        """
        self._ensureBuiltinInstr('_setp')
        self.addEvent('_setp', start=start, dur=0, args=[1 if isinstance(p1, str) else 0, p1, pindex, value])

    def automatePfield(self, p1: int | float | str, pindex: int, pairs: Sequence[float], start: float) -> None:
        """
        Automate the pfield of a scheduled event

        Args:
            p1: the instr number/name of the event
            pindex: the index of the pfield
            pairs: a flat sequence of breakpoints of the form [time0, value0, time1, value1, ...]
            start: absolute time to start the automation
        """
        raise RuntimeError("Not supported yet")
        # self._ensureBuiltinInstr('_automatep')


def _cropScore(events: list[ScoreLine], start=0., end=0.) -> list:
    """
    Crop the score so that no event exceeds the given limits

    Args:
        events: a list of events, where each event is a sequence
            representing the pargs [p1, p2, p3, ...]
        start: the min. start time for any event
        end: the max. end time for any event

    Returns:
        the score events which are between start and end
    """
    scoreend = max(_ for ev in events
                   if (_ := ev.end) is not None)
    assert scoreend is not None and scoreend > 0, f"Invalid score duration ({scoreend}): {events}"
    if end == 0:
        end = scoreend
    cropped = []
    for ev in events:
        kind = ev.kind
        if kind == 'e' or kind == 'f':
            if start <= ev.start < end:
                cropped.append(ev)
        elif kind != 'i':
            cropped.append(ev)
            continue
        else:
            assert kind == 'i', f"Invalid kind: {kind=}, {ev=}"
            evstart = ev.start
            evdur = ev.dur
            evend = evstart + evdur if evdur >= 0 else float('inf')
            if evend < start or evstart > end:
                continue

            if start <= evstart and evend <= end:
                cropped.append(ev)
            else:
                xstart, xend = emlib.mathlib.intersection(start, end, evstart, evend)
                if xstart is not None:
                    if xend == float('inf'):
                        dur = -1
                    else:
                        dur = xend - xstart
                    ev2 = ev.copy()
                    ev2.pfields[1] = xstart
                    ev2.pfields[2] = dur
                    cropped.append(ev2)
    return cropped
