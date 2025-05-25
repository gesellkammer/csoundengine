from __future__ import annotations

import math
import os
import tempfile
import textwrap
import contextlib
from dataclasses import dataclass
from functools import cache

import numpy as np

from . import (
    csoundparse,
    engineorc,
    internal,
    tools
    )
from .config import config, logger
from .enginebase import TableInfo, _EngineBase
from .engineorc import BUSKIND_AUDIO, BUSKIND_CONTROL
from .errors import CsoundError
from .renderjob import RenderJob


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from csoundengine.csd import Csd
    import libcsound
    from typing import Sequence


__all__ = [
    'OfflineEngine'
]


class _OfflineComponent:
    """
    Base class for keeping track of offline events

    The history can then be used to generate a csd file from
    an offline engine
    """
    def apply(self, csd: Csd) -> None:
        pass


@dataclass
class _CompileEvent(_OfflineComponent):
    code: str

    def apply(self, csd: Csd) -> None:
        csd.addGlobalCode(self.code)


@dataclass
class _ScoreEvent(_OfflineComponent):
    kind: str
    pfields: list[float | str]
    comment: str = ''

    def __post_init__(self):
        assert self.kind in 'ifCed', f"Invalid event kind, got {self.kind}"

    def apply(self, csd: Csd) -> None:
        if self.kind == 'i':
            start = self.pfields[1]
            dur = self.pfields[2]
            assert isinstance(start, (int, float)) and isinstance(dur, (int, float))
            csd.addEvent(instr=self.pfields[0], start=start, dur=dur,
                         args=self.pfields[3:], comment=self.comment)
        else:
            pfields = [self.kind]
            pfields.extend(f'"{p}"' if isinstance(p, str) else str(p) for p in self.pfields)
            line = ' '.join(pfields)
            csd.addScoreLine(line)


@dataclass
class _SchedEvent(_OfflineComponent):
    instr: int | float | str
    delay: float
    dur: float
    args: list
    eventid: float
    comment: str = ''

    def apply(self, csd: Csd) -> None:
        csd.addEvent(instr=self.instr,
                     start=self.delay,
                     dur=self.dur,
                     args=self.args,
                     comment=self.comment)


@dataclass
class _TableDataEvent(_OfflineComponent):
    data: np.ndarray
    tabnum: int
    delay: float = 0.
    sr: int = 0

    def apply(self, csd: Csd) -> None:
        csd.addTableFromData(data=self.data, tabnum=self.tabnum, start=self.delay, sr=self.sr)


@dataclass
class _ReadSoundfileEvent(_OfflineComponent):
    path: str
    tabnum: int
    delay: float = 0.
    skiptime: float = 0.
    chan: int = 0

    def apply(self, csd: Csd) -> None:
        csd.addSndfile(sndfile=self.path, tabnum=self.tabnum, start=self.delay,
                       skiptime=self.skiptime, chan=self.chan)


@dataclass
class _SetParamEvent(_OfflineComponent):
    p1: int | float | str
    pindex: int
    value: int | float
    delay: float

    def apply(self, csd: Csd) -> None:
        csd.setPfield(p1=self.p1, pindex=self.pindex, value=self.value, start=self.delay)


@dataclass
class _AutomateEvent(_OfflineComponent):
    p1: int | float | str
    pfield: int | str
    pairs: Sequence[float] | np.ndarray
    delay: float = 0.
    overtake: bool = False

    def apply(self, csd: Csd) -> None:
        assert isinstance(self.pfield, int), f"Pfield must be an integer, got {self.pfield}"
        pairs = self.pairs.tolist() if isinstance(self.pairs, np.ndarray) else self.pairs
        csd.automatePfield(p1=self.p1,
                           pindex=self.pfield,
                           pairs=pairs,
                           start=self.delay)


class OfflineEngine(_EngineBase):
    """
    Non-real-time engine using the csound API

    This is similar to an :class:`csoundengine.engine.Engine` but
    it renders offline to a soundfile

    Args:
        sr: sample rate
        ksmps: samples per cycle
        outfile: soundfile to render to. If not given a tempfile is used
        nchnls: number of output channels
        a4: reference frequency
        numAudioBuses: number of audio buses (see :ref:`Bus Opcodes<busopcodes>`)
        numControlBuses: number of control buses (see :ref:`Bus Opcodes<busopcodes>`)
        withBusSupport: if True, add bus support at creation time. This should be True
            if you are using the bus opcodes from csound code before creating
            any bus from python. Otherwise, the creation of any bus (via assignBus)
            adds bus support automatically
        quiet: if True, suppress output of csound (-m 0)
        includes: a list of files to include. Can be added later via :meth:`~OfflineEngine.includeFile`
        sampleAccurate: use sample-accurate scheduling
        commandlineOptions: command line options passed verbatim to the
            csound process when started
        encoding: the sample encoding of the rendered file, given as 'pcm<bits>' or 'float<bits>',
            where bits presents the bit-depth (for example, 'pcm16', 'pcm24', 'float32', etc).
            If no encoding is given a suitable default for the sample format is chosen.


    """
    def __init__(self,
                 sr=44100,
                 ksmps: int = 64,
                 outfile: str = '',
                 nchnls=2,
                 a4: int = 0,
                 globalcode='',
                 numAudioBuses: int | None = None,
                 numControlBuses: int | None = None,
                 withBusSupport=False,
                 quiet=True,
                 includes: list[str] | None = None,
                 sampleAccurate=False,
                 encoding='',
                 nosound=False,
                 commandlineOptions: list[str] | None = None):
        super().__init__(sr=sr,
                         ksmps=ksmps,
                         a4=a4 or config['A4'],
                         nchnls=nchnls,
                         numAudioBuses=numAudioBuses if numAudioBuses is not None else config['num_audio_buses'],
                         numControlBuses=numControlBuses if numControlBuses is not None else config['num_control_buses'],
                         sampleAccurate=sampleAccurate)
        from . import csoundlib
        self.outfile = outfile or tempfile.mktemp(prefix='csoundengine-', suffix='.wav') if not nosound else ''
        self.globalcode = globalcode
        self.numAudioBuses = numAudioBuses if numAudioBuses is not None else config['num_audio_buses']
        self.numControlBuses = numControlBuses if numControlBuses is not None else config['num_control_buses']
        self.includes = includes if includes is not None else []
        self.encoding = encoding or csoundlib.bestSampleEncodingForExtension(os.path.splitext(self.outfile)[1][1:])
        self.version = 0

        self._renderjob: RenderJob | None = None

        self._strToIndex: dict[str, int] = {}
        self._indexToStr: dict[int, str] = {}
        self._tableInfo: dict[int, TableInfo] = {}
        self._soundfilesLoaded: dict[tuple[str, int, float], int] = {}
        self._parsedInstrs: dict[str, csoundparse.ParsedInstrBody] = {}
        # maps (path, chan, skiptime) -> tablenumber

        self._instanceCounters: dict[int, int] = {}
        self._fracnumdigits = 4  # number of fractional digits used for unique instances
        self._strLastIndex = 20
        self._stopped = False
        self._history: list[_OfflineComponent] = []
        self._trackHistory = True
        self._usesBuses = False  # does any instr uses the bus system?
        self._hasBusSupport = withBusSupport

        self._reservedInstrnums: set[int] = set()
        self._reservedInstrnumRanges: list[tuple[str, int, int]] = [
            ('builtinorc', engineorc.CONSTS['reservedInstrsStart'], engineorc.CONSTS['userInstrsStart'] - 1)]

        self._shouldPerform = False
        self._endtime = 0.
        self._tableCounter = engineorc.CONSTS['reservedTablesStart']
        self.nosound = nosound
        self.options = ["-d"]
        if quiet:
            self.options.extend(["--messagelevel=0", "--m-amps=0", "--m-range=0"])

        if nchnls == 0 or nosound:
            self.options.append('--nosound')
            self.nosound = True

        if sampleAccurate:
            self.options.append('--sample-accurate')

        if commandlineOptions:
            self.options.extend(commandlineOptions)

        self.csound: libcsound.Csound

        self._start()

        for s in ["cos", "linear", "smooth", "smoother"]:
            self.strSet(s)

    @property
    def endtime(self) -> float:
        return self._endtime

    def _makeIncludeBlock(self) -> str:
        if not self.includes:
            return ''
        includelines = [f'#include "{include}"' for include in self.includes]
        return "\n".join(includelines)

    def _compile(self, code: str) -> None:
        err = self.csound.compileOrc(code)
        if err:
            logger.error(internal.addLineNumbers(code))
            raise CsoundError(f"Error compiling base ochestra, error: {err}")
        self._history.append(_CompileEvent(code))

    def _start(self) -> None:
        import libcsound
        self.version = libcsound.VERSION
        self.csound = csound = libcsound.Csound()
        for option in self.options:
            csound.setOption(option)
        if not self.nosound:
            csound.setOption(f'-o{self.outfile}')

        header = engineorc.makeOrcHeader(sr=self.sr, ksmps=self.ksmps, nchnls=self.nchnls, nchnls_i=0, a4=self.a4)
        csound.compileOrc(header)
        orc, instrmap = engineorc.makeOrc(globalcode=self.globalcode,
                                          includestr=self._makeIncludeBlock())

        self._builtinInstrs = instrmap
        self._reservedInstrnums.update(set(instrmap.values()))
        self._compile(orc)
        csound.start()
        if self._hasBusSupport:
            self.addBusSupport()

    def evalCode(self, code: str) -> float:
        return self.csound.evalCode(code)

    def addBusSupport(self, numAudioBuses: int|None = None, numControlBuses: int|None = None) -> None:
        numAudioBuses = numAudioBuses if numAudioBuses is not None else self.numAudioBuses
        numControlBuses = numControlBuses if numControlBuses is not None else self.numControlBuses

        startInstr = max(instrnum for instrnum in self._builtinInstrs.values() if instrnum < engineorc.CONSTS['postProcInstrnum']) + 1
        postInstrnum = 1 + max(max(self._builtinInstrs.values()), engineorc.CONSTS['postProcInstrnum'])

        busorc, businstrs = engineorc.makeBusOrc(numAudioBuses=self.numAudioBuses,
                                                 numControlBuses=self.numControlBuses,
                                                 startInstr=startInstr,
                                                 postInstr=postInstrnum)
        self._builtinInstrs.update(businstrs)
        self._reservedInstrnumRanges.append(('busorc', min(businstrs.values()), max(businstrs.values())))
        self._hasBusSupport = True
        self.numAudioBuses = numAudioBuses
        self.numControlBuses = numControlBuses
        chanptr, error = self.csound.channelPtr("_busTokenCount", kind='control', mode='rw')
        if error:
            raise RuntimeError(f"Error in csound.channelPtr: {error}")
        assert isinstance(chanptr, np.ndarray)
        self._busTokenCountPtr = chanptr
        self._compile(busorc)
        kbustable = int(self.csound.evalCode("return gi__bustable"))
        self._kbusTable = self.getTableData(kbustable)

    def hasBusSupport(self) -> bool:
        """
        Returns True if this Engine was started with bus support

        .. seealso::

            :meth:`~csoundengine.engine.Engine.assignBus`
            :meth:`~csoundengine.engine.Engine.writeBus`
            :meth:`~csoundengine.engine.Engine.readBus`
        """
        return self._hasBusSupport and (self.numAudioBuses > 0 or self.numControlBuses > 0)

    def compile(self, code: str) -> None:
        """
        Send orchestra code to csound

        The code sent can be any orchestra code

        Args:
            code: the code to compile
        """
        if not self.csound:
            raise RuntimeError("This OfflineEngine does not have an associated csound process")

        codeblocks = csoundparse.parseOrc(code)
        for codeblock in codeblocks:
            if codeblock.kind == 'instr':
                parsedbody = csoundparse.instrParseBody(csoundparse.instrGetBody(codeblock.lines))
                self._parsedInstrs[codeblock.name] = parsedbody
                if codeblock.name[0].isdigit():
                    instrnum = int(codeblock.name)
                    for rangename, mininstr, maxinstr in self._reservedInstrnumRanges:
                        if mininstr <= instrnum < maxinstr:
                            logger.error(f"Instrument number {instrnum} is reserved. Code:")
                            logger.error("\n" + textwrap.indent(codeblock.text, "    "))
                            raise ValueError(f"Cannot use instrument number {instrnum}, "
                                             f"the range {mininstr} - {maxinstr} is reserved for '{rangename}'")
                    if instrnum in self._reservedInstrnums:
                        raise ValueError("Cannot compile instrument with number "
                                         f"{instrnum}: this is a reserved instr and "
                                         f"cannot be redefined. Reserved instrs: "
                                         f"{sorted(self._reservedInstrnums)}")

        self.csound.compileOrc(code)
        self._shouldPerform = True
        self._addHistory(_CompileEvent(code))

    def _addHistory(self, component: _OfflineComponent) -> None:
        if self._trackHistory:
            self._history.append(component)

    @cache
    def instrNum(self, name: str) -> int:
        assert not self._stopped
        return int(self.csound.evalCode(f'return nstrnum("{name}")'))

    def assignInstanceNum(self, instr: int | str) -> int:
        if isinstance(instr, str):
            instr = self.instrNum(instr)
        c = self._instanceCounters.get(instr, 0) + 1
        self._instanceCounters[instr] = c
        instancenum = (c % int(10 ** self._fracnumdigits - 2)) + 1
        return instancenum

    def assignEventId(self, instr: int | str) -> float:
        """
        Assign a unique p1 value for the given instrument number

        This is used internally to assign a fractional p1 when
        called sched with unique=True, but it can also be used
        to generate unique p1 for :meth:`OfflineEngine.inputMessage`
        or when using csound's ``schedule`` with :meth:`OfflineEngine.compile`

        Args:
            instr: the instrument number

        Returns:
            a unique fractional p1
        """
        instancenum = self.assignInstanceNum(instr=instr)
        frac = (instancenum / (10 ** self._fracnumdigits)) % 1
        return (instr if isinstance(instr, int) else self.instrNum(instr)) + frac

    def sched(self,
              instr: int | float | str,
              delay=0.,
              dur=-1.,
              *pfields,
              args: np.ndarray | list[float | str] | None = None,
              unique=False,
              comment='',
              relative=True,
              **namedpfields
              ) -> float | str:
        """
        Schedule an instrument

        Args:
            instr : the instrument number/name. If it is a fractional number,
                that value will be used as the instance number.
                An integer or a string will result in a unique instance assigned
                by csound if unique is True. Named instruments
                with a fractional number can also be scheduled (for example,
                for an instrument named "myinstr" you canuse "myinstr.001")
            delay: start time, relative to the elapsed time. (see :attr:`OfflineEngine.now`)
            dur: duration of the event
            args: any other args expected by the instrument, starting with p4
                (as a list of floats/strings, or a numpy array). Any
                string arguments will be converted to a string index via strSet. These
                can be retrieved in csound via strget
            unique: if True, assign a unique p1
            namedpfields: pfields can be given as keyword arguments of the form p4=..., p6=...
                Defaults are filled with values defined via ``pset``

        Returns:
            a fractional p1 of the instr started, which identifies this event.
            If instr is a fractional named instr, like "synth.01", then this
            same instr is returned as eventid (as a string).

        Example
        -------

        .. code-block :: python

            from csoundengine import *
            e = Engine()
            e.compile(r'''
              instr 100
                pset 0, 0, 0, 1000, 2000, 0, 9.0
                kfreq = p4
                kcutoff = p5
                Smode strget p6
                iq = p7
                asig vco2 0.1, kfreq
                if strcmp(Smode, "lowpass") == 0 then
                  asig moogladder2 asig, kcutoff, 0.95
                else
                  asig K35_hpf asig, kcutoff, iq
                endif
                outch 1, asig
              endin
            ''')
            eventid = e.sched(100, 2, args=[200, 400, "lowpass"])
            # simple automation in python
            for cutoff in range(400, 3000, 10):
                e.setp(eventid, 5, cutoff)
                time.sleep(0.01)
            e.unsched(eventid)
            #
            # To ensure simultaneity between events:
            now = e.elapsedTime()
            for t in np.arange(2, 4, 0.2):
                e.sched(100, t+now, 0.2, relative=False, p7=9.8)

        .. seealso:: :meth:`~csoundengine.engine.Engine.unschedAll`
        """
        if not self.csound:
            raise RuntimeError("This OfflineEngine has no associated csound process")

        if pfields and args:
            raise ValueError("Either pfields as positional arguments or args can be given, "
                             "got both")
        elif pfields:
            args = pfields

        if namedpfields:
            instrdef = self._parsedInstrs.get(str(int(instr)) if isinstance(instr, (int, float)) else instr)
            if instrdef:
                kwargs = csoundparse.normalizeNamedPfields(namedpfields, instrdef.pfieldNameToIndex)
            else:
                assert all(csoundparse.isPfield(key) for key in namedpfields)
                kwargs = {int(key[1:]):value for key, value in namedpfields.items()}
            args = csoundparse.fillPfields(args, kwargs, defaults=instrdef.pfieldIndexToValue if instrdef else None)

        if unique:
            if isinstance(instr, str):
                p1 = self.assignEventId(instr)
                frac = round(math.modf(p1)[0], self._fracnumdigits)
                instr = instr + '.' + str(frac).split('.')[1]
            elif isinstance(instr, int):
                instr = p1 = self.assignEventId(instr)
            elif isinstance(instr, float) and int(instr) == instr:
                instr = p1 = self.assignEventId(int(instr))
            else:
                raise TypeError(f"Expected an instrument number of name, got {instr}")
        else:
            p1 = instr if not isinstance(instr, str) else self.instrNum(instr)

        now = self.elapsedTime()
        if not relative:
            # We always convert to relative
            delay = delay - now
            if delay < 0:
                raise ValueError(f"The time offset is in the pase, elapsed time: {now}, absolute offset given: {delay + now}")

        if dur >= 0 and delay + dur > self._endtime:
            self._endtime = delay + dur
        elif delay > self._endtime:
            self._endtime = delay

        if isinstance(args, np.ndarray):
            pfields = np.empty((len(args) + 3,), dtype=float)
            pfields[0] = p1
            pfields[1] = delay
            pfields[2] = dur
            pfields[3:] = args
        else:
            pfields = [p1, delay, dur]
            if args:
                with self.nohistory():
                    # do not keep track of strsets for scheduling events in the history
                    # since these end up in the score and csound does the string handling for us
                    pfields.extend(float(a) if not isinstance(a, str) else self.strSet(a) for a in args)
        self.csound.scoreEvent("i", pfields=pfields)
        # self.csound.scoreEventAbsolute(type_='i', pFields=pfields, timeOffset=timeOffset)
        self._addHistory(_SchedEvent(instr=instr, delay=delay+now, dur=dur, args=args,
                                     eventid=p1, comment=comment))

        self._shouldPerform = True
        return instr

    def unsched(self, p1: float | str, delay: float = 0) -> None:
        """
        Stop a playing event

        If p1 is a round number, all events with the given number
        are unscheduled. Otherwise only an exact matching event
        is unscheduled, if it exists

        Args:
            p1: the instrument number/name to stop
            delay: absolute time to turnoff the given event. A value of 0 means to
                unschedule it at the current time

        Example
        ~~~~~~~

        >>> from csoundengine import *
        >>> e = Engine(...)
        >>> e.compile(r'''
        ... instr sine
        ...   a0 oscili 0.1, 1000
        ...   outch 1, a0
        ... endin
        ... ''')
        >>> # sched an event with indefinite duration
        >>> eventid = e.sched(10, 0, -1)
        >>> e.unsched(eventid, 10)

        .. seealso:: :meth:`~Engine.unschedAll`

        """
        if (isinstance(p1, int) and int(p1) != p1) or (isinstance(p1, str) and "." in p1):
            mode = 4
        else:
            mode = 0
        self.sched(self._builtinInstrs['turnoff'], delay, 0, p1, mode)

    @contextlib.contextmanager
    def nohistory(self):
        """
        A context manager to suppress tracking history
        """
        try:
            self._trackHistory = False
            yield self
        finally:
            self._trackHistory = True

    def setEndMarker(self, time: float) -> None:
        """
        Set the end marker

        When calling :meth:`OfflineEngine.perform`, performance
        will be advanced up to this absolute time

        The current end time can be queried via the :attr:`OfflineEngine.endtime`
        attribute
        """
        self._endtime = time

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._endtime and self._shouldPerform:
            self.perform()

    def perform(self, endtime=0., extratime=0.) -> None:
        """
        Advance the time, performing any scheduled events within the current time and the end time

        By default, perform until the end marker. The end marker is set by the last
        end time of an event. **NB**: Events with infinite duration do not modify the
        end marker. Use ``extratime`` to assign rendering time for such events.

        This method is called implicitely at object destruction. To prevent implicit rendering,
        call :meth:`OfflineEngine.stop` with ``renderPending=False``

        The rendering process is finished when :meth:`OfflineEngine.stop` is called, which
        happens implicitely at object destrution. If the rendered audiofile needs to be accessed
        immediately, then :meth:`OfflineEngine.stop` needs to be called explicitely

        .. note::
            :meth:`~OfflineEngine.stop` will call :meth:`~OfflineEngine.perform` if there
            are any pending events, so in order to render a process to completion, only
            :meth:`~OfflineEngine.stop` needs to be called

        Args:
            endtime: if given, overrides the endtime of the engine
            extratime: time added to the current end marker, it is not taken into account
                if endtime is given

        Example
        ~~~~~~~

        .. code-block :: python

            from csoundengine import OfflineEngine
            import sndfileio
            e = OfflineEngine()
            e.compile(r'''
            instr pitchtrack
              itabnum = p4   ; the sound source
              itabfreq = p5  ; the table where to put tracked pitch. The table must be big enough
              itabsize = ftlen(itabfreq)
              ifftsize = 2048
              ifftsize2 = ifftsize * 4  ; this second anaylsis helps smooth the result
              iwtype = 0        ; hamming
              kcount init 1     ; the first element is the element count
              aouts[] loscilx 1, 1, itabnum, 4, 1
              a0 = aouts[0]     ; only analyze first channel
              a0d = delay(a0, ifftsize2 * 0.5 / sr)  ; shift the original signal
              fsig  pvsanal a0d, ifftsize, 512, ifftsize, iwtype
              fsig2 pvsanal a0, ifftsize2, 512, ifftsize2, iwtype
              kfreq, kamp pvspitch fsig, 0.1
              kfreq2, kamp2 pvspitch fsig2, 0.08
              if kcount >= itabsize || detectsilence(a0, db(-90), 0.1) == 1 then
                turnoff
              endif
              if (kfreq2 == 0 ? 1 : kfreq / kfreq2) < 0.5 then
                kfreq = kfreq2
              endif
              tabw kfreq, kcount, itabfreq
              kcount += 1
              tabw kcount, 0, itabfreq
              ; To validate the analysis, we output the original sound and the resynthesized audio
              outs delay(a0, ifftsize + ifftsize2*0.5), vco2(kamp, kfreq)
            endin
            ''')

            sndtab = e.readSoundfile(sndfile)
            duration = sndfileio.info(sndfile).duration
            numcycles = int((duration * e.sr / e.ksmps)
            freqtab = e.makeEmptyTable(numcycles + 100)
            e.sched('pitchtrack', 0, duration + 0.1, sndtab, freqtab)
            # Advance time until analysis has been done
            e.perform()
            # Now retrieve information
            outarr = e.getTableData(freqtab)
            datalen = outarr[0]
            freqs = outarr[1:1+datalen]
            # At any moment the user can access the generated soundfile via the
            # renderjob variable, even if the engine is still active.
            e.renderjob.openOutfile()
            # Stop the engine to clean up
            e.stop()

        """
        assert not self._stopped
        endtime = (endtime or self._endtime) + extratime
        if endtime == 0.:
            logger.debug("The render time is 0, nothing to perform.")
            return

        if self._shouldPerform:
            if endtime > self._endtime:
                self._endtime = endtime
            maxsamples = int(endtime * self.sr)
            cs = self.csound
            while not cs.performKsmps() and cs.currentTimeSamples() < maxsamples:
                pass
            self._shouldPerform = False
        else:
            logger.debug("Nothing to perform")

    @property
    def renderjob(self) -> RenderJob | None:
        if not self._renderjob:
            info = tools.sndfileInfo(self.outfile)
            if info.duration == 0:
                return None
            renderjob = RenderJob(outfile=self.outfile, samplerate=self.sr, encoding=self.encoding)
            self._renderjob = renderjob
        return self._renderjob

    def cancel(self, remove=True) -> None:
        """
        Cancel performance of this engine

        Args:
            remove: if True, remove the generated outfile
        """
        if self._stopped:
            logger.info("This engine is already stopped")
            return
        self.csound.stop()
        self.csound.cleanup()
        if remove and os.path.exists(self.outfile):
            logger.debug(f"Removing outfile '{self.outfile}'")
            os.remove(self.outfile)

    def stop(self, render=True, extratime=0.) -> RenderJob | None:
        """
        Stop this csound process, optionally rendering any pending events

        Args:
            render: if True, advance the clock to render any pending scheduled
                event.
            extratime: if render is True, add extra render time to allow for
                decays

        Returns:
            a RenderJob containing information about the rendered file, or None
            if no render took place.
        """
        if self._stopped:
            return None
        if self._shouldPerform and render:
            self.perform(extratime=extratime)
        self.csound.stop()
        self.csound.reset()
        self._stopped = True
        if not os.path.exists(self.outfile):
            raise RuntimeError(f"Did not find rendered file '{self.outfile}'")

        info = tools.sndfileInfo(self.outfile)
        if info.duration > 0:
            renderjob = RenderJob(outfile=self.outfile, samplerate=self.sr, encoding=self.encoding)
            self._renderjob = renderjob
            return renderjob
        else:
            return None

    def __del__(self):
        if self.csound and self._shouldPerform:
            self.stop(render=True)

    def strSet(self, s: str) -> int:
        """
        Assign a numeric index to a string to be used inside csound

        Args:
            s: the string to set

        Returns:
            the index associated with ``s``. When passed to a csound instrument
            it can be retrieved via ``strget``.
        """
        stringIndex = self._strToIndex.get(s)
        if stringIndex:
            return stringIndex
        stringIndex = self._strLastIndex
        self._strLastIndex += 1
        self._strToIndex[s] = stringIndex
        self._indexToStr[stringIndex] = s
        self.compile(fr'strset {stringIndex}, "{s}"')
        return stringIndex

    def inputMessage(self, msg: str) -> None:
        """
        Schedule a score line

        At the moment, due to a limitation in the csound scheduler input
        messages are only scheduled at the beginning of the performance.
        Any meesage scheduled after :meth:`OfflineEngine.perform` has
        been called will be ignored

        Args:
            msg: the message in score format (``'i1 0.5 3 ...'``)

        Returns:
            the p1, will be 0 if a named instrument is used
        """
        if not self.csound:
            raise RuntimeError("This OfflineEngine does not have an associated csound process")
        if self.elapsedTime() > 0:
            logger.warning(f"Performance has already been started (elapsed time: {self.elapsedTime()}, "
                           f"this message ('{msg}') will probably be ignored.")
        self.csound.inputMessage(msg)
        parts = csoundparse.splitScoreLine(msg, quote=False)
        kind = parts[0]
        start = parts[2]
        dur = parts[3]
        assert isinstance(start, (int, float)), f"Invalid start time, got {start}"
        assert isinstance(dur, (int, float)), f"Invalid duration, got {dur}"
        if dur > 0:
            end = start + dur
            if end > self._endtime:
                self.setEndMarker(end)
        assert isinstance(kind, str) and kind in 'ife'
        self._addHistory(_ScoreEvent(kind=kind, pfields=parts[1:]))

    def _assignTableNumber(self) -> int:
        """
        Return a free table number and mark that as being used.
        To release the table, call unassignTable

        Returns:
            the table number (an integer)
        """
        self._tableCounter += 1
        return self._tableCounter - 1

    def readSoundfile(self, path: str, tabnum=0, chan=0, skiptime=0., delay=0., unique=True
                      ) -> int:
        """
        Read a soundfile into a table (via GEN1), returns the table number

        Args:
            path: the path to the output -- **"?" to open file interactively**
            tabnum: if given, a table index. If None, an index is
                autoassigned
            chan: the channel to read. 0=read all channels
            skiptime: time to skip at the beginning of the soundfile.
            delay: performance time at which to read the soundfile
            unique: if False and the same file with the same params is already loaded,
                the existing table is returned

        Returns:
            the table number. Information about the read soundfile can be obtained
            via :meth:`OfflineEngine.tableInfo`

        """
        assert not self._stopped
        if not unique and tabnum == 0:
            if existingtab := self._soundfilesLoaded.get((path, chan, skiptime)):
                return existingtab
        if not tabnum:
            tabnum = self._assignTableNumber()
        self._tableInfo[tabnum] = TableInfo.get(path)
        self._soundfilesLoaded[(path, chan, skiptime)] = tabnum

        self.csound.compileOrc(fr'itab ftgen {tabnum}, {delay}, 0, -1, "{path}", {skiptime}, 0, {chan}')
        self._addHistory(_ReadSoundfileEvent(path=path, tabnum=tabnum, delay=delay, skiptime=skiptime, chan=chan))
        return tabnum

    def queryVariable(self, variable: str) -> float:
        """
        Query the value of a csound variable via the ``return`` opcode

        Example
        ~~~~~~~

            >>> engine.compile(r'''
            ... gkfoo init 0
            ... instr 100
            ...   gkfoo = p4
            ... endin
            ... ''')
            >>> engine.sched(100, 0, 0, 314)
            >>> engine.perform()
            >>> engine.queryVariable('gkfoo')
            314.0

        """
        assert not self._stopped
        return self.csound.evalCode(fr'return {variable}\n')

    def call(self, func: str) -> float:
        """
        Call an init-time csound function, returns the result

        Args:
            func: the function to call, like 'ftgen(200, 0, 100, 2, 0)'. It must be
                an init-time function returning one init value

        Returns:
            the value returned by the function

        Example
        ~~~~~~~

            # Create an empty table manually with size 100
            >>> engine.compile(r'i0 = ftgen(200, 0, 100, 2, 0)')
            # Check that the size is correct
            >>> engine.call('ftlen(200)')
            100

        """
        assert not self._stopped
        return self.csound.evalCode(fr'return {func}')

    def tableInfo(self, tabnum: int, cache=True) -> TableInfo:
        info = self._tableInfo.get(tabnum)
        if info and cache:
            return info
        assert not self._stopped
        sr = self.csound.evalCode(f'return ftsr({tabnum})')
        numchannels = self.csound.evalCode(f'return ftchnls({tabnum})')
        tablen = self.csound.evalCode(f'return ftlen({tabnum}')
        return TableInfo(sr=int(sr), size=int(tablen), nchnls=int(numchannels))

    def makeEmptyTable(self, size: int, numchannels=1, sr=0, delay=0.) -> int:
        """
        Create an empty table, returns the index of the created table

        Args:
            size: the size of the table
            numchannels: if the table will be used to hold audio, the
                number of channels of the audio
            sr: the samplerate of the audio, if the table is used to hold audio
            delay: when to create the table

        Returns:
            the table number
        """
        tabnum = self._assignTableNumber()
        self._tableInfo[tabnum] = TableInfo(sr=sr, size=size, nchnls=numchannels)
        self.compile(f'itab ftgen {tabnum}, {delay}, {-size}, -2, 0')
        if sr > 0:
            self.sched(instr=self._builtinInstrs['ftsetparams'], delay=delay, dur=0,
                       args=[tabnum, sr, numchannels])
        return tabnum

    def getTableData(self, idx: int) -> np.ndarray:
        assert not self._stopped
        arr = self.csound.table(idx)
        if arr is None:
            raise ValueError(f"Table {idx} does not exist")
        return arr

    def channelPointer(self, channel: str, kind='control', mode='rw') -> np.ndarray:
        if kind != 'control' and kind != 'audio':
            raise NotImplementedError("Only kind 'control' and 'audio' are implemented "
                                      "at the moment")
        assert not self._stopped
        ptr, err = self.csound.channelPtr(channel, kind, mode)
        if err:
            raise RuntimeError(f"Error while trying to retrieve/create a channel pointer: {err}")
        assert isinstance(ptr, np.ndarray)
        return ptr

    def initChannel(self,
                    channel: str,
                    value: float | str | np.ndarray = 0,
                    kind='',
                    mode="r") -> None:
        modei = {"r": 1, "w": 2, "rw": 3, "wr": 3}[mode]
        if not kind:
            if isinstance(value, (int, float)):
                kind = 'k'
            elif isinstance(value, str):
                kind = 'S'
            elif isinstance(value, np.ndarray):
                kind = 'a'
        if kind == 'k':
            self.compile(f'chn_k "{channel}", {modei}\n')
            self.setChannel(channel, value)
        elif kind == 'a':
            self.compile(f'chn_a "{channel}", {modei}')
            if value:
                self.setChannel(channel, value)
        elif kind == 'S':
            self.compile(f'chn_S "{channel}", {modei}\n')
            self.setChannel(channel, value)
        else:
            raise TypeError(f"Expected an initial value of type float or string, got {value}")

    def getControlChannel(self, channel: str) -> float:
        assert not self._stopped
        value, err = self.csound.controlChannel(channel)
        if err != 0:
            raise KeyError(f"Control channel '{channel}' not found, error: {err}, value: {value}")
        return value

    def setChannel(self, channel: str, value: float | str | np.ndarray, delay=0.):
        assert not self._stopped
        if delay == 0.:
            if isinstance(value, (int, float)):
                self.csound.setControlChannel(channel, value)
            elif isinstance(value, str):
                self.csound.setStringChannel(channel, value)
            else:
                self.csound.setAudioChannel(channel, value)
        else:
            if isinstance(value, (int, float)):
                instrnum = self._builtinInstrs['chnset']
                self.sched(instrnum, delay, 0, args=[channel, value])
            elif isinstance(value, str):
                instrnum = self._builtinInstrs['chnsets']
                self.sched(instrnum, delay, 0, args=[channel, value])
            else:
                raise TypeError(f"Expected a number or a str as value, got {value}")

    def fillTable(self, tabnum: int, data: np.ndarray | Sequence[float]) -> None:
        """
        Fill an existing table with data

        If data is bigger than the table itself, then only the
        data which fits in the table is copied. If, on the
        contrary, the data is smaller than the table, then
        the rest of the table is left unmodified. For
        more control considere using :meth:`~OfflineEngine.getTableData`

        Args:
            tabnum: the table number of an already existing table
            data: the data to put into the table

        """
        tablearray = self.getTableData(tabnum)
        maxidx = min(len(tablearray), len(data))
        tablearray[:maxidx] = data[:maxidx]

    def makeTable(self,
                  data: np.ndarray | Sequence[float],
                  sr: int = 0,
                  tabnum: int = -1,
                  delay=0.
                  ) -> int:
        """
        Create a new table and fill it with data.

        Args:
            data: the data used to fill the table
            tabnum: the table number. If -1, a number is assigned by the engine.
                If 0, a number is assigned by csound (this operation will be blocking
                if no callback was given)
            sr: only needed if filling sample data. If given, it is used to fill the
                table metadata in csound, as if this table had been read via gen01
            delay: when to allocate the table

        Returns:
            the index of the new table
        """
        if not self.csound:
            raise RuntimeError("This OfflineEngine does not have an associated csound process")

        if tabnum == -1:
            tabnum = self._assignTableNumber()
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        numchannels = 1 if len(data.shape) == 1 else data.shape[1]
        flatdata = data if numchannels == 1 else data.flatten()
        arr = np.zeros((len(flatdata)+4,), dtype=float)
        if delay > 0 and tabnum == 0:
            raise ValueError("Cannot schedule a table in the future without assigning a "
                             "table number. Set tabnum=-1 to autoassign a number")
        if delay > 0 and sr == 0 and numchannels == 1:
            arr[0:4] = [tabnum, delay, len(flatdata), -2]
            arr[4:] = flatdata
            self.csound.scoreEvent("f", arr)
        else:
            if delay > 0:
                logger.warning("delay will be ignored")

            if tabnum == 0:
                tabnum = int(self.csound.evalCode(fr'return ftgen(0, 0, {len(flatdata)}, -2, 0)'))
            else:
                self.csound.compileOrc(fr'i0_ ftgen {tabnum}, 0, {len(flatdata)}, -2, 0')
            tabptr = self.csound.table(tabnum)
            assert tabptr is not None
            tabptr[:] = flatdata
            if sr > 0 or numchannels > 0:
                self.csound.compileOrc(fr'ftsetparams {tabnum}, {sr}, {numchannels}')

        self._tableInfo[tabnum] = TableInfo(sr=sr, size=len(flatdata), nchnls=numchannels)
        self._addHistory(_TableDataEvent(data=data, delay=delay, tabnum=tabnum, sr=sr))
        return int(tabnum)

    def freeTable(self, tableindex: int, delay=0.) -> None:
        self.sched(self._builtinInstrs['freetable'], delay, 0., tableindex)

    @property
    def now(self) -> float:
        """
        The current elapsed time

        This is the same as :meth:`OfflineEngine.elapsedTime`
        """
        return self.elapsedTime()

    def elapsedTime(self) -> float:
        """
        Reports the logical elapsed time of this engine
        """
        return self.csound.currentTimeSamples() / self.sr

    def playSample(self, tabnum: int, delay=0., chan=1, speed=1., gain=1., fade=0.,
                   starttime=0., lagtime=0.01, dur=-1.
                   ) -> float:
        """
        Play a sample already loaded into a table.

        Speed and gain can be modified via setp while playing

        Args:
            tabnum: the table where the sample data was loaded
            delay: when to start playback
            chan: the first channel to send output to (channels start with 1)
            speed: the playback speed
            gain: a gain applied to this sample
            fade: fadein/fadeout time in seconds
            starttime: playback can be started from anywhere within the table
            lagtime: a lag value for dynamic pfields (see below)
            dur: the duration of playback. Use -1 to play until the end

        Returns:
            the instance number of the playing instrument.

        .. admonition:: Dynamic Fields
            :class: important

            - **p4**: `gain`
            - **p5**: `speed`

        Example
        ~~~~~~~

            >>> from csoundengine import *
            >>> e = Engine()
            >>> import sndfileio
            >>> sample, sr = sndfileio.sndread("stereo.wav")
            >>> # modify the samples in python
            >>> sample *= 0.5
            >>> table = e.makeTable(sample, sr=sr, block=True)
            >>> eventid = e.playSample(table)
            ... # gain (p4) and speed (p5) can be modified while playing
            >>> e.setp(eventid, 5, 0.5)   # Play at half speed

        """
        info = self._tableInfo.get(tabnum)
        if not info:
            raise ValueError(f"Invalid table number {tabnum}, available tables "
                             f"are {self._tableInfo.keys()}")
        if dur < 0:
            if info.sr == 0:
                sr = self.sr
            else:
                sr = self.sr
            sampledur = info.numFrames / sr
            estimatedDuration = sampledur / speed
            endtime = delay + estimatedDuration
            if self._endtime < endtime:
                self._endtime = endtime
        args = [gain, speed, tabnum, chan, fade, starttime, lagtime]
        eventid = self.sched(self._builtinInstrs['playgen1'], delay=delay, dur=dur,
                             args=args, unique=True)
        assert isinstance(eventid, (int, float))
        return eventid

    def automatep(self,
                  p1: float | str,
                  pfield: int | str,
                  pairs: Sequence[float] | np.ndarray,
                  mode='linear',
                  delay=0.,
                  overtake=False
                  ) -> float:
        """
        Automate a pfield of a scheduled event

        Args:
            p1: the fractional instr number of a running event, or an int number
                to modify all running instances of that instr
            pfield: the pfield index. For example, if the pfield to modify if p4,
                pidx should be 4. Values of 1, 2, and 3 are not allowed.
            pairs: the automation data is given as a flat data. of pairs (time, value).
                Times are relative to the start of the automation event
            mode: one of 'linear', 'cos', 'expon(xx)', 'smooth'. See the csound opcode
                `interp1d` for more information
                (https://csound-plugins.github.io/csound-plugins/opcodes/interp1d.html)
            delay: the time delay to start the automation, relative to the elapsed time
            overtake: if True, the first value of pairs is replaced with
                the current value in the running instance

        Returns:
            the p1 associated with the automation synth

        Example
        ~~~~~~~

        >>> e = OfflineEngine()
        >>> e.compile(r'''
        ... instr 100
        ...   kfreq = p4
        ...   outch 1, oscili:a(0.1, kfreq)
        ... endin
        ... ''')
        >>> eventid = e.sched(100, 0, 10, args=(1000,))
        >>> e.automatep(eventid, 4, [0, 1000, 3, 200, 5, 200])

        .. seealso:: :meth:`~Engine.setp`, :meth:`~Engine.automateTable`
        """
        if math.isnan(pairs[1]):
            overtake = True

        if len(pairs) % 2 != 0:
            raise ValueError(f"Pairs needs to be a flat sequence of floats with the form"
                             f" t0, value0, t1, value1, ..., with an even number of "
                             f"total elements. Got {len(pairs)} items: {pairs}")
        if isinstance(pfield, str):
            instrdef = self._parsedInstrs.get(str(int(p1)) if isinstance(p1, (int, float)) else p1.split(".")[0])
            if not instrdef:
                raise ValueError(f"Could not find definition for instr '{p1}'")
            pidx = instrdef.pfieldNameToIndex.get(pfield)
            if pidx is None:
                raise ValueError(f"Invalid pfield name '{pfield}'. Valid pfields: {instrdef.pfieldNameToIndex.keys()}")
        else:
            pidx = pfield

        self._addHistory(_AutomateEvent(p1, pfield=pfield, pairs=pairs, delay=delay, overtake=overtake))

        if self.version >= 7000 or len(pairs) <= 1900:
            if isinstance(pairs, np.ndarray):
                pairs = pairs.tolist()
            p1IsString = int(isinstance(p1, str))
            args = [p1, pidx, mode, int(overtake), len(pairs), p1IsString, *pairs]
            eventid = self.sched(self._builtinInstrs['automatePargViaPargs'],
                                 delay=delay,
                                 dur=pairs[-2] + self.ksmps / self.sr,
                                 args=args)
            assert isinstance(eventid, (int, float))
            return eventid
        else:
            if isinstance(pairs, np.ndarray):
                pairs = list(pairs)
            events = [self.automatep(p1=p1, pfield=pfield, pairs=subgroup, mode=mode, delay=delay + subdelay,
                                     overtake=overtake)
                      for subdelay, subgroup in internal.splitAutomation(pairs, 1900 // 2)]
            return events[0]

    def renderHistory(self, outfile='') -> RenderJob:
        csd = self.generateCsd()
        return csd.render(outfile=outfile)

    def rewind(self, offset=0.) -> None:
        """
        Unschedule future events and set the time pointer to the given offset
        """
        if offset > 0:
            self.csound.setScoreOffsetSeconds(offset)
        self.csound.rewindScore()

    def generateCsd(self) -> Csd:
        """
        Generate a :class:`~csoundengine.csd.Csd` from this OfflineEngine

        This can be used to export a project file and use the csound
        binary to render it. It might also be useful for debugging

        Returns:
            the generated :class:`~csoundengine.csd.Csd`
        """
        from csoundengine.csd import Csd
        csd = Csd(sr=self.sr, ksmps=self.ksmps, nchnls=self.nchnls, a4=self.a4,
                  options=self.options, nchnls_i=0)
        csd.setSampleEncoding(self.encoding)
        for event in self._history:
            event.apply(csd)
        return csd

    def write(self, outfile: str) -> None:
        """
        Dump this OfflineEngine as a .csd

        This will include all code, instruments and opcodes compiled until now
        as well as any events scheduled, etc.

        Any data files added are written to a folder ``'<csdfile>.assets'`` besides the
        generated .csd file.

        Args:
            outfile: the path of the generated .csd file
        """
        self.generateCsd().write(outfile)

    def includeFile(self, include: str) -> None:
        """
        Adds an #include file to this engine and evaluates the file contents

        Args:
            include: the path to the file to be included and evaluated
        """
        abspath = os.path.abspath(include)
        for f in self.includes:
            if abspath == f:
                return
        self.includes.append(abspath)
        code = open(abspath).read()
        self.compile(code)

    def playSoundFromDisk(self, path: str, delay=0., dur=-1, chan=0, speed=1., fade=0.01
                          ) -> float:
        """
        Play a soundfile from disk via diskin2

        Args:
            path: the path to the output
            delay: time offset to start playing
            dur: duration of playback, use -1 to play until the
                end of the file
            chan: first channel to output to
            speed: playback speed (2.0 will sound an octave higher)
            fade: fadein/out in seconds

        Returns:
            the instance number of the scheduled event

        .. seealso::
            * :meth:`~OfflineEngine.readSoundfile`
            * :meth:`~OfflineEngine.playSample`

        """
        assert not self._stopped
        if dur < 0:
            info = tools.sndfileInfo(path)
            sampledur = info.duration
            estimatedDuration = sampledur / speed
            endtime = delay + estimatedDuration
            if self._endtime < endtime:
                self._endtime = endtime

        eventid = self.sched(instr=self._builtinInstrs['playsndfile'],
                             delay=delay,
                             dur=dur,
                             args=[path, chan, speed, fade],
                             unique=True)
        assert isinstance(eventid, float)
        return eventid

    def setp(self, p1: int | float | str, *pairs, delay=0.) -> None:
        """
        Modify a pfield of an active note

        Multiple pfields can be modified simultaneously. It only makes sense to
        modify a pfield if a control-rate (k) variable was assigned to this pfield
        (see example)

        Args:
            p1: the p1 of the instrument to automate. A float or a "<name>.<instanceid>" will set
                the value for a specific instance, an int or a unqualified name will set
                the value of the given parameter for all instances
            *pairs: each pair consists of a pfield index and a value. The index is an int,
                matching the pfield number (4=p4, 5=p5, etc), the value can be a number
                (string values are not supported)
            delay: when to start the automation

        .. rubric:: Example

        .. code-block:: python

            >>> engine = OfflineEngine(...)
            >>> engine.compile(r'''
            ... instr foo
            ...   kamp = p5
            ...   kfreq = p6
            ...   a0 oscili kamp, kfreq
            ...   outch 1, a0
            ... endin
            ... ''')
            >>> p1 = engine.sched('foo', args=[0.1, 440], unique=True)
            >>> p1
            'foo.0001'
            >>> engine.setp(p1, 5, 0.2, delay=0.5)
        """
        numpairs = len(pairs) // 2
        if len(pairs) % 2 == 1:
            raise ValueError(f"Pairs needs to be even, got {pairs}")
        if numpairs > 5:
            # split and schedule the parts
            # TODO
            raise ValueError(f"Only up to 5 pairs supported, got {pairs}")
        args = [1 if isinstance(p1, str) else 0, p1, numpairs]
        args.extend(pairs)
        instr = self._builtinInstrs['pwrite']
        self.sched(instr, delay=delay, dur=0, args=args)
        for pair in range(numpairs):
            self._addHistory(_SetParamEvent(p1=p1, pindex=pairs[pair*2], value=pairs[pair*2+1], delay=delay))

    def _getBusIndex(self, bus: int) -> int | None:
        bus = int(bus)
        if (index := self._busIndexes.get(bus)) is not None:
            return index
        busindex = int(self.csound.evalCode(f'return dict_get:i(gi__bustoken2num, {bus})'))
        if busindex < 0:
            return None
        self._busIndexes[bus] = busindex
        return busindex

    def assignBus(self,
                  kind='',
                  value: float | None = None,
                  persist=False
                  ) -> int:
        """
        Assign one audio/control bus, returns the bus number.

        Audio buses are always mono.

        Args:
            kind: the kind of bus, "audio" or "control". If left unset and value
                is not given it defaults to an audio bus. Otherwise, if value
                is given a control bus is created. Explicitely asking for an
                audio bus and setting an initial value will raise an expection
            value: for control buses it is possible to set an initial value for
                the bus. If a value is given the bus is created as a control
                bus. For audio buses this should be left as None
            persist: if True the bus created is kept alive until the user
                calls :meth:`~Engine.releaseBus`. Otherwise, the bus is
                reference counted and is released after the last
                user releases it.

        Returns:
            the bus token, can be passed to any instrument expecting a bus
            to be used with the built-in opcodes :ref:`busin`, :ref:`busout`, etc.

        A bus created here can be used together with the built-in opcodes :ref:`busout`,
        :ref:`busin` and :ref:`busmix`. A bus can also be created directly in csound by
        calling :ref:`busassign`

        A non-persistent bus is reference counted: it is kept alive as long as there
        are clients using it and it is released when it is not used anymore. At
        creation the bus is "parked", waiting to be used by any client.
        As long as no clients use it, the bus stays in this state and is ready to
        be used. A persistent bus stays alive until it is freed via :meth:`OfflineEngine.releaseBus`

        Order of evaluation is important: **audio buses are cleared at the end of each
        performance cycle** and can only be used to communicate from a low
        priority to a high priority instrument.

        For more information, see :ref:`Bus Opcodes<busopcodes>`

        Example
        ~~~~~~~

        Pass audio from one instrument to another. The bus will be released after the events
        are finished.

        >>> e = OfflineEngine(...)
        >>> e.compile(r'''
        ... instr 100
        ...   ibus = p4
        ...   kfreq = 1000
        ...   asig vco2 0.1, kfreq
        ...   busout(ibus, asig)
        ... endin
        ... ''')
        >>> e.compile(r'''
        ... instr 110
        ...   ibus = p4
        ...   asig = busin(ibus)
        ...   ; do something with asig
        ...   asig *= 0.5
        ...   outch 1, asig
        ... endin
        ... ''')
        >>> bus = e.assignBus("audio")
        >>> s1 = e.sched(100, 0, 4, (bus,))
        >>> s2 = e.sched(110, 0, 4, (bus,))
        >>> e.perform()


        Modulate one instr with another, at k-rate. **NB: control buses act like global
        variables, the are not cleared at the end of each cycle**.

        >>> e = OfflineEngine(...)
        >>> e.compile(r'''
        ... instr 130
        ...   ibus = p4
        ...   ; lfo between -0.5 and 0 at 6 Hz
        ...   kvibr = linlin(lfo:k(1, 6), -0.5, 0, -1, 1)
        ...   busout(ibus, kvibr)
        ... endin
        ...
        ... instr 140
        ...   itranspbus = p4
        ...   kpitch = p5
        ...   ktransp = busin:k(itranspbus, 0)
        ...   kpitch += ktransp
        ...   asig vco2 0.1, mtof(kpitch)
        ...   outch 1, asig
        ... endin
        ... ''')
        >>> bus = e.assignBus()
        >>> s1 = e.sched(140, 0, -1, (bus, 67))
        >>> s2 = e.sched(130, 0, -1, (bus,))  # start moulation
        >>> e.unsched(s2)        # remove modulation
        >>> e.writeBus(bus, 0)   # reset value
        >>> e.unschedAll()

        .. seealso:: :meth:`~Engine.writeBus`, :meth:`~Engine.readBus`, :meth:`~Engine.releaseBus`

        """
        if not self.hasBusSupport():
            raise RuntimeError("This Engine was created without bus support")

        if kind:
            if value is not None and kind == 'audio':
                raise ValueError(f"You asked to assign an audio bus but gave an initial "
                                 f"value ({value})")
        else:
            kind = 'audio' if value is None else 'control'

        bustoken = int(self._busTokenCountPtr[0])
        assert isinstance(bustoken, int) and (kind == 'audio' or kind == 'control')
        self._busTokenToKind[bustoken] = kind
        ikind = BUSKIND_AUDIO if kind == 'audio' else BUSKIND_CONTROL
        ivalue = value if value is not None else 0.
        self._busTokenCountPtr[0] = bustoken + 1
        synctoken = 0
        pfields = [synctoken, bustoken, ikind, int(persist), ivalue]
        self.sched(self._builtinInstrs['busassign'], delay=self.elapsedTime(), dur=0, args=pfields)
        self._usesBuses = True
        return bustoken

    def releaseBus(self, bus: int, delay: float | None = None) -> None:
        """
        Release a persistent bus

        Args:
            bus: the bus to release, as returned by :meth:`OfflineEngine.assignBus`
            delay: when to release the bus. None means now

        .. seealso:: :meth:`~OfflineEngine.assignBus`
        """
        # bus is the bustoken
        if not self.hasBusSupport():
            raise RuntimeError("This OfflineEngine was created without bus support")
        now = self.elapsedTime()
        if delay is None:
            delay = now
        elif delay < now:
            raise ValueError(f"The delay given ({delay} lies in the past ({now=})")
        self.sched(self._builtinInstrs['busrelease'], delay, 0, int(bus))

    def writeBus(self, bus: int, value: float, delay=0.) -> None:
        if not self.hasBusSupport():
            raise RuntimeError("This engine does not have bus support")

        bus = int(bus)
        kind = self._busTokenToKind.get(bus)
        if not kind:
            logger.warning(f"Bus token {bus} not known")
        elif kind != 'control':
            raise ValueError(f"Only control buses can be written to, got {kind}")
        if delay == 0 and (busindex := self._getBusIndex(bus)) is not None:
            assert self._kbusTable is not None
            self._kbusTable[busindex] = value
        else:
             self.sched(self._builtinInstrs['busoutk'], delay=delay, dur=self.ksmps/self.sr*2, args=[bus, value])

    def openOutfile(self, app='') -> None:
        """
        Open the generated soundfile in an external app
        """
        if not self.renderjob:
            raise RuntimeError("No render job found")
        if not os.path.exists(self.renderjob.outfile):
            raise FileNotFoundError(f"The rendered outfile '{self.renderjob.outfile}' "
                                    f"does not exists")
        self.renderjob.openOutfile(app=app)
