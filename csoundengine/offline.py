
"""
This module implements offline rendering using the same
interface as a :class:`~csoundengine.session.Session`. Any realtime
code run via a :class:`~csoundengine.session.Session` can be rendered offline
by replacing the Session via a :class:`~csoundengine.offline.Renderer`

Example
=======

.. code-block:: python

    from csoundengine import *
    renderer = Renderer(sr=44100, nchnls=2)

    Instr('saw', r'''
      kmidi = p5
      outch 1, oscili:a(0.1, mtof:k(kfreq))
    ''').register(renderer)

    score = [('saw', 0,   2, 60),
             ('saw', 1.5, 4, 67),
             ('saw', 1.5, 4, 67.1)]
    events = [renderer.sched(ev[0], delay=ev[1], dur=ev[2], pargs=ev[3:])
              for ev in score]

    # offline events can be modified just like real-time events
    renderer.automatep(events[0], 'kmidi', pairs=[0, 60, 2, 59])
    renderer.setp(events[1], 3, 'kmidi', 67.2)
    renderer.render("out.wav")

"""

from __future__ import annotations
import os
import sys
from dataclasses import dataclass
from .config import config
from . import csoundlib
from .instr import Instr
from . import internalTools
from . import engineorc

from emlib import misc, iterlib
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING or "sphinx" in sys.modules:
    from typing import *
import textwrap as _textwrap



__all__ = ["Renderer", "ScoreEvent"]

@dataclass
class ScoreEvent:
    """
    A ScoreEvent represent a csound event. It is used by the
    offline renderer to keep track of scheduled events

    NB: instances of this class are **NOT** created by the used directly, they
    are generated when scheduling events

    Attributes:
        p1: a unique (fractional) instr number
        start: start time of this event (p2)
        dur: duration of this event (p3)
        args: rest of pargs, starting with p4
        eventId: a unique identifier for this event
        paramTable: if the instrument of this event has a parameters table,
            this attribute points to the table index (0 if no parameters table).
            Normally, if the inst has a parameters table the table index is
            passed as p4, so paramTable == args[0]
        renderer: the renderer which scheduled this score event (if any)
    """
    p1: float
    start: float
    dur: float
    args: List[float]
    eventId: int = 0
    paramTable: int = 0
    renderer: Optional[Renderer] = None

    def setp(self, delay:float, *args, **kws) -> None:
        """
        Modify a parg of this synth.

        Multiple pargs can be modified simultaneously. It only makes sense
        to modify a parg if a k-rate variable was assigned to this parg
        (see Renderer.setp for an example). A parg can be referred to via an integer,
        corresponding to the p index (5 would refer to p5), or to the name
        of the assigned k-rate variable as a string (for example, if there
        is a line "kfreq = p6", both 6 and "kfreq" refer to the same parg).

        Example
        =======

            >>> from csoundengine import *
            >>> r = Renderer()
            >>> Instr('sine', r'''
            ... |kamp=0.1, kfreq=1000|
            ... outch 1, oscili:ar(kamp, freq)
            ... ''')
            >>> event = r.sched('sine', 0, dur=4, pargs=[0.1, 440])
            >>> event.setp(2, kfreq=880)
            >>> event.setp(3, 'kfreq', 660, 'kamp', 0.5)

        """
        if not self.renderer:
            raise RuntimeError("This ScoreEvent has no associated renderer")
        self.renderer.setp(self, delay=delay, *args, **kws)

    def automatep(self, param: str, pairs: Union[List[float], np.ndarray],
                  mode="linear", delay: float = None) -> None:
        """
        Automate a named parg

        See Also
        ~~~~~~~~

        * :meth: `csoundengine.offline.Renderer.automatep`
        """
        self.renderer.automatep(self, param=param, pairs=pairs, mode=mode,
                                delay=delay)

    def automateTable(self, param: str, pairs: Union[List[float], np.ndarray],
                      mode="linear", delay=0.) -> None:
        """
        Automate the event's parameter table with the given pairs

        See Also
        ~~~~~~~~

        * :meth:`csoundengine.offline.Renderer.automateTable`
        """
        self.renderer.automateTable(self, param=param, pairs=pairs, mode=mode,
                                    delay=delay)


_offlineOrc = r'''
gi__soundfontIndexes dict_new "str:float"
gi__soundfontIndexCounter init 1000

chn_k "_soundfontPresetCount", 3

instr _automatePargViaTable
  ; automates a parg from a table
  ip1 = p4
  ipindex = p5
  itabpairs = p6  ; a table containing flat pairs t0, y0, t1, y1, ...
  imode = p7;  interpolation method
  Sinterpmethod = strget(imode)
  if ftexists:i(itabpairs) == 0 then
    initerror sprintf("Table with pairs %d does not exists", itabpairs)
  endif 
  ftfree itabpairs, 1

  kt timeinsts
  kidx bisect kt, itabpairs, 2, 0
  ky interp1d kidx, itabpairs, Sinterpmethod, 2, 1
  pwrite ip1, ipindex, ky
endin 

instr _automateTableViaTable
  ; automates a slot within a table from another table
  itabnum = p4
  ipindex = p5
  itabpairs = p6
  imode = p7
  Sinterpmethod = strget(imode)
  if ftexists:i(itabpairs) == 0 then
    initerror sprintf("Table with pairs %d does not exists", itabpairs)
  endif 
  ftfree itabpairs, 1
  kt timeinsts
  kidx bisect kt, itabpairs, 2, 0
  ky interp1d kidx, itabpairs, Sinterpmethod, 2, 1
  tabw ky, ipindex, itabnum
endin 
  

instr _pwrite
  ip1 = p4
  inumpairs = p5
  if inumpairs == 1 then
    pwrite ip1, p(6), p(7)
  elseif inumpairs == 2 then
    pwrite ip1, p(6), p(7), p(8), p(9)
  elseif inumpairs == 3 then
    pwrite ip1, p(6), p(7), p(8), p(9), p(10), p(11)
  elseif inumpairs == 4 then
    pwrite ip1, p(6), p(7), p(8), p(9), p(10), p(11), p(12), p(13)
  elseif inumpairs == 5 then
    pwrite ip1, p(6), p(7), p(8), p(9), p(10), p(11), p(12), p(13), p(14), p(15)
  else
    initerror sprintf("Max. pairs is 5, got %d", inumpairs)
  endif
  turnoff
endin

opcode sfloadonce, i, S
  Spath xin
  Skey_ strcat "SFLOAD:", Spath
  iidx dict_get gi__soundfontIndexes, Skey_, -1
  if (iidx == -1) then
      iidx sfload Spath
      dict_set gi__soundfontIndexes, Skey_, iidx
  endif
  xout iidx
endop

opcode sfPresetIndex, i, Sii
  Spath, ibank, ipresetnum xin
  isf sfloadonce Spath
  Skey sprintf "SFIDX:%d:%d:%d", isf, ibank, ipresetnum  
  iidx dict_get gi__soundfontIndexes, Skey, -1  
  if iidx == -1 then
      iidx chnget "_soundfontPresetCount"
      chnset iidx+1, "_soundfontPresetCount"
      i0 sfpreset ipresetnum, ibank, isf, iidx
      if iidx != i0 then
        prints "???: iidx = %d, i0 = %d\n", iidx, i0
      endif
      dict_set gi__soundfontIndexes, Skey, i0
  endif
  xout iidx
endop
'''

class Renderer:
    """
    A Renderer is used when rendering offline.

    Instruments with higher priority are assured to be evaluated later
    in the chain. Instruments within a given priority are evaluated in
    the order they are defined (first defined is evaluated first)

    Args:
        sr: the sampling rate
        nchnls: number of channels
        ksmps: csound ksmps
        a4: reference frequency
        maxpriorities: max. number of priority groups. This will determine
            how long an effect chain can be
        bucketsize: max. number of instruments per priority group

    Example
    =======

    .. code-block:: python

        from csoundengine import *
        renderer = Renderer(sr=44100, nchnls=2)

        Instr('saw', r'''
          kmidi = p5
          outch 1, oscili:a(0.1, mtof:k(kfreq))
        ''').register(renderer)

        score = [('saw', 0,   2, 60),
                 ('saw', 1.5, 4, 67),
                 ('saw', 1.5, 4, 67.1)]
        events = [renderer.sched(ev[0], delay=ev[1], dur=ev[2], pargs=ev[3:])
                  for ev in score]

        # offline events can be modified just like real-time events
        renderer.automatep(events[0], 'kmidi', pairs=[0, 60, 2, 59])
        renderer.setp(events[1], 3, 'kmidi', 67.2)
        renderer.render("out.wav")
    """


    def __init__(self, sr: int = None, nchnls: int = 2, ksmps: int = None,
                 a4: float = None, maxpriorities=10, bucketsize=100,
                 numAudioBuses=1000):
        """

        """
        self.sr = sr
        self.nchnls = nchnls
        self.ksmps = ksmps
        self.a4 = a4
        # maps eventid -> ScoreEvent.
        self.scheduledEvents: Dict[int, ScoreEvent] = {}
        self._idCounter = 0
        a4 = a4 or config['A4']
        sr = sr or config['rec.sr']
        ksmps = ksmps or config['rec.ksmps']
        self.csd = csoundlib.Csd(sr=sr, nchnls=nchnls, ksmps=ksmps, a4=a4)
        self._nameAndPriorityToInstrnum: Dict[Tuple[str, int], int] = {}
        self._instrnumToNameAndPriority: Dict[int, Tuple[str, int]] = {}
        self._numbuckets = maxpriorities
        self._bucketCounters = [0]*maxpriorities
        self._bucketSize = bucketsize
        self._instrdefs: Dict[str, Instr] = {}
        self._instanceCounters: Dict[int, int] = {}
        self._numInstancesPerInstr = 10000
        self._numAudioBuses = numAudioBuses
        self._numControlBuses = 10000

        self.csd.addGlobalCode(_textwrap.dedent(_offlineOrc))
        self._busSystemInitialized = False
        self._busTokenCount = 0

    def _commitInstrument(self, instrname: str, priority=1) -> int:
        """
        Creaaate concrete instrument at the given priority

        Returns the instr number

        Args:
            instrname: the name of the previously defined instrument to commit
            priority: the priority of this version, will define the order
                of execution (higher priority is evaluated later)

        Returns:
            The instr number (as in "instr xx ... endin" in a csound orc)

        """
        assert 1<=priority<=self._numbuckets

        instrnum = self._nameAndPriorityToInstrnum.get((instrname, priority))
        if instrnum is not None:
            return instrnum

        instrdef = self._instrdefs.get(instrname)
        if not instrdef:
            raise KeyError(f"instrument {instrname} is not defined")

        count = self._bucketCounters[priority]
        if count>self._bucketSize:
            raise ValueError(
                f"Too many instruments ({count}) defined, max. is {self._bucketSize}")

        self._bucketCounters[priority] += 1
        instrnum = priority*self._bucketSize+count
        self._nameAndPriorityToInstrnum[(instrname, priority)] = instrnum
        self._instrnumToNameAndPriority[instrnum] = (instrname, priority)
        self.csd.addInstr(instrnum, instrdef.body)
        return instrnum

    def isInstrDefined(self, instrname: str) -> bool:
        """
        Returns True if an Instr with the given name has been registered
        """
        return instrname in self._instrdefs

    def registerInstr(self, instr: Instr) -> None:
        """
        Register an Instr to be used in this Renderer

        Example
        =======

            >>> from csoundengine import *
            >>> renderer = Renderer(sr=44100, nchnls=2)
            >>> instrs = [
            ... Instr('vco', r'''
            ...   |kmidi=60|
            ...   outch 1, vco2:a(0.1, mtof:k(kmidi))
            ...   '''),
            ... Instr('sine', r'''
            ...   |kmidi=60|
            ...   outch 1, oscili:a(0.1, mtof:k(kmidi))
            ... ''')]
            >>> for instr in instrs:
            ...     renderer.registerInstr(instr)
            >>> renderer.sched('vco', dur=4, kmidi=67)
            >>> renderer.sched('sine', 2, dur=3, kmidi=68)
            >>> renderer.render('out.wav')

        """
        self._instrdefs[instr.name] = instr

    def defInstr(self, name: str, body: str, **kws) -> Instr:
        """
        Create an :class:`~csoundengine.instr.Instr` and register it with this renderer

        Args:
            name (str): the name of the created instr
            body (str): the body of the instrument. It can have named
                pfields (see example) or a table declaration
            kws: any keywords are passed on to the Instr constructor.
                See the documentation of Instr for more information.

        Returns:
            the created Instr. If needed, this instr can be registered
            at any other Renderer/Session

        Example
        =======

            >>> from csoundengine import *
            >>> renderer = Renderer()
            # An Instr with named pfields
            >>> renderer.defInstr('synth', '''
            ... |ibus, kamp=0.5, kmidi=60|
            ... kfreq = mtof:k(lag:k(kmidi, 1))
            ... a0 vco2 kamp, kfreq
            ... a0 *= linsegr:a(0, 0.1, 1, 0.1, 0)
            ... busout ibus, a0
            ... ''')
            # An instr with named table args
            >>> renderer.defInstr('filter', '''
            ... {bus=0, cutoff=1000, resonance=0.9}
            ... a0 = busin(kbus)
            ... a0 = moogladder2(a0, kcutoff, kresonance)
            ... outch 1, a0
            ... ''')

            >>> bus = renderer.assignBus()
            >>> synth = renderer.sched('sine', 0, dur=10, ibus=bus, kmidi=67)
            >>> synth.setp(kmidi=60, delay=2)

            >>> filt = renderer.sched('filter', 0, dur=synth.dur, priority=synth.priority+1,
            ...                       tabargs={'bus': bus, 'cutoff': 1000})
            >>> filt.automateTable('cutoff', [3, 1000, 6, 200, 10, 4000])
        """
        instr = Instr(name=name, body=body, **kws)
        self.registerInstr(instr)
        return instr

    def registeredInstrs(self) -> Dict[str, Instr]:
        """
        Returns a dict (instrname: Instr) with all registered Instrs
        """
        return self._instrdefs

    def getInstr(self, name) -> Optional[Instr]:
        """
        Find a registered Instr, by name

        Returns None if no such Instr was registered
        """
        return self._instrdefs.get(name)

    def addGlobalCode(self, code: str) -> None:
        """
        Add global code (instr 0)

        Example
        =======

        >>> from csoundengine import *
        >>> renderer = Renderer(...)
        >>> renderer.addGlobalCode("giMelody[] fillarray 60, 62, 64, 65, 67, 69, 71")
        """
        self.csd.addGlobalCode(code, acceptDuplicates=False)

    def _getUniqueP1(self, instrnum: int) -> float:
        count = self._instanceCounters.get(instrnum, 0)
        count = 1+((count+1)%self._numInstancesPerInstr-1)
        p1 = instrnum+count/self._numInstancesPerInstr
        self._instanceCounters[instrnum] = count
        return p1

    def sched(self,
              instrname: str,
              delay=0.,
              dur=-1.,
              priority=1,
              pargs: Union[List[float], Dict[str, float]] = None,
              tabargs: Dict[str, float] = None,
              **pkws) -> ScoreEvent:
        """
        Schedule an event

        Args:
            instrname: the name of the already registered instrument
            priority: determines the order of execution
            delay: time offset
            dur: duration of this event. -1: endless
            pargs: pargs beginning with p5
                (p1: instrnum, p2: delay, p3: duration, p4: reserved)
            tabargs: a dict of the form param: value, to initialize
                values in the parameter table (if defined by the given
                instrument)

        Returns:
            a ScoreEvent, holding the csound event (p1, start, dur, args)


        Example
        =======

            >>> from csoundengine import *
            >>> renderer = Renderer(sr=44100, nchnls=2)
            >>> instrs = [
            ... Instr('vco', r'''
            ...   |kmidi=60|
            ...   outch 1, vco2:a(0.1, mtof:k(kmidi))
            ...   '''),
            ... Instr('sine', r'''
            ...   |kamp=0.1, kmidi=60|
            ...   outch 1, oscili:a(kamp, mtof:k(kmidi))
            ... ''')]
            >>> for instr in instrs:
            ...     renderer.registerInstr(instr)
            >>> renderer.sched('vco', dur=4, kmidi=67)
            >>> renderer.sched('sine', 2, dur=3, kmidi=68)
            >>> renderer.render('out.wav')

        """
        instr = self._instrdefs.get(instrname)
        if not instr:
            raise KeyError(f"instrument {instrname} is not defined")
        instrnum = self._commitInstrument(instrname, priority)
        tabnum = 0
        if instr.hasParamTable():
            tabnum = self.csd.addTableFromData(instr.overrideTable(tabargs),
                                               start=max(0., delay - 2.))
        args = internalTools.instrResolveArgs(instr, tabnum, pargs, pkws)
        p1 = self._getUniqueP1(instrnum)
        self.csd.addEvent(p1, start=delay, dur=dur, args=args)
        eventId = self._generateEventId()
        event = ScoreEvent(p1, delay, dur, args, eventId, paramTable=tabnum,
                           renderer=self)
        self.scheduledEvents[eventId] = event
        return event

    def _initBusSystem(self) -> None:
        if self._busSystemInitialized:
            return
        code = engineorc.busSupportCode(numAudioBuses=self._numAudioBuses,
                                        clearBusesInstrnum=engineorc.CONSTS['postProcInstrnum'],
                                        numControlBuses=self._numControlBuses)
        self.csd.addGlobalCode(code)
        self._busSystemInitialized = True

    def assignBus(self, kind='audio') -> int:
        """
        Assign a bus number

        Example
        =======

            >>> from csoundengine.offline import Renderer
            >>> r = Renderer()
            >>> Instr('vco', r'''
            ...     kfreq=p5
            ...     kamp=p6
            ...     asig vco2 1, kfreq
            ...     aenv = linsegr:a(0, 0.01, 1, 0.01, 0)
            ...     aenv *= lag:a(a(kamp), 0.1)
            ...     asig *= aenv
            ...     outch 1, asig
            ... ''').register(r)
        """
        if kind != 'audio':
            raise ValueError("Only audio buses are supported at the moment")
        self._initBusSystem()
        token = self._busTokenCount
        self._busTokenCount += 1
        return token

    def _generateEventId(self) -> int:
        out = self._idCounter
        self._idCounter += 1
        return out

    def setCsoundOptions(self, *options: str) -> None:
        """
        Set any command line options

        Args:
            *options (str): any option will be passed directly to csound when rendering

        Examples
        ========

            >>> from csoundengine.offline import Renderer
            >>> renderer = Renderer()
            >>> instr = Instr("sine", ...)
            >>> renderer.registerInstr(instr)
            >>> renderer.sched("sine", ...)
            >>> renderer.setCsoundOptions("--omacro:MYMACRO=foo")
            >>> renderer.render("outfile.wav")
        """
        self.csd.setOptions(*options)

    def render(self, outfile: str = None, samplefmt: str = None,
               wait=True, quiet:bool=None, openWhenDone=False) -> str:
        """
        Render to a soundfile

        To further customize the render set any csound options via
        :meth:`Renderer.setCsoundOptions`

        By default, if the output is an uncompressed file (.wav, .aif)
        the sample format is set to float32 (csound defaults to 16 bit pcm)

        Args:
            outfile: the output file to render to. None will render to a temp file
            samplefmt: the sample format of the rendered file, given as
                'pcmXX' or 'floatXX', where XX represent the bit-depth
                ('pcm16', 'float32', etc)
            wait: if True this method will block until the underlying process exits
            quiet: if True, all output from the csound subprocess is supressed
            openWhenDone: open the file in the default application after rendering

        Returns:
            the path of the rendered file
        """
        if outfile is None:
            import tempfile
            outfile = tempfile.mktemp(suffix=".wav")
        if not self.csd.score:
            raise ValueError("score is empty")
        quiet = quiet if quiet is not None else config['rec_suppress_output']
        if quiet:
            run_suppressdisplay = True
            run_piped = True
        else:
            run_suppressdisplay = False
            run_piped = False

        if samplefmt is None:
            ext = os.path.splitext(outfile)[1]
            samplefmt = csoundlib.bestSampleFormatForExtension(ext)
        self.csd.setSampleFormat(samplefmt)
        proc = self.csd.run(output=outfile,
                            suppressdisplay=run_suppressdisplay,
                            nomessages=run_suppressdisplay,
                            piped=run_piped)
        if openWhenDone:
            proc.wait()
            misc.open_with_app(outfile, wait=True)
        elif wait:
            proc.wait()
        return outfile

    def writeCsd(self, outfile: str) -> None:
        """
        Generate the csd for this renderer, write it to `outfile`

        Args:
            outfile: the path of the generated csd

        If this csd includes any datafiles (tables with data exceeding
        the limit to include the data 'inline') or soundfiles defined
        relative to the csd, these datafiles are written to a subfolder
        with the name ``{outfile}.assets``, where outfile is the
        outfile given as argument

        For example, if we call ``writeCsd`` as ``renderer.writeCsd('~/foo/myproj.csd')`` ,
        any datafiles will be saved in ``'~/foo/myproj.assets'`` and referenced
        with relative paths as ``'myproj.assets/datafile.gen23'`` or
        ``'myproj.assets/mysnd.wav'``
        """
        self.csd.write(outfile)

    def getEventById(self, eventid: int) -> Optional[ScoreEvent]:
        """
        Retrieve a scheduled event by its eventid

        Args:
            eventid: the event id, as returned by sched

        Returns:
            the ScoreEvent if it exists, or None
        """
        return self.scheduledEvents.get(eventid)

    def getEventsByP1(self, p1: float) -> List[ScoreEvent]:
        """
        Retrieve all scheduled events which have the given p1

        Args:
            p1: the p1 of the scheduled event. This can be a fractional
                value

        Returns:
            a list of all scheduled events with the given p1

        """
        return [ev for ev in self.scheduledEvents.values() if ev.p1 == p1]

    def strSet(self, s: str) -> int:
        """
        Set a string in this renderer. The string can be retrieved in any
        instrument via strget. The index is determined by the Renderer itself,
        and it is guaranteed that calling strSet with the same string will
        result in the same index

        Args:
            s: the string to set

        Returns:
            the string id. This can be passed to any instrument to retrieve
            the given string
        """
        return self.csd.strset(s)

    def _instrFromEvent(self, event: ScoreEvent) -> Instr:
        instrNameAndPriority = self._instrnumToNameAndPriority.get(int(event.p1))
        if not instrNameAndPriority:
            raise ValueError(f"Unknown instrument for instance {event.p1}")
        instr = self._instrdefs[instrNameAndPriority[0]]
        return instr

    def setp(self, event: ScoreEvent, delay: float, *args, **kws):
        """
        Modify a pfield of a scheduled event at the given time

        **NB**: the instr needs to have assigned the pfield to a k-rate variable
        (example: ``kfreq = p5``)

        Args:
            event: the event to modify
            delay: time offset of the modification

        Example
        -------

            >>> from csoundengine.offline import Renderer
            >>> renderer = Renderer()
            >>> instr = Instr("sine", '''
            ... |kmidi=60|
            ... outch 1, oscili:a(0.1, mtof:k(kmidi))
            ... ''')
            >>> renderer.registerInstr(instr)
            >>> event = renderer.sched("sine", pargs={'kmidi': 62})
            >>> renderer.setp(event, 10, kmidi=67)
            >>> renderer.render("outfile.wav")
        """
        instr = self._instrFromEvent(event)
        pairsd = {}
        if not kws:
            assert len(args)%2 == 0
            for i in range(len(args)//2):
                k = args[i*2]
                v = args[i*2+1]
                idx = instr.pargIndex(k)
                pairsd[idx] = v
        else:
            for k, v in kws.items():
                idx = instr.pargIndex(k)
                pairsd[idx] = v
        pairs = iterlib.flatdict(pairsd)
        pargs = [event.p1, len(pairs)//2]
        pargs.extend(pairs)
        self.csd.addEvent("_pwrite", start=delay, dur=0.1, args=pargs)

    def makeTable(self, data: Union[np.ndarray, List[float]] = None,
                  size: int = 0, tabnum: int = 0, sr: int = 0,
                  delay=0.
                  ) -> int:
        """
        Create a table with given data or an empty table of the given size

        Args:
            data (np.ndarray | List[float]): the data of the table. Use None
                if the table should be empty
            size (int): if not data is given, sets the size of the empty table created
            tabnum (int): 0 to let csound determine a table number, -1 to self assign
                a value
            sr (int): the samplerate of the data, if applicable.
            delay: when to create this table

        Returns:
            the table number
        """
        if data is not None:
            return self.csd.addTableFromData(data=data, tabnum=tabnum, start=delay, sr=sr)
        else:
            assert size > 0
            return self.csd.addEmptyTable(size=size, sr=sr)

    def readSoundfile(self, path: str, tabnum:int=None, chan=0, start=0., skiptime=0.
                      ) -> int:
        """
        Add code to this offline renderer to load a soundfile

        Args:
            path: the path of the soundfile to load
            tabnum: the table number to assign, or None to autoassign a number
            chan: the channel to read, or 0 to read all channels
            start: moment in the score to read this soundfile
            skiptime: skip this time at the beginning of the soundfile

        Returns:
            the assigned table number
        """
        return self.csd.addSndfile(sndfile=path, tabnum=tabnum,
                                   start=start, skiptime=skiptime,
                                   chan=chan)

    def playSample(self, source:Union[int, str, np.ndarray],
                   delay=0., chan=1, speed=1., gain=1.,
                   fade=0., starttime=0., dur=-1, sr=0
                   ) -> None:
        """
        Play a table or a soundfile

        Adds an instrument definition and an event to play the given
        table as sound (assumes that the table was allocated via
        :meth:`~Renderer.readSoundFile` or any other GEN1 ftgen

        Args:
            source: the table number to play, the path of a soundfile or a numpy array
                holding audio samples (in this case, sr must be given)
            delay: when to start playback
            chan: the channel to play
            speed: the speed to play at
            gain: apply a gain to playback
            fade: fade-in / fade-out ramp, in seconds
            starttime: playback does not start at the beginning of
                the table but at `starttime`
            dur: duration of playback. -1=until end of sample
            sr: when using audio samples (a numpy array) as source, sr must be given
        """
        if isinstance(source, np.ndarray):
            assert sr > 0
            tabnum = self.makeTable(data=source, sr=sr)
            return self.playSample(source=tabnum, delay=delay, chan=chan, speed=speed,
                                   gain=gain, fade=fade, starttime=starttime, dur=dur)

        if isinstance(source, str):
            tabnum = self.readSoundfile(path=source, start=delay, skiptime=starttime)
        else:
            tabnum = source
        assert tabnum > 0
        self.csd.playTable(tabnum=source, start=delay, dur=dur,
                           gain=gain, speed=speed, chan=chan, fade=fade,
                           skip=starttime)

    def automatep(self, event: ScoreEvent, param: str,
                  pairs: Union[List[float], np.ndarray],
                  mode="linear", delay: float = None) -> None:
        """
        Automate a pfield of a scheduled event

        Args:
            event: the event to automate, as returned by sched
            param (str): the name of the parameter to automate. The instr should
                have a corresponding line of the sort "kparam = pn"
            pairs: the automateion data as a flat list ``[t0, y0, t1, y1, ...]``, where
                the times are relative to the start of the automation event
            mode (str): one of "linear", "cos", "smooth", "exp=xx" (see interp1d)
            delay: start time of the automation event. If None is given, the start
                time of the automated event will be used.
        """
        if delay is None:
            delay = event.start
        instr = self._instrFromEvent(event)
        pindex = instr.pargIndex(param)
        dur = pairs[-2]-pairs[0]
        epsilon = self.csd.ksmps / self.csd.sr * 3
        start = max(0., delay-epsilon)
        if event.dur>0:
            # we clip the duration of the automation to the lifetime of the automated event
            end = min(event.start+event.dur, start+dur+epsilon)
            dur = end-start
        modeint = self.strSet(mode)
        # we schedule the table to be created prior to the start of the automation
        tabpairs = self.csd.addTableFromData(pairs, start=start)
        args = [event.p1, pindex, tabpairs, modeint]
        self.csd.addEvent("_automatePargViaTable", start=delay, dur=dur, args=args)

    def automateTable(self, event: ScoreEvent, param: str,
                      pairs: Union[List[float], np.ndarray],
                      mode="linear", delay: float = None) -> None:
        """
        Automate a slot of an event param table

        Args:
            event: the event to modify. Its instrument should define a param table
            param: the named slot to modify
            pairs: the automation data is given as a flat sequence of pairs (time,
              value). Times are relative to the start of the automation event.
              The very first value can be a NAN, in which case the current value
              in the table is used.
            mode: one of 'linear', 'cos', 'expon(xx)', 'smooth'. See the opcode
              `interp1d` for more information
            delay: the time delay to start the automation.

        Example
        =======

        >>> from csoundengine import offline
        >>> r = offline.Renderer()
        >>> r.defInstr('oscili', r'''
        ... {kamp=0.1, kfreq=1000}
        ... outch 1, oscili:a(kamp, kfreq)
        ... ''')
        >>> ev = r.sched('oscili', delay=1, tabargs={'kfreq': 440})
        >>> r.automateTable(ev, 'kfreq', pairs=[0, 440, 2, 880])

        See Also
        ~~~~~~~~

        :meth:`~Renderer.setp`
        :meth:`~Renderer.automatep`
        """

        if delay is None:
            delay = event.start
        instr = self._instrFromEvent(event)
        if not instr.hasParamTable():
            raise RuntimeError(f"instr {instr.name} does not define a parameters table")
        if event.paramTable == 0:
            raise RuntimeError(f"instr {instr.name} should have a parameters table, but"
                               f"no table has been assigned (p1: {event.p1}")
        pindex = instr.paramTableParamIndex(param)
        assert pindex >= 0
        dur = pairs[-2] - pairs[0]
        epsilon = self.csd.ksmps / self.csd.sr * 3
        start = max(0., delay - epsilon)
        if event.dur > 0:
            # we clip the duration of the automation to the lifetime of the automated event
            end = min(event.start + event.dur, start + dur + epsilon)
            dur = end - start
        modeint = self.strSet(mode)
        # we schedule the table to be created prior to the start of the automation
        tabpairs = self.csd.addTableFromData(pairs, start=start)
        args = [event.paramTable, pindex, tabpairs, modeint]
        self.csd.addEvent("_automateTableViaTable", start=delay, dur=dur, args=args)

