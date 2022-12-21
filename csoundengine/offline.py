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
    events = [renderer.sched(ev[0], delay=ev[1], dur=ev[2], args=ev[3:])
              for ev in score]

    # offline events can be modified just like real-time events
    renderer.automatep(events[0], 'kmidi', pairs=[0, 60, 2, 59])
    renderer.setp(events[1], 3, 'kmidi', 67.2)
    renderer.render("out.wav")

"""

from __future__ import annotations

import copy
import os
import sys
import sndfileio
from .errors import RenderError
from .config import config
from . import csoundlib
from .instr import Instr
from . import internalTools
from . import engineorc
from . import sessioninstrs
from . import state as _state
import logging

import emlib.misc
import emlib.filetools
import emlib.mathlib
import emlib.iterlib
import numpy as np
import textwrap as _textwrap
from .baseevent import BaseEvent

from typing import TYPE_CHECKING
if TYPE_CHECKING or "sphinx" in sys.modules:
    from typing import Optional, Union
    import subprocess


logger = logging.getLogger("csoundengine")


__all__ = ["Renderer", "ScoreEvent"]


class ScoreEvent(BaseEvent):
    """
    A ScoreEvent represent a csound event.

    It is used by the offline renderer to keep track of scheduled events

    .. note::
        instances of this class are **NOT** created by the used directly, they
        are generated when scheduling events

    """
    __slots__ = ('uniqueId', 'paramTable', 'renderer')

    def __init__(self,
                 p1: Union[float, str],
                 start: float,
                 dur: float,
                 args: list[float],
                 uniqueId: int,
                 paramTable: int = 0,
                 renderer: Renderer = None):
        super().__init__(p1, start, dur, args)
        self.uniqueId = uniqueId
        """A unique id of this event, as integer"""

        self.paramTable = paramTable
        """Table number of a parameter table, if any"""

        self.renderer = renderer
        """The Renderer to which this event belongs (can be None)"""

    def clone(self, **kws) -> ScoreEvent:
        """Clone this event"""
        out = copy.copy(self)
        for kw, value in kws.items():
            setattr(out, kw, value)
        return out

    def setp(self, delay:float, *args, **kws) -> None:
        """
        Modify a parg of this synth.

        Multiple pfields can be modified simultaneously. It only makes sense
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
            >>> event = r.sched('sine', 0, dur=4, args=[0.1, 440])
            >>> event.setp(2, kfreq=880)
            >>> event.setp(3, 'kfreq', 660, 'kamp', 0.5)

        """
        assert self.renderer
        self.renderer.setp(self, delay=delay, *args, **kws)

    def automatep(self, param: str, pairs: Union[list[float], np.ndarray],
                  mode="linear", delay: float = None) -> None:
        """
        Automate a named parg

        See Also
        ~~~~~~~~

        * :meth: `csoundengine.offline.Renderer.automatep`
        """
        assert self.renderer
        self.renderer.automatep(self, param=param, pairs=pairs, mode=mode,
                                delay=delay)

    def automateTable(self, param: str, pairs: Union[list[float], np.ndarray],
                      mode="linear", delay=0.) -> None:
        """
        Automate the event's parameter table with the given pairs

        See Also
        ~~~~~~~~

        * :meth:`csoundengine.offline.Renderer.automateTable`
        """
        self.renderer.automateTable(self, param=param, pairs=pairs, mode=mode,
                                    delay=delay)

    def stop(self, delay=0.) -> None:
        assert self.renderer
        self.renderer.unsched(self, delay=delay)


_offlineOrc = r'''
gi__soundfontIndexes dict_new "str:float"
gi__soundfontIndexCounter init 1000

gi__subgains   ftgen 0, 0, 100, -2, 0
ftset gi__subgains, 1


chn_k "_soundfontPresetCount", 3

opcode _panweights, kk, k
    kpos xin   
    kampL = bpf:k(kpos, 0, 1.4142, 0.5, 1, 1, 0)
    kampR = bpf:k(kpos, 0, 0,      0.5, 1, 1, 1.4142)
    xout kampL, kampR
endop 

instr _stop
    ; turnoff inum (match instr number exactly, allow release)
    inum = p4
    turnoff2_i inum, 4, 1
    turnoff
endin

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
  iidx dict_get gi__soundfontIndexes, Spath, -1
  if (iidx == -1) then
      iidx sfload Spath
      dict_set gi__soundfontIndexes, Spath, iidx
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

    In most cases a :class:`Renderer` is a drop-in replacement of a
    :class:`~csoundengine.session.Session` when rendering offline
    (see :meth:`~csoundengine.session.Session.makeRenderer`).

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
        events = [renderer.sched(ev[0], delay=ev[1], dur=ev[2], args=ev[3:])
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
        self.scheduledEvents: dict[int, ScoreEvent] = {}
        self._idCounter = 0
        a4 = a4 or config['A4']
        sr = sr or config['rec_sr']
        ksmps = ksmps or config['rec_ksmps']
        self.csd = csoundlib.Csd(sr=sr, nchnls=nchnls, ksmps=ksmps, a4=a4)
        self._nameAndPriorityToInstrnum: dict[tuple[str, int], int] = {}
        self._instrnumToNameAndPriority: dict[int, tuple[str, int]] = {}
        self._numbuckets = maxpriorities
        self._bucketCounters = [0]*maxpriorities
        self._bucketsize = bucketsize
        self._instrdefs: dict[str, Instr] = {}
        self._instanceCounters: dict[int, int] = {}
        self._numInstancesPerInstr = 10000
        self._numAudioBuses = numAudioBuses
        self._numControlBuses = 10000
        self._lastUserInstr = self._numbuckets * self._bucketsize
        self._numReservedInstrs = 100
        self._ndarrayHashToTabnum: dict[str, int] = {}

        self.csd.addGlobalCode(_textwrap.dedent(_offlineOrc))
        self._busSystemInitialized = False
        self._busTokenCount = 0
        self._endMarker = 0.

        for instrname in ['.playSample']:
            instr = sessioninstrs.builtinInstrIndex[instrname]
            self.registerInstr(instr)

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
        if count>self._bucketsize:
            raise ValueError(
                f"Too many instruments ({count}) defined, max. is {self._bucketsize}")

        self._bucketCounters[priority] += 1
        instrnum = priority * self._bucketsize + count
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
            >>> event = renderer.sched('sine', 0, dur=10, ibus=bus, kmidi=67)
            >>> event.setp(kmidi=60, delay=2)

            >>> filt = renderer.sched('filter', 0, dur=synth.dur, priority=synth.priority+1,
            ...                       tabargs={'bus': bus, 'cutoff': 1000})
            >>> filt.automateTable('cutoff', [3, 1000, 6, 200, 10, 4000])
        """
        instr = Instr(name=name, body=body, **kws)
        self.registerInstr(instr)
        return instr

    def registeredInstrs(self) -> dict[str, Instr]:
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
              args: list[float] | dict[str, float] = None,
              tabargs: dict[str, float] = None,
              **pkws) -> ScoreEvent:
        """
        Schedule an event

        Args:
            instrname: the name of the already registered instrument
            priority: determines the order of execution
            delay: time offset
            dur: duration of this event. -1: endless
            args: pfields **beginning with p5**
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
        args = internalTools.instrResolveArgs(instr, tabnum, args, pkws)
        p1 = self._getUniqueP1(instrnum)
        self.csd.addEvent(p1, start=delay, dur=dur, args=args)
        eventId = self._generateEventId()
        event = ScoreEvent(p1, delay, dur, args, eventId, paramTable=tabnum,
                           renderer=self)
        self.scheduledEvents[eventId] = event
        return event

    def unsched(self, event: Union[int, float, ScoreEvent], delay: float) -> None:
        """
        Stop a scheduled event

        This schedule the stop of a playing event. The event
        can be an indefinite event (dur=-1) or it can be used
        to stop an event before its actual end

        Args:
            event: the event to stop
            delay: when to stop the given event
        """
        p1 = event.p1 if isinstance(event, ScoreEvent) else event
        self.csd.addEvent("_stop", start=delay, dur=0, args=[p1])

    def _initBusSystem(self) -> None:
        if self._busSystemInitialized:
            return
        busorc, instrIndex = engineorc.busSupportCode(numAudioBuses=self._numAudioBuses,
                                                      numControlBuses=self._numControlBuses,
                                                      postInstrNum=self._lastUserInstr + self._numReservedInstrs,
                                                      startInstr=self._lastUserInstr)
        self.csd.addGlobalCode(busorc)
        self._busSystemInitialized = True

    def assignBus(self) -> int:
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

    def scoreTimeRange(self) -> tuple[float, float]:
        """
        Returns a tuple (score start time, score end time)

        If any event is of indeterminate duration (``dur==-1``) the
        end time will be *infinite* unless the end marker has been set
        (see :meth:`~Renderer.setEndMarker`)

        Returns:
            a tuple (start of earliest event, end of last event)
        """
        events = self.scheduledEvents.values()
        start = min(event.start for event in events)
        end = max(event.end for event in events)
        if end < float("inf"):
            end = max(end, self._endMarker)
        return start, end

    def setEndMarker(self, time: float) -> None:
        """
        Set the end marker for the score

        The end marker will extend the rendering time if it is placed after the
        end of the last event. It will also crop any *infinite* event. It does not
        have any effect if there are events with determinate duration ending after
        it. In this case the end time of the render will be the end of the latest
        event.
        """
        self._endMarker = time
        self.csd.setEndMarker(time)

    def render(self, outfile: str = None, endtime: float = 0, encoding: str = None,
               wait=True, quiet:bool=None, openWhenDone=False, starttime: float=0,
               compressionBitrate: int = None
               ) -> tuple[str, subprocess.Popen]:
        """
        Render to a soundfile

        To further customize the render set any csound options via
        :meth:`Renderer.setCsoundOptions`

        By default, if the output is an uncompressed file (.wav, .aif)
        the sample format is set to float32 (csound defaults to 16 bit pcm)

        Args:
            outfile: the output file to render to. The extension will determine
                the format (wav, flac, etc). None will render to a temp wav file.
            encoding: the sample encoding of the rendered file, given as
                'pcmXX' or 'floatXX', where XX represent the bit-depth
                ('pcm16', 'float32', etc). If no encoding is given a suitable default for
                the sample format is chosen
            wait: if True this method will block until the underlying process exits
            quiet: if True, all output from the csound subprocess is supressed
            endtime: stop rendering at the given time. This will either extend or crop
                the rendering.
            starttime: start rendering at the given time. Any event ending previous to
                this time will not be rendered and any event between starttime and
                endtime will be cropped
            compressionBitrate: used when rendering to ogg
            openWhenDone: open the file in the default application after rendering. At
                the moment this will force the operation to be blocking, waiting for
                the render to finish.

        Returns:
            a tuple (path of the rendered file, subprocess.Popen object). The Popen object
            is only meaningful if wait is False, in which case it can be further queried,
            waited, etc.
        """
        if not self.csd.score:
            raise ValueError("score is empty")

        if outfile is None:
            import tempfile
            outfile = tempfile.mktemp(suffix=".wav")
        elif outfile == '?':
            outfile = _state.saveSoundfile(title="Select soundfile for rendering",
                                           ensureSelection=True)
        outfile = emlib.filetools.normalizePath(outfile)
        outfiledir = os.path.split(outfile)[0]
        if not os.path.isdir(outfiledir) or not os.path.exists(outfiledir):
            raise FileNotFoundError(f"The path '{outfiledir}' where the rendered soundfile should "
                                    f"be generated does not exist (outfile: '{outfile}')")
        scorestart, scoreend = self.scoreTimeRange()
        renderend = endtime if endtime > 0 else scoreend
        if renderend == float('inf'):
            raise RenderError("Cannot render an infinite score. Set an endtime when calling "
                              ".render(...) or use ")
        if renderend <= scorestart:
            raise RenderError(f"No score to render (start: {scorestart}, end: {renderend})")
        if scoreend != renderend:
            previousEndMarker = self._endMarker
            self.setEndMarker(renderend)
        else:
            previousEndMarker = None

        quiet = quiet if quiet is not None else config['rec_suppress_output']
        if quiet:
            run_suppressdisplay = True
            run_piped = True
        else:
            run_suppressdisplay = False
            run_piped = False

        if encoding is None:
            ext = os.path.splitext(outfile)[1]
            encoding = csoundlib.bestSampleEncodingForExtension(ext[1:])

        if encoding:
            self.csd.setSampleEncoding(encoding)

        if compressionBitrate:
            self.csd.setCompressionBitrate(compressionBitrate)

        proc = self.csd.run(output=outfile,
                            suppressdisplay=run_suppressdisplay,
                            nomessages=run_suppressdisplay,
                            piped=run_piped)
        if openWhenDone:
            if not wait:
                logger.info("Waiting for the render to finish...")
            proc.wait()
            emlib.misc.open_with_app(outfile, wait=True)
        elif wait:
            proc.wait()
        if previousEndMarker is not None:
            self.setEndMarker(previousEndMarker)
        return outfile, proc

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

    def generateCsdString(self) -> str:
        """
        Returns the csd as a string

        Returns:
            the csd as str
        """
        return self.csd.dump()

    def getEventById(self, eventid: int) -> Optional[ScoreEvent]:
        """
        Retrieve a scheduled event by its eventid

        Args:
            eventid: the event id, as returned by sched

        Returns:
            the ScoreEvent if it exists, or None
        """
        return self.scheduledEvents.get(eventid)

    def getEventsByP1(self, p1: float) -> list[ScoreEvent]:
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
            >>> event = renderer.sched("sine", args={'kmidi': 62})
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
        pairs = emlib.iterlib.flatdict(pairsd)
        pargs = [event.p1, len(pairs)//2]
        pargs.extend(pairs)
        self.csd.addEvent("_pwrite", start=delay, dur=0.1, args=pargs)

    def makeTable(self, data: Union[np.ndarray, list[float]] = None,
                  size: int = 0, tabnum: int = 0, sr: int = 0,
                  delay=0.
                  ) -> int:
        """
        Create a table with given data or an empty table of the given size

        Args:
            data (np.ndarray | list[float]): the data of the table. Use None
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
            arrayhash = internalTools.ndarrayhash(data)
            if tabnum := self._ndarrayHashToTabnum.get(arrayhash, 0):
                return tabnum
            tabnum = self.csd.addTableFromData(data=data, tabnum=tabnum, start=delay, sr=sr)
            self._ndarrayHashToTabnum[arrayhash] = tabnum
            return tabnum
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


    def playSample(self, source:Union[int, str, tuple[np.ndarray, int]],
                   delay=0., dur=0,
                   chan=1, speed=1., loop=False, pan=-1, gain=1.,
                   fade=0., skip=0.,
                   compensateSamplerate=True,
                   crossfade=0.02, **kws
                   ) -> ScoreEvent:
        """
        Play a table or a soundfile

        Adds an instrument definition and an event to play the given
        table as sound (assumes that the table was allocated via
        :meth:`~Renderer.readSoundFile` or any other GEN1 ftgen

        Args:
            source: the table number to play, the path of a soundfile or a
                tuple (numpy array, sr)
            delay: when to start playback
            chan: the channel to output to. If the sample is stereo/multichannel, this indicates
                the first of a set of consecutive channels to output to.
            loop: if True, sound will loop
            speed: the speed to play at
            pan: a value between 0-1. -1=default, which is 0 for mono, 0.5 for stereo. For
                multichannel samples panning is not taken into account at the moment
            gain: apply a gain to playback
            fade: fade-in / fade-out ramp, in seconds
            skip: playback does not start at the beginning of
                the table but at `starttime`
            dur: duration of playback. -1=indefinite duration, will stop at the end of the
                sample if no looping was set; 0=definite duration, the event is scheduled
                with dur=sampledur/speed
            compensateSamplerate: if True, adjust playback rate in order to preserve
                the sample's original pitch if there is a sr mismatch between the
                sample and the engine.
            crossfade: if looping, this indicates the length of the crossfade
        """
        if loop and dur == 0:
            logger.warning(f"playSample was called with loop=True, but the duration ({dur}) given "
                           f"will result in no looping taking place")
        if isinstance(source, tuple):
            assert len(source) == 2 and isinstance(source[0], np.ndarray)
            data, sr = source
            tabnum = self.makeTable(data=source[0], sr=sr)
            if dur == 0:
                dur = (len(data)/sr) / speed

        elif isinstance(source, str):
            tabnum = self.readSoundfile(path=source, start=delay, skiptime=skip)
            if dur == 0:
                dur = sndfileio.sndinfo(source).duration/speed
        else:
            tabnum = source
            if dur == 0:
                dur = -1
        assert tabnum > 0
        if not loop:
            crossfade = -1
        args = dict(isndtab=tabnum, istart=skip,
                    ifade=fade, icompensatesr=int(compensateSamplerate),
                    kchan=chan, kspeed=speed, kpan=pan, kgain=gain,
                    ixfade=crossfade)
        return self.sched('.playSample', delay=delay, dur=dur, args=args)

    def automatep(self, event: ScoreEvent, param: str,
                  pairs: Union[list[float], np.ndarray],
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
                      pairs: Union[list[float], np.ndarray],
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


def cropScore(events: list[ScoreEvent], start=0, end=0) -> list[ScoreEvent]:
    """
    Crop the score so that no event exceeds the given limits

    Args:
        events: a list of ScoreEvents
        start: the min. start time for any event
        end: the max. end time for any event
    """
    scoreend = max(ev.end for ev in events)
    if end == 0:
        end = scoreend
    cropped = []
    for ev in events:
        if ev.end < start or ev.start > end:
            continue

        if start <= ev.start <= ev.end:
            cropped.append(ev)
        else:
            xstart, xend = emlib.mathlib.intersection(start, end, ev.start, ev.end)
            if xend == float('inf'):
                dur = -1
            else:
                dur = xend - xstart
            ev = ev.clone(start=xstart, dur=dur)
            cropped.append(ev)
    return cropped


