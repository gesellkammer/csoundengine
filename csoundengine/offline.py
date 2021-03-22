"""
This module provides an interface to offline rendering using the same
mechanisms as a Session.

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
import dataclasses
from typing import TYPE_CHECKING, List, Dict, Tuple, KeysView, Union as U, Optional as Opt
import textwrap as _textwrap

from .config import config
from . import csoundlib
from .instr import Instr
from . import tools
from emlib import misc, iterlib
import numpy as np

__all__ = ["Renderer", "ScoreEvent"]

@dataclasses.dataclass
class ScoreEvent:
    """
    A ScoreEvent represent a csound event. It is used by the
    offline renderer to keep track of scheduled events

    NB: instances of this class are **NOT** created by the used directly, they
    are generated when scheduling events

    * eventId: a unique identifier
    """
    p1: float
    start: float
    dur: float
    args: List[float]
    eventId: int = 0


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
        maxpriorities: max. groups
        bucketsize: max. number of instruments per priority group

    """
    _builtinInstrs = '''
        instr _automatePargViaTable
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
          println "kt: %f, kidx: %f, ky: %f", kt, kidx, ky
          pwrite ip1, ipindex, ky
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

    '''

    def __init__(self, sr: int = None, nchnls: int = 2, ksmps: int = None,
                 a4: float = None,
                 maxpriorities=10, bucketsize=100):
        """

        """
        self._idCounter = 0
        self._eventsIndex: Dict[int, ScoreEvent] = {}
        a4 = a4 or config['A4']
        sr = sr or config['rec.sr']
        ksmps = ksmps or config['rec.ksmps']
        self._csd = csoundlib.Csd(sr=sr, nchnls=nchnls, ksmps=ksmps, a4=a4)
        self._nameAndPriorityToInstrnum: Dict[Tuple[str, int], int] = {}
        self._instrnumToNameAndPriority: Dict[int, Tuple[str, int]] = {}
        self._numbuckets = maxpriorities
        self._bucketCounters = [0]*maxpriorities
        self._bucketSize = bucketsize
        self._instrdefs: Dict[str, Instr] = {}
        self._instanceCounters: Dict[int, int] = {}
        self._numInstancesPerInstr = 10000

        # a list of i events, starting with p1
        self.events: List[List[float]] = []
        self.unscheduledEvents: List[List[float]] = []
        self._csd.addGlobalCode(_textwrap.dedent(self._builtinInstrs))

    def _commitInstrument(self, instrname: str, priority=1) -> int:
        """
        Generates a concrete version of the instrument
        (with the given priority).

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
        self._csd.addInstr(instrnum, instrdef.body)
        return instrnum

    def registerInstr(self, instr: Instr) -> None:
        """
        Register an Instr to be used in this Renderer
        """
        self._instrdefs[instr.name] = instr

    def registeredInstrs(self) -> KeysView:
        """
        Returns a seq. with the names of all registered Instrs
        """
        return self._instrdefs.keys()

    def addGlobalCode(self, code: str) -> None:
        """
        Add global code (instr 0)
        """
        self._csd.addGlobalCode(code)

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
              pargs: U[List[float], Dict[str, float]] = None,
              tabargs: Dict[str, float] = None,
              unique: bool = True,
              **pkws) -> ScoreEvent:
        """
        Schedule an event

        Args:
            instrname: the name of the already registered instrument
            priority: the priority 1-9, will decide the order of
                execution
            delay: time offset
            dur: duration of this event. -1: endless
            pargs: pargs beginning with p5
                (p1: instrnum, p2: delay, p3: duration, p4: tabnum)
            tabargs: a dict of the form param: value, to initialize
                values in the exchange table (if defined by the given
                instrument)
            unique: if True, schedule a unique instance.

        Returns:
            a ScoreEvent, holding the csound event (p1, start, dur, args)
        """
        instr = self._instrdefs.get(instrname)
        if not instr:
            raise KeyError(f"instrument {instrname} is not defined")
        instrnum = self._commitInstrument(instrname, priority)
        if instr.hasExchangeTable():
            tableinit = instr.overrideTable(tabargs)
            tabnum = self._csd.addTableFromSeq(tableinit)
        else:
            tabnum = 0
        args = tools.instrResolveArgs(instr, tabnum, pargs, pkws)
        p1 = self._getUniqueP1(instrnum) if unique else instrnum
        self._csd.addEvent(p1, start=delay, dur=dur, args=args)
        eventId = self._generateEventId()
        event = ScoreEvent(p1, delay, dur, args, eventId)
        self._eventsIndex[eventId] = event
        return event

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

            >>> renderer = Renderer(...)
            >>> instr = Instr("sine", ...)
            >>> renderer.registerInstr(instr)
            >>> renderer.sched("sine", ...)
            >>> renderer.setCsoundOptions("--omacro:MYMACRO=foo")
            >>> renderer.render("outfile.wav")
        """
        self._csd.setOptions(*options)

    def render(self, outfile: str = None, samplefmt: str = None,
               wait=True, quiet=False, openWhenDone=False) -> str:
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
        import tempfile
        if outfile is None:
            outfile = tempfile.mktemp(suffix=".wav")
        if not self._csd.score:
            raise ValueError("score is empty")
        kws = {}
        if quiet:
            kws['supressdisplay'] = True
            kws['piped'] = True
        if samplefmt is None:
            ext = os.path.splitext(outfile)[1]
            samplefmt = csoundlib.bestSampleFormatForExtension(ext)
        self._csd.setSampleFormat(samplefmt)
        proc = self._csd.run(output=outfile, **kws)
        if openWhenDone:
            proc.wait()
            misc.open_with_standard_app(outfile, wait=True)
        elif wait:
            proc.wait()
        return outfile

    def generateCsd(self) -> str:
        """
        Generate the csd for this renderer as string

        Returns:
            the csd as string
        """
        import io
        stream = io.StringIO()
        self._csd.writeCsd(stream)
        return stream.getvalue()

    def writeCsd(self, outfile: str) -> None:
        """
        Generate the csd for this renderer, write it to `outfile`
        """
        with open(outfile, "w") as f:
            csd = self.generateCsd()
            f.write(csd)

    def getEventById(self, eventid: int) -> Opt[ScoreEvent]:
        """
        Retrieve a scheduled event by its eventid

        Args:
            eventid: the event id, as returned by sched

        Returns:
            the ScoreEvent if it exists, or None
        """
        return self._eventsIndex.get(eventid)

    def getEventsByP1(self, p1: float) -> List[ScoreEvent]:
        """
        Retrieve all scheduled events which have the given p1

        Args:
            p1: the p1 of the scheduled event. This can be a fractional
                value

        Returns:
            a list of all scheduled events with the given p1

        """
        return [ev for ev in self._eventsIndex.values() if ev.p1 == p1]

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
        return self._csd.strset(s)

    def _instrFromEvent(self, event: ScoreEvent) -> Instr:
        instrNameAndPriority = self._instrnumToNameAndPriority.get(int(event.p1))
        if not instrNameAndPriority:
            raise ValueError(f"Unknown instrument for instance {event.p1}")
        instr = self._instrdefs[instrNameAndPriority[0]]
        return instr

    def setp(self, event: ScoreEvent, delay: float, *args, **kws):
        """ Modify a pfield of a scheduled event at the given time
        NB: the instr needs to have assigned the pfield to a k-rate variable
        (example: kfreq = p4"""
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
        args = [event.p1, len(pairs)//2]
        args.extend(pairs)
        self._csd.addEvent("_pwrite", start=delay, dur=0.1, args=args)

    def automatep(self, event: ScoreEvent, param: str,
                  pairs: U[List[float], np.ndarray],
                  mode="linear", delay: float = None) -> None:
        """
        Automate a pfield of a scheduled event

        Args:
            event: the event to automate, as returned by sched
            param (str): the name of the parameter to automate. The instr should
                have a corresponding line of the sort "kparam = pn"
            pairs: the automateion data as a flat list [t0, y0, t1, y1, ...], where
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
        epsilon = self._csd.ksmps/self._csd.sr*3
        start = max(0., delay-epsilon)
        if event.dur>0:
            # we clip the duration of the automation to the lifetime of the automated event
            end = min(event.start+event.dur, start+dur+epsilon)
            dur = end-start
        modeint = self.strSet(mode)
        # we schedule the table to be created prior to the start of the automation
        tabpairs = self._csd.addTableFromSeq(pairs, start=start)
        args = [event.p1, pindex, tabpairs, modeint]
        self._csd.addEvent("_automatePargViaTable", start=delay, dur=dur, args=args)