"""
Offline rendering is implemented via the class :class:`~csoundengine.offline.Renderer`,
which has the same interface as a :class:`~csoundengine.session.Session` and
can be used as a drop-in replacement.

Example
=======

.. code-block:: python

    from csoundengine import *
    renderer = Renderer(sr=44100, nchnls=2)

    renderer.defInstr('saw', r'''
      kmidi = p5
      outch 1, oscili:a(0.1, mtof:k(kfreq))
    ''')

    score = [('saw', 0,   2, 60),
             ('saw', 1.5, 4, 67),
             ('saw', 1.5, 4, 67.1)]

    events = [renderer.sched(ev[0], delay=ev[1], dur=ev[2], args=ev[3:])
              for ev in score]

    # offline events can be modified just like real-time events
    events[0].automate('kmidi', (0, 60, 2, 59))

    # When rendering offline, an event needs to set the time offset
    # of the set operation. This is a difference from a Synth, in which
    # the delay value is optional.
    events[1].set(3, kmidi=67.2)
    events[2].set(kmidi=80, delay=4)
    renderer.render("out.wav")

"""

from __future__ import annotations

import copy
import os
import sys
import sndfileio
import bpf4
import numpy as np
from functools import cache
from dataclasses import dataclass

from .errors import RenderError
from .config import config, logger
from .instr import Instr
from .baseevent import BaseEvent
from .abstractrenderer import AbstractRenderer
from . import csoundlib
from . import internalTools
from . import engineorc
from . import sessioninstrs
from . import state as _state
from . import offlineorc
from . import tableproxy

import emlib.misc
import emlib.filetools
import emlib.mathlib
import emlib.iterlib

from typing import TYPE_CHECKING
if TYPE_CHECKING or "sphinx" in sys.modules:
    from typing import Callable, Sequence, Iterator
    import subprocess


__all__ = (
    "Renderer",
    "ScoreEvent",
    "EventGroup"
)


class ScoreEvent(BaseEvent):
    """
    A ScoreEvent represent a csound event.

    It is used by the offline renderer to keep track of scheduled events

    .. note::
        instances of this class are **NOT** created by the used directly, they
        are generated when scheduling events

    """
    __slots__ = ('uniqueId', 'paramTable', 'renderer', 'instrname', 'priority', 'args', 'p1')

    def __init__(self,
                 p1: float | str,
                 start: float,
                 dur: float,
                 args: list[float],
                 uniqueId: int,
                 paramTable: int = 0,
                 renderer: Renderer = None,
                 instrname: str = '',
                 priority: int = 0):
        super().__init__(start=start, dur=dur)

        self.p1 = p1
        """p1 of this event"""

        self.args = args
        """Args used for this event (p4, p5, ...)"""

        self.uniqueId = uniqueId
        """A unique id of this event, as integer"""

        self.paramTable = paramTable
        """Table number of a parameter table, if any"""

        self.renderer = renderer
        """The Renderer to which this event belongs (can be None)"""

        self.instrname = instrname
        """The instrument template this ScoreEvent was created from, if applicable"""

        self.priority = priority
        """The priority of this ScoreEvent, if applicable"""

    def __hash__(self) -> int:
        return hash((self.p1, self.uniqueId, self.instrname, self.priority, hash(tuple(self.args))))

    def __repr__(self):
        parts = [f"p1={self.p1}, start={self.start}, dur={self.dur}, uniqueId={self.uniqueId}"]
        if self.args:
            parts.append(f'args={self.args}')
        if self.instrname:
            parts.append(f'instrname={self.instrname}')
        if self.priority:
            parts.append(f'priority={self.priority}')
        partsstr = ', '.join(parts)
        return f"ScoreEvent({partsstr})"

    def clone(self, **kws) -> ScoreEvent:
        """Clone this event"""
        out = copy.copy(self)
        for kw, value in kws.items():
            setattr(out, kw, value)
        return out

    def setp(self, delay=0., strict=True, **kws) -> None:
        """
        Modify a parg of this synth (offline).

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
            >>> event.setp(3, kfreq=660, kamp=0.5)

        .. seealso:: :meth:`ScoreEvent.set`

        """
        if self.renderer is None:
            raise RuntimeError("This ScoreEvent is not assigned to a Renderer")
        if strict:
            _checkParams(kws.keys(), self.dynamicParams(), obj=self)

        self.renderer.setp(self, delay=delay, pairs=kws)

    def getInstr(self) -> Instr | None:
        """
        The Instr corresponding to this Event, if applicable
        """
        if self.instrname:
            return self.renderer.getInstr(self.instrname)
        try:
            return self.renderer._instrFromEvent(self)
        except ValueError:
            return None

    def namedParams(self) -> set[str]:
        instr = self.getInstr()
        return set(instr.namedParams().keys()) if instr else set()

    def dynamicParams(self) -> set[str]:
        instr = self.getInstr()
        return instr.dynamicParamKeys() if instr else set()

    def automate(self,
                 param: str,
                 pairs: Sequence[float] | np.ndarray,
                 mode="linear",
                 delay: float = None,
                 overtake=False,
                 strict=True
                 ) -> None:
        if self.renderer is None:
            raise RuntimeError("This ScoreEvent is not assigned to a Renderer")
        if strict:
            _checkParams((param,), self.dynamicParams(), obj=self)
        self.renderer.automate(self, param=param, pairs=pairs, mode=mode,
                               delay=delay)

    def stop(self, delay=0.) -> None:
        if self.renderer is None:
            raise RuntimeError("This ScoreEvent is not assigned to a Renderer")
        self.renderer.unsched(self, delay=delay)


class EventGroup(BaseEvent):
    """
    An EventGroup represents a group of offline events

    These events can be controlled together, similar
    to a SynthGroup
    """
    def __init__(self, events: list[ScoreEvent]):
        if not events:
            raise ValueError("No events given")

        start = min(ev.start for ev in events)
        end = max(ev.end for ev in events)
        dur = end - start
        super().__init__(start=start, dur=dur)
        self.events = events

    def __len__(self):
        return len(self.events)

    def stop(self, delay=0.) -> None:
        for ev in self.events:
            ev.stop(delay=delay)

    def setp(self, delay=0., strict=True, **kws) -> None:
        if strict:
            _checkParams(kws.keys(), self.dynamicParams(), obj=self)

        for ev in self.events:
            ev.setp(delay=delay, strict=False, **kws)

    @cache
    def namedParams(self) -> set[str]:
        allparams = set()
        for ev in self.events:
            allparams.update(ev.namedParams())
        return allparams

    @cache
    def dynamicParams(self) -> set[str]:
        params = set()
        for ev in self.events:
            params.update(ev.dynamicParams())
        return params

    def __hash__(self):
        return hash(tuple(hash(ev) for ev in self.events))

    def automate(self,
                 param: str,
                 pairs: Sequence[float] | np.ndarray,
                 mode="linear",
                 delay: float = None,
                 overtake=False,
                 strict=True
                 ) -> None:
        if strict:
            _checkParams((param,), self.namedParams(), obj=self)

        for ev in self.events:
            if param in ev.namedParams():
                ev.automate(param=param, pairs=pairs, mode=mode, delay=delay, strict=False)


@dataclass
class RenderJob:
    outfile: str
    samplerate: int
    encoding: str = ''
    starttime: float = 0.
    endtime: float = 0.

    def openOutfile(self, wait=True):
        emlib.misc.open_with_app(self.outfile, wait=wait)


class Renderer(AbstractRenderer):
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
        events[0].automate('kmidi', pairs=[0, 60, 2, 59])
        events[1].set(3, 'kmidi', 67.2)
        renderer.render("out.wav")

    """
    def __init__(self,
                 sr: int | None = None,
                 nchnls: int = 2,
                 ksmps: int | None = None,
                 a4: float | None = None,
                 numpriorities=10,
                 numAudioBuses=1000):

        self.sr = sr or config['rec_sr']
        """samplerate"""

        self.nchnls = nchnls
        """number of output channels"""

        self.ksmps = ksmps or config['rec_ksmps']
        """samples per cycle"""

        self.a4 = a4 or config['A4']
        """reference frequency"""

        # maps eventid -> ScoreEvent.
        self.scheduledEvents: dict[int, ScoreEvent] = {}
        """All events scheduled in this Renderer, mapps token to event"""

        self.renderedJobs: list[RenderJob] = []
        """A stack of rendered jobs"""

        self.csd = csoundlib.Csd(sr=sr, nchnls=nchnls, ksmps=ksmps, a4=a4)
        """Csd structure for this renderer (see :class:`~csoundengine.csoundlib.Csd`"""

        self._idCounter = 0
        self._nameAndPriorityToInstrnum: dict[tuple[str, int], int] = {}
        self._instrnumToNameAndPriority: dict[int, tuple[str, int]] = {}
        self._numbuckets = numpriorities
        self._bucketCounters = [0] * numpriorities
        self._startUserInstrs = 20
        self._instrdefs: dict[str, Instr] = {}
        self._instanceCounters: dict[int, int] = {}
        self._numInstancesPerInstr = 10000
        self._numAudioBuses = numAudioBuses
        self._numControlBuses = 10000
        self._ndarrayHashToTabnum: dict[str, int] = {}

        bucketSizeCurve = bpf4.expon(0.7, 1, 500, numpriorities, 50)
        bucketSizes = [int(size) for size in bucketSizeCurve.map(numpriorities)]

        self._bucketSizes = bucketSizes
        """Size of each bucket, by bucket index"""

        self._bucketIndices = [self._startUserInstrs + sum(bucketSizes[:i])
                               for i in range(numpriorities)]
        self._postUserInstrs = self._bucketIndices[-1] + self._bucketSizes[-1]
        """Start of 'post' instruments (instruments at the end of the processing chain)"""

        self._busTokenCount = 0
        self._endMarker = 0.
        self._exitCallbacks: set[Callable] = set()
        self._stringRegistry: dict[str, int] = {}
        self._includes: set[str] = set()
        self._builtinInstrs: dict[str, int] = {}
        self._soundfileRegistry: dict[str, tableproxy.TableProxy] = {}

        self.csd.addGlobalCode(offlineorc.prelude())

        if self.hasBusSupport():
            busorc, instrIndex = engineorc.busSupportCode(numAudioBuses=self._numAudioBuses,
                                                          numControlBuses=self._numControlBuses,
                                                          postInstrNum=self._postUserInstrs,
                                                          startInstr=1)
            self._builtinInstrs.update(instrIndex)
            self.csd.addGlobalCode(busorc)

        self.csd.addGlobalCode(offlineorc.orchestra())

        for instrname in ['.playSample']:
            instr = sessioninstrs.builtinInstrIndex[instrname]
            self.registerInstr(instr)

        if self.hasBusSupport():
            self.csd.addEvent(self._builtinInstrs['clearbuses_post'], start=0, dur=-1)

    def renderMode(self) -> str:
        return 'offline'

    def commitInstrument(self, instrname: str, priority=1) -> int:
        """
        Create concrete instrument at the given priority.

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
        if count > self._bucketSizes[priority]:
            raise ValueError(
                f"Too many instruments ({count}) defined, max. is {self._bucketSizes[priority]}")

        self._bucketCounters[priority] += 1
        instrnum = self._bucketIndices[priority] + count
        self._nameAndPriorityToInstrnum[(instrname, priority)] = instrnum
        self._instrnumToNameAndPriority[instrnum] = (instrname, priority)
        self.csd.addInstr(instrnum, instrdef.body)
        return instrnum

    def _registerExitCallback(self, callback) -> None:
        """
        Register a function to be called when exiting this Renderer as context manager
        """
        self._exitCallbacks.add(callback)

    def isInstrDefined(self, instrname: str) -> bool:
        """
        Returns True if an Instr with the given name has been registered
        """
        return instrname in self._instrdefs

    def registerInstr(self, instr: Instr) -> None:
        """
        Register an Instr to be used in this Renderer

        Example
        ~~~~~~~

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
            ...     instr.register(renderer)   # This will call .registerInstr
            >>> renderer.sched('vco', dur=4, kmidi=67)
            >>> renderer.sched('sine', 2, dur=3, kmidi=68)
            >>> renderer.render('out.wav')

        """
        self._instrdefs[instr.name] = instr

    def defInstr(self,
                 name: str,
                 body: str,
                 args: dict[str, float|str] = None,
                 init: str = None,
                 priority: int = None,
                 tabargs: dict[str, float] | None = None,
                 **kws) -> Instr:
        """
        Create an :class:`~csoundengine.instr.Instr` and register it with this renderer

        Args:
            name (str): the name of the created instr
            body (str): the body of the instrument. It can have named
                pfields (see example) or a table declaration
            args: args: pfields with their default values. Only needed if not using inline
                args
            init: init (global) code needed by this instr (read soundfiles,
                load soundfonts, etc)
            kws: any keywords are passed on to the Instr constructor.
                See the documentation of Instr for more information.

        Returns:
            the created Instr. If needed, this instr can be registered
            at any other Renderer/Session

        .. seealso: :class:`~csoundengine.instr.Instr`, :meth:`Session.defInstr <csoundengine.session.Session.defInstr>`

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
            ... {ibus=0, kcutoff=1000, kresonance=0.9}
            ... a0 = busin(ibus)
            ... a0 = moogladder2(a0, kcutoff, kresonance)
            ... outch 1, a0
            ... ''')

            >>> bus = renderer.assignBus()
            >>> event = renderer.sched('synth', 0, dur=10, ibus=bus, kmidi=67)
            >>> event.set(kmidi=60, delay=2)  # This will set the kmidi param

            >>> filt = renderer.sched('filter', 0, dur=event.dur, priority=event.priority+1,
            ...                       tabargs={'ibus': bus, 'kcutoff': 1000})
            >>> filt.automate('kcutoff', [3, 1000, 6, 200, 10, 4000])
        """
        instr = Instr(name=name, body=body, args=args, init=init, **kws)
        self.registerInstr(instr)
        return instr

    def registeredInstrs(self) -> dict[str, Instr]:
        """
        Returns a dict (instrname: Instr) with all registered Instrs
        """
        return self._instrdefs

    def getInstr(self, name) -> Instr | None:
        """
        Find a registered Instr, by name

        Returns None if no such Instr was registered
        """
        return self._instrdefs.get(name)

    def includeFile(self, path: str) -> None:
        """
        Add an #include clause to this offline renderer

        Args:
            path: the path to the include file
        """
        if path in self._includes:
            return
        self._includes.add(path)
        if not os.path.exists(path):
            logger.warning(f"Adding an include '{path}', but this path does not exist")
        self.csd.addGlobalCode(f'#include "{path}"')

    def addGlobalCode(self, code: str) -> None:
        """
        Add global code (instr 0)

        Example
        ~~~~~~~

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
              **kws) -> ScoreEvent:
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
            kws: any named argument passed to the instr

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
        instrnum = self.commitInstrument(instrname, priority)
        tabnum = 0
        if instr.hasParamTable():
            tabnum = self.csd.addTableFromData(instr.overrideTable(tabargs),
                                               start=max(0., delay - 2.))
        args = internalTools.instrResolveArgs(instr, tabnum, args, kws)
        p1 = self._getUniqueP1(instrnum)
        return self._schedEvent(p1=p1, start=float(delay), dur=float(dur), args=args,
                                instrname=instrname, priority=priority)

    def _schedEvent(self, p1: float|int, start: float, dur: float, 
                    args: list[float|str],
                    instrname: str = '', priority=0
                    ) -> ScoreEvent:
        self.csd.addEvent(p1, start=start, dur=dur, args=args)
        eventid = self._generateEventId()
        event = ScoreEvent(p1, start=start, dur=dur, args=args, uniqueId=eventid, renderer=self,
                           priority=priority, instrname=instrname)
        self.scheduledEvents[eventid] = event
        return event

    def unsched(self, event: int|float|ScoreEvent, delay: float) -> None:
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

    def hasBusSupport(self):
        """
        Returns True if this Engine was started with bus suppor

        """
        return (self._numAudioBuses > 0 or self._numControlBuses > 0)

    def assignBus(self, kind='audio', persist=False) -> int:
        """
        Assign a bus number

        Example
        ~~~~~~~

        TODO
        """
        if kind != 'audio':
            raise ValueError("offline rendering has no control bus support yet...")

        assert self.hasBusSupport()
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

    def renderDuration(self) -> float:
        """
        Returns the actual duration of the rendered score

        Returns:
            the duration of the render, in seconds
        """
        _, end = self.scoreTimeRange()
        if self._endMarker:
            end = min(end, self._endMarker)
        return end

    def scoreTimeRange(self) -> tuple[float, float]:
        """
        Returns a tuple (score start time, score end time)

        If any event is of indeterminate duration (``dur==-1``) the
        end time will be *infinite*. Notice that the end marker is not taken
        into consideration here

        Returns:
            a tuple (start of the earliest event, end of last event). If no events, returns
            (0, 0)
        """
        if not self.scheduledEvents:
            return (0., 0.)
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

    def render(self,
               outfile='',
               endtime=0.,
               encoding='',
               wait=True,
               quiet: bool | None = None,
               openWhenDone=False,
               starttime=0.,
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
            raise ValueError("Score is empty")

        if not outfile:
            import tempfile
            outfile = tempfile.mktemp(suffix=".wav")
            logger.info(f"Rendering to temporary file: '{outfile}'. See renderer.renderedJobs")
        elif outfile == '?':
            outfile = _state.saveSoundfile(title="Select soundfile for rendering",
                                           ensureSelection=True)
        outfile = emlib.filetools.normalizePath(outfile)
        outfiledir = os.path.split(outfile)[0]
        if not os.path.isdir(outfiledir) or not os.path.exists(outfiledir):
            raise FileNotFoundError(f"The path '{outfiledir}' where the rendered soundfile should "
                                    f"be generated does not exist "
                                    f"(outfile: '{outfile}')")
        scorestart, scoreend = self.scoreTimeRange()
        if endtime == 0:
            endtime = scoreend
        endmarker = self._endMarker or endtime
        renderend = min(endtime, scoreend, endmarker)

        if renderend == float('inf'):
            raise RenderError("Cannot render an infinite score. Set an endtime when calling "
                              ".render(...) or use ")
        if renderend <= scorestart:
            raise RenderError(f"No score to render (start: {scorestart}, end: {renderend})")

        if encoding or compressionBitrate or scoreend > renderend:
            csd = self.csd.copy()
        else:
            csd = self.csd

        previousEndMarker = self._endMarker
        if scoreend < renderend:
            self.setEndMarker(renderend)
        elif scoreend > renderend:
            csd.cropScore(end=renderend)

        quiet = quiet if quiet is not None else config['rec_suppress_output']
        if quiet:
            runSuppressdisplay = True
            runPiped = True
        else:
            runSuppressdisplay = False
            runPiped = False

        if encoding is None:
            ext = os.path.splitext(outfile)[1]
            encoding = csoundlib.bestSampleEncodingForExtension(ext[1:])

        # We create a copy so that we can modify encoding/compression/etc
        if encoding:
            csd.setSampleEncoding(encoding)

        if compressionBitrate:
            csd.setCompressionBitrate(compressionBitrate)

        proc = csd.run(output=outfile,
                       suppressdisplay=runSuppressdisplay,
                       nomessages=runSuppressdisplay,
                       piped=runPiped)
        if openWhenDone:
            if not wait:
                logger.info("Waiting for the render to finish...")
            proc.wait()
            emlib.misc.open_with_app(outfile, wait=True)
        elif wait:
            proc.wait()

        if previousEndMarker is not None:
            self.setEndMarker(previousEndMarker)

        self.renderedJobs.append(RenderJob(outfile=outfile, encoding=encoding, samplerate=self.sr,
                                           endtime=endtime, starttime=starttime))
        return outfile, proc

    def lastRender(self) -> str | None:
        """
        Returns the last rendered soundfile, or None if no jobs were rendered
        """
        return self.renderedJobs[-1].outfile if self.renderedJobs else None

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

    def getEventById(self, eventid: int) -> ScoreEvent | None:
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

    def strSet(self, s: str, index: int = None) -> int:
        """
        Set a string in this renderer.

        The string can be retrieved in any instrument via strget. The index is
        determined by the Renderer itself, and it is guaranteed that calling
        strSet with the same string will result in the same index

        Args:
            s: the string to set
            index: if given, it will force the renderer to use this index.

        Returns:
            the string id. This can be passed to any instrument to retrieve
            the given string via the opcode "strget"
        """
        return self.csd.strset(s, index=index)

    def _instrFromEvent(self, event: ScoreEvent) -> Instr:
        instrNameAndPriority = self._instrnumToNameAndPriority.get(int(event.p1))
        if not instrNameAndPriority:
            raise ValueError(f"Unknown instrument for instance {event.p1}")
        instr = self._instrdefs[instrNameAndPriority[0]]
        return instr

    def setp(self, event: ScoreEvent, delay: float, pairs: dict[int|str: float]):
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
            >>> renderer.defInstr("sine", '''
            ... |kmidi=60|
            ... outch 1, oscili:a(0.1, mtof:k(kmidi))
            ... ''')
            >>> event = renderer.sched("sine", args={'kmidi': 62})
            >>> renderer.setp(event, 10, {'kmidi': 67})
            >>> renderer.render("outfile.wav")
        """
        instr = self._instrFromEvent(event)
        pairsd = {}
        for k, v in pairs.items():
            if (idx:=instr.pargIndex(k, 0)) > 0:
                pairsd[idx] = v
        if pairsd:
            flatpairs = emlib.iterlib.flatdict(pairsd)
            pargs = [event.p1, len(pairs)]
            pargs.extend(flatpairs)
            self.csd.addEvent("_pwrite", start=delay, dur=0.1, args=pargs)

    def makeTable(self,
                  data: np.ndarray | list[float] | None = None,
                  size: int = 0,
                  tabnum: int = 0,
                  sr: int = 0,
                  delay: float = 0.
                  ) -> int:
        """
        Create a table with given data or an empty table of the given size

        Args:
            data: the data of the table. Use None if the table should be empty
            size: if not data is given, sets the size of the empty table created
            tabnum: 0 to self assign a table number
            sr: the samplerate of the data, if applicable.
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
            return self.csd.addEmptyTable(size=size, sr=sr, tabnum=tabnum)

    def readSoundfile(self,
                      path: str='?',
                      chan: int = 0,
                      start: float = 0.,
                      skiptime: float = 0.,
                      force=False
                      ) -> tableproxy.TableProxy:
        """
        def readSoundfile(self, path="?", chan=0, free=False, force=False, skiptime: float=0
                      ) -> TableProxy:

        Add code to this offline renderer to load a soundfile

        Args:
            path: the path of the soundfile to load. Use '?' to select a file using a GUI
                dialog
            chan: the channel to read, or 0 to read all channels
            start: moment in the score to read this soundfile
            skiptime: skip this time at the beginning of the soundfile
            force: if True, add the soundfile to this renderer even if the same
                soundfile has already been added

        Returns:
            an instance of :
            the assigned table number
        """
        if path == "?":
            path = _state.openSoundfile()

        tabproxy = self._soundfileRegistry.get(path)
        if tabproxy is not None and tabproxy.skiptime == skiptime:
            return tabproxy

        tabnum = self.csd.addSndfile(sndfile=path,
                                     start=start,
                                     skiptime=skiptime,
                                     chan=chan)
        info = sndfileio.sndinfo(path)
        tabproxy = tableproxy.TableProxy(tabnum=tabnum, engine=None, numframes=info.nframes,
                                         sr=info.samplerate, nchnls=info.channels, path=path,
                                         skiptime=skiptime)
        # TODO: keep registry of tabproxy for given soundfile
        self._soundfileRegistry[path] = tabproxy
        return tabproxy


    def playSample(self,
                   source: int|str|tuple[np.ndarray, int],
                   delay=0., dur=0,
                   chan=1, speed=1., loop=False, pan=-1, gain=1.,
                   fade=0., skip=0.,
                   compensateSamplerate=True,
                   crossfade=0.02,
                   **kws
                   ) -> ScoreEvent:
        """
        Play a table or a soundfile

        Adds an instrument definition and an event to play the given
        table as sound (assumes that the table was allocated via
        :meth:`~Renderer.readSoundFile` or any other GEN1 ftgen)

        Args:
            source: the table number to play, the path of a soundfile or a
                tuple (numpy array, sr). Use '?' to select a file using a GUI dialog
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
            tabproxy = self.readSoundfile(path=source, start=delay, skiptime=skip)
            tabnum = tabproxy.tabnum
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

    def automate(self,
                 event: ScoreEvent,
                 param: str,
                 pairs: Sequence[float] | np.ndarray,
                 mode="linear",
                 delay: float = None
                 ) -> None:
        """
        Automate a parameter of a scheduled event

        Args:
            event: the event to automate, as returned by sched
            param (str): the name of the parameter to automate. The instr should
                have a corresponding line of the sort "kparam = pn".
                Call :meth:`ScoreEvent.dynamicParams` to query the set of accepted
                parameters
            pairs: the automateion data as a flat list ``[t0, y0, t1, y1, ...]``, where
                the times are relative to the start of the automation event
            mode (str): one of "linear", "cos", "smooth", "exp=xx" (see interp1d)
            delay: start time of the automation event. If None is given, the start
                time of the automated event will be used.
        """
        if delay is None:
            delay = event.start

        automStart = delay + pairs[0]
        automEnd = delay + pairs[-2]
        if automEnd <= event.start or automStart >= event.end:
            # automation line ends before the actual event!!
            logger.debug(f"Automation times outside of this event: {param=}, "
                         f"automation start-end: {automStart} - {automEnd}, "
                         f"event: {event}")
            return

        if automStart > event.start or automEnd < event.end:
            pairs, delay = internalTools.cropDelayedPairs(pairs=internalTools.aslist(pairs), delay=delay, start=automStart, end=automEnd)
            if not pairs:
                return

        if pairs[0] > 0:
            pairs, delay = internalTools.consolidateDelay(pairs, delay)

        instr = self._instrFromEvent(event)
        if instr.paramMode() == 'table':
            return self._automateTable(event=event, param=param, pairs=pairs, mode=mode, delay=delay)

        if len(pairs) > 1900:
            pairgroups = internalTools.splitPairs(pairs, 1900)
            for pairs in pairgroups:
                self.automate(event=event, param=param, pairs=pairs, mode=mode, delay=delay)
            return

        pindex = instr.pargIndex(param)
        dur = pairs[-2]-pairs[0]
        epsilon = self.csd.ksmps / self.csd.sr * 3
        start = max(0., delay-epsilon)
        if event.dur > 0:
            # we clip the duration of the automation to the lifetime of the automated event
            end = min(event.start+event.dur, start+dur+epsilon)
            dur = end-start
        modeint = self.strSet(mode)
        args = [event.p1, pindex, modeint, 0, len(pairs)]
        args.extend(pairs)
        self.csd.addEvent('_automatePargViaPargs', start=delay, dur=dur, args=args)
        return

        # we schedule the table to be created prior to the start of the automation
        #tabpairs = self.csd.addTableFromData(pairs, start=start)
        #args = [event.p1, pindex, tabpairs, modeint]
        #self.csd.addEvent("_automatePargViaTable", start=delay, dur=dur, args=args)

    def _automateTable(self,
                       event: ScoreEvent,
                       param: str,
                       pairs: list[float]|np.ndarray,
                       mode="linear",
                       delay: float = None
                       ) -> None:
        """
        Automate a slot of an event param table

        This is called when :meth:`Renderer.automate` is called for an instrument
        which defines a parameter table

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
        >>> r.automate(ev, 'kfreq', pairs=[0, 440, 2, 880])

        See Also
        ~~~~~~~~

        :meth:`~Renderer.setp`
        :meth:`~Renderer._automatep`
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

    def __enter__(self):
        if not self._exitCallbacks:
            logger.debug("Called Renderer as context, will render with default values at exit")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._exitCallbacks:
            for func in self._exitCallbacks:
                func(self)
        else:
            self.render()

    def _repr_html_(self) -> str:
        blue = internalTools.safeColors['blue1']

        def _(s):
            return f'<code style="color:{blue}">{s}</code>'

        if self.renderedJobs and os.path.exists(self.renderedJobs[-1].outfile):
            last = self.renderedJobs[-1]
            sndfile = last.outfile
            soundfileHtml = internalTools.soundfileHtml(sndfile)
            info = f'sr={_(self.sr)}, renderedJobs={_(self.renderedJobs)}'
            htmlparts = (
                f'<strong>Renderer</strong>({info})',
                soundfileHtml
            )
            return '<br>'.join(htmlparts)
        else:
            info = f'sr={_(self.sr)}'
            return f'<strong>Renderer</strong>({info})'

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


def _checkParams(params: Iterator[str], dynamicParams: set[str], obj=None) -> None:
    for param in params:
        if param not in dynamicParams:
            if obj:
                msg = f"Parameter {param} not known for {obj}. Possible parameters: {params}"
            else:
                msg = f"Parameter {param} not known. Possible parameters: {params}"
            raise KeyError(msg)
