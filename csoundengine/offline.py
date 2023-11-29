"""
Offline rendering is implemented via the class :class:`~csoundengine.offline.Renderer`,
which has the same interface as a :class:`~csoundengine.session.Session` and
can be used as a drop-in replacement.

Example
=======

.. code-block:: python

    from csoundengine import *
    from pitchtools import *

    renderer = Renderer(sr=44100, nchnls=2)

    renderer.defInstr('saw', r'''
      kmidi = p5
      outch 1, oscili:a(0.1, mtof:k(kfreq))
    ''')

    events = [
        renderer.sched('saw', 0, 2, kmidi=ntom('C4')),
        renderer.sched('saw', 1.5, 4, kmidi=ntom('4G')),
        renderer.sched('saw', 1.5, 4, kmidi=ntom('4G+10'))
    ]

    # offline events can be modified just like real-time events
    events[0].automate('kmidi', (0, 0, 2, ntom('B3')), overtake=True)

    events[1].set(delay=3, kmidi=67.2)
    events[2].set(kmidi=80, delay=4)
    renderer.render("out.wav")

"""

from __future__ import annotations

import os
import sys
import sndfileio
import bpf4
import numpy as np
from functools import cache
from dataclasses import dataclass
import textwrap

from .errors import RenderError
from .config import config, logger
from .instr import Instr
from .schedevent import SchedEvent, SchedEventGroup
from .event import Event
from .abstractrenderer import AbstractRenderer
from . import csoundlib
from . import internalTools
from . import sessioninstrs
from . import state as _state
from . import offlineorc
from . import instrtools
from . import busproxy
from . import engineorc
from .engineorc import BUSKIND_CONTROL, BUSKIND_AUDIO
from .tableproxy import TableProxy


import emlib.misc
import emlib.filetools
import emlib.mathlib
import emlib.iterlib
import emlib.textlib

from typing import TYPE_CHECKING
if TYPE_CHECKING or "sphinx" in sys.modules:
    from typing import Callable, Sequence, Iterator, Any
    import subprocess


__all__ = (
    "Renderer",
    "SchedEvent",
    "SchedEventGroup",
    "RenderJob"
)


_EMPTYDICT: dict[str, Any] = {}


@dataclass
class ChannelDef:
    """
    A csound channel definition
    """
    name: str
    "The name of the channel"

    kind: str
    "The type, one of k, S or a"

    mode: str
    "The mode, one of r, w, rw"

    def __post_init__(self):
        assert self.kind in ('k', 'S', 'a')
        assert self.mode in ('r', 'w', 'rw', 'wr')


@dataclass
class RenderJob:
    """
    Represent an offline render process

    A RenderJob is generated each time :meth:`Renderer.render` is called.
    Each new process is appended to :attr:`Renderer.renderedJobs`. The
    last render job can be accesses via :meth:`Renderer.lastRenderJob`
    """
    outfile: str
    """The soundfile rendered / being rendererd"""

    samplerate: int
    """Samplerate of the rendered soundfile"""

    encoding: str = ''
    """Encoding of the rendered soundfile"""

    starttime: float = 0.
    """Start time of the rendered timeline"""

    endtime: float = 0.
    """Endtime of the rendered timeline"""

    process: subprocess.Popen | None = None
    """The csound subprocess used to render the soundfile"""

    def openOutfile(self, timeout=None, appwait=True, app=''):
        """
        Open outfile in external app

        Args:
            timeout: if still rendering, timeout after this number of seconds. None
                means to wait until rendering is finished
            app: if given, use the given application. Otherwise the default
                application
            appwait: if True, wait until the external app exits before returning
                from this method
        """
        self.wait(timeout=timeout)
        emlib.misc.open_with_app(self.outfile, wait=appwait, app=app)

    def wait(self, timeout: float | None = None):
        """Wait for the render process to finish"""
        if self.process is not None:
            self.process.wait(timeout=timeout)

    def _repr_html_(self):
        self.wait()
        blue = internalTools.safeColors['blue1']

        def _(s, color=blue):
            return f'<code style="color:{color}">{s}</code>'

        if not os.path.exists(self.outfile):
            info = (f"outfile='{self.outfile}' (not found), sr={_(self.samplerate)}, "
                    f"encoding={self.encoding}")
            return f'<string>RenderJob</strong>({info})'
        else:
            sndfile = self.outfile
            soundfileHtml = internalTools.soundfileHtml(sndfile, withHeader=False)
            info = (f"outfile='{_(self.outfile)}' (not found), sr={_(self.samplerate)}, "
                    f"encoding='{self.encoding}'")
            htmlparts = (
                f'<strong>RenderJob</strong>({info})',
                soundfileHtml
            )
            return '<br>'.join(htmlparts)


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
        sr: the sampling rate. If not given, the value in the config is used
            (see :ref:`config['rec_sr'] <config_rec_sr>`)
        nchnls: number of channels.
        ksmps: csound ksmps. If not given, the value in the config is used (see :ref:`config['ksmps'] <config_ksmps>`)
        a4: reference frequency. (see :ref:`config['A4'] <config_a4>`)
        priorities: max. number of priority groups. This will determine
            how long an effect chain can be
        numAudioBuses: max. number of audio buses. This is the max. number of simultaneous
            events using an audio bus

    Example
    ~~~~~~~

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
                 priorities: int = None,
                 numAudioBuses=1000,
                 dynamicArgsPerInstr: int = 16,
                 dynamicArgsSlots: int = None):

        if priorities is None:
            priorities = config['session_priorities']

        self.sr = sr or config['rec_sr']
        """Samplerate"""

        self.nchnls = nchnls
        """Number of output channels"""

        self.ksmps = ksmps or config['rec_ksmps']
        """Samples per cycle"""

        self.a4 = a4 or config['A4']
        """Reference frequency"""

        # maps eventid -> ScoreEvent.
        self.scheduledEvents: dict[int, SchedEvent] = {}
        """All events scheduled in this Renderer, mapps token to event"""

        self.renderedJobs: list[RenderJob] = []
        """A stack of rendered jobs"""

        self.csd = csoundlib.Csd(sr=self.sr, nchnls=nchnls, ksmps=self.ksmps, a4=self.a4)
        """Csd structure for this renderer (see :class:`~csoundengine.csoundlib.Csd`"""

        self.controlArgsPerInstr = dynamicArgsPerInstr or config['max_dynamic_args_per_instr']
        """The max. number of dynamic controls per instr"""

        self.instrs: dict[str, Instr] = {}
        """Maps instr name to Instr instance"""

        self.numPriorities: int = priorities
        """Number of priorities in this Renderer"""

        self._idCounter = 0
        self._nameAndPriorityToInstrnum: dict[tuple[str, int], int] = {}
        self._instrnumToNameAndPriority: dict[int, tuple[str, int]] = {}
        self._numbuckets = priorities
        self._bucketCounters = [0] * priorities
        self._startUserInstrs = 50
        self._instanceCounters: dict[int, int] = {}
        self._numInstancesPerInstr = 10000
        self._numAudioBuses = numAudioBuses
        self._numControlBuses = 10000
        self._ndarrayHashToTabproxy: dict[str, TableProxy] = {}

        self._channelRegistry: dict[str, ChannelDef] = {}
        """Dict mapping channel name to tuple (valuetype, channeltype)
        valuetype is one of 'k', 'S'; channeltype is 'r', 'w', 'rw', """

        self._dynargsNumSlices = dynamicArgsSlots or config['dynamic_args_num_slots']
        "Number of dynamic control slices"

        self._dynargsSliceSize = dynamicArgsPerInstr or config['max_dynamic_args_per_instr']
        """Number of dynamic args per instr"""

        self._dynargsTokenCounter = 0

        bucketSizeCurve = bpf4.expon(0.7, 1, 500, priorities, 50)
        bucketSizes = [int(size) for size in bucketSizeCurve.map(priorities)]

        self._bucketSizes = bucketSizes
        """Size of each bucket, by bucket index"""

        self._bucketIndices = [self._startUserInstrs + sum(bucketSizes[:i])
                               for i in range(priorities)]
        self._postUserInstrs = self._bucketIndices[-1] + self._bucketSizes[-1]
        """Start of 'post' instruments (instruments at the end of the processing chain)"""

        self._busTokenCount = 1
        self._endMarker = 0.
        self._exitCallbacks: set[Callable] = set()
        self._stringRegistry: dict[str, int] = {}
        self._includes: set[str] = set()
        self._builtinInstrs: dict[str, int] = {}
        self._soundfileRegistry: dict[str, TableProxy] = {}

        prelude = offlineorc.prelude(controlNumSlots=self._dynargsNumSlices,
                                     controlArgsPerInstr=self._dynargsSliceSize)
        self.csd.addGlobalCode(prelude)
        self.csd.addGlobalCode(offlineorc.orchestra())

        if self.hasBusSupport():
            # The bus code should make room for the named instruments in the offline
            # orchestra
            busorc, instrIndex = engineorc.busSupportCode(numAudioBuses=self._numAudioBuses,
                                                          numControlBuses=self._numControlBuses,
                                                          postInstrNum=self._postUserInstrs,
                                                          startInstr=20)
            self._builtinInstrs.update(instrIndex)
            self.csd.addGlobalCode(busorc)

        for instr in sessioninstrs.builtinInstrs:
            self.registerInstr(instr)

        self._dynargsTabnum = self.makeTable(size=self.controlArgsPerInstr * self._dynargsNumSlices).tabnum
        self.setChannel('.dynargsTabnum', self._dynargsTabnum)

        if self.hasBusSupport():
            self.csd.addEvent(self._builtinInstrs['clearbuses_post'], start=0, dur=-1)

    def renderMode(self) -> str:
        return 'offline'

    def initChannel(self,
                    channel: str,
                    value: float | str | None = None,
                    kind='',
                    mode='rw'):
        """
        Create a channel and, optionally set its initial value

        Args:
            channel: the name of the channel
            value: the initial value of the channel,
                will also determine the type (k, S)
            kind: One of 'k', 'S', 'a'. Leave unset to auto determine the channel type.
            mode: r for read, w for write, rw for both.

        .. note::
                the `mode` determines the communication direction between csound and
                a host when running csound via its api. For offline rendering and when
                using channels for internal communication this is irrelevant

        """
        if mode not in ('r', 'w', 'rw'):
            raise ValueError(f"Invalid mode '{mode}', it should be one of 'r', 'w', 'rw'")

        if not value and not kind:
            raise ValueError(f"Either a value or a kind must be given")

        if value is not None:
            valuetype = csoundlib.channelTypeFromValue(value)
            assert valuetype in 'kS'
            if kind and kind != valuetype:
                raise ValueError(f"A value of type '{valuetype}' was given, but it is not "
                                 f"compatible with kind '{kind}'")
            kind = valuetype

        channelDef = ChannelDef(name=channel, kind=kind, mode=mode)
        previousDef = self._channelRegistry.get(channel)
        if previousDef is not None:
            logger.warning(f"Channel '{channel}' already defined: {previousDef}. Skipiing")
            return
        self._channelRegistry[channel] = channelDef
        self.addGlobalCode(f'chn_{kind} "{channel}" "{mode}"')
        if value is not None:
            self.setChannel(channel=channel, value=value, delay=0.)

    def setChannel(self, channel: str, value: float | str, delay=0.
                   ) -> None:
        """
        Set the value of a software channel

        Args:
            channel: the name of the channel
            value: the new value, should match the type of the channel. Audio channels
                are not allowed offline
            delay: when to perform the operation. A delay of 0 will generate a chnset
                instruction at the instr0 level
        """
        if delay > 0:
            self.csd.addEvent(instr='_chnset', start=delay, dur=0., args=[channel, value])
        else:
            if isinstance(value, str):
                self.addGlobalCode(f'chnset "{value}", "{channel}"')
            else:
                self.addGlobalCode(f'chnset {value}, "{channel}"')

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
        assert 1 <= priority <= self._numbuckets
        instrnum = self._nameAndPriorityToInstrnum.get((instrname, priority))
        if instrnum is not None:
            return instrnum

        instrdef = self.instrs.get(instrname)
        if not instrdef:
            raise KeyError(f"instrument {instrname} is not defined")
        priority0 = priority - 1
        count = self._bucketCounters[priority0]
        if count > self._bucketSizes[priority0]:
            raise ValueError(
                f"Too many instruments ({count}) defined, max. is {self._bucketSizes[priority0]}")

        self._bucketCounters[priority0] += 1
        instrnum = self._bucketIndices[priority0] + count
        self._nameAndPriorityToInstrnum[(instrname, priority)] = instrnum
        self._instrnumToNameAndPriority[instrnum] = (instrname, priority)
        body = self.generateInstrBody(instr=instrdef)
        self.csd.addInstr(instr=instrnum, body=body, instrComment=instrname)
        return instrnum

    @staticmethod
    @cache
    def defaultInstrBody(instr: Instr) -> str:
        body = instr._preprocessedBody
        parts = []
        docstring, body = csoundlib.splitDocstring(body)
        if docstring:
            parts.append(docstring)

        if instr.controls:
            code = _namedControlsGenerateCodeOffline(instr.controls)
            parts.append(code)

        if instr.pfieldIndexToName:
            pfieldstext, body, docstring = instrtools.generatePfieldsCode(body, instr.pfieldIndexToName)
            if pfieldstext:
                parts.append(pfieldstext)
        parts.append(body)
        out = emlib.textlib.joinPreservingIndentation(parts)
        return textwrap.dedent(out)

    def generateInstrBody(self, instr: Instr) -> str:
        return Renderer.defaultInstrBody(instr)

    def _registerExitCallback(self, callback) -> None:
        """
        Register a function to be called when exiting this Renderer as context manager
        """
        self._exitCallbacks.add(callback)

    def registerInstr(self, instr: Instr) -> bool:
        """
        Register an Instr to be used in this Renderer

        Args:
            instr: the insturment to register

        Returns:
            true if the instrument was registered, False if
            it was already registered in the current form

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
        oldinstr = self.instrs.get(instr.name)
        if oldinstr is not None and instr == oldinstr:
            return False
        self.instrs[instr.name] = instr
        return True

    def defInstr(self,
                 name: str,
                 body: str,
                 args: dict[str, float|str] = None,
                 init: str = '',
                 priority: int = None,
                 doc: str = '',
                 includes: list[str] | None = None,
                 aliases: dict[str, str] = None,
                 useDynamicPfields: bool = None,
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
            priority: has no effect for offline rendering, only here to maintain
                the same interface with Session
            doc: documentation describing what this instr does
            includes: list of files to be included in order for this instr to work
            aliases: a dict mapping arg names to real argument names.
            useDynamicPfields: if True, use pfields to implement dynamic arguments (arguments
                given as k-variables). Otherwise dynamic args are implemented as named controls,
                using a big global table
            kws: any keywords are passed on to the Instr constructor.
                See the documentation of Instr for more information.

        Returns:
            the created Instr. If needed, this instr can be registered
            at any other Renderer/Session

        .. seealso: :class:`~csoundengine.instr.Instr`, :meth:`Session.defInstr <csoundengine.session.Session.defInstr>`

        Example
        ~~~~~~~

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
            ...                       args={'ibus': bus, 'kcutoff': 1000})
            >>> filt.automate('kcutoff', [3, 1000, 6, 200, 10, 4000])
        """
        instr = Instr(name=name, body=body, args=args, init=init,
                      includes=includes, aliases=aliases,
                      useDynamicPfields=useDynamicPfields,
                      **kws)
        self.registerInstr(instr)
        return instr

    def registeredInstrs(self) -> dict[str, Instr]:
        """
        Returns a dict (instrname: Instr) with all registered Instrs
        """
        return self.instrs

    def getInstr(self, name) -> Instr | None:
        """
        Find a registered Instr, by name

        Returns None if no such Instr was registered
        """
        return self.instrs.get(name)

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
        count = 1 + ((count+1) % self._numInstancesPerInstr - 1)
        p1 = instrnum+count/self._numInstancesPerInstr
        self._instanceCounters[instrnum] = count
        return p1

    def schedEvent(self, event: Event) -> SchedEvent:
        kws = event.kws or {}
        schedevent = self.sched(instrname=event.instrname,
                                delay=event.delay,
                                dur=event.dur,
                                priority=event.priority,
                                args=event.args,
                                **kws)
        if event.automations:
            for autom in event.automations:
                schedevent.automate(param=autom.param, pairs=autom.pairs,
                                    delay=autom.delay, mode=autom.interpolation,
                                    overtake=autom.overtake)
        return schedevent

    def sched(self,
              instrname: str,
              delay=0.,
              dur=-1.,
              priority=1,
              args: Sequence[float | str] | dict[str, float] | None = None,
              whenfinished: Callable = None,
              relative=True,
              **kwargs
              ) -> SchedEvent:
        """
        Schedule an event

        Args:
            instrname: the name of the already registered instrument
            priority: determines the order of execution
            delay: time offset
            dur: duration of this event. -1: endless
            args: pfields **beginning with p5**
                (p1: instrnum, p2: delay, p3: duration, p4: reserved)
            whenfinished: not relevant in the context of offline rendering
            relative: not relevant for offline rendering
            kwargs: any named argument passed to the instr

        Returns:
            a ScoreEvent, holding the csound event (p1, start, dur, args)


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
            ...   |kamp=0.1, kmidi=60|
            ...   outch 1, oscili:a(kamp, mtof:k(kmidi))
            ... ''')]
            >>> for instr in instrs:
            ...     renderer.registerInstr(instr)
            >>> renderer.sched('vco', dur=4, kmidi=67)
            >>> renderer.sched('sine', 2, dur=3, kmidi=68)
            >>> renderer.render('out.wav')

        """
        instr = self.getInstr(instrname)
        if instr is None:
            raise KeyError(f"Instrument '{instrname}' is not defined. Known instruments: "
                           f"{self.instrs.keys()}")

        pfields5, dynargs = instr.parseSchedArgs(args=args, kws=kwargs)
        event = self.makeEvent(start=float(delay), dur=float(dur), pfields5=pfields5,
                               instr=instr, priority=priority)
        self.csd.addEvent(event.p1, start=event.start, dur=event.dur, args=event.args)
        self.scheduledEvents[event.uniqueId] = event

        if instr.hasControls():
            controlvalues = instr.overrideControls(d=dynargs)
            self.csd.addEvent(instr='_initDynamicControls',
                              start=max(delay - self.ksmps / self.sr, 0.), dur=0,
                              args=[event.controlsSlot, len(controlvalues), *controlvalues])
        return event

    def makeEvent(self,
                  start: float,
                  dur: float,
                  pfields5: list[float|str],
                  instr: str | Instr,
                  priority: int = 1,
                  ) -> SchedEvent:
        """
        Create a SchedEvent for this Renderer

        This method does not schedule the event, it only creates it. It must
        be scheduled via :meth:`Renderer.schedEvent`

        Args:
            start: the start time
            dur: the duration
            pfields5: pfields, starting at p5
            instr: the name of the instr or the actual Instr instance
            priority: the priority
        """
        _instr = instr if isinstance(instr, Instr) else self.getInstr(instr)
        if _instr is None:
            raise ValueError(f"instrument '{instr}' not known")
        controlsSlot = self._dynargsAssignToken() if _instr.hasControls() else -1
        instrnum = self.commitInstrument(_instr.name, priority)
        p1 = self._getUniqueP1(instrnum)
        pfields4 = [controlsSlot, *pfields5]
        return SchedEvent(p1=p1, uniqueId=self._generateEventId(), start=start,
                          dur=dur, args=pfields4, instrname=_instr.name,
                          parent=self, priority=priority, controlsSlot=controlsSlot)

    def _dynargsAssignToken(self) -> int:
        self._dynargsTokenCounter = (self._dynargsTokenCounter + 1) % 2**32
        return self._dynargsTokenCounter

    def _schedEvent_old(self,
                        p1: float | int,
                        start: float,
                        dur: float,
                        args: list[float | str],
                        instrname: str = '',
                        priority=0,
                        controlsSlot=0
                        ) -> SchedEvent:
        self.csd.addEvent(p1, start=start, dur=dur, args=args)
        eventid = self._generateEventId()
        event = SchedEvent(p1=p1, start=start, dur=dur, args=args,
                           uniqueId=eventid, parent=self,
                           priority=priority, instrname=instrname,
                           controlsSlot=controlsSlot)
        self.scheduledEvents[eventid] = event
        return event

    def unsched(self, event: int | float | SchedEvent, delay: float) -> None:
        """
        Stop a scheduled event

        This schedule the stop of a playing event. The event
        can be an indefinite event (dur=-1) or it can be used
        to stop an event before its actual end

        Args:
            event: the event to stop
            delay: when to stop the given event
        """
        p1 = event.p1 if isinstance(event, SchedEvent) else event
        self.csd.addEvent("_stop", start=delay, dur=0, args=[p1])

    def hasBusSupport(self):
        """
        Returns True if this Engine was started with bus suppor

        """
        return (self._numAudioBuses > 0 or self._numControlBuses > 0)

    def assignBus(self, kind='', value=None, persist=False) -> busproxy.Bus:
        """
        Assign a bus

        Args:
            kind: the bus kind, one of 'audio' or 'control'. The value, if given,
                will determine the kind if `kind` is left unset
            value: an initial value for the bus, only valid for control buses
            persist: if True, the bus exists until it is manually released.
                Otherwise the bus exists as long as it is unused and remains
                alive as long as there are instruments using it

        Example
        ~~~~~~~

        .. code-block:: python

            from csoundengine import *
            r = Renderer()

            r.defInstr('sender', r'''
              ibus = p5
              ifreqbus = p6
              kfreq = busin:k(ifreqbus)
              asig vco2 0.1, kfreq
              busout(ibus, asig)
            ''')

            r.defInstr('receiver', r'''
              ibus  = p5
              kgain = p6
              asig = busin:a(ibus)
              asig *= a(kgain)
              outch 1, asig
            ''')

            bus = r.assignBus('audio')
            freqbus = s.assignBus(value=880)
            chain = [r.sched('sender', ibus=bus.token, ifreqbus=freqbus.token),
                     r.sched('receiver', priority=2, ibus=bus.token, kgain=0.5)]

            # Make a glissando
            freqbus.automate((0, 880, 5, 440))

        """
        assert self.hasBusSupport()

        if kind:
            if value is not None and kind == 'audio':
                raise ValueError(f"An audio bus cannot have a scalar value")
        else:
            kind = 'audio' if value is None else 'control'

        token = self._busTokenCount
        self._busTokenCount += 1
        ikind = BUSKIND_AUDIO if kind == 'audio' else BUSKIND_CONTROL
        ivalue = float(value) if value is not None else 0.
        args = [0, token, ikind, int(persist), ivalue]
        self.csd.addEvent(self._builtinInstrs['busassign'], 0, 0, args=args)
        return busproxy.Bus(token=token, kind=kind, renderer=self, bound=False)

    def _writeBus(self, bus: busproxy.Bus, value: float, delay=0.) -> None:
        if bus.kind != 'control':
            raise ValueError("This operation is only valid for control buses")
        self.csd.addEvent(self._builtinInstrs['busoutk'], start=delay, dur=0,
                          args=[bus.token, value])

    def _automateBus(self, bus: busproxy.Bus, pairs: Sequence[float],
                     mode='linear', delay=0., overtake=False):
        maxDataSize = config['max_pfields'] - 10
        if len(pairs) <= maxDataSize:
            args = [int(bus), self.strSet(mode), int(overtake), len(pairs), *pairs]
            self.csd.addEvent(self._builtinInstrs['automateBusViaPargs'],
                              start=delay,
                              dur=pairs[-2] + self.ksmps/self.sr,
                              args=args)
        else:
            for groupdelay, subgroup in internalTools.splitAutomation(pairs, maxDataSize//2):
                self._automateBus(bus=bus, pairs=subgroup, delay=groupdelay+delay,
                                  mode=mode, overtake=overtake)

    def _readBus(self, bus: busproxy.Bus) -> float | None:
        "Reading from a bus is not supported in offline mode"
        logger.error("Reading from a bus is not supported in offline mode")
        return None

    def _releaseBus(self, bus: busproxy.Bus) -> None:
        """
        The python counterpart of a bus does not need to be released in offline mode

        The csound bus itself will release itself
        """
        return None

    def _generateEventId(self) -> int:
        out = self._idCounter
        self._idCounter += 1
        return out

    def setCsoundOptions(self, *options: str) -> None:
        """
        Set any command line options to use by all render operations

        Options can also be set while calling :meth:`Renderer.render`

        Args:
            *options (str): any option will be passed directly to csound when rendering

        Examples
        ~~~~~~~~

            >>> from csoundengine.offline import Renderer
            >>> renderer = Renderer()
            >>> instr = Instr("sine", ...)
            >>> renderer.registerInstr(instr)
            >>> renderer.sched("sine", ...)
            >>> renderer.setCsoundOptions("--omacro:MYMACRO=foo")
            >>> renderer.render("outfile.wav")
        """
        self.csd.addOptions(*options)

    def renderDuration(self) -> float:
        """
        Returns the actual duration of the rendered score, considering an end marker

        Returns:
            the duration of the render, in seconds

        .. seealso:: :meth:`Renderer.setEndMarker`
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

        The end marker will **extend the rendering time** if it is placed **after** the
        end of the last event; it will also **crop** any *infinite* event. It does not
        have any effect if there are events with determinate duration ending after
        it. In this case **the end time of the render will be the end of the latest
        event**.

        .. note::

            To render only part of a score use the `starttime` and / or `endtime`
            parameters when calling :meth:`Renderer.render`
        """
        self._endMarker = time
        self.csd.setEndMarker(time)

    def render(self,
               outfile='',
               endtime=0.,
               encoding='',
               wait=True,
               verbose: bool | None = None,
               openWhenDone=False,
               starttime=0.,
               compressionBitrate: int = None,
               sr: int = None,
               ksmps: int = None,
               tail=0.,
               numthreads=0,
               csoundoptions: list[str] = None
               ) -> RenderJob:
        """
        Render to a soundfile

        To further customize the render set any csound options via
        :meth:`Renderer.setCsoundOptions`

        By default, if the output is an uncompressed file (.wav, .aif)
        the sample format is set to float32 (csound defaults to 16 bit pcm)

        Args:
            outfile: the output file to render to. The extension will determine
                the format (wav, flac, etc). None will render to a temp wav file.
            sr: the sample rate used for recording, overrides the samplerate of
                the renderer
            encoding: the sample encoding of the rendered file, given as
                'pcmXX' or 'floatXX', where XX represent the bit-depth
                ('pcm16', 'float32', etc). If no encoding is given a suitable default for
                the sample format is chosen
            wait: if True this method will block until the underlying process exits
            verbose: if True, all output from the csound subprocess is logged
            endtime: stop rendering at the given time. This will either extend or crop
                the rendering.
            tail: extra time at the end, usefull when rendering long reverbs
            starttime: start rendering at the given time. Any event ending previous to
                this time will not be rendered and any event between starttime and
                endtime will be cropped
            compressionBitrate: used when rendering to ogg
            openWhenDone: open the file in the default application after rendering. At
                the moment this will force the operation to be blocking, waiting for
                the render to finish.
            numthreads: number of threads to use for rendering. If not given, the
                value in ``config['rec_numthreads']`` is used
            csoundoptions: a list of options specific to this render job. Options
                given to the Renderer itself will be included in all render jobs

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
        renderend = min(endtime, scoreend, endmarker) + tail

        if renderend == float('inf'):
            raise RenderError("Cannot render an infinite score. Set an endtime when calling "
                              ".render(...)")
        if renderend <= scorestart:
            raise RenderError(f"No score to render (start: {scorestart}, end: {renderend})")

        if numthreads == 0:
            numthreads = config['rec_numthreads'] or config['numthreads']

        csd = self.csd.copy()

        if numthreads > 1:
            csd.numthreads = numthreads

        if csoundoptions:
            csd.addOptions(*csoundoptions)

        # if scoreend < renderend:
        #    csd.setEndMarker(renderend)
        if scoreend > renderend:
            csd.cropScore(end=renderend)

        csd.setEndMarker(renderend)

        verbose = verbose if verbose is not None else not config['rec_suppress_output']
        if verbose:
            runSuppressdisplay = False
            runPiped = False
        else:
            runSuppressdisplay = True
            runPiped = True

        if encoding is None:
            ext = os.path.splitext(outfile)[1]
            encoding = csoundlib.bestSampleEncodingForExtension(ext[1:])

        # We create a copy so that we can modify encoding/compression/etc
        if sr:
            if csd.sr != sr:
                logger.warning(f"Rendering with a different sr ({sr}) as the sr of this Renderer "
                               f"{self.sr}. This might result in unexpected results")
            csd.sr = sr

        if ksmps:
            csd.ksmps = ksmps

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

        renderjob = RenderJob(outfile=outfile, encoding=encoding, samplerate=self.sr,
                              endtime=endtime, starttime=starttime, process=proc)

        self.renderedJobs.append(renderjob)
        return renderjob

    def openLastSoundfile(self, app='') -> None:
        lastjob = self.lastRenderJob()
        if lastjob:
            lastjob.openOutfile(app=app)

    def lastRenderJob(self) -> RenderJob | None:
        """
        Returns the last RenderJob spawned by :meth:`Renderer.render`

        Returns:
            the last :class:`RenderJob` or None if no rendering has been
            performed yet

        .. seealso:: :meth:`Renderer.render`
        """
        return self.renderedJobs[-1] if self.renderedJobs else None

    def lastRenderedSoundfile(self) -> str | None:
        """
        Returns the last rendered soundfile, or None if no jobs were rendered
        """
        job = self.lastRenderJob()
        return job.outfile if job else None

    def writeCsd(self, outfile: str) -> None:
        """
        Generate the csd project for this renderer, write it to `outfile`

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

    def getEventById(self, eventid: int) -> SchedEvent | None:
        """
        Retrieve a scheduled event by its eventid

        Args:
            eventid: the event id, as returned by sched

        Returns:
            the ScoreEvent if it exists, or None
        """
        return self.scheduledEvents.get(eventid)

    def getEventsByP1(self, p1: float) -> list[SchedEvent]:
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

    def _instrFromEvent(self, event: SchedEvent) -> Instr:
        instrNameAndPriority = self._instrnumToNameAndPriority.get(int(event.p1))
        if not instrNameAndPriority:
            raise ValueError(f"Unknown instrument for instance {event.p1}")
        instr = self.instrs[instrNameAndPriority[0]]
        return instr

    def _setNamedControl(self,
                         event: SchedEvent,
                         param: str,
                         value: float,
                         delay: float = 0.):
        paramindex = event.instr.controlIndex(param)
        self.csd.addEvent("_setControl", start=delay, dur=0,
                          args=[event.controlsSlot, paramindex, value])

    def _setPfield(self, event: SchedEvent, delay: float, param: str, value: float
                   ) -> None:
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
            ... kmidi = p5
            ... outch 1, oscili:a(0.1, mtof:k(kmidi))
            ... ''')
            >>> event = renderer.sched("sine", args={'kmidi': 62})
            # .set invokes _setPfield in the renderer
            >>> event.set(delay=10, kmidi=67)

        """
        instr = self._instrFromEvent(event)
        if param not in instr.dynamicPfieldNames():
            if param not in instr.pfieldNames():
                raise ValueError(f"'{param}' is not a known pfield. Known pfields for "
                                 f"instr '{instr.name}' are: {instr.pfieldNames()}")
            else:
                raise ValueError(f"'{param}' is not a dynamic pfield. Modifying its "
                                 f"value via setp (pwrite) will have no effect")

        pfieldIndex = instr.pfieldIndex(param, 0)
        self.csd.addEvent("_pwrite", start=delay, dur=0.,
                          args=[event.p1, pfieldIndex, value])

    def makeTable(self,
                  data: np.ndarray | list[float] | None = None,
                  size: int = 0,
                  tabnum: int = 0,
                  sr: int = 0,
                  delay: float = 0.,
                  unique=True
                  ) -> TableProxy:
        """
        Create a table with given data or an empty table of the given size

        Args:
            data: the data of the table. Use None if the table should be empty
            size: if not data is given, sets the size of the empty table created
            tabnum: 0 to self assign a table number
            sr: the samplerate of the data, if applicable.
            delay: when to create this table
            unique: if True, create a table even if a table exists with the
                same data.

        Returns:
            a TableProxy
        """
        if data is not None:
            if isinstance(data, list):
                data = np.array(data)
            arrayhash = internalTools.ndarrayhash(data)
            if not unique and (tabproxy := self._ndarrayHashToTabproxy.get(arrayhash)) is not None:
                return tabproxy
            tabnum = self.csd.addTableFromData(data=data, tabnum=tabnum, start=delay, sr=sr)
            tabproxy = TableProxy(tabnum=tabnum, numframes=len(data), sr=sr)
            self._ndarrayHashToTabproxy[arrayhash] = tabproxy
            return tabproxy
        else:
            assert size > 0
            tabnum = self.csd.addEmptyTable(size=size, sr=sr, tabnum=tabnum)
            return TableProxy(tabnum=tabnum, numframes=size, sr=sr)

    def freeTable(self,
                  table: int | TableProxy,
                  delay: float = 0.) -> None:
        tabnum = table if isinstance(table, int) else table.tabnum
        self.csd.freeTable(tabnum, time=delay)

    def readSoundfile(self,
                      path='?',
                      chan: int = 0,
                      skiptime: float = 0.,
                      delay: float = 0.,
                      force=False
                      ) -> TableProxy:
        """
        Add code to this offline renderer to load a soundfile

        Args:
            path: the path of the soundfile to load. Use '?' to select a file using a GUI
                dialog
            chan: the channel to read, or 0 to read all channels
            delay: moment in the score to read this soundfile
            skiptime: skip this time at the beginning of the soundfile
            force: if True, add the soundfile to this renderer even if the same
                soundfile has already been added

        Returns:
            a TableProxy, representing the table holding the soundfile
        """
        if path == "?":
            path = _state.openSoundfile()

        tabproxy = self._soundfileRegistry.get(path)
        if tabproxy is not None:
            logger.warning(f"Soundfile '{path}' has already been added to this project")
            if not force and tabproxy.skiptime == skiptime:
                return tabproxy

        tabnum = self.csd.addSndfile(sndfile=path,
                                     start=delay,
                                     skiptime=skiptime,
                                     chan=chan)
        info = sndfileio.sndinfo(path)
        tabproxy = TableProxy(tabnum=tabnum, parent=self, numframes=info.nframes,
                              sr=info.samplerate, nchnls=info.channels, path=path,
                              skiptime=skiptime)
        self._soundfileRegistry[path] = tabproxy
        return tabproxy

    def playSample(self,
                   source: int | str | TableProxy | tuple[np.ndarray, int],
                   delay=0.,
                   dur=0,
                   chan=1,
                   gain=1.,
                   speed=1.,
                   loop=False,
                   pan=0.5,
                   skip=0.,
                   fade: float | tuple[float, float] | None = None,
                   crossfade=0.02,
                   ) -> SchedEvent:
        """
        Play a table or a soundfile

        Adds an instrument definition and an event to play the given
        table as sound (assumes that the table was allocated via
        :meth:`~Renderer.readSoundFile` or any other GEN1 ftgen)

        Args:
            source: the table number to play, a :class:`~csoundengine.TableProxy`,
                the path of a soundfile or a tuple (numpy array, sr).
                Use '?' to select a file using a GUI dialog
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
            crossfade: if looping, this indicates the length of the crossfade
        """
        if loop and dur == 0:
            logger.warning(f"playSample was called with loop=True, but the duration ({dur}) given "
                           f"will result in no looping taking place")
        if isinstance(source, tuple):
            assert len(source) == 2 and isinstance(source[0], np.ndarray)
            data, sr = source
            tabproxy = self.makeTable(data=source[0], sr=sr)
            tabnum = tabproxy.tabnum
            if dur == 0:
                dur = (len(data)/sr) / speed
        elif isinstance(source, TableProxy):
            tabnum = source.tabnum
        elif isinstance(source, str):
            tabproxy = self.readSoundfile(path=source, delay=delay, skiptime=skip)
            tabnum = tabproxy.tabnum
            if dur == 0:
                dur = tabproxy.duration() / speed
        elif isinstance(source, int):
            tabnum = source
            if dur == 0:
                dur = -1
        else:
            raise TypeError(f"Not a valid source: {source}, expected a TableProxy, "
                            f"path, table number or sample data as (samples, sr: int)")
        assert tabnum > 0
        if not loop:
            crossfade = -1

        if isinstance(fade, (int, float)):
            fadein = fadeout = fade
        elif isinstance(fade, tuple):
            fadein, fadeout = fade
        elif fade is None:
            fadein = fadeout = config['sample_fade_time']
        else:
            raise TypeError(f"fade should be None to use default, or a time or a tuple "
                            f"(fadein, fadeout), got {fade}")

        args = dict(isndtab=tabnum, istart=skip,
                    ifadein=fadein, ifadeout=fadeout,
                    kchan=chan, kspeed=speed, kpan=pan, kgain=gain,
                    ixfade=crossfade)
        return self.sched('.playSample', delay=delay, dur=dur, args=args)

    def automate(self,
                 event: SchedEvent,
                 param: str,
                 pairs: Sequence[float] | np.ndarray,
                 mode="linear",
                 delay: float = None,
                 overtake=False
                 ) -> float:
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
            overtake: if True, the first value is not used, the current value
                for the given parameter is used in its place.
        """
        instr = event.instr
        pairs = internalTools.aslist(pairs)
        param = instr.unaliasParam(param, param)
        params = instr.dynamicParamNames(aliases=False)
        if param not in params:
            raise KeyError(f"Unknown parameter '{param}' for {event}. Possible "
                           f"parameters: {params}")

        if delay is None:
            delay = event.start

        automStart = delay + pairs[0]
        automEnd = delay + pairs[-2]
        if automEnd <= event.start or automStart >= event.end:
            # automation line ends before the actual event!!
            logger.debug(f"Automation times outside of this event: {param=}, "
                         f"automation start-end: {automStart} - {automEnd}, "
                         f"event: {event}")
            return 0

        if automStart > event.start or automEnd < event.end:
            pairs, delay = internalTools.cropDelayedPairs(pairs=pairs, delay=delay,
                                                          start=automStart, end=automEnd)
            if not pairs:
                logger.warning("There is no intersection between event and automation data")
                return 0.

        if pairs[0] > 0:
            pairs, delay = internalTools.consolidateDelay(pairs, delay)

        instr = event.instr
        assert instr is not None
        maxDataSize = config['max_pfields'] - 10
        if len(pairs) > maxDataSize:
            for subdelay, subgroup in internalTools.splitAutomation(pairs, maxDataSize//2):
                self.automate(event=event, param=param, pairs=subgroup, mode=mode,
                              delay=delay+subdelay, overtake=overtake)
            return 0.

        if isinstance(param, int):
            param = f'p{param}'

        if instr.hasControls() and param in instr.controlNames(aliases=False):
            self._automateTable(event=event, param=param, pairs=pairs, mode=mode, delay=delay,
                                overtake=overtake)
        elif csoundlib.isPfield(param) or param in instr.pfieldNames(aliases=False):
            self._automatePfield(event=event, param=param, pairs=pairs, mode=mode, delay=delay,
                                 overtake=overtake)
        else:
            raise KeyError(f"Parameter '{param}' not known. Controls: {instr.controlNames(aliased=True)}, "
                           f"pfields: {instr.pfieldNames()}")
        return 0.

    def _getTableData(self, table: int | TableProxy) -> np.ndarray | None:
        logger.error("An offline renderer cannot access table data")
        return None

    def _automatePfield(self,
                        event: SchedEvent,
                        param: str,
                        pairs: Sequence[float] | np.ndarray,
                        delay: float,
                        mode="linear",
                        overtake=False
                        ) -> None:
        instr = event.instr
        pfieldindex = instr.pfieldIndex(param)
        dur = pairs[-2]-pairs[0]
        epsilon = self.csd.ksmps / self.csd.sr * 3
        start = max(0., delay-epsilon)
        if event.dur > 0:
            # we clip the duration of the automation to the lifetime of the automated event
            end = min(event.start+event.dur, start+dur+epsilon)
            dur = end-start
        args = [event.p1, pfieldindex, self.strSet(mode), int(overtake), len(pairs), *pairs]
        self.csd.addEvent('_automatePargViaPargs', start=delay, dur=dur, args=args)

    def _automateTable(self,
                       event: SchedEvent,
                       param: str,
                       pairs: Sequence[float]|np.ndarray,
                       delay: float,
                       mode="linear",
                       overtake=False
                       ) -> None:
        """
        Automate a named control of an event
        """
        # splitting is done in automate
        assert len(pairs) < config['max_pfields'] and len(pairs) % 2 == 0
        instr = event.instr
        paramindex = instr.controlIndex(param)
        dur = pairs[-2] - pairs[0]
        epsilon = self.csd.ksmps / self.csd.sr * 3
        assert delay is not None
        start = max(0., delay - epsilon)
        if event.dur > 0:
            # we clip the duration of the automation to the lifetime of the event
            end = min(event.start + event.dur, start + dur + epsilon)
            dur = end - start
        imode = self.strSet(mode)
        args = [event.controlsSlot, paramindex, imode, int(overtake), len(pairs), *pairs]
        self.csd.addEvent('_automateControlViaPargs', start=delay, dur=dur, args=args)

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

    def playPartials(self,
                     source: int | str | TableProxy | np.ndarray,
                     delay=0.,
                     dur=-1,
                     speed=1.,
                     freqscale=1.,
                     gain=1.,
                     bwscale=1.,
                     loop=False,
                     chan=1,
                     start=0.,
                     stop=0.,
                     minfreq=0,
                     maxfreq=0,
                     maxpolyphony=50,
                     gaussian=False,
                     interpfreq=True,
                     interposcil=True,
                     position=0.
                     ) -> SchedEvent:
        """
        Play a packed spectrum

        A packed spectrum is a 2D numpy array representing a fixed set of
        oscillators. After partial tracking analysis, all partials are arranged
        into such a matrix where each row represents the state of all oscillators
        over time.

        The **loristrck** packge is needed for both partial-tracking analysis and
        packing. It can be installed via ``pip install loristrck`` (see
        https://github.com/gesellkammer/loristrck). This is an optional dependency


        Args:
            source: a table number, tableproxy, path to a .mtx or .sdif file, or
                a numpy array containing the partials data
            delay: when to start the playback
            dur: duration of the synth (-1 will play indefinitely if looping or until
                the end of the last partial or the end of the selection
            speed: speed of playback (does not affect pitch)
            loop: if True, loop the selection or the entire spectrum
            chan: channel to send the output to
            start: start of the time selection
            stop: stop of the time selection (0 to play until the end)
            minfreq: lowest frequency to play
            maxfreq: highest frequency to play
            gaussian: if True, use gaussian noise for residual resynthesis
            interpfreq: if True, interpolate frequency between cycles
            interposcil: if True, use linear interpolation for the oscillators
            maxpolyphony: if a sdif is passed, compress the partials to max. this
                number of simultaneous oscillators
            position: pan position
            freqscale: frequency scaling factor
            gain: playback gain
            bwscale: bandwidth scaling factor

        Returns:
            the playing Synth

        Example
        ~~~~~~~

            >>> import loristrck as lt
            >>> import csoundengine as ce
            >>> samples, sr = lt.util.sndread("/path/to/soundfile")
            >>> partials = lt.analyze(samples, sr, resolution=50)
            >>> lt.util.partials_save_matrix(partials, outfile='packed.mtx')
            >>> session = ce.Engine().session()
            >>> session.playPartials(source='packed.mtx', speed=0.5)

        """
        iskip, inumrows, inumcols = -1, 0, 0

        if isinstance(source, int):
            tabnum = source
        elif isinstance(source, TableProxy):
            tabnum = source.tabnum
        elif isinstance(source, str):
            # a .mtx file
            ext = os.path.splitext(source)[1]
            if ext == '.mtx':
                table = self.readSoundfile(source)
                tabnum = table.tabnum
            elif ext == '.sdif':
                try:
                    import loristrck as lt
                    partials, labels = lt.read_sdif(source)
                    tracks, matrix = lt.util.partials_save_matrix(partials=partials, maxtracks=maxpolyphony)
                    tabnum = self.makeTable(matrix).tabnum
                except ImportError:
                    raise ImportError("loristrck is needed in order to read a .sdif file. "
                                      "Install it via `pip install loristrck` (see https://loristrck.readthedocs.io "
                                      "for more information)")
            else:
                raise ValueError(f"Expected a .mtx file or .sdif file, got {source}")

        elif isinstance(source, np.ndarray):
            assert len(source.shape) == 2
            array = source.flatten()
            table = self.makeTable(array, unique=False)
            tabnum = table.tabnum
            iskip = 0
            inumrows, inumcols = source.shape

        else:
            raise TypeError(f"Expected int, TableProxy or str, got {source}")

        flags = 1 * int(gaussian) + 2 * int(interposcil) + 4 * int(interpfreq)
        return self.sched('.playPartials',
                          delay=delay,
                          dur=dur,
                          args=dict(ifn=tabnum,
                                    iskip=iskip,
                                    inumrows=inumrows,
                                    inumcols=inumcols,
                                    kspeed=speed,
                                    kloop=int(loop),
                                    kminfreq=minfreq,
                                    kmaxfreq=maxfreq,
                                    ichan=chan,
                                    istart=start,
                                    istop=stop,
                                    kfreqscale=freqscale,
                                    iflags=flags,
                                    iposition=position,
                                    kbwscale=bwscale,
                                    kgain=gain))

# ------------------ end Renderer ---------------------


def cropScore(events: list[SchedEvent], start=0., end=0.) -> list[SchedEvent]:
    """
    Crop the score so that no event exceeds the given limits

    Args:
        events: a list of ScoreEvents
        start: the min. start time for any event
        end: the max. end time for any event

    Returns:
        a list with the cropped events
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
            if xstart is not None:
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


def _namedControlsGenerateCodeOffline(controls: dict) -> str:
    """
    Generates code for an instr to read named controls offline

    Args:
        controls: a dict mapping control name to default value. The
            keys are valid csound k-variables

    Returns:
        the generated code
    """

    lines = [fr'''
    ; --- start generated code for dynamic args
    i__token__ = p4
    i__tabnum__ = gi__dynargsTable
    i__slot__ = _getControlSlot(i__token__)
    i__slicestart__ = i__slot__ * gi__dynargsSliceSize
    atstop "_releaseDynargsToken", 0, 0, i__token__
    ''']
    idx = 0
    for key, value in controls.items():
        assert key.startswith('k')
        lines.append(f"    {key} tab i__slicestart__ + {idx}, i__tabnum__")
        idx += 1
    lines.append("    ; --- end generated code\n")
    out = emlib.textlib.stripLines(emlib.textlib.joinPreservingIndentation(lines))
    return out
