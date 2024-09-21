"""
A :class:`Session` provides a high-level interface to control an underlying
csound process. A :class:`Session` is associated with an
:class:`~csoundengine.engine.Engine` (there is one Session per Engine)
"""

from __future__ import annotations
import weakref
import os
from dataclasses import dataclass
from collections import deque
import numpy as np
from functools import cache
import textwrap
import queue as _queue
import threading
import sndfileio

import emlib.dialogs as _dialogs
import emlib.textlib as _textlib
import emlib.numpytools as _numpytools
import bpf4

from .abstractrenderer import AbstractRenderer
from .event import Event
from .schedevent import SchedEvent
from .errors import CsoundError
from .engine import Engine, getEngine
from . import engineorc
from .instr import Instr
from .synth import Synth, SynthGroup
from .tableproxy import TableProxy
from .config import config, logger
from . import internal as _tools
from .sessioninstrs import builtinInstrs
from . import state as _state
from . import jupytertools
from . import instrtools
from . import csoundlib
from . import offline
from . import busproxy
from .sessionhandler import SessionHandler

from typing import Callable, Sequence


__all__ = [
    'Session',
    'Event'
]


@dataclass
class _ReifiedInstr:
    """
    A _ReifiedInstr is just a marker of a concrete instr sent to the
    engine for a given Instr template.

    An Instr is an abstract declaration without a specific instr number and thus
    without a specific order of execution. To be able to schedule an instrument
    at different places in the chain, the same instrument is redeclared (lazily)
    as different instrument numbers depending on the priority. When an instr
    is scheduled at a given priority for the first time a ReifiedInstr is created
    to mark that and the code is sent to the engine
    """

    instrnum: int
    """the actual instrument number inside csound"""

    priority: int
    """the priority of this instr"""

    def __post_init__(self):
        assert isinstance(self.instrnum, int)


class _RenderingSessionHandler(SessionHandler):
    """
    Adapts a Session for offline rendering
    """
    def __init__(self, renderer: offline.OfflineSession):
        self.renderer = renderer

    def schedEvent(self, event: Event):
        return self.renderer.schedEvent(event)

    def makeTable(self,
                  data: np.ndarray | list[float] | None = None,
                  size: int | tuple[int, int] = 0,
                  sr: int = 0,
                  ) -> TableProxy:
        return self.renderer.makeTable(data=data, size=size, sr=sr)

    def readSoundfile(self,
                      path: str,
                      chan=0,
                      skiptime=0.,
                      delay=0.,
                      force=False,
                      ) -> TableProxy:
        return self.renderer.readSoundfile(path)


class Session(AbstractRenderer):
    """
    A Session is associated (exclusively) to a running
    :class:`~csoundengine.engine.Engine` and manages instrument declarations
    and scheduled events. An Engine can be thought of as a low-level interface
    for managing a csound instance, whereas a Session allows a higher-level control

    **A user normally does not create a Session manually**: the normal way to create a
    Session for a given Engine is to call :meth:`~csoundengine.engine.Engine.session`
    (see example below)

    Once a Session is created for an existing Engine,
    calling :meth:`~csoundengine.engine.Engine.session` again will always return the
    same Session object.

    Example
    -------

    In order to add an instrument to a :class:`~csoundengine.session.Session`,
    an :class:`~csoundengine.instr.Instr` is created and registered with the Session.
    Alternatively, the shortcut :meth:`~Session.defInstr` can be used to create and
    register an :class:`~csoundengine.instr.Instr` at once.

    .. code::

        s = Engine().session()
        s.defInstr('sine', r'''
            |kfreq=440, kamp=0.1|
            a0 = oscili:a(kamp, kfreq)
            outch 1, a0
        ''')
        synth = s.sched('sine', kfreq=500)
        synth.stop()


    An :class:`~csoundengine.instr.Instr` can define default values for any of
    parameters. By default, any dynamic argument (any argument starting with 'k')
    will be implemented as a dynamic control and not as a pfield. On the contrary,
    any init-time argument will be implemented as a pfield.

    .. code::

        s = Engine().session()
        s.defInstr('sine', args={'kamp': 0.1, 'kfreq': 1000}, body=r'''
            a0 = oscili:a(kamp, kfreq)
            outch 1, a0
        ''')
        # We schedule an event of sine, kamp will take the default (0.1)
        synth = s.sched('sine', kfreq=440)
        synth.stop()

    An inline args declaration can set both parameter name and default value:

    .. code::

        s = Engine().session()
        Intr('sine', r'''
            |kamp=0.1, kfreq=1000|
            a0 = oscili:a(kamp, kfreq)
            outch 1, a0
        ''').register(s)
        synth = s.sched('sine', kfreq=440)
        synth.stop()

    To force usage of pfields for dynamic args you need to use manual declaration:

    .. code::

        s.defInstr('sine', r'''
        ;                     p5   p6
        pset p1, p2, p3, 0,   0.1, 1000
        kamp = p5
        kfreq = p6
        outch 1, oscili:a(kamp, kfreq)
        ''')
        synth = s.sched('sine', kfreq=440)


    """
    def __new__(cls,
                engine: str | Engine | None = None,
                priorities: int | None = None,
                dynamicArgsPerInstr: int | None = None,
                dynamicArgsSlots: int | None = None,
                **enginekws):

        if isinstance(engine, str):
            _engine = Engine.activeEngines.get(engine)
            if not _engine:
                raise KeyError(f"Engine {engine} does not exist!")
        else:
            _engine = engine if isinstance(engine, Engine) else None
        if _engine and _engine._session:
            return _engine._session
        return super().__new__(cls)

    def __init__(self,
                 engine: str | Engine | None = None,
                 priorities: int | None = None,
                 maxControlsPerInstr: int | None = None,
                 numControlSlots: int | None = None,
                 **enginekws
                 ) -> None:
        """
        A Session controls a csound Engine

        Normally a user does not create a Session directly, but calls the
        :meth:`Engine.session() <csoundengine.engine.Engine.session>`` method

        Args:
            engine: the parent engine. If no engine is given, an engine with
                default parameters will be created. To customize the engine,
                the canonical way of creating a session is to use
                ``session = Engine(...).session()``
            priorities: the max. number of priorities for scheduled instrs
            maxControlsPerInstr: the max. number of named controls per instr
            numControlSlots: the total number of slots allocated for
                dynamic parameters. Each synth which declares named controls is
                assigned a slot, used to hold all its named controls. This is also
                the max. number of simultaneous events with named controls.
            enginekws: any keywords are used to create an Engine, if no engine
                has been provided. See docs for :class:`~csoundengine.engine.Engine`
                for available keywords.

        Example
        ~~~~~~~

            >>> from csoundengine import *
            >>> session = Engine(nchnls=4, nchnls_i=2).session()

        This is the same as

            >>> engine = Engine(nchnls=4, nchnls_i=2)
            >>> session = Session(engine=engine)
        """
        if not engine:
            _engine = Engine(**enginekws)
            logger.debug(f"Creating an Engine with default arguments: {engine}")
        elif isinstance(engine, str):
            _engine = Engine.activeEngines.get(engine)
            if _engine is None:
                raise ValueError(f"Engine '{engine}' does not exist")
            if _engine._session is not None:
                raise ValueError(f"The given engine already has an active session: {_engine._session}")
        elif isinstance(engine, Engine):
            _engine = engine
        else:
            raise TypeError(f"Expected an Engine or an engine name, got {engine}")

        self.engine: Engine = _engine
        """The Engine corresponding to this Session"""

        self.name: str = _engine.name
        """The name of this Session/Engine"""

        self.instrs: dict[str, Instr] = {}
        "maps instr name to Instr"

        self.numPriorities: int = priorities if priorities else config['session_priorities']
        "Number of priorities in this Session"

        if not isinstance(self.numPriorities, int) or self.numPriorities < 2:
            raise ValueError(f"Invalid number of priorites. Expected an int >= 2, got "
                             f"{self.numPriorities}")

        self._instrIndex: dict[int, Instr] = {}
        """A dict mapping instr id to Instr. This keeps track of defined instruments"""

        self._sessionInstrStart = engineorc.CONSTS['sessionInstrsStart']
        """Start of the reserved instr space for session"""

        bucketSizeCurve = bpf4.expon(0.7, 1, 500, self.numPriorities, 50)
        bucketSizes = [int(size) for size in bucketSizeCurve.map(self.numPriorities)]
        bucketIndices = [self._sessionInstrStart + sum(bucketSizes[:i])
                         for i in range(self.numPriorities)]

        self._bucketSizes = bucketSizes
        """Size of each bucket, by bucket index"""

        self._bucketIndices = bucketIndices
        """The start index of each bucket"""

        self._buckets: list[dict[str, int]] = [{} for _ in range(self.numPriorities)]

        self._reifiedInstrDefs: dict[str, dict[int, _ReifiedInstr]] = {}
        "A dict of the form {instrname: {priority: reifiedInstr }}"

        self._synths: dict[float, Synth] = {}
        self._whenfinished: dict[float, Callable] = {}
        self._initCodes: list[str] = []
        self._tabnumToTabproxy: dict[int, TableProxy] = {}
        self._pathToTabproxy: dict[str, TableProxy] = {}
        self._ndarrayHashToTabproxy: dict[str, TableProxy] = {}
        self._offlineRenderer: offline.OfflineSession | None = None
        self._inbox: _queue.Queue[Callable] = _queue.Queue()
        self._acceptingMessages = True
        self._dispatchingThread: threading.Thread | None = None

        self._notificationUseOsc = False
        self._notificationOscPort = 0
        self._includes: set[str] = set()
        self._lockedLatency: float | None = None
        self._handler: SessionHandler | None = None

        self.maxDynamicArgs = maxControlsPerInstr or config['max_dynamic_args_per_instr']
        """The max. number of dynamic parameters per instr"""

        self._dynargsNumSlots = numControlSlots or config['dynamic_args_num_slots']
        self._dynargsTabnum = _engine.makeEmptyTable(size=self.maxDynamicArgs * self._dynargsNumSlots)
        _engine.sync()
        _engine.setChannel(".dynargsTabnum", self._dynargsTabnum)
        self._dynargsArray = _engine.getTableData(self._dynargsTabnum)

        # We don't use slice 0. We use a deque as pool instead of a list, this helps
        # debugging
        self._dynargsSlotPool: deque[int] = deque(range(1, self._dynargsNumSlots))

        _engine.registerOutvalueCallback("__dealloc__", self._deallocCallback)
        if config['define_builtin_instrs']:
            self._defBuiltinInstrs()
        mininstr, maxinstr = self._reservedInstrRange()
        _engine._reserveInstrRange('session', mininstr, maxinstr)
        _engine._session = self

        # dispatchingThread = threading.Thread(target=self._dispatcher)
        # dispatchingThread.start()
        # self._dispatchingThread = dispatchingThread

    def __del__(self):
        if self._dispatchingThread:
            self._acceptingMessages = False
            self._dispatchingThread.join(timeout=1)

    def __hash__(self):
        return id(self)

    def isRendering(self) -> bool:
        """Is an offline renderer attached to this session?"""
        return self._offlineRenderer is not None

    def hasHandler(self) -> bool:
        """
        Does this session have a handler to redirect actions?

        .. seealso:: :meth:`Session.setHandler`
        """
        return self._handler is not None

    def stop(self) -> None:
        """Stop this session and the underlying engine"""
        self.engine.stop()
        self.engine._session = None

    def _dispatcher(self):
        while self._acceptingMessages:
            task = self._inbox.get()
            if callable(task):
                task()
            elif isinstance(task, str):
                print(f"Message: '{task}'")
            else:
                print("error", task)

    def hasBusSupport(self) -> bool:
        """Does the underlying engine have bus support?"""
        return self.engine.hasBusSupport()

    def getSynthById(self, token: int) -> Synth | None:
        return self._synths.get(token)

    @property
    def now(self) -> float:
        return self.engine.elapsedTime()

    def automate(self,
                 event: SchedEvent,
                 param: str,
                 pairs: Sequence[float] | np.ndarray | tuple[np.ndarray, np.ndarray],
                 mode='linear',
                 delay: float | None = None,
                 overtake=False,
                 ) -> float:
        """
        Automate any named parameter of this Synth

        Raises KeyError if the parameter is unknown

        Args:
            event: the event to automate
            param: the name of the parameter to automate
            pairs: automation data as a flat array with the form [time0, value0, time1, value1, ...]
            mode: one of 'linear', 'cos'. Determines the curve between values
            delay: relatime time from now to start the automation. If None is given, sync the start
                of the automation to the start of the given event.
            overtake: if True, do not use the first value in pairs but overtake the current value

        Returns:
            the eventid of the automation event.
        """
        now = self.engine.elapsedTime()
        relstart = delay if delay is not None else event.start - now

        if isinstance(pairs, np.ndarray):
            # TODO: check (in general) if this converting between numpy/list/numpy
            # can be a bottleneck when automation lines get big
            pairs = pairs.tolist()
        elif isinstance(pairs, tuple) and isinstance(pairs[0], np.ndarray):
            assert len(pairs) == 2
            pairs = _numpytools.interlace(*pairs).tolist()

        assert isinstance(pairs, (list, tuple))

        if len(pairs) == 2:
            t0 = float(pairs[0])
            event.set(param=param, delay=relstart + t0, value=float(pairs[1]))
            return 0

        absAutomStart = now + relstart + pairs[0]
        absAutomEnd = now + relstart + pairs[-2]
        if absAutomStart < event.start or absAutomEnd > event.end:
            pairs, absdelay = _tools.cropDelayedPairs(pairs=pairs, delay=now + relstart, start=absAutomStart,
                                                      end=absAutomEnd)
            if not pairs:
                return 0
            relstart = absdelay - now

        if pairs[0] > 0:
            pairs, relstart = _tools.consolidateDelay(pairs, relstart)

        if csoundlib.isPfield(param):
            return self._automatePfield(event=event, param=param, pairs=pairs, mode=mode, delay=relstart,
                                        overtake=overtake)

        param = event.unaliasParam(param, param)
        instr = event.instr
        params = instr.dynamicParams(aliases=False)
        if param not in params:
            raise KeyError(f"Unknown parameter '{param}' for {self}. Possible parameters: {params}")

        if (controlnames := instr.controlNames(aliases=False)) and param in controlnames:
            return self._automateTable(event=event, param=param, pairs=pairs, mode=mode,
                                       delay=relstart, overtake=overtake)
        elif (pargs := instr.pfieldNames(aliases=False)) and param in pargs:
            return self._automatePfield(event=event, param=param, pairs=pairs, mode=mode,
                                        delay=relstart, overtake=overtake)
        else:
            raise KeyError(f"Unknown parameter '{param}', supported parameters: {instr.dynamicParamNames()}")

    def _automatePfield(self,
                        event: SchedEvent,
                        param: int | str,
                        pairs: Sequence[float] | np.ndarray,
                        mode='linear',
                        delay=0.,
                        overtake=False):
        if event.playStatus() == 'stopped':
            raise RuntimeError(f"The event {event} has already stopped, cannot automate")

        if isinstance(param, str):
            assert event.instr is not None
            pidx = event.instr.pfieldIndex(param)
            if not pidx:
                raise KeyError(f"pfield '{param}' not known. Known pfields: {event.instr.pfieldIndexToName}")
        else:
            pidx = param
        assert isinstance(event.p1, float)
        synthid = self.engine.automatep(event.p1, pidx=pidx, pairs=pairs, mode=mode, delay=delay, overtake=overtake)
        return synthid

    def _automateTable(self,
                       event: SchedEvent,
                       param: str,
                       pairs: Sequence[float] | np.ndarray,
                       mode="linear",
                       overtake=False,
                       delay=0.) -> float:
        """
        Automate a dynamic parameter of a synth

        Args:
            event: the synth to automate
            param: the parameter name
            pairs: a flat sequence of the form (t0, value0, t1, value1, ...)
                where times are relative to the start of the automation line.
                Normally t0 is 0.
            mode: interpolation mode, one of 'linear' or 'cos'
            overtake: if True, the first value is not used and instead the
                current value of the parameter is used. This same overtake can
            delay: when to start the automation line.

        Returns:
            the id of the automation event, as float
        """
        assert event.instr is not None
        slot = event.instr.controlIndex(param)
        if slot is None:
            raise KeyError(f"Unknown parameter '{param}' for instr {event.instr.name}. "
                           f"Possible parameters: {event.instr.dynamicParamNames()}")
        if event.playStatus() == 'stopped':
            logger.error(f"Synth {self} has already stopped, cannot "
                         f"mset param '{param}'")
            return 0.

        idx = event.controlsSlot * self.maxDynamicArgs + slot
        return self.engine.automateTable(tabnum=self._dynargsTabnum,
                                         idx=idx,
                                         pairs=pairs,
                                         mode=mode,
                                         delay=delay,
                                         overtake=overtake)

    def renderMode(self) -> str:
        """The render mode of this Renderer"""
        return 'online'

    def _reservedInstrRange(self) -> tuple[int, int]:
        lastinstrnum = self._bucketIndices[-1] + self._bucketSizes[-1]
        return self._sessionInstrStart, lastinstrnum

    def __repr__(self):
        active = len(self.activeSynths())
        return f"Session({self.name}, synths={active})"

    def _repr_html_(self):
        active = len(self.activeSynths())
        if active and jupytertools.inside_jupyter():
            jupytertools.displayButton("Stop Synths", self.unschedAll)
        name = jupytertools.htmlName(self.name)
        return f"Session({name}, synths={active})"

    def _deallocSynthResources(self, synthid: int | float, delay=0.) -> None:
        """
        Deallocates resources associated with synth

        The actual csound event is not freed, since this function is
        called by "atstop" when a synth is actually stopped

        Args:
            synthid: the id (p1) of the synth
        """
        if delay > 0:
            def callback(session=self, synthid=synthid):
                session._deallocSynthResources(synthid=synthid)
            self.engine.callLater(delay, callback=callback)

        synth = self._synths.pop(synthid, None)
        if synth is None:
            return
        synth._scheduled = False
        if synth.controlsSlot:
            assert synth.args and synth.controlsSlot * self.maxDynamicArgs == synth.args[0]
            self._dynargsReleaseSlot(int(synth.controlsSlot))

        if (callback := self._whenfinished.pop(synthid, None)) is not None:
            callback(synthid)

    def _deallocCallback(self, _, synthid):
        """ This is called by csound when a synth is deallocated """
        self._deallocSynthResources(synthid)

    def _registerInstrAtPriority(self, instrname: str, priority=1) -> int:
        """
        Get the instrument number corresponding to this name and the given priority

        Args:
            instrname: the name of the instr as given to defInstr
            priority: the priority, an int from 1 to 10. Instruments with
                low priority are executed before instruments with high priority

        Returns:
            the instrument number (an integer)
        """
        if not 1 <= priority <= self.numPriorities:
            raise ValueError(f"Priority {priority} out of range (allowed range: 1 - "
                             f"{self.numPriorities})")
        bucketidx = priority - 1
        bucket = self._buckets[bucketidx]
        instrnum = bucket.get(instrname)
        if instrnum is not None:
            return instrnum
        bucketstart = self._bucketIndices[bucketidx]
        idx = len(bucket) + 1
        if idx >= self._bucketSizes[bucketidx]:
            raise RuntimeError(f"Too many instruments defined with priority {priority}")
        instrnum = bucketstart + idx
        bucket[instrname] = instrnum
        return instrnum

    def setHandler(self, handler: SessionHandler | None
                   ) -> SessionHandler | None:
        """
        Set a SessionHandler for this session

        This is used internally to delegate actions to an offline renderer
        when this session is rendering.
        """
        prevhandler = self._handler
        self._handler = handler
        return prevhandler

    def defInstr(self,
                 name: str,
                 body: str,
                 args: dict[str, float|str] | None = None,
                 init='',
                 priority: int | None = None,
                 doc='',
                 includes: list[str] | None = None,
                 aliases: dict[str, str] | None = None,
                 useDynamicPfields: bool | None = None,
                 **kws) -> Instr:
        """
        Create an :class:`~csoundengine.instr.Instr` and register it at this session

        Any init code given is compiled and executed at this point

        Args:
            name (str): the name of the created instr
            body (str): the body of the instrument. It can have named
                pfields (see example) or a table declaration
            args: pfields with their default values
            init: init (global) code needed by this instr (read soundfiles,
                load soundfonts, etc)
            priority: if given, the instrument is prepared to be executed
                at this priority
            doc: documentation describing what this instr does
            includes: list of files to be included in order for this instr to work
            aliases: a dict mapping arg names to real argument names. It enables
                to define named args for an instrument using any kind of name instead of
                following csound name
            useDynamicPfields: if True, use pfields to implement dynamic arguments (arguments
                given as k-variables). Otherwise dynamic args are implemented as named controls,
                using a big global table
            kws: any keywords are passed on to the Instr constructor.
                See the documentation of Instr for more information.

        Returns:
            the created Instr. If needed, this instr can be registered
            at any other running Session via session.registerInstr(instr)

        .. note::

            An instr is not compiled at the moment of definition: only
            when an instr is actually scheduled to be run at a given
            priority the code is compiled. There might be a small delay
            the first time an instr is scheduled at a given
            priority. To prevent this a user can give a default priority
            when calling :meth:`Session.defInstr`, or call
            :meth:`Session.prepareSched` to explicitely compile the instr


        Example
        ~~~~~~~

            >>> session = Engine().session()
            # An Instr with named parameters
            >>> session.defInstr('filter', r'''
            ... a0 = busin(kbus)
            ... a0 = moogladder2(a0, kcutoff, kresonance)
            ... outch 1, a0
            ... ''', args=dict(kbus=0, kcutoff=1000, kresonance=0.9))
            # Parameters can be given inline. Parameters do not necessarily need
            # to define defaults
            >>> session.defInstr('synth', r'''
            ... |ibus, kamp=0.5, kmidi=60|
            ... kfreq = mtof:k(lag:k(kmidi, 1))
            ... a0 vco2 kamp, kfreq
            ... a0 *= linsegr:a(0, 0.1, 1, 0.1, 0)
            ... busout ibus, a0
            ... ''')

            >>> bus = session.engine.assignBus()
            # Named params can be given as keyword arguments
            >>> synth = session.sched('sine', 0, dur=10, ibus=bus, kmidi=67)
            >>> synth.set(kmidi=60, delay=2)
            >>> filt = session.sched('filter', 0, dur=synth.dur, priority=synth.priority+1,
            ...                      args={'kbus': bus, 'kcutoff': 1000})
            >>> filt.automate('kcutoff', [3, 1000, 6, 200, 10, 4000])

        See Also
        ~~~~~~~~

        :meth:`~Session.sched`
        """
        oldinstr = self.instrs.get(name)
        instr = Instr(name=name, body=body, args=args, init=init,
                      doc=doc, includes=includes, aliases=aliases,
                      maxNamedArgs=self.maxDynamicArgs,
                      useDynamicPfields=useDynamicPfields,
                      **kws)
        if oldinstr is not None and oldinstr == instr:
            return oldinstr
        self.registerInstr(instr)
        if priority:
            self.prepareSched(name, priority, block=True)
        return instr

    def registeredInstrs(self) -> dict[str, Instr]:
        """
        Returns a dict (instrname: Instr) with all registered Instrs
        """
        return self.instrs

    def isInstrRegistered(self, instr: Instr) -> bool:
        """
        Returns True if *instr* is already registered in this Session

        To check that a given instrument name is defined, use
        ``session.getInstr(instrname) is not None``

        .. seealso:: :meth:`~Session.getInstr`, :meth:`~Session.registerInstr`
        """
        return instr.id in self._instrIndex

    def registerInstr(self, instr: Instr) -> bool:
        """
        Register the given Instr in this session.

        It evaluates any init code, if necessary

        Args:
            instr: the Instr to register

        Returns:
            True if the action was performed, False if this instr was already
            defined in its current form

        See Also
        ~~~~~~~~

        :meth:`~Session.defInstr`

        """
        if instr.id in self._instrIndex:
            logger.debug(f"Instr {instr.name} already defined")
            return False

        if instr.name in self.instrs:
            logger.info(f"Redefining instr {instr.name}")
            oldinstr = self.instrs[instr.name]
            del self._instrIndex[oldinstr.id]

        if instr.includes:
            for include in instr.includes:
                self.engine.includeFile(include)

        if instr.init and instr.init not in self._initCodes:
            # compile init code if we haven't already
            try:
                self.engine.compile(instr.init)
                self._initCodes.append(instr.init)
            except CsoundError:
                raise CsoundError(f"Could not compile init code for instr {instr.name}")
        self._clearCacheForInstr(instr.name)
        self.instrs[instr.name] = instr
        self._instrIndex[instr.id] = instr
        return True

    def _clearCacheForInstr(self, instrname: str) -> None:
        if instrname in self._reifiedInstrDefs:
            self._reifiedInstrDefs[instrname].clear()

    def _resetSynthdefs(self, name):
        self._reifiedInstrDefs[name] = {}

    def _registerReifiedInstr(self, name: str, priority: int, rinstr: _ReifiedInstr
                              ) -> None:
        registry = self._reifiedInstrDefs.setdefault(name, {})
        registry[priority] = rinstr

    def _makeReifiedInstr(self, name: str, priority: int, block=True) -> _ReifiedInstr:
        """
        A ReifiedInstr is a version of an instrument with a given priority
        """
        assert isinstance(priority, int) and 1 <= priority <= 10
        instr = self.instrs.get(name)
        if instr is None:
            raise ValueError(f"instrument {name} not registered")
        instrnum = self._registerInstrAtPriority(name, priority)
        body = self.generateInstrBody(instr=instr)
        instrtxt = _tools.instrWrapBody(body=body,
                                        instrid=instrnum)
        try:
            self.engine._compileInstr(instrnum, instrtxt, block=block)
        except CsoundError as e:
            logger.error(str(e))
            raise CsoundError(f"Could not compile body for instr '{name}'")
        rinstr = _ReifiedInstr(instrnum, priority)
        self._registerReifiedInstr(name, priority, rinstr)
        return rinstr

    def getInstr(self, instrname: str) -> Instr | None:
        """
        Returns the :class:`~csoundengine.instr.Instr` defined under name

        Returns None if no Instr is defined with the given name

        Args:
            instrname: the name of the Instr - **use "?" to select interactively**

        See Also
        ~~~~~~~~

        :meth:`~Session.defInstr`
        """
        if instrname == "?":
            instrname = _dialogs.selectItem(list(self.instrs.keys()))
            if instrname is None:
                return None
        return self.instrs.get(instrname)

    def _getReifiedInstr(self, name: str, priority: int) -> _ReifiedInstr | None:
        registry = self._reifiedInstrDefs.get(name)
        if not registry:
            return None
        return registry.get(priority)

    def _isInstrPrepared(self, instrname: str, priority: int = 1) -> bool:
        """
        Is an instrument with the given name prepared for the given priorty=

        Args:
            instrname: the name of the instrument
            priority: the priority to use

        Returns:
            True if the instrument is prepared
        """
        if registry := self._reifiedInstrDefs.get(instrname):
            return priority in registry
        return False

    def prepareSched(self,
                     instr: str | Instr,
                     priority: int = 1,
                     block=False
                     ) -> tuple[_ReifiedInstr, bool]:
        """
        Prepare an instrument template for being scheduled

        The only use case to call this method explicitely is when the user
        is certain to need the given instrument at the specified priority and
        wants to avoid the delay needed for the first time an instr
        is called, since this first call implies compiling the code in csound.

        Args:
            instr: the name of the instrument to send to the csound engine or the Instr itself
            priority: the priority of the instr
            block: if True, this method will block until csound is ready to
                schedule the given instr at the given priority

        Returns:
            a tuple (_ReifiedInstr, needssync: bool)
        """
        rinstr = self._getReifiedInstr(instr if isinstance(instr, str) else instr.name, priority)
        needssync = False
        if isinstance(instr, Instr):
            instr = instr.name
        if rinstr is None:
            rinstr = self._makeReifiedInstr(instr, priority, block=block)
            if block:
                self.engine.sync()
            else:
                needssync = True
        return rinstr, needssync

    def instrnum(self, instrname: str, priority: int = 1) -> int:
        """
        Return the instr number for the given Instr at the given priority

        For a defined :class:`~csoundengine.instr.Instr` (identified by `instrname`)
        and a priority, return the concrete instrument number for this instrument.

        This returned instrument number will not be a unique (fractional)
        instance number.

        Args:
            instrname: the name of a defined Instr
            priority: the priority at which an instance of this Instr should
                be scheduled. An instance with a higher priority is evaluated
                later in the chain. This is relevant when an instrument performs
                some task on data generated by a previous instrument.

        Returns:
            the actual (integer) instrument number inside csound

        See Also
        ~~~~~~~~

        :meth:`~Session.defInstr`
        """
        assert isinstance(priority, int) and 1 <= priority <= 10
        assert instrname in self.instrs
        rinstr, needssync = self.prepareSched(instrname, priority)
        return rinstr.instrnum

    def assignBus(self, kind='', value: float | None = None, persist=False
                  ) -> busproxy.Bus:
        """
        Creates a bus in the engine

        This is a wrapper around
        :meth:`Engine.assignBus() <csoundengine.engine.Engine.assignBus>`. Instead of returning
        a raw bus token it returns a :class:`~csoundengine.busproxy.Bus`, which can be used
        to write, read or automate a bus. To pass the bus to an instrument expecting
        a bus, use its :attr:`~csoundengine.busproxy.Bus.token` attribute.

        Within csound a bus is reference counted and is kept alive as long as there are
        events using it via any of the builtin bus opcdodes: :ref:`busin<busin>`,
        :ref:`busout<busout>`, :ref:`busmix<busmix>`. A :class:`~csoundengine.busproxy.Bus`
        can hold itself a reference to the bus if called with ``persist=True``, which means
        that the csound bus will be kept alive as long as python holds a reference to the
        Bus object.

        For more information on the bus-opcodes, see :ref:`Bus Opcodes<busopcodes>`

        Args:
            kind: the kind of bus, "audio" or "control". If left unset and value
                is not given it defaults to an audio bus. Otherwise, if value
                is given a control bus is created. Explicitely asking for an
                audio bus and setting an initial value will raise an expection
            value: for control buses it is possible to set an initial value for
                the bus. If a value is given the bus is created as a control
                bus. For audio buses this should be left as None
            persist: if True, the bus is valid until manually released or until
                the returned Bus object is freed.

        Returns:
            a Bus, representing the bus created. The returned object can be
            used to modify/read/automate the bus

        .. seealso:: :meth:`csoundengine.engine.Engine.assignBus`, :class:`csoundengine.busproxy.Bus`


        Example
        ~~~~~~~

        .. code-block:: python

            from csoundengine import *

            s = Engine().session()

            s.defInstr('sender', r'''
            ibus = p5
            ifreqbus = p6
            kfreq = busin:k(ifreqbus)
            asig vco2 0.1, kfreq
            busout(ibus, asig)
            ''')

            s.defInstr('receiver', r'''
            ibus  = p5
            kgain = p6
            asig = busin:a(ibus)
            asig *= a(kgain)
            outch 1, asig
            ''')

            bus = s.assignBus('audio')
            freqbus = s.assignBus(value=880)

            # The receiver needs to have a higher priority in order to
            # receive the audio of the sender
            chain = [s.sched('sender', ibus=bus.token, ifreqbus=freqbus.token),
                     s.sched('receiver', priority=2, ibus=bus.token, kgain=0.5)]

            # Make a glissando
            freqbus.automate((0, 880, 5, 440))

        """
        if kind:
            if value is not None and kind == 'audio':
                raise ValueError(f"An audio bus cannot have a scalar value")
        else:
            kind = 'audio' if value is None else 'control'
        bustoken = self.engine.assignBus(kind=kind, value=value, persist=persist)
        return busproxy.Bus(token=bustoken, kind=kind, renderer=self, bound=persist)

    def _writeBus(self, bus: busproxy.Bus, value: float, delay=0.) -> None:
        self.engine.writeBus(bus=bus.token, value=value, delay=delay)

    def _readBus(self, bus: busproxy.Bus, default: float | None = None
                 ) -> float | None:
        return self.engine.readBus(bus=bus.token, default=default)

    def _releaseBus(self, bus: busproxy.Bus) -> None:
        self.engine.releaseBus(bus.token)

    def _automateBus(self, bus: busproxy.Bus, pairs: Sequence[float],
                     mode='linear', delay=0., overtake=False) -> None:
        self.engine.automateBus(bus=bus.token, pairs=pairs, mode=mode,
                                delay=delay, overtake=overtake)

    def schedEvents(self, events: Sequence[Event]) -> SynthGroup:
        """
        Schedule multiple events

        Args:
            events: the events to schedule

        Returns:
            a SynthGroup with the synths corresponding to the given events
        """
        sync = False
        for event in events:
            _, needssync = self.prepareSched(instr=event.instrname,
                                             priority=event.priority,
                                             block=False)
            if needssync:
                sync = True
        if sync:
            self.engine.sync()
        with self.engine.lockedClock():
            synths = [self.schedEvent(event) for event in events]
        return SynthGroup(synths)

    def schedEvent(self, event: Event) -> Synth:
        """
        Schedule an event

        An Event can be generated to store a Synth's data.

        Args:
            event: the event to schedule. An :class:`csoundengine.event.Event`
                represents an unscheduled event.

        Returns:
            the generated Synth

        Example
        ~~~~~~~

        >>> from csoundengine import *
        >>> s = Engine().session()
        >>> s.defInstr('simplesine', r'''
        ... |ifreq=440, iamp=0.1, iattack=0.2|
        ... asig vco2 0.1, ifreq
        ... asig *= linsegr:a(0, iattack, 1, 0.1, 0)
        ... outch 1, asig
        ... ''')
        >>> event = Event('simplesine', args=dict(ifreq=1000, iamp=0.2, iattack=0.2))
        >>> synth = s.schedEvent(event)
        ...
        >>> synth.stop()

        .. seealso:: :class:`csoundengine.synth.Synth`, :class:`csoundengine.schedevent.SchedEvent`

        """
        kws = event.kws or {}
        synth = self.sched(instrname=event.instrname,
                           delay=event.delay,
                           dur=event.dur,
                           priority=event.priority,
                           args=event.args,
                           whenfinished=event.whenfinished,
                           relative=event.relative,
                           **kws)
        if event.automations:
            for automation in event.automations:
                synth.automate(param=automation.param,
                               pairs=automation.pairs,
                               delay=automation.delay or 0.,
                               mode=automation.interpolation,
                               overtake=automation.overtake)
        return synth

    def lockedClock(self, latency: float | None) -> Session:
        """
        context manager to ensure sync

        .. seealso:: :meth:`csoundengine.engine.Engine.lockClock`
        """
        self._lockedLatency = latency
        return self

    def __enter__(self):
        if self.engine.isClockLocked():
            logger.warning("This session is already locked")
        else:
            latency = self._lockedLatency if self._lockedLatency is not None else min(0.2, self.engine.extraLatency*2)
            self.engine.pushLock(latency)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.engine.isClockLocked():
            self.engine.popLock()
        self._lockedLatency = None

    def rendering(self,
                  outfile='',
                  sr=0,
                  nchnls: int | None = None,
                  ksmps=0,
                  encoding='',
                  starttime=0.,
                  endtime=0.,
                  tail=0.,
                  openWhenDone=False,
                  verbose: bool | None = None
                  ) -> offline.OfflineSession:
        """
        A context-manager for offline rendering

        All scheduled events are rendered to `outfile` when exiting the
        context. The :class:`~csoundengine.offline.OfflineSession` returned by the
        context manager has the same interface as a :class:`Session` and can
        be used as a drop-in replacement. Any instrument or resource declared
        within this Session is available for offline rendering.

        Args:
            outfile: the soundfile to generate after exiting the context
            sr: the samplerate. If not given, the samplerate of the session will be used
            nchnls: the number of channels. If not given, the number of channels of the
                session will be used
            ksmps: samples per cycle to use for rendering
            encoding: the sample encoding of the rendered file, given as
                'pcmXX' or 'floatXX', where XX represent the bit-depth
                ('pcm16', 'float32', etc.). If no encoding is given a suitable default
                for the sample format is chosen
            starttime: start rendering at the given time. Any event ending previous to
                this time will not be rendered and any event between starttime and
                endtime will be cropped
            endtime: stop rendering at the given time. This will either extend or crop
                the rendering.
            tail: extra render time at the end, to accomodate extended releases
            openWhenDone: open the file in the default application after rendering.
            verbose: if True, output rendering information. If None uses the value
                specified in the config (``config['rec_suppress_output']``)

        Returns:
            a :class:`csoundengine.offline.OfflineSession`

        Example
        ~~~~~~~

            >>> from csoundengine import *
            >>> s = Engine().session()
            >>> s.defInstr('simplesine', r'''
            ... |kfreq=440, kgain=0.1, iattack=0.05|
            ... asig vco2 1, ifreq
            ... asig *= linsegr:a(0, iattack, 1, 0.1, 0)
            ... asing *= kgain
            ... outch 1, asig
            ... ''')
            >>> with s.rendering('out.wav') as r:
            ...     r.sched('simplesine', 0, dur=2, kfreq=1000)
            ...     r.sched('simplesine', 0.5, dur=1.5, kfreq=1004)
            >>> # Generate the corresponding csd
            >>> r.writeCsd('out.csd')

        .. seealso:: :class:`~csoundengine.offline.OfflineSession`
        """
        renderer = self.makeRenderer(sr=sr or self.engine.sr,
                                     nchnls=nchnls or self.engine.nchnls,
                                     ksmps=ksmps)
        handler = _RenderingSessionHandler(renderer=renderer)
        self.setHandler(handler)

        def atexit(r: offline.OfflineSession, outfile: str, session: Session) -> None:
            r.render(outfile=outfile, endtime=endtime, encoding=encoding,
                     starttime=starttime, openWhenDone=openWhenDone,
                     tail=tail, verbose=verbose)
            session._offlineRenderer = None
            session.setHandler(None)

        renderer._registerExitCallback(lambda renderer: atexit(r=renderer, outfile=outfile, session=self))
        self._offlineRenderer = renderer
        return renderer

    def _dynargsAssignSlot(self) -> int:
        """
        Assign a slice for the dynamic args of a synth
        """
        try:
            return self._dynargsSlotPool.pop()
        except IndexError:
            raise IndexError("Tried to assign a slice for dynamic controls but the pool"
                             " is empty.")

    def _dynargsReleaseSlot(self, slicenum: int) -> None:
        assert 1 <= slicenum < self._dynargsNumSlots
        assert slicenum not in self._dynargsSlotPool   # Remove this after testing
        self._dynargsSlotPool.appendleft(slicenum)

    @staticmethod
    def defaultInstrBody(instr: Instr) -> str:
        body = instr._preprocessedBody
        parts = []
        docstring, body = csoundlib.splitDocstring(body)
        if docstring:
            parts.append(docstring)

        if instr.controls:
            code = _namedControlsGenerateCode(instr.controls)
            parts.append(code)

        if instr.pfieldIndexToName:
            pfieldstext, body, docstring = instrtools.generatePfieldsCode(body, instr.pfieldIndexToName)
            if pfieldstext:
                parts.append(pfieldstext)
        parts.append(body)
        # deallocInstr = self.engine._builtinInstrs['notifyDealloc']
        # parts.append(f'atstop {deallocInstr}, 0.01, 0, p1')
        parts.append('atstop dict_get:i(gi__builtinInstrs, "notifyDealloc"), 0, 0, p1')
        if instr.controls:
            parts.append('__exit:')
        out = _textlib.joinPreservingIndentation(parts)
        return textwrap.dedent(out)

    @cache
    def generateInstrBody(self, instr: Instr) -> str:
        """
        Generate the actual body for a given instr

        This task is done by a Session/Renderer because the actual
        body might be different if we are rendering in realtime,
        as is the case of a session, or if its offline

        Args:
            instr: the Instr for which to generate the instr body

        Returns:
            the generated body. This is the text which must be
            wrapped between instr/endin
        """
        body = instr._preprocessedBody
        parts = []
        docstring, body = csoundlib.splitDocstring(body)
        if docstring:
            parts.append(docstring)

        if instr.controls:
            code = _namedControlsGenerateCode(instr.controls)
            parts.append(code)

        if instr.pfieldIndexToName:
            pfieldstext, body, docstring = instrtools.generatePfieldsCode(body, instr.pfieldIndexToName)
            if pfieldstext:
                parts.append(pfieldstext)
        parts.append(body)
        if not self._notificationUseOsc:
            # Use outvalue for deallocation
            deallocInstr = self.engine._builtinInstrs['notifyDealloc']
            parts.append(f'atstop {deallocInstr}, 0.01, 0, p1')
        else:
            # Use osc
            assert self._notificationOscPort > 0
            deallocInstr = self.engine._builtinInstrs['notifyDeallocOsc']
            parts.append(f'atstop {deallocInstr}, 0.01, 0, p1, {self._notificationOscPort}')

        if instr.controls:
            parts.append('__exit:')
        out = _textlib.joinPreservingIndentation(parts)
        return textwrap.dedent(out)

    def sched(self,
              instrname: str,
              delay=0.,
              dur=-1.,
              *pfields,
              args: Sequence[float|str] | dict[str, float] | None = None,
              priority: int = 1,
              whenfinished: Callable | None = None,
              relative=True,
              **kwargs
              ) -> Synth:
        """
        Schedule an instance of an instrument

        Args:
            instrname: the name of the instrument, as defined via defInstr.
                **Use "?" to select an instrument interactively**
            delay: time offset of the scheduled instrument
            dur: duration (-1 = forever)
            pfields: pfields passed as positional arguments. Pfields can also
                be given as a list/array passed to the ``args`` argument or
                as keyword arguments
            args: arguments passed to the instrument, a dict of the
                form {'argname': value}, where argname can be any px string or the name
                of the variable (for example, if the instrument has a line
                'kfreq = p5', then 'kfreq' can be used as key here). Alternatively, a list of
                positional arguments, starting with p5
            priority: the priority (1 to 10)
            whenfinished: a function of the form f(synthid) -> None
                if given, it will be called when this instance stops
            relative: if True, delay is relative to the current time. Otherwise delay
                is interpreted as an absolute time from the start time of the Engine.
            kwargs: keyword arguments are interpreted as named parameters. This is needed when
                passing positional and named arguments at the same time

        Returns:
            a :class:`~csoundengine.synth,Synth`, which is a handle to the instance
            (can be stopped, etc.)

        Example
        ~~~~~~~

        >>> from csoundengine import *
        >>> s = Engine().session()
        >>> s.defInstr('simplesine', r'''
        ... pset 0, 0, 0, 440, 0.1, 0.05
        ... ifreq = p5
        ... iamp = p6
        ... iattack = p7
        ... asig vco2 0.1, ifreq
        ... asig *= linsegr:a(0, iattack, 1, 0.1, 0)
        ... outch 1, asig
        ... ''')
        # NB: in a Session, pfields start at p5 since p4 is reserved
        >>> synth = s.sched('simplesine', args=[1000, 0.2], iattack=0.2)
        ...
        >>> synth.stop()

        See Also
        ~~~~~~~~

        :meth:`~csoundengine.synth.Synth.stop`
        """
        if pfields and args:
            raise ValueError(f"Either pfields as positional arguments or args can be given, "
                             f"got both")
        elif pfields:
            args = pfields

        if self._handler:
            event = Event(instrname=instrname, delay=delay, dur=dur, priority=priority,
                          args=args, whenfinished=whenfinished, relative=relative, kws=kwargs)
            return self._handler.schedEvent(event)

        if self.isRendering():
            raise RuntimeError("Session blocked during rendering")

        assert isinstance(priority, int) and 1 <= priority <= self.numPriorities
        assert self._dynargsArray is not None
        abstime = delay if not relative else self.engine.elapsedTime() + delay + self.engine.extraLatency

        if instrname == "?":
            selected = _dialogs.selectItem(list(self.instrs.keys()),
                                           title="Select Instr",
                                           ensureSelection=True)
            assert selected is not None
            instrname = selected

        instr = self.getInstr(instrname)
        if instr is None:
            raise ValueError(f"Instrument '{instrname}' not defined. "
                             f"Known instruments: {', '.join(self.instrs.keys())}")

        rinstr, needssync = self.prepareSched(instrname, priority, block=True)
        pfields5, dynargs = instr.parseSchedArgs(args=args, kws=kwargs)
        if instr.controls:
            slicenum = self._dynargsAssignSlot()
            values = instr._controlsDefaultValues if not dynargs else instr.overrideControls(dynargs)
            assert isinstance(values, list)
            idx0 = p4 = slicenum * self.maxDynamicArgs
            if delay < 1:
                self._dynargsArray[idx0:idx0+len(values)] = values
            else:
                self.engine.sched(self.engine._builtinInstrs['initDynamicControls'],
                                  delay=abstime-self.engine.ksmps/self.engine.sr,
                                  dur=0.01,
                                  args=[p4, len(values), *values],
                                  relative=False)
        else:
            p4 = 0
            slicenum = 0

        pfields4 = [p4, *pfields5]

        if needssync:
            self.engine.sync()
        synthid = self.engine.sched(rinstr.instrnum, delay=abstime, dur=dur, args=pfields4,
                                    relative=False, unique=True)
        synth = Synth(session=self,
                      p1=synthid,
                      instr=instr,
                      start=abstime,
                      dur=dur,
                      args=pfields5,
                      controlsSlot=slicenum,
                      priority=priority,
                      controls=dynargs)

        if whenfinished is not None:
            self._whenfinished[synthid] = whenfinished
        self._synths[synthid] = synth
        return synth

    def _getNamedControl(self, slicenum: int, paramslot: int) -> float | None:
        idx = slicenum * self.maxDynamicArgs + paramslot
        if 0 <= idx < len(self._dynargsArray):
            return float(self._dynargsArray[idx])
        else:
            raise IndexError(f"Named control index out of range, "
                             f"slicenum: {slicenum}, slot: {paramslot}")

    def _setPfield(self, event: SchedEvent, delay: float, param: str, value: float
                   ) -> None:
        assert event.instr is not None
        idx = event.instr.pfieldIndex(param, default=0)
        if idx == 0:
            raise KeyError(f"Unknown parameter {param} for {event}. "
                           f"Possible parameters: {event.dynamicParamNames()}")
        assert isinstance(event.p1, (int, float))
        timeoffset = event.start - self.engine.elapsedTime()
        if timeoffset > delay:
            # The event will not have started by the time this operation is performed. pwrite will not find
            # the instrument and will do nothing.
            # Instead, we schedule an automation on the future, starting somewhat before the event
            # and ending just after the event has started.
            # self.engine.setp(event.p1, idx, value, delay=timeoffset)
            self._automatePfield(event, param=idx, pairs=[max(0., timeoffset-0.25), value, timeoffset+0.01, value])
        else:
            self.engine.setp(event.p1, idx, value, delay=delay)

    def _setNamedControl(self,
                         event: SchedEvent,
                         param: str,
                         value: float,
                         delay: float = 0.
                         ) -> None:
        instr = event.instr
        assert instr is not None
        paramindex = instr.controlIndex(param)
        slot = event.controlsSlot
        if not slot:
            raise RuntimeError(f"This synth ({event}) has no associated controls slot")
        assert paramindex < self.maxDynamicArgs
        assert slot < self._dynargsNumSlots
        idx = slot * self.maxDynamicArgs + paramindex
        if delay > 0:
            self.engine.tableWrite(tabnum=self._dynargsTabnum,
                                   idx=idx, value=value, delay=delay)
        else:
            self._dynargsArray[idx] = value

    def activeSynths(self, sortby="start") -> list[Synth]:
        """
        Returns a list of playing synths

        Args:
            sortby: either "start" (sort by start time) or None (unsorted)

        Returns:
            a list of active :class:`Synths<csoundengine.synth.Synth>`
        """
        synths = [synth for synth in self._synths.values() if synth.playStatus() != 'stopped']
        if sortby == "start":
            synths.sort(key=lambda synth: synth.start)
        return synths

    def scheduledSynths(self) -> list[Synth]:
        """
        Returns all scheduled synths (both active and future)
        """
        return list(self._synths.values())

    def unsched(self, event: int | float | SchedEvent, delay=0.) -> None:
        """
        Stop a scheduled instance.

        This will stop an already playing synth or a synth
        which has been scheduled in the future

        Normally the user should not call :meth:`.unsched`. This method
        is called by a :class:`~csoundengine.synth.Synth` when
        :meth:`~csoundengine.synth.Synth.stop` is called.

        Args:
            event: the event to stop, either a Synth or the p1
            delay: how long to wait before stopping them
        """
        #if self.isRendering():
        #    raise RuntimeError("This Session is blocked while in rendering mode. Call .unsched on the renderer instead")

        synthid = event if isinstance(event, (int, float)) else event.p1
        assert isinstance(synthid, (int, float))
        synth = self._synths.get(synthid)
        if not synth:
            logger.error(f"Event {event} not found, cannot unschedule")
            return
        elif synth.finished():
            logger.debug(f"Event {event} already finished, cannot unschedule")
            return
        elif synth.playing():
            self.engine.unsched(synthid, delay=delay)
            # Normally the outvalue callback calls the dealloc sequence itself
            # But it seems that when the event is turned off (via the turnoff
            # opcode) the outvalue callback is not triggered.
            # TODO: this needs to be investigated further.
            # self._deallocSynthResources(synthid, delay)
        elif synth.start > self.engine.elapsedTime():
            if delay == 0:
                self.engine.unschedFuture(synth.p1)
                self._deallocSynthResources(synthid, delay)
            else:
                self.engine.unsched(synthid, delay=delay)
        else:
            self._deallocSynthResources(synthid, delay)

    def unschedAll(self, future=False) -> None:
        """
        Unschedule all playing synths

        Args:
            future: if True, cancel also synths which are already scheduled
                but have not started playing yet
        """
        synthids = [synth.p1 for synth in self._synths.values()]
        futureSynths = [synth for synth in self._synths.values() if not synth.playing()]
        for synthid in synthids:
            self.unsched(synthid, delay=0)

        if future and futureSynths:
            self.engine.unschedAll()
            self._synths.clear()

    def includeFile(self, path: str) -> None:
        if path in self._includes:
            return
        self._includes.add(path)
        self.engine.includeFile(include=path)

    def readSoundfile(self,
                      path="?",
                      chan=0,
                      skiptime=0.,
                      delay=0.,
                      force=False,
                      block=False
                      ) -> TableProxy:
        """
        Read a soundfile, store its metadata in a :class:`~csoundengine.tableproxy.TableProxy`

        The life-time of the returned TableProxy object is not bound to the csound table.
        If the user needs to free the table, this needs to be done manually by calling
        :meth:`csoundengine.tableproxy.TableProxy.free`

        Args:
            path: the path to a soundfile. **"?" to open file via a gui dialog**
            chan: the channel to read, or 0 to read all channels into a multichannel table.
                Within a multichannel table, samples are interleaved
            force: if True, the soundfile will be read and added to the session even if the
                same path has already been read before.#
            delay: when to read the soundfile (0=now)
            skiptime: start playback from this time instead of the beginning
            block: block execution while reading the soundfile

        Returns:
            a TableProxy, holding information like
            .source: the table number
            .path: the path you just passed
            .nchnls: the number of channels in the output
            .sr: the sample rate of the output

        Example
        ~~~~~~~

            >>> import csoundengine as ce
            >>> session = ce.Engine().session()
            >>> table = session.readSoundfile("path/to/soundfile.flac")
            >>> table
            TableProxy(source=100, sr=44100, nchnls=2,
                       numframes=88200, path='path/to/soundfile.flac',
                       freeself=False)
            >>> table.duration()
            2.0
            >>> session.playSample(table)

        """
        if self.isRendering():
            raise RuntimeError("This Session is blocked during rendering. Call .readSoundFile on the offline "
                               "renderer instead")
        if path == "?":
            path = _state.openSoundfile()
        if (table := self._pathToTabproxy.get(path)) is not None and not force:
            return table
        tabnum = self.engine.readSoundfile(path=path, chan=chan, skiptime=skiptime, block=block)
        info = sndfileio.sndinfo(path)
        table = TableProxy(tabnum=tabnum,
                           path=path,
                           sr=info.samplerate,
                           nchnls=info.channels,
                           parent=self,
                           numframes=info.nframes)

        self._registerTable(table)
        return table

    def _registerTable(self, tabproxy: TableProxy) -> None:
        self._tabnumToTabproxy[tabproxy.tabnum] = tabproxy
        if tabproxy.path:
            self._pathToTabproxy[tabproxy.path] = tabproxy

    def makeTable(self,
                  data: np.ndarray | list[float] | None = None,
                  size: int | tuple[int, int] = 0,
                  tabnum: int = 0,
                  sr: int = 0,
                  delay: float = 0.,
                  unique=True,
                  freeself=False,
                  block=False,
                  callback=None,
                  ) -> TableProxy:
        """
        Create a table with given data or an empty table of the given size

        Args:
            data: the data of the table. Use None if the table should be empty
            size: if not data is given, sets the size of the empty table created. Either
                a size as int or a tuple (numchannels: int, numframes: int). In the latter
                case, the actual size of the table is numchannels * numframes.
            tabnum: 0 to let csound determine a table number, -1 to self assign
                a value
            block: if True, wait until the operation has been finished
            callback: function called when the table is fully created
            sr: the samplerate of the data, if applicable.
            freeself: if True, the underlying csound table will be freed
                whenever the returned TableProxy ceases to exist.
            unique: if False, do not create a table if there is a table with the same data
            delay: when to allocate the table. This has little use in realtime but is here
                to comply to the signature.

        Returns:
            a TableProxy object

        """
        if self.isRendering():
            raise RuntimeError("This Session is in rendering mode. Call .makeTable on the renderer instead "
                               "(with session.rendering() as r: ... r.makeTable(...)")

        if self._handler is not None:
            try:
                return self._handler.makeTable(data=data, size=size, sr=sr)
            except NotImplementedError:
                # The handler does not implement makeTable, so we need to do that here
                pass

        # TODO: check block / callback for empty table
        if delay > 0:
            logger.info(f"Delay parameter ignored ({delay=} when allocating table")

        if size:
            assert not data
            if isinstance(size, int):
                tabsize = size
                numchannels = 1
            elif isinstance(size, tuple) and len(size) == 2:
                numchannels, tabsize = size
            else:
                raise TypeError(f"Expected a size as int or a tuple (numchannels, size), got {size}")
            tabnum = self.engine.makeEmptyTable(size=tabsize, numchannels=numchannels, sr=sr)
            tabproxy = TableProxy(tabnum=tabnum, sr=sr, nchnls=numchannels, numframes=tabsize,
                                  parent=self, freeself=freeself)
            if block or callback:
                logger.info("blocking / callback for this operation is not implemented")
        elif data is None:
            raise ValueError("Either data or a size must be given")
        else:
            if isinstance(data, list):
                nchnls = 1
                data = np.asarray(data, dtype=float)
            else:
                assert isinstance(data, np.ndarray)
                nchnls = _tools.arrayNumChannels(data)
            if not unique:
                datahash = _tools.ndarrayhash(data)
                if (tabproxy := self._ndarrayHashToTabproxy.get(datahash)) is not None:
                    return tabproxy
            else:
                datahash = None
            numframes = len(data)
            tabnum = self.engine.makeTable(data=data, tabnum=tabnum, block=block,
                                           callback=callback, sr=sr)
            tabproxy = TableProxy(tabnum=tabnum, sr=sr, nchnls=nchnls, numframes=numframes,
                                  parent=self, freeself=freeself)
            if datahash is not None:
                self._ndarrayHashToTabproxy[datahash] = tabproxy

        self._registerTable(tabproxy)
        return tabproxy

    def _getTableData(self, table: int | TableProxy) -> np.ndarray | None:
        tabnum = table if isinstance(table, int) else table.tabnum
        assert self.engine.csound is not None
        return self.engine.csound.table(tabnum)

    def freeTable(self,
                  table: int | TableProxy,
                  delay: float = 0.) -> None:
        """
        Free the given table

        Args:
            table: the table to free (a table number / a :class:`TableProxy`)
            delay: when to free it (0=now)
        """
        tabnum = table if isinstance(table, int) else table.tabnum
        self.engine.freeTable(tabnum, delay=delay)

    def testAudio(self, dur=20, mode='noise', verbose=True, period=1, gain=0.1):
        """
        Schedule a test synth to test the engine/session

        The test iterates over each channel outputing audio to the
        channel for a specific time period

        Args:
            dur: the duration of the test synth
            mode: the test mode, one of 'noise', 'sine'
            period: the duration of each iteration
            gain: the gain of the output
        """
        imode = {
            'noise': 0,
            'sine': 1
        }.get(mode)
        if imode is None:
            raise ValueError(f"mode {mode} is invalid. Possible modes are 'noise', 'sine'")
        return self.sched('.testAudio', dur=dur,
                          args=dict(imode=imode, iperiod=period, igain=gain, iverbose=int(verbose)))

    def playPartials(self,
                     source: int | TableProxy | str | np.ndarray,
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
                     position=0.,
                     freqoffset=0.,
                     minbw=0.,
                     maxbw=1.,
                     minamp=0.
                     ) -> Synth:
        """
        Play a packed spectrum

        A packed spectrum is a 2D numpy array representing a fixed set of
        oscillators. After partial tracking analysis, all partials are arranged
        into such a matrix where each row represents the state of all oscillators
        over time.

        The **loristrck** package is needed for both partial-tracking analysis and
        packing. It can be installed via ``pip install loristrck`` (see
        https://github.com/gesellkammer/loristrck). This is an optional dependency


        Args:
            source: a table number, TableProxy, path to a .mtx or .sdif file, or
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
            minbw: breakpoints with bw less than this value are not played
            maxbw: breakpoints with bw higher than this value are not played
            freqoffset: an offset to add to all frequencies, shifting them by a fixed amount. Notice
                that this will make a harmonic spectrum inharmonic
            minamp: exclude breanpoints with an amplitude less than this value

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
        if self.isRendering():
            raise RuntimeError("This Session is blocked during rendering")

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
                    raise ImportError("loristrck is needed in order to play a .sdif file"
                                      ". Install it via `pip install loristrck`")
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
                                    kgain=gain,
                                    kminbw=minbw,
                                    kmaxbw=maxbw,
                                    kfreqoffset=freqoffset,
                                    kminamp=minamp))

    def playSample(self,
                   source: int | TableProxy | str | tuple[np.ndarray, int],
                   delay=0.,
                   dur=0.,
                   chan=1,
                   gain=1.,
                   speed=1.,
                   loop=False,
                   pan=0.5,
                   skip=0.,
                   fade: float | tuple[float, float] | None = None,
                   crossfade=0.02,
                   blockread=True
                   ) -> Synth:

        """
        Play a sample.

        This method ensures that the sample is played at the original pitch,
        independent of the current samplerate. The source can be a table,
        a soundfile or a :class:`~csoundengine.tableproxy.TableProxy`. If a path
        to a soundfile is given, the 'diskin2' opcode is used by default

        Args:
            source: table number, a path to a sample or a TableProxy, or a tuple
                (numpy array, samplerate).
            dur: the duration of playback (-1 to play until the end of the sample
                or indefinitely if loop==True).
            chan: the channel to play the sample to. In the case of multichannel
                  samples, this is the first channel
            pan: a value between 0-1. -1 means default, which is 0 for mono,
                0.5 for stereo. For multichannel (3+) samples, panning is not
                taken into account
            gain: gain factor.
            speed: speed of playback. Pitch will be changed as well.
            loop: True/False or -1 to loop as defined in the file itself (not all
                file formats define loop points)
            delay: time to wait before playback starts
            skip: the starting playback time (0=play from beginning)
            fade: fade in/out in secods. None=default. Either a fade value or a tuple
                (fadein, fadeout)
            crossfade: if looping, this indicates the length of the crossfade
            blockread: block while reading the source (if needed) before playback is scheduled

        Returns:
            A Synth with the following mutable parameters: kgain, kspeed, kchan, kpan

        """
        if self.isRendering():
            raise RuntimeError("This Session is in rendering mode. Call .playSample on the renderer instead")

        if fade is None:
            fadein = fadeout = config['sample_fade_time']
        elif isinstance(fade, tuple):
            fadein, fadeout = fade
        elif isinstance(fade, (int, float)):
            fadein, fadeout = fade, fade
        else:
            raise TypeError(f"Expected a fade value in seconds or a tuple (fadein, fadeout), got {fade}")

        if loop and dur == 0:
            dur = -1

        if isinstance(source, str):
            if not loop and dur == 0:
                info = sndfileio.sndinfo(source)
                dur = info.duration / speed + fadeout
            return self.sched('.diskin',
                              delay=delay,
                              dur=dur,
                              Spath=source,
                              ifadein=fadein,
                              ifadeout=fadeout,
                              iloop=int(loop),
                              kspeed=speed,
                              kpan=pan,
                              ichan=chan)
        elif isinstance(source, int):
            tabnum = source
        elif isinstance(source, TableProxy):
            tabnum = source.tabnum
        elif isinstance(source, tuple) and isinstance(source[0], np.ndarray) and isinstance(source[1], int):
            table = self.makeTable(source[0], sr=source[1], unique=False, block=blockread)
            tabnum = table.tabnum
        else:
            raise TypeError(f"Expected table number as int, TableProxy, a path to a soundfile as str or a "
                            f"tuple (samples: np.ndarray, sr: int), got {source}")

        assert isinstance(tabnum, int) and tabnum >= 1
        return self.sched('.playSample',
                          delay=delay,
                          dur=dur,
                          args=dict(isndtab=tabnum,
                                    istart=skip,
                                    ifadein=fadein,
                                    ifadeout=fadeout,
                                    kchan=chan,
                                    kspeed=speed,
                                    kpan=pan,
                                    kgain=gain,
                                    ixfade=crossfade))

    def makeRenderer(self, sr=0, nchnls: int | None = None, ksmps=0,
                     addTables=True, addIncludes=True
                     ) -> offline.OfflineSession:
        """
        Create an :class:`~csoundengine.offline.OfflineSession` with
        the instruments defined in this Session

        To schedule events, use the :meth:`~csoundengine.offline.OfflineSession.sched` method
        of the renderer

        Args:
            sr: the samplerate (see config['rec_sr'])
            ksmps: ksmps used for rendering (see also config['rec_ksmps']). 0 uses
                the default defined in the config
            nchnls: the number of output channels. If not given, nchnls is taken
                from the session
            addTables: if True, any soundfile read via readSoundFile will be made
                available to the renderer. The TableProxy corresponding to that
                soundfile can be queried via :attr:`csoundengine.offline.OfflineSession.soundfileRegistry`.
                Notice that data tables will not be exported to the renderer
            addIncludes:
                add any ``#include`` file declared in this session to the created renderer

        Returns:
            a Renderer

        Example
        -------

            >>> from csoundengine import *
            >>> s = Engine().session()
            >>> s.defInstr('sine', r'''
            ... |kamp=0.1, kfreq=1000|
            ... outch 1, oscili:ar(kamp, freq)
            ... ''')
            >>> renderer = s.makeRenderer()
            >>> event = renderer.sched('sine', 0, dur=4, args=[0.1, 440])
            >>> event.set(delay=2, kfreq=880)
            >>> renderer.render("out.wav")

        """
        renderer = offline.OfflineSession(sr=sr or config['rec_sr'],
                                          nchnls=nchnls if nchnls is not None else self.engine.nchnls,
                                          ksmps=ksmps or config['rec_ksmps'],
                                          a4=self.engine.a4,
                                          dynamicArgsPerInstr=self.maxDynamicArgs,
                                          dynamicArgsSlots=self._dynargsNumSlots)
        for instrname, instrdef in self.instrs.items():
            renderer.registerInstr(instrdef)
        if addIncludes:
            for include in self._includes:
                renderer.includeFile(include)
        if addTables:
            for path in self._pathToTabproxy:
                renderer.readSoundfile(path)
        return renderer


    def _defBuiltinInstrs(self):
        for csoundInstr in builtinInstrs:
            self.registerInstr(csoundInstr)


def _namedControlsGenerateCode(controls: dict) -> str:
    """
    Generates code for an instr to read named controls

    Args:
        controls: a dict mapping control name to default value. The
            keys are valid csound k-variables

    Returns:
        the generated code
    """

    lines = [fr'''
    ; --- start generated code for dynamic args
    i__slicestart__ = p4
    i__tabnum__ chnget ".dynargsTabnum"
    if i__tabnum__ == 0 then
        initerror sprintf("Session table does not exist (p1: %f)", p1)
        goto __exit
    endif
    ''']
    idx = 0
    for key, value in controls.items():
        assert key.startswith('k')
        lines.append(f"    {key} tab i__slicestart__ + {idx}, i__tabnum__")
        idx += 1
    lines.append("    ; --- end generated code\n")
    out = _textlib.stripLines(_textlib.joinPreservingIndentation(lines))
    return out
