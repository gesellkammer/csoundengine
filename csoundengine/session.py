"""
A Session provides a high-level interface to control an underlying
csound process. A Session is associated with an Engine (there is one
Session per Engine)

**Features**

*   A Session uses instrument templates (:class:`~csoundengine.instr.Instr`), which
    enable an instrument to be instantiated at any place in the evaluation chain.
*   An instrument template within a Session can also declare default values for pfields
*   Session instruments can also have an associated table (a parameter table) to pass
    and modify parameters dynamically without depending on pfields. In fact, all
    :class:`~csoundengine.instr.Instr` reserve ``p4`` for the table number of this
    associated table

1. Instrument Templates
-----------------------

In csound (and within an :class:`~csoundengine.engine.Engine`) there is a direct
mapping between an instrument declaration and its order of evaluation. Within a
:class:`Session`, on the other hand, it is possible to declare an instrument which
is used as a template and can be instantiated at any order, making it possibe to
create chains of processing units.

Example
~~~~~~~

.. code-block:: python

    s = Engine().session()
    # Notice: the filter is declared before the generator. If these were
    # normal csound instruments, the filter would receive an instr number
    # lower and thus could never process audio generated by `myvco`
    Instr('filt', r'''
        Schan strget p5
        kcutoff = p6
        a0 chnget Schan
        a0 moogladder2 a0, kcutoff, 0.9
        outch 1, a0
        chnclear Schan
    ''').register(s)

    Intr('myvco', r'''
        kfreq = p5
        kamp = p6
        Schan strget p7
        a0 = vco2:a(kamp, kfreq)
        chnset a0, Schan
    ''').register(s)
    synth = s.sched('myvco', kfreq=440, kamp=0.1, Schan="chan1")
    # The filter is instantiated with a priority higher than the generator and
    # thus is evaluated later in the chain.
    filt = s.sched('filt', priority=synth.priority+1, kcutoff=1000, Schan="chan1")

2. Named pfields with default values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An :class:`~csoundengine.instr.Instr` (also declared via :meth:`~Session.defInstr`)
can define default values for its pfields. When scheduling an event the user only
needs to fill the values for those pfields which differ from the given default

.. code::

    s = Engine().session()
    s.defInstr('sine', r'''
        kamp = p5
        kfreq = p6
        a0 = oscili:a(kamp, kfreq)
        outch 1, a0
    ''', args={'kamp': 0.1, 'kfreq': 1000})
    # We schedule an event of sine, kamp will take the default (0.1)
    synth = s.sched('sine', kfreq=440)
    # pfields can be modified by name
    synth.setp(kamp=0.5)


3. Inline arguments
~~~~~~~~~~~~~~~~~~~

An :class:`~csoundengine.instr.Instr` can set both pfield name and default value
as inline declaration:

.. code::

    s = Engine().session()
    Intr('sine', r'''
        |kamp=0.1, kfreq=1000|
        a0 = oscili:a(kamp, kfreq)
        outch 1, a0
    ''').register(s)
    synth = s.sched('sine', kfreq=440)
    synth.stop()

This will generate the needed code:

.. code-block:: csound

    kamp = p5
    kfreq = p6

And will set the defaults.

4. Parameter Table
~~~~~~~~~~~~~~~~~~

Pfields are modified via the opcode ``pwrite``, which writes directly to
the memory where the event holds its parameter values. A Session provides
an alternative way to provide dynamic, named parameters, by defining a table
(an actual csound table) attached to each created event. Such tables define
names and default values for each parameters. The param table and the instrument
instance are created in tandem and the event reads the value from the table. ``p4``
is always reserved for a param table and should not be used for any other parameter,
even if the :class:`~csoundengine.instr.Instr` does not define a parameter table.


.. code::

    s = Engine().session()
    Intr('sine', r'''
        a0 = oscili:a(kamp, kfreq)
        outch 1, a0
    ''', tabledef=dict(amp=0.1, freq=1000)
    ).register(s)
    synth = s.sched('sine', tabargs=dict(amp=0.4, freq=440))
    synth.stop()

In this example, prior to scheduling the event a table is created and filled
with the values ``[0.4, 440]``. Code is generated to read these values from the table
(the actual code is somewhat different, for example, variables are mangled to avoid
any possible name clashes, etc):

.. code-block:: csound

    iparamTabnum = p4
    kamp  tab 0, iparamTabnum
    kfreq tab 1, iparamTabnum

An inline syntax exists also for tables:

.. code::

    Intr('sine', r'''
        {amp=0.1, freq=1000}
        a0 = oscili:a(kamp, kfreq)
        outch 1, a0
    ''')
"""

from __future__ import annotations
import dataclasses
from typing import TYPE_CHECKING, Dict, List, Union as U, Optional as Opt, Callable
from .engine import Engine, getEngine, CsoundError
from .instr import Instr
from .synth import AbstrSynth, Synth, SynthGroup
from .tableproxy import TableProxy
from .paramtable import ParamTable
from .config import config, logger
from . import internalTools as tools
from .sessioninstrs import builtinInstrs
import time
import numpy as np

if TYPE_CHECKING:
    from .offline import Renderer

__all__ = [
    'Session',
    'getSession',
]

@dataclasses.dataclass
class _ReifiedInstr:
    """
    A _ReifiedInstr is just a marker of a concrete instr sent to the
    engine for a given Instr template. An Instr is an abstract declaration without
    a specific instr number and thus without a specific order of execution.
    To be able to schedule an instrument at different places in the chain,
    the same instrument is redeclared (lazily) as different instrument numbers
    depending on the priority. When an instr. is scheduled at a given priority for
    the first time a ReifiedInstr is created to mark that and the code is sent
    to the engine
    """
    qname: str
    instrnum: int
    priority: int


class Session:
    """
    A Session is associated (exclusively) to a running
    :class:`~csoundengine.engine.Engine` and manages instrument declarations
    and scheduled events. An Engine can be thought of as a low-level interface
    to managing a csound instance, whereas a Session allows a higher-level control

    .. note:: 
    
        The user **does not** create an instance of this class directly.
        It is returned by either calling :meth:`engine.session()<csoundengine.engine.Engine.session>`
        or :func:`~csoundengine.session.getSession`
    
    Args:
        name: the name of the Engine. Only one Session per Engine can be created
    
    Example
    =======

    In order to add an instrument to a :class:`~csoundengine.session.Session`,
    an :class:`~csoundengine.instr.Instr` is created and registered with the Session.
    Alternatively, the shortcut :meth:`~Session.defInstr` can be used to create and
    register an :class:`~csoundengine.instr.Instr` at once.

    .. code::

        s = Engine().session()
        Intr('sine', r'''
            kfreq = p5
            kamp = p6
            a0 = oscili:a(kamp, kfreq)
            outch 1, a0
        ''').register(s)
        synth = s.sched('sine', kfreq=440, kamp=0.1)
        synth.stop()

    An :class:`~csoundengine.instr.Instr` can define default values for any of its
    p-fields:

    .. code::

        s = Engine().session()
        s.defInstr('sine', args={'kamp': 0.1, 'kfreq': 1000, body=r'''
            kamp = p5
            kfreq = p6
            a0 = oscili:a(kamp, kfreq)
            outch 1, a0
        ''')
        # We schedule an event of sine, kamp will take the default (0.1)
        synth = s.sched('sine', kfreq=440)
        synth.stop()

    An inline args declaration can set both pfield name and default value:

    .. code::

        s = Engine().session()
        Intr('sine', r'''
            |kamp=0.1, kfreq=1000|
            a0 = oscili:a(kamp, kfreq)
            outch 1, a0
        ''').register(s)
        synth = s.sched('sine', kfreq=440)
        synth.stop()

    The same can be achieved via an associated table:

    .. code-block:: python

        s = Engine().session()
        Intr('sine', r'''
            a0 = oscili:a(kamp, kfreq)
            outch 1, a0
        ''', tabledef=dict(amp=0.1, freq=1000
        ).register(s)
        synth = s.sched('sine', tabargs=dict(freq=440))
        synth.stop()

    This will create a table and fill it will the given/default values,
    and generate code to read from the table and free the table after
    the event is done. Call :meth:`~csoundengine.instr.Instr.dump` to see
    the generated code:

    .. code-block:: csound

        i_params = p4
        if ftexists(i_params) == 0 then
            initerror sprintf("params table (%d) does not exist", i_params)
        endif
        i__paramslen = ftlen(i_params)
        if i__paramslen < {maxidx} then
            initerror sprintf("params table is too small (size: %d, needed: {maxidx})", i__paramslen)
        endif
        kamp tab 0, i_params
        kfreq tab 1, i_params
        a0 = oscili:a(kamp, kfreq)
        outch 1, a0

    An inline syntax exists also for tables, using ``{...}``:

    .. code::

        Intr('sine', r'''
            {amp=0.1, freq=1000}
            a0 = oscili:a(kamp, kfreq)
            outch 1, a0
        ''')
    """
    _activeSessions: Dict[str, Session] = {}

    def __init__(self, name: str) -> None:
        """

        Args:
            name (str): the name of the Session, which corresponds to an existing
                Engine with the same name
        """
        assert name not in self._activeSessions
        assert name in Engine.activeEngines, f"Engine {name} does not exist!"

        self.name: str = name
        self.instrRegistry: Dict[str, Instr] = {}

        self._bucketsize: int = 1000
        self._numbuckets: int = 10
        self._buckets: List[Dict[str, int]] = [{} for _ in range(self._numbuckets)]

        # A dict of the form: {instrname: {priority: reifiedInstr }}
        self._reifiedInstrDefs: Dict[str, Dict[int, _ReifiedInstr]] = {}

        self._synths: Dict[float, Synth] = {}
        self._isDeallocCallbackSet = False
        self._whenfinished: Dict[float, Callable] = {}
        self._initCodes: List[str] = []
        self._tabnumToTable: Dict[int, TableProxy] = {}
        self._pathToTable: Dict[str, TableProxy] = {}
        self.engine = self._getEngine()

        if config['define_builtin_instrs']:
            self._defBuiltinInstrs()

        self._activeSessions[name] = self

    def __repr__(self):
        active = len(self.activeSynths())
        return f"Session({self.name}, synths={active})"

    def _deallocSynth(self, synthid: U[int, float], delay=0.) -> None:
        """
        Deallocates (frees) a synth in the engine and its proxy in the session

        Args:
            synthid: the id (p1) of the synth
            delay: when to deallocate the csound event.

        """
        synth = self._synths.pop(synthid, None)
        if synth is None:
            return
        logger.debug(f"Synth({synth.instr.name}, id={synthid}) deallocated")
        self.engine.unsched(synthid, delay)
        synth._playing = False
        if synth.table:
            if synth.instr.mustFreeTable:
                self.engine.freeTable(synth.table.tableIndex, delay=delay)
            else:
                self.engine._releaseTableNumber(synth.table.tableIndex)
        callback = self._whenfinished.pop(synthid, None)
        if callback:
            callback(synthid)

    def _deallocCallback(self, _, synthid):
        """ This is called by csound when a synth is deallocated """
        self._deallocSynth(synthid)

    def _getEngine(self) -> Engine:
        """
        Returns the associated engine and sets needed callbacks. The
        result is cached in Session.engine
        """
        engine = getEngine(self.name)
        assert engine is not None
        if not self._isDeallocCallbackSet:
            engine.registerOutvalueCallback("__dealloc__", self._deallocCallback)
            self._isDeallocCallbackSet = True
        return engine

    def _registerInstrAtPriority(self, instrname: str, priority=1) -> int:
        """
        Get the instrument number corresponding to this name and
        the given priority

        Args:
            instrname: the name of the instr as given to defInstr
            priority: the priority, an int from 1 to 10. Instruments with
                low priority are executed before instruments with high priority

        Returns:
            the instrument number (an integer)
        """
        assert 1<=priority<self._numbuckets-1
        bucket = self._buckets[priority]
        instrnum = bucket.get(instrname)
        if instrnum is not None:
            return instrnum
        idx = len(bucket)+1
        instrnum = self._bucketsize*priority+idx
        bucket[instrname] = instrnum
        return instrnum

    def _makeInstrTable(self,
                        instr: Instr,
                        overrides: Dict[str, float] = None,
                        wait=True) -> int:
        """
        Create and init the table associated with instr, returns the index

        Args:
            instr: the instrument to create a table for
            overrides: a dict of the form param:value, which overrides the defaults
                in the table definition of the instrument
            wait: if True, wait until the table has been created

        Returns:
            the index of the created table
        """
        values = instr._tableDefaultValues
        if overrides:
            values = instr.overrideTable(overrides)
        assert values is not None
        if len(values)<1:
            logger.warning(f"instr table with no init values (instr={instr})")
            return self.engine.makeTable(size=config['associated_table_min_size'],
                                         block=wait)
        else:
            logger.debug(f"Making table with init values: {values} ({overrides})")
            return self.engine.makeTable(data=values, block=wait)

    def defInstr(self, name: str, body: str, **kws) -> Instr:
        """
        Create an :class:`~csoundengine.instr.Instr` and register it with this session

        Args:
            name (str): the name of the created instr
            body (str): the body of the instrument. It can have named
                pfields (see example) or a table declaration
            kws: any keywords are passed on to the Instr constructor.
                See the documentation of Instr for more information.

        Returns:
            the created Instr. If needed, this instr can be registered
            at any other running Session via session.registerInstr(instr)

        Example
        =======

            >>> session = Engine().session()
            # An Instr with named pfields
            >>> session.defInstr('synth', '''
            ... |ibus, kamp=0.5, kmidi=60|
            ... kfreq = mtof:k(lag:k(kmidi, 1))
            ... a0 vco2 kamp, kfreq
            ... a0 *= linsegr:a(0, 0.1, 1, 0.1, 0)
            ... busout ibus, a0
            ... ''')
            # An instr with named table args
            >>> session.defInstr('filter', '''
            ... {bus=0, cutoff=1000, resonance=0.9}
            ... a0 = busin(kbus)
            ... a0 = moogladder2(a0, kcutoff, kresonance)
            ... outch 1, a0
            ... ''')

            >>> bus = session.engine.assignBus()
            >>> synth = session.sched('sine', 0, dur=10, ibus=bus, kmidi=67)
            >>> synth.setp(kmidi=60, delay=2)

            >>> filt = session.sched('filter', 0, dur=synth.dur, priority=synth.priority+1,
            ...                      tabargs={'bus': bus, 'cutoff': 1000})
            >>> filt.automateTable('cutoff', [3, 1000, 6, 200, 10, 4000])
            >>> bus.free()
        """
        instr = Instr(name=name, body=body, **kws)
        self.registerInstr(instr)
        return instr

    def registerInstr(self, instr: Instr) -> None:
        """
        Register the given Instr in this session.

        It evaluates any init code, if necessary

        Args:
            instr: the Instr to register

        """
        oldinstr = self.instrRegistry.get(instr.name)
        if instr.init and (oldinstr is None or instr.init != oldinstr.init):
            try:
                self.engine.compile(instr.init)
                self._initCodes.append(instr.init)
            except CsoundError:
                raise CsoundError(f"Could not compile init code for instr {instr.name}")
        self._clearCacheForInstr(instr.name)
        self.instrRegistry[instr.name] = instr

    def _clearCacheForInstr(self, instrname: str) -> None:
        if instrname in self._reifiedInstrDefs:
            self._reifiedInstrDefs[instrname].clear()

    def _resetSynthdefs(self, name):
        self._reifiedInstrDefs[name] = {}

    def _registerReifiedInstr(self, name: str, priority: int, rinstr: _ReifiedInstr
                              ) -> None:
        registry = self._reifiedInstrDefs.get(name)
        if registry:
            registry[priority] = rinstr
        else:
            registry = {priority:rinstr}
            self._reifiedInstrDefs[name] = registry

    def _makeReifiedInstr(self, name: str, priority: int) -> _ReifiedInstr:
        """
        A ReifiedInstr is a version of an instrument with a given priority
        """
        assert isinstance(priority, int) and 1<=priority<=10
        qname = f"{name}:{priority}"
        instrdef = self.instrRegistry.get(name)
        if instrdef is None:
            raise ValueError(f"instrument {name} not registered")
        instrnum = self._registerInstrAtPriority(name, priority)
        instrtxt = tools.instrWrapBody(instrdef.body, instrnum, addNotificationCode=True)
        try:
            self.engine.compile(instrtxt)
        except CsoundError:
            raise CsoundError(f"Could not compile body for instr {name}")
        rinstr = _ReifiedInstr(qname, instrnum, priority)
        self._registerReifiedInstr(name, priority, rinstr)
        return rinstr

    def getInstr(self, name: str) -> Instr:
        """
        Returns the :class:`~csoundengine.instr.Instr` defined under name
        """
        return self.instrRegistry[name]

    def _getReifiedInstr(self, name: str, priority: int) -> Opt[_ReifiedInstr]:
        registry = self._reifiedInstrDefs.get(name)
        if not registry:
            return None
        return registry.get(priority)

    def _prepareSched(self, instrname: str, priority: int = 1) -> _ReifiedInstr:
        rinstr = self._getReifiedInstr(instrname, priority)
        if rinstr is None:
            rinstr = self._makeReifiedInstr(instrname, priority)
        return rinstr

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
        """
        assert isinstance(priority, int) and 1<=priority<=10
        assert instrname in self.instrRegistry
        rinstr = self._prepareSched(instrname, priority)
        return rinstr.instrnum

    def assignBus(self, kind='audio') -> int:
        """ Creates a bus in the engine

        Example
        =======

        >>> from csoundengine import *
        >>> s = Engine().session()
        >>> s.defInstr('sender', r'''
        ... ibus = p5
        ... asig vco2 0.1, 1000
        ... busout(ibus, asig)
        ... ''')
        >>> s.defInstr('receiver', r'''
        ... ibus = p5
        ... asig = busin:a(ibus)
        ... asig *= 0.5
        ... outch 1, asig
        ... ''')
        >>> bus = s.assignBus()
        >>> chain = [s.sched('sender', ibus=bus),
        ...          s.sched('reveiver', ibus=bus)]
        >>> for synth in chain:
        ...     synth.stop()
        """
        if kind != 'audio':
            raise ValueError("Only audio buses are supported")
        return self.engine.assignBus()

    def sched(self,
              instrname: str,
              delay=0.,
              dur=-1.,
              priority: int = 1,
              pargs: U[List[float], Dict[str, float]] = [],
              tabargs: Dict[str, float] = None,
              whenfinished=None,
              **pkws
              ) -> Synth:
        """
        Schedule the instrument identified by *instrname*

        Args:
            instrname: the name of the instrument, as defined via defInstr
            priority: the priority (1 to 10)
            delay: time offset of the scheduled instrument
            dur: duration (-1 = for ever)
            pargs: pargs passed to the instrument (p5, p6, ...) or a dict of the
                form {'parg': value}, where parg can be any px string or the name
                of the variable (for example, if the instrument has a line
                'kfreq = p5', then 'kfreq' can be used as key here.
            tabargs: args to set the initial state of the associated table. Any
                arguments here will override the defaults in the instrument definition
            whenfinished: a function of the form f(synthid) -> None
                if given, it will be called when this instance stops

        Returns:
            a :class:`~csoundengine.synth,Synth`, which is a handle to the instance
            (can be stopped, etc.)
        """
        assert isinstance(priority, int) and 1<=priority<=10
        assert instrname in self.instrRegistry
        instr = self.getInstr(instrname)
        table: Opt[ParamTable]
        if instr._tableDefaultValues is not None:
            # the instruments has an associated table
            tableidx = self._makeInstrTable(instr, overrides=tabargs, wait=True)
            table = ParamTable(engine=self.engine, idx=tableidx,
                               mapping=instr._tableNameToIndex)
        else:
            tableidx = 0
            table = None
        # tableidx is always p4
        allargs = tools.instrResolveArgs(instr, tableidx, pargs, pkws)
        instrnum = self.instrnum(instrname, priority)
        synthid = self.engine.sched(instrnum, delay=delay, dur=dur, args=allargs)
        if whenfinished is not None:
            self._whenfinished[synthid] = whenfinished
        synth = Synth(engine=self.engine,
                      synthid=synthid,
                      instr=instr,
                      starttime=time.time()+delay,
                      dur=dur,
                      table=table,
                      pargs=allargs,
                      priority=priority)
        self._synths[synthid] = synth
        return synth

    def activeSynths(self, sortby="start") -> List[Synth]:
        """
        Returns a list of playing synths

        Args:
            sortby: either "start" (sort by start time) or None (unsorted)

        Returns:
            a list of active :class:`Synths<csoundengine.synth.Synth>`
        """
        synths = [synth for synth in self._synths.values() if synth.isPlaying()]
        if sortby == "start":
            synths.sort(key=lambda synth:synth.startTime)
        return synths

    def scheduledSynths(self) -> List[Synth]:
        """
        Returns all scheduled synths (both active and future)
        """
        return list(self._synths.values())

    def unsched(self, *synthids: float, delay=0.) -> None:
        """
        Stop an already scheduled instrument

        Args:
            synthids: one or many synthids to stop
            delay: how long to wait before stopping them
        """
        for synthid in synthids:
            synth = self._synths.get(synthid)
            if synth and synth.isPlaying() or delay>0:
                # We just need to unschedule it from csound. If the synth is playing,
                # it will be deallocated and the callback will be fired
                self.engine.unsched(synthid, delay)
            else:
                self._deallocSynth(synthid, delay)

    def unschedLast(self, n=1, unschedParent=True) -> None:
        """
        Unschedule last synth

        Args:
            n: number of synths to unschedule
            unschedParent: if the synth belongs to a group, unschedule the
                whole group

        """
        activeSynths = self.activeSynths(sortby="start")
        for i in range(n):
            if activeSynths:
                last = activeSynths[-1]
                assert last.synthid in self._synths
                last.stop(stopParent=unschedParent)

    def unschedByName(self, instrname: str) -> None:
        """
        Unschedule all playing synths created from given instr
        """
        synths = self.findSynthsByName(instrname)
        for synth in synths:
            self.unsched(synth.synthid)

    def unschedAll(self, cancel_future=True, force=False) -> None:
        """
        Unschedule all playing synths
        """
        synthids = [synth.synthid for synth in self._synths.values()]
        futureSynths = [synth for synth in self._synths.values() if not synth.isPlaying()]
        for synthid in synthids:
            self.unsched(synthid, delay=0)

        if force or (cancel_future and futureSynths):
            self.engine.unschedAll()
            self._synths.clear()

    def findSynthsByName(self, instrname: str) -> List[Synth]:
        """
        Return a list of active Synths created from the given instr
        """
        out = []
        for synthid, synth in self._synths.items():
            if synth.instr.name == instrname:
                out.append(synth)
        return out

    def restart(self) -> None:
        """
        Restart the associated engine

        """
        self.engine.restart()
        for i, initcode in enumerate(self._initCodes):
            print(f"code #{i}: initCode")
            self.engine.compile(initcode)

    def readSoundfile(self, path: str, chan=0, free=False) -> TableProxy:
        """
        Read a soundfile, store its metadata in a :class:`~csoundengine.tableproxy.TableProxy`

        Args:
            path: the path to a soundfile
            chan: the channel to read, or 0 to read all channels into a
                (possibly) stereo or multichannel table
            free: free the table when the returned TableDef is itself deallocated

        Returns:
            a TableProxy, holding information like
            .tabnum: the table number
            .path: the path you just passed
            .nchnls: the number of channels in the soundfile
            .sr: the sample rate of the soundfile

        """
        table = self._pathToTable.get(path)
        if table:
            return table
        tabnum = self.engine.readSoundfile(path=path, chan=chan)
        import sndfileio
        info = sndfileio.sndinfo(path)
        table = TableProxy(tabnum=tabnum,
                           path=path,
                           sr=info.samplerate,
                           nchnls=info.channels,
                           session=self,
                           numframes=info.nframes,
                           freeself=free)

        self._registerTable(table)
        return table

    def _registerTable(self, tabproxy: TableProxy) -> None:
        self._tabnumToTable[tabproxy.tabnum] = tabproxy
        if tabproxy.path:
            self._pathToTable[tabproxy.path] = tabproxy

    def makeTable(self, data: U[np.ndarray, List[float]] = None,
                  size: int = 0, tabnum: int = 0,
                  block=True, callback=None, sr: int = 0,
                  freeself=True,
                  _instrnum: float = -1,
                  ) -> TableProxy:
        """
        Create a table with given data or an empty table of the given
        size

        Args:
            data (np.ndarray | List[float]): the data of the table. Use None
                if the table should be empty
            size (int): if not data is given, sets the size of the empty table created
            tabnum (int): 0 to let csound determine a table number, -1 to self assign
                a value
            block (bool): if True, wait until the operation has been finished
            callback (func): if given, this function will be called when the table
                is fully created
            sr (int): the samplerate of the data, if applicable.
            freeself (bool): if True, the underlying csound table will be freed
                whenever the returned TableProxy ceases to exist.
            _instrnum (float): private, used internally for argument tables to
                keep record of the instrument instance a given table is assigned
                to

        Returns:
            a TableProxy object
        """
        tabnum = self.engine.makeTable(data=data, size=size, tabnum=tabnum,
                                       _instrnum=_instrnum, block=block,
                                       callback=callback, sr=sr)
        if data is not None:
            if isinstance(data, np.ndarray):
                nchnls = tools.arrayNumChannels(data)
            elif isinstance(data, list):
                nchnls = 1
                assert isinstance(data[0], float)
            else:
                raise TypeError(f"data should be np.ndarray or list[float], got "
                                f"{type(data)} = {data=}")
            numframes = len(data)
        else:
            numframes = size
            nchnls = 1

        return TableProxy(tabnum=tabnum, sr=sr, nchnls=nchnls, numframes=numframes,
                          session=self, freeself=freeself)

    def playSample(self, sample: U[int, TableProxy, str],
                   chan=1, gain=1.,
                   dur=-1., speed=1., loop=False, delay=0., pan=-1.,
                   start=0., fade: float = None, gaingroup=0) -> Synth:
        """
        Play a sample. If a path is given, the soundfile will be read and the sample
        data will be cached. This cache can be evicted via XXX

        Args:
            sample: a path to a sample, a TableProxy
            dur: the duration of playback (-1 to play the whole sample)
            chan: the channel to play the sample to. In the case of multichannel
                  samples, this is the first channel
            pan: a value between 0-1. -1 means default, which is 0 for mono,
                0.5 for stereo. For multichannel (3+) samples, panning is not
                taken into account
            gain: gain factor. See also: gaingroup
            speed: speed of playback
            loop: True/False or -1 to loop as defined in the file itself (not all
                file formats define loop points)
            delay: time to wait before playback starts
            start: the starting playback time (0=play from beginning)
            fade: fade in/out in secods. None=default
            gaingroup: the idx of a gain group. The gain of all samples routed to the
                same group are scaled by the same value and can be altered as a group
                via Engine.setSubGain(idx, gain)

        Returns:
            A Synth with the following modulatable parameters: gain, speed, chan, pan

        """
        if isinstance(sample, int):
            tabnum = sample
        elif isinstance(sample, TableProxy):
            tabnum = sample.tabnum
        else:
            table = self.readSoundfile(sample, free=False)
            tabnum = table.tabnum
        # isndtab, iloop, istart, ifade
        if fade is None:
            fade = config['sample_fade_time']
        return self.sched('.playSample',
                          delay=delay,
                          dur=dur,
                          pargs=dict(isndtab=tabnum, iloop=int(loop), istart=start,
                                     ifade=fade, igaingroup=gaingroup,
                                     kchan=chan, kspeed=speed, kpan=pan, kgain=gain))

    def makeRenderer(self, sr: int = None, ksmps: int = None) -> Renderer:
        """
        Create a Renderer (to render offline) with the instruments defined
        in this Session

        To schedule events, use the .sched method of the renderer

        Args:
            sr: the samplerate (see config['rec.sr'])
            ksmps: ksmps used for rendering (see also config['rec.ksmps'])

        Returns:
            a Renderer

        TODO: add example

        """
        renderer = Renderer(sr=sr or config['rec.sr'],
                            nchnls=self.engine.nchnls,
                            ksmps=ksmps or config['rec.ksmps'],
                            a4=self.engine.a4)
        for instrname, instrdef in self.instrRegistry.items():
            renderer.registerInstr(instrdef)
        return renderer

    def _defBuiltinInstrs(self):
        for csoundInstr in builtinInstrs:
            self.registerInstr(csoundInstr)


def getSession(name="default") -> Opt[Session]:
    '''
    Get/create a Session. A Session controls a series of instruments
    and is associated to an Engine. In order to create a Session an
    engine must habe been started before.

    Example::

        # create a session/engine with udp support enabled
        >>> engine = Engine("foo", udpserver=True)
        >>> s = getSession("foo")
        # Alternatively:
        >>> s = Engine("foo", ...).session()

        # an instrument is defined with the code inside instr/endin
        # (don't use p4, it is reserved)
        >>> Instr("sine", """
        ... kamp = p5
        ... kfreq = p6
        ... outch 1, oscili:a(kamp, kfreq)
        ... """).register(s)
        >>> synth = s.sched('sine', kamp=0.1, kfreq=442)
        >>> synth.setp(kfreq=800)
        >>> synth.stop()
        # A Session exists as long as the underlying engine exists
        >>> s2 = getSession("foo")
        >>> s2 is s
        True

    Args:
        name: the name of the Engine to which this Session belongs.
    '''
    engine = getEngine(name)
    session = Session._activeSessions.get(name)
    if session:
        assert engine is not None and engine.started
        return session
    if engine is None:
        raise ValueError(f"Engine {name} does not exist.")
    return Session(name)


def groupSynths(synths: List[AbstrSynth]) -> SynthGroup:
    """
    Groups synths together to form a SynthGroup

    Example
    =======

    >>> from csoundengine import *
    >>> s = Engine().session()
    >>> s.defInstr('sender', r'''
    ... ibus = p5
    ... kfreq = p6
    ... asig vco2 0.1, 1000
    ... busout(ibus, asig)
    ... ''')
    >>> s.defInstr('receiver', r'''
    ... ibus = p5
    ... asig = busin:a(ibus)
    ... asig *= 0.5
    ... outch 1, asig
    ... ''')
    >>> bus = s.assignBus()
    >>> chain = groupSynths([s.sched('sender', ibus=bus.busnum),
    ...                    s.sched('reveiver', ibus=bus.busnum)])
    >>> chain[0].setp('kfreq', 440)
    >>> chain.stop()
    """
    realsynths: List[Synth] = []
    for synth in synths:
        if isinstance(synth, Synth):
            realsynths.append(synth)
        elif isinstance(synth, SynthGroup):
            realsynths.extend(synth)
    return SynthGroup(synths=realsynths)