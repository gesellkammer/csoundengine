"""
An Engine implements a simple interface to run and control a csound process.

.. code::

    from csoundengine import Engine
    # create an engine with default options for the platform
    engine = Engine()
    engine.defInstr('''
      instr synth
        kmidinote = p4
        kamp = p5
        kcutoff = p6
        kfreq = mtof:k(kmidinote)
        asig = vco2:a(kamp, kfreq)
        asig = moogladder2(asig, kcutoff, 0.9)
        aenv = linsegr:a(0, 0.1, 1, 0.1, 0)
        asig *= aenv
        outs asig, asig
      endin
    ''')
    # start a synth with indefinite duration
    event = engine.sched("synth", args=[67, 0.1, 3000])

    # any parameter can be modified afterwords:
    # change midinote
    engine.setp(event, 4, 67)

    # modify cutoff
    engine.setp(event, 6, 1000, delay=4)

    # stop the synth:
    engine.unsched(event)


"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional as Opt, Union as U, Sequence as Seq, \
    KeysView, Callable, Dict, List, Tuple, Set
import ctypes as _ctypes
import atexit as _atexit
import queue as _queue

import time

import numpy as np

from emlib import iterlib, net
from emlib.containers import IntPool
import numpyx

from .config import config, logger
from . import csoundlib
from . import ujacktools as jacktools
from . import tools
from . import engineorc
from .engineorc import CONSTS
from .errors import CsoundError
if TYPE_CHECKING:
    from . import session as _session

try:
    import ctcsound

    _MYFLTPTR = _ctypes.POINTER(ctcsound.MYFLT)

except:
    print("Using mocked ctcsound, this should only happen when building"
          "the sphinx documentation")
    from sphinx.ext.autodoc.mock import _MockObject
    ctcsound = _MockObject()


__all__ = [
    'Engine',
    'getEngine',
    'activeEngines',
    'config',
    'logger',
    'csoundlib'
]




def _generateUniqueEngineName(prefix="engine") -> str:
    for i in range(10000):
        name = f"{prefix}{i}"
        if name not in Engine.activeEngines:
            return name

def _asEngine(e: U[str, Engine]) -> Engine:
    if isinstance(e, Engine):
        return e
    return getEngine(e)


class Engine:
    """
    Create a csound Engine, which controls an underlying csound process.
    Default values can be configured via `config.edit()`.

    .. note::

        When using csoundengine inside jupyter all csound output (including
        any error messages) will be shown in the terminal where jupyter was
        started

    Example
    =======

        >>> from csoundengine import *
        # Create an Engine with default options. The most appropriate backend for the
        # platform (from the available backends) will be chosen.
        >>> engine = Engine()
        # The user can specify many options, if needed, or defaults can be set
        # via config.edit()
        >>> engine = Engine(backend='portaudio', buffersize=256, ksmps=64, nchnls=2)
        >>> engine.defInstr('testoutput', r'''
        ... instr test
        ...     iperiod = p4
        ...     kchan init 0
        ...     if metro(1/iperiod) == 1 then
        ...         kchan = (kchan + 1) % nchnls
        ...     endif
        ...     asig pinker
        ...     asig *= linsegr:a(0, 0.1, 1, 0.1, 0)
        ...     outch ifirstchan+kchan, asig
        ... endin
        ... ''')
        >>> eventid = engine.sched('testoutput', args=[1.])
        >>> # wait, then evaluate next line to stop
        >>> engine.unsched(eventid)
        >>> engine.stop()


    Any option with a default value of None has a corresponding slot in the
    config.

    Args:
        name: the name of the engine
        sr: sample rate
        ksmps: samples per k-cycle
        backend: passed to -+rtaudio
        outdev: the audio output device, passed to -o
        indev: the audio input device, passed to -i
        a4: freq of a4
        nchnls: number of output channels (passed to nchnls)
        nchnls_i: number of input channels
        buffersize: samples per buffer, corresponds to csound's -b option
        numbuffers: the number of buffers to fill. Together with the buffersize determines
            the latency of csound and any communication between csound and the python
            host
        globalcode: code to evaluate as instr0
        includes: a list of files to include
        numAudioBuses: number of audio buses
        quiet: if True, suppress output of csound (-m 0)
        udpserver: if True, start a udp server for communication (see udpport)
        udpport: the udpport to use for real-time messages. 0=autoassign port
        commandlineOptions: extra command line options passed verbatim to the
            csound process when started

    Attributes:
        activeEngines (dict): a dictionary of active engines (name:Engine)
        started: is this Engine active?
        udpport: will hold the udp port if the server is started with UDP support

    """
    activeEngines: Dict[str, Engine] = {}

    _reservedInstrnums = set(engineorc.BUILTIN_INSTRS.values())
    _builtinInstrs = engineorc.BUILTIN_INSTRS
    _builtinTables = engineorc.BUILTIN_TABLES

    def __init__(self,
                 name:str = None,
                 sr:int = None,
                 ksmps:int = None,
                 backend:str = None,
                 outdev:str=None,
                 indev:str=None,
                 a4:int = None,
                 nchnls:int = None,
                 nchnls_i:int=None,
                 buffersize:int=None,
                 numbuffers:int=None,
                 globalcode:str = "",
                 numAudioBuses:int=None,
                 quiet:bool=None,
                 udpserver:bool=None,
                 udpport:int=0,
                 commandlineOptions:List[str]=None,
                 includes:List[str]=None,
                 autostart=True):
        if name is None:
            name = _generateUniqueEngineName()
        elif name in Engine.activeEngines:
            raise KeyError(f"engine {name} already exists")
        cfg = config
        if backend is None or backend == 'default':
            backend = cfg[f'{tools.platform}.backend']
        backends = csoundlib.getAudioBackendNames()
        if backend not in backends:
            # should we fallback?
            fallback_backend = cfg['fallback_backend']
            if not fallback_backend:
                raise CsoundError(f"The backend {backend} is not available, "
                                  f"possible backends: {backends}. "
                                  f"no fallback backend defined")
            logger.info(f"The backend {backend} is not available. "
                         f"possible backends: {backends}. "
                         f"Fallback backend: {fallback_backend}")
            backend = fallback_backend
        if outdev is None or indev is None:
            defaultin, defaultout = csoundlib.defaultDevicesForBackend(backend)
            if outdev is None:
                outdev = defaultout
            if indev is None:
                indev = defaultin

        commandlineOptions = commandlineOptions if commandlineOptions is not None else []
        sr = sr if sr is not None else cfg['sr']
        if sr == 0:
            sr = csoundlib.getSamplerateForBackend(backend)
            if not sr:
                # failed to get sr for backend
                sr = 44100
                logger.error(f"Failed to get sr for backend {backend}, using default: {sr}")
        if a4 is None: a4 = cfg['A4']
        if ksmps is None: ksmps = cfg['ksmps']
        if nchnls_i is None:
            nchnls_i = cfg['nchnls_i']
        if nchnls is None:
            nchnls = cfg['nchnls']

        if nchnls == 0 or nchnls_i == 0:
            print("*******", backend, outdev, indev)
            inchnls, outchnls = csoundlib.getNchnls(backend, device=outdev, indevice=indev)
            nchnls = nchnls or outchnls
            nchnls_i = nchnls_i or inchnls

        assert nchnls > 0
        assert nchnls_i >= 0

        commandlineOptions = commandlineOptions if commandlineOptions is not None else []
        if quiet is None: quiet = cfg['suppress_output']
        if quiet:
            commandlineOptions.append('-m0')
            commandlineOptions.append('-d')
        self.name = name
        self.sr = sr
        self.backend = backend
        self.a4 = a4
        self.ksmps = ksmps
        self.outdev = outdev
        self.nchnls = nchnls
        self.nchnls_i = nchnls_i
        self.globalcode = globalcode
        self.started = False
        self.extraOptions = commandlineOptions
        self.includes = includes
        self.numAudioBuses = numAudioBuses or config['num_audio_buses']
        if buffersize is None:
            buffersize =  cfg['buffersize']
            if buffersize == 0:
                buffersize = ksmps * 2
        self.buffersize = buffersize
        self.numbuffers = (numbuffers or
                           config['numbuffers'] or
                           tools.determineNumbuffers(self.backend, buffersize=buffersize))
        if udpserver is None: udpserver = config['start_udp_server']
        if udpserver:
            self.udpport = udpport or net.findport()
            self._udpsocket = net.udpsocket()
            self._sendAddr = ("127.0.0.1", self.udpport)
        else:
            self.udpport = None
            self._udpsocket = None
            self._sendAddr = None

        self._perfThread: Opt[ctcsound.CsoundPerformanceThread] = None
        self.csound: Opt[ctcsound.Csound] = None            # the csound object
        self._fracnumdigits = 4        # number of fractional digits used for unique instances
        self._exited = False           # are we still running?

        # the template to create new engines
        self._csdstr = engineorc.ORC_TEMPLATE

        # counters to create unique instances for each instrument
        self._instanceCounters = {}

        # Maps instrname/number: code
        self._instrRegistry:Dict[U[str, int], str] = {}

        self._outvalueCallbacks = {}   # a dict of callbacks, reacting to outvalue opcodes

        # Maps used for strSet / strGet
        self._indexToStr: Dict[int:str] = {}
        self._strToIndex: Dict[str:int] = {}
        self._strLastIndex = 20

        # global code added to this engine
        self._globalcode = {}

        # this will be a numpy array pointing to a csound table of
        # NUMTOKENS size. When an instrument wants to return a value to the
        # host, the host sends a token, the instr sets table[token] = value
        # and calls 'outvale "__sync__", token' to signal that an answer is
        # ready
        self._responsesTable: Opt[np.ndarray] = None

        # a table with sub-mix gains which can be used to group
        # synths, samples, etc.
        self._subgainsTable: Opt[np.ndarray] = None

        # tokens start at 1, leave token 0 to signal that no sync is needed
        # tokens are used as indices to _responsesTable, which is an alias of
        # gi__responses
        self._tokens = list(range(1, CONSTS['numtokens']))

        # a pool of reserved table numbers
        reservedTablesStart = CONSTS['reservedTablesStart']
        self._tablePool = IntPool(CONSTS['numReservedTables'], start=reservedTablesStart)

        # a dict of token:callback, used to register callbacks when asking for
        # feedback from csound
        self._responseCallbacks = {}

        # a dict mapping tableindex to fractional instr number
        self._assignedTables: Dict[int, float] = {}
        self.activeEngines[name] = self

        self._tableCache: Dict[int, np.ndarray] = {}

        self._audioBusPool = IntPool(self.numAudioBuses)
        self._audioBusRefsTable: Opt[np.ndarray] = None
        self._instrNumCache: Dict[str, int] = {}

        self._session: Opt[_session.Session] = None
        self.started = False

        if autostart:
            self.start()
            self.sync()

    def __repr__(self):
        return f"Engine(name={self.name}, backend={self.backend}, " \
               f"out={self.outdev}, nchnls={self.nchnls})"

    def __del__(self):
        self.stop()

    def _getToken(self) -> int:
        """
        Get a unique token, to pass to csound for a sync response
        """
        return self._tokens.pop()

    def _releaseToken(self, token:int) -> None:
        """ Release token back to pool when done """
        self._tokens.append(token)

    def _assignTableNumber(self, p1=-1.) -> int:
        """
        Return a free table number and mark that as being used.
        To release the table, call unassignTable

        Args:
            p1: the p1 of an instr instance if this table is assigned to
                a specific instance, or -1 to mark is as a free-standing table
                (the default)
            
        Returns:
            the table number (an integer)
        """
        if len(self._tablePool) == 0:
            raise RuntimeError("Table pool is empty")

        tabnum = self._tablePool.pop()
        assert tabnum not in self._assignedTables and tabnum is not None
        self._assignedTables[tabnum] = p1
        return tabnum

    def _assignEventId(self, instrnum: U[int, str]) -> float:
        """
        This is not really a unique instance, there might be conflicts
        with a previously scheduled event. To really generate a unique instance
        we would need to call uniqinstance, which creates a roundtrip to csound

        Args:
            instrnum (int): the instrument number

        """
        if isinstance(instrnum, str):
            instrnum = self._instrNameToNumber(instrnum)
        c = self._instanceCounters.get(instrnum, 0)
        c += 1
        self._instanceCounters[instrnum] = c
        instancenum = (c % int(10 ** self._fracnumdigits - 2)) + 1
        return self._makeEventId(instrnum, instancenum)

    def _makeEventId(self, num:int, instance:int) -> float:
        frac = (instance / (10**self._fracnumdigits)) % 1
        return num + frac
        
    def _startCsound(self) -> None:
        buffersize = self.buffersize
        optB = buffersize*self.numbuffers
        if self.backend == 'jack':
            if not jacktools.jack_running():
                logger.error("jack is not running")
                raise RuntimeError("jack is not running")
            jackinfo = jacktools.get_info()
            self.sr = jackinfo.samplerate
            if optB < jackinfo.blocksize*2:
                optB = jackinfo.blocksize*2
                self.numbuffers = optB // self.buffersize
                logger.warning(f"Using -b {self.buffersize}, -B {optB}")

        options = ["-d", "-odac", f"-b{buffersize}", f"-B{optB}",
                   "-+rtaudio=%s" % self.backend]
        if self.extraOptions:
            options.extend(self.extraOptions)

        if self.backend == 'jack':
            if self.name is not None:
                clientname = self.name.strip().replace(" ", "_")
                options.append(f'-+jack_client=csoundengine.{clientname}')

        if self.udpport is not None:
            options.append(f"--port={self.udpport}")

        cs = ctcsound.Csound()
        for opt in options:
            cs.setOption(opt)
        if self.includes:
            includelines = [f'#include "{include}"' for include in self.includes]
            includestr = "\n".join(includelines)
        else:
            includestr = ""
        orc = engineorc.ORC_TEMPLATE.format(sr=self.sr,
                                            ksmps=self.ksmps,
                                            nchnls=self.nchnls,
                                            nchnls_i=self.nchnls_i,
                                            backend=self.backend,
                                            a4=self.a4,
                                            globalcode=self.globalcode,
                                            includes=includestr,
                                            numAudioBuses=self.numAudioBuses)
        logger.debug("--------------------------------------------------------------")
        logger.debug("  Starting performance thread. ")
        logger.debug(f"     Options: {options}")
        logger.debug(orc)
        logger.debug("--------------------------------------------------------------")
        cs.compileOrc(orc)
        logger.info(f"Starting csound with options: {options}")
        cs.start()
        pt = ctcsound.CsoundPerformanceThread(cs.csound())
        pt.play()
        self._orc = orc
        self.csound = cs
        self._perfThread = pt
        if config['set_sigint_handler']:
            tools.setSigintHandler()
        self._setupCallbacks()

    def _setupGlobalInstrs(self):
        self._perfThread.scoreEvent(0, "i", [self._builtinInstrs['cleanbuses'], 0, -1])

    def stop(self):
        """
        Stop this Engine
        """
        if not self.started or self._exited:
            return
        self._perfThread.stop()
        self.csound.stop()
        self.csound.cleanup()
        self._exited = True
        self.csound = None
        self._perfThread = None
        self._instanceCounters = {}
        self._instrRegistry = {}
        self.activeEngines.pop(self.name, None)
        self.started = False

    def start(self):
        """ Start this engine. The call to .start() is only needed if
         the Engine was created with autostart=False """
        if self.started:
            return
        logger.info(f"Starting engine {self.name}")
        self._startCsound()
        priorengine = self.activeEngines.get(self.name)
        if priorengine:
            priorengine.stop()
        self.activeEngines[self.name] = self
        self._subgainsTable = self.csound.table(self._builtinTables['subgains'])
        self._responsesTable = self.csound.table(self._builtinTables['responses'])
        self._audioBusRefsTable = self.csound.table(self._builtinTables['busrefs'])
        self._setupGlobalInstrs()

        self.started = True
        strsets = ["cos", "linear", "smooth", "smoother"]
        for s in strsets:
            self.strSet(s)

    def restart(self) -> None:
        """ Restart this engine. All defined instrs / tables are removed"""
        self.stop()
        time.sleep(2)
        self.start()
        
    def _outcallback(self, _, channelName, valptr, chantypeptr):
        funcOrFuncs = self._outvalueCallbacks.get(channelName)
        if not funcOrFuncs:
            return
        if callable(funcOrFuncs):
            val = _ctypes.cast(valptr, _MYFLTPTR).contents.value
            funcOrFuncs(channelName, val)
            return
        for i, func in enumerate(funcOrFuncs):
            val = _ctypes.cast(valptr, _MYFLTPTR).contents.value
            func(channelName, val)

    def _setupCallbacks(self) -> None:

        def _syncCallback(_, token):
            """ Called with outvalue __sync__, the value is put
            in gi__responses at token idx, then __sync__ is
            called with token to signal that a response is
            waiting. The value can be retrieved via self._responsesTable[token]
            """
            token = int(token)
            callback = self._responseCallbacks.get(token)
            if callback:
                del self._responseCallbacks[token]
                callback(token)
                self._releaseToken(token)
            else:
                logger.error(f"Unknown sync token: {token}")

        self._outvalueCallbacks[bytes("__sync__", "ascii")] = _syncCallback
        self.csound.setOutputChannelCallback(self._outcallback)

    def registerOutvalueCallback(self, chan:str, func: Callable[[str, float], None]) -> None:
        """
        Register a function `(channelname, newvalue) -> None`, which will be called
        whenever the given channel is modified via the "outvalue" opcode. Multiple
        functions per channel can be registered

        Args:
            chan: the name of a channel
            func: a function of the form `func(chan:str, newvalue:float) -> None`

        """
        key = bytes(chan, "ascii")
        previousCallback = self._outvalueCallbacks.get(key)
        if chan.startswith("__"):
            if previousCallback:
                logger.warning("Attempting to set a reserved callback, but one "
                               "is already present. The new one will replace the old one")
            self._outvalueCallbacks[key] = func
        else:
            if not previousCallback:
                self._outvalueCallbacks[key] = [func]
            else:
                assert isinstance(previousCallback, list)
                previousCallback.append(func)

    def controlLatency(self) -> float:
        """
        The latency (in seconds) of the communication to the underlying csound process.
        This latency depens on the buffersize and number of buffers
        """
        return self.buffersize/self.sr * self.numbuffers

    def sync(self) -> None:
        """
        Block until csound is responsive

        Example
        =======

            >>> from csoundengine import *
            >>> e = Engine(...)
            >>> tables = [e.makeEmptyTable(size=1000) for _ in range(10)]
            >>> e.sync()
            >>> # do something with the tables
        """
        self._perfThread.flushMessageQueue()
        token = self._getToken()
        pargs = [self._builtinInstrs['pingback'], 0, 0.01, token]
        self._eventNotify(token, pargs)

    def defInstr(self, instrcode:str) -> None:
        """
        Compile a csound instrument

        Args:
            instrcode : the instrument definition, beginning with 'instr xxx'

        Example
        =======

            >>> from csoundengine import *
            >>> e = Engine(...)
            >>> e.defInstr(r'''
            ...   instr vco
            ...     kfreq = p4
            ...     kamp = p5
            ...     asig = oscili:a(kamp, kfreq)
            ...     outch 1, asig
            ...   endin''')
            >>> e.sched('vco', dur=4, args=[1000, 0.1])
        """
        assert self.started
        instrn = csoundlib.instrNames(instrcode)
        if isinstance(instrn, list):
           raise ValueError("Only one name/number is allowed. Use multiple defInstrs")
        assert isinstance(instrn, (int, str))
        if instrn in self._reservedInstrnums:
            raise ValueError(f"Instrument {instrn} is reserved")
        self._instrRegistry[instrn] = instrcode
        logger.debug(f"------ defInstr (compileOrc): {instrn}")
        logger.debug(instrcode)
        logger.debug("------ end")
        self.sendCode(instrcode)
        if isinstance(instrn, str):
            self._cacheInstrnumForNamedInstr(instrn)

    def sendCode(self, code:str, block=False) -> None:
        """
        Send (compile) code to the running csound instance. The code sent
        can be any orchestra code

        Args:
            code (str): the code to send
            block (bool): if True, this method will block until the code
                has taken effect

        .. note::

            If this instance has been started with a UDP port and
            the config option 'prefer_udp' is true, the code will be sent
            via udp. Otherwise the API is used. This might have an impact in
            the resulting latency of the operation, since using the API when
            running a performance thread can cause delays under certain
            circumstances

        Example
        =======

            >>> e = Engine()
            >>> e.sendCode("giMelody[] fillarray 60, 62, 64, 65, 67, 69, 71")
            >>> code = open("myopcodes.udo").read()
            >>> e.sendCode(code)
        """
        if self.udpport is not None and config['prefer_udp']:
            self._udpSend(code)
            if block:
                time.sleep(self.controlLatency())
        else:
            if not block:
                self.csound.compileOrc(code)
            else:
                self.csound.evalCode(code)

    def evalCode(self, code:str, once=False) -> float:
        """
        Evaluate code, return the result of the evaluation

        Args:
            code (str): the code to evaluate
            once (bool): if True, any code will be evaluated only once

        Returns:
            the result of the evaluation

        Example
        =======

            >>> e = Engine()
            >>> e.defInstr(r'''
            ... instr myinstr
            ...   prints "myinstr!"
            ...   turnoff
            ... ''')
            >>> e.sendCode(r'''
            ... opcode getinstrnum, i, S
            ...   Sinstr xin
            ...   inum nstrnum
            ...   xout inum
            ... endop''')
            >>> e.evalCode('return getinstrnum("myinstr")')

        """
        assert self.started
        if once:
            out = self._globalcode.get(code)
            if out is not None:
                return out
        logger.debug(f"evalCode: \n{code}")
        self._globalcode[code] = out = self.csound.evalCode(code)
        return out

    def tableWrite(self, tabnum:int, idx:int, value:float, delay:float=0.) -> None:
        """
        Write to a specific index of a table

        Args:
            tabnum (int): the table number
            idx (int): the index to modify
            value (float): the new value
            delay (float): delay time in seconds
        """
        assert self.started
        if delay == 0:
            arr = self.getTableData(tabnum)
            if arr is None:
                raise ValueError(f"table {tabnum} not found")
            arr[idx] = value
        else:
            pargs = [self._builtinInstrs['tabwrite'], delay, 1, tabnum, idx, value]
            self._perfThread.scoreEvent(0, "i", pargs)

    def getTableData(self, idx:int) -> Opt[np.ndarray]:
        """
        Returns a numpy array pointing to the data of the table. Any modifications
        to this array will modify the table itself

        Args:
            idx (int): the table index

        Returns:
            a numpy array pointing to the data array of the table, or None
            if the table was not found

        Example
        =======

        >>> # TODO
        """
        return self.csound.table(idx)

    def sched(self, instr:U[int, float, str], delay:float = 0, dur:float = -1,
              args:Seq[U[float, str]] = None) -> float:
        """
        Schedule an instrument

        Args:

            instr : the instrument number/name. If it is a fractional number,
                that value will be used as the instance number.
            delay    : time to wait before instrument is started
            dur      : duration of the event
            args     : any other args expected by the instrument, starting with p4. Any
                string arguments will be converted to a string index via strSet. These
                can be retrieved via strget in the csound instrument

        Returns: 
            a fractional p1 of the instr started, which identifies this event

        Example
        =======

        >>> from csoundengine import *
        >>> e = Engine()
        >>> e.defInstr(r'''
        ...   instr 10
        ...     kfreq = p4
        ...     kcutoff = p5
        ...     Smode strget p6
        ...     asig vco2 0.1, kfreq
        ...     if strcmp(Smode, "lowpass") == 0 then
        ...       asig moogladder2 asig, kcutoff, 0.95
        ...     else
        ...       asig K35_hpf asig, kcutoff, 9.0
        ...     endif
        ...     outch 1, asig
        ...   endif
        ... ''')
        >>> eventid = e.sched(10, 2, args=[200, 400, "lowpass"])
        >>> # simple automation in python
        >>> for cutoff in range(400, 3000, 10):
        ...     e.setp(eventid, 5, cutoff)
        ...     time.sleep(0.01)
        >>> e.unsched(eventid)
        """
        assert self.started
        if isinstance(instr, float):
            instrfrac = instr
        else:
            instrfrac = self._assignEventId(instr)
        pargs = [instrfrac, delay, dur]
        if args:
            pargs.extend(a if not isinstance(a, str) else self.strSet(a) for a in args)
            logger.debug(f"Engine.sched: scoreEvent(0, 'i', {pargs})  -> {instrfrac}")
        self._perfThread.scoreEvent(0, "i", pargs)
        return instrfrac

    def _cacheInstrnumForNamedInstr(self, instrname:str) -> int:
        token = self._getToken()
        msg = f'i {self._builtinInstrs["nstrnum"]} 0 0.1 {token} "{instrname}"'
        instrnum = int(self._inputMessageNotify(token, msg))
        self._instrNumCache[instrname] = instrnum
        return instrnum

    def _instrNameToNumber(self, instrname:str) -> int:
        if instrnum := self._instrNumCache.get(instrname):
            return instrnum
        return self._cacheInstrnumForNamedInstr(instrname)

    def unsched(self, p1:U[float, str], delay:float = 0) -> None:
        """
        Stop a playing event

        Args:
            p1: the instrument number/name to stop
            delay: if 0, remove the instance as soon as possible

        Example
        =======

        >>> from csoundengine import *
        >>> e = Engine(...)
        >>> e.defInstr(r'''
        ... instr sine
        ...   a0 oscili 0.1, 1000
        ...   outch 1, a0
        ... endin
        ... ''')
        >>> # sched an event with indefinite duration
        >>> eventid = e.sched(10, 0, -1)
        >>> e.unsched(eventid, 10)

        """
        if isinstance(p1, str):
            p1 = self._instrNameToNumber(p1)
        pfields = [self._builtinInstrs['turnoff'], delay, 0.1, p1]
        self._perfThread.scoreEvent(0, "i", pfields)

    def unschedFuture(self) -> None:
        """
        Remove all future notes
        """
        self.csound.rewindScore()

    def session(self):
        """
        Return the Session corresponding to this Engine

        Returns:
            the corresponding Session

        Example
        =======

        >>> from csoundengine import *
        >>> session = Engine(...).session()
        >>> session.defInstr("synth", r'''
        ... kamp = p4
        ... kmidi = p5
        ... asig vco2 kamp, mtof:k(kmidi)
        ... chnmix asig, "mix1"
        ... ''')
        >>> session.defInstr("post", r'''
        ... a1 chnget "mix1"
        ... a2 chnget "mix2"
        ... aL, aR reverbsc a1, a2, 0.85, 12000, sr, 0.5, 1
        ... outch 1, aL, 2, aR
        ... chnclear "mix1", "mix2"
        ... ''')
        >>> session.sched("post", priority=2)
        >>> for i, midi in enumerate([60, 64, 67]):
        ...     session.sched("synth", delay=i, dur=4, kamp=0.1, kmidi=midi)
        """
        from .session import Session
        if self._session is None:
            self._session = Session(self.name)
        return self._session

    def makeEmptyTable(self, size, numchannels=1, sr=0, instrnum=-1) -> int:
        """
        Create an empty table, returns the index of the created table

        Example
        =======

        Use a table to control amplitude of synths

        >>> from csoundengine import *
        >>> e = Engine()
        >>> tabnum = e.makeEmptyTable(128)
        >>> e.defInstr(r'''
        ... instr 10
        ...   imidi = p4
        ...   iamptab = p5
        ...   islot = p6
        ...   kamp table islot, iamptab
        ...   asig = oscili:a(interp(kamp), mtof(imidi))
        ...   outch 1, asig
        ... endin
        ... ''')
        >>> tabarray = e.getTableData(tabnum)
        >>> tabarray[0] = 0.5
        >>> eventid = e.sched(10, args=[67, tabnum, 0])
        # fade out
        >>> e.automateTable(tabnum=tabnum, idx=0, pairs=[1, 0.5, 5, 0.])
        """
        tabnum = self._assignTableNumber(p1=instrnum)
        pargs = [tabnum, 0, size, -2, 0]
        self._perfThread.scoreEvent(0, "f", pargs)
        self._perfThread.flushMessageQueue()
        if numchannels > 1 or sr > 0:
            self.setTableMetadata(tabnum, numchannels=numchannels, sr=sr)
        return tabnum

    def makeTable(self, data:U[Seq[float], np.ndarray]=None,
                  size:int = 0, tabnum:int=0, sr:int=0,
                  block=True, callback=None,
                  _instrnum=-1.
                  ) -> int:
        """
        Create a new table and fill it with data.

        Args:
            data: the data used to fill the table, or None if creating an empty table
            size: the size of the table (will only be used if no data is supplied)
            tabnum: the table number. If -1, a number is assigned by the engine.
                If 0, a number is assigned by csound (only possible in block or
                callback mode)
            block: wait until the table is actually created
            callback: call this function when ready - f(token, tablenumber) -> None
            sr: only needed if filling sample data. If given, it is used to fill the
                table metadata in csound, as if this table had been read via gen01
            _instrnum: the instrument this table should be assigned to, if applicable

        Returns:
            the index of the new table, if wait is True

        Example
        =======

        >>> from csoundengine import *
        >>> e = Engine()
        >>> import sndfileio
        >>> sample, sr = sndfileio.sndread("stereo.wav")
        >>> # modify the sample in python
        >>> sample *= 0.5
        >>> tabnum = e.makeTable(sample, sr=sr, block=True)
        >>> e.playSample(tabnum)
        """
        if tabnum < 0:
            tabnum = self._assignTableNumber(p1=_instrnum)
        elif tabnum == 0 and not callback:
            block = True
        if block or callback:
            assignedTabnum = self._makeTableNotify(data=data, size=size, sr=sr,
                                                   tabnum=tabnum, callback=callback)
            assert assignedTabnum > 0
            return assignedTabnum

        # Create a table asynchronously
        assert tabnum > 0
        if not data:
            # an empty table
            assert size > 0
            pargs = [tabnum, 0, size, -2, 0]
            self._perfThread.scoreEvent(0, "f", pargs)
            self._perfThread.flushMessageQueue()
        elif len(data) < 1900:
            # data can be passed as p-args directly
            pargs = np.zeros((len(data)+4,), dtype=float)
            pargs[0:4] = [tabnum, 0, len(data), -2]
            pargs[4:] = data
            self._perfThread.scoreEvent(0, "f", pargs)
            self._perfThread.flushMessageQueue()
        else:
            self._makeTableNotify(data=data, tabnum=tabnum, sr=sr)

        return int(tabnum)

    def _makeTableViaScore(self, tabnum:int, data: np.ndarray):
        size = data.shape[0]
        if len(data.shape) > 1:
            size *= data.shape[1]
        self._perfThread.scoreEvent(0, "f", [tabnum, 0, size, -2, 0])
        self._perfThread.flushMessageQueue()
        self._fillTableViaScore(data, tabnum=tabnum)
        self._perfThread.flushMessageQueue()

    def setTableMetadata(self, tabnum:int, sr:int, numchannels:int = 1) -> None:
        """
        Set metadata for a table holding sound samples. When csound reads a soundfile
        into a table, it stores some additional data, like samplerate and number of
        channels. A table created by other means and then filled with samples normally
        does not have this information. In most of the times this is ok, but there are
        some opcodes which need this information (loscil, for example). This
        method allows to set this information for such tables.
        """
        pargs = [self._builtinInstrs['ftsetparams'], 0, -1, tabnum, sr, numchannels]
        self._perfThread.scoreEvent(0, "i", pargs)

    def _registerSync(self, token:int) -> _queue.Queue:
        q = _queue.Queue()
        table = self._responsesTable
        self._responseCallbacks[token] = lambda token, q=q, t=table: q.put(t[token])
        return q

    def schedCallback(self, delay:float, callback:Callable) -> None:
        """
        Call callback after delay, triggered by csound scheduler

        Args:
            delay (float): the delay time, in seconds
            callback (callable): the callback, a function of the sort () -> None

        Example
        =======

        >>> from csoundengine import *
        >>> import time
        >>> e = Engine()
        >>> startTime = time.time()
        >>> e.schedCallback(2, lambda:print(f"Elapsed time: {time.time() - startTime}"))
        """
        token = self._getToken()
        pargs = [self._builtinInstrs['pingback'], delay, 0.01, token]
        self._eventNotify(token, pargs, callback=lambda token: callback())

    def _eventNotify(self, token:int, pargs, callback=None, timeout=1) -> Opt[float]:
        """
        Create a csound "i" event with the given pargs with the possibility
        of receiving a notification from the instrument

        The event is passed a token as p4 and can set a return value by:

            itoken = p4
            tabw kreturnValue, itoken, gi__responses
            ; or tabw_i ireturnValue, itoken, gi__responses
            outvalue "__sync__", itoken

        Args:
            token: a token as returned by self._getToken()
            pargs: the pfields passed to the event (beginning by p1)
            callback: if a callback is not passed, this method will block until a response
                is received from the csound event
            timeout: how long to wait for a response in blocking mode

        Returns:
            in blocking mode, it returns the value returned by the instrument. If
            a callback is given the callback will be called with the return value
            as argument
        """
        assert token == pargs[3]
        if callback:
            self._responseCallbacks[token] = callback
            self._perfThread.scoreEvent(0, "i", pargs)
            return None

        q = self._registerSync(token)
        self._perfThread.scoreEvent(0, "i", pargs)
        try:
            outvalue = q.get(block=True, timeout=timeout)
            return outvalue
        except _queue.Empty:
            raise TimeoutError(f"{token=}, {pargs=}")


    def _inputMessageNotify(self, token:int, inputMessage:str, callback=None,
                            timeout=1) -> Opt[float]:
        if callback:
            self._responseCallbacks[token] = callback
            self._perfThread.inputMessage(inputMessage)
            return None

        q = self._registerSync(token)
        self._perfThread.inputMessage(inputMessage)
        try:
            value = q.get(block=True, timeout=timeout)
            return value
        except _queue.Empty:
            raise TimeoutError(f"{token=}, {inputMessage=}")

    def _makeTableNotify(self, data:U[List[float], np.ndarray]=None, size=0, tabnum=0, callback=None, sr:int=0,
                         timeout=1) -> int:
        """
        Create a table with data (or an empty table of the given size).
        Let csound generate a table index if needed

        Args:
            data: the data to put in the table
            tabnum: the table number to create, 0 to let csound generate
                a table number
            callback: a callback of the form (token, value) -> None
                where value will hold the table number. If no callback
                is given this method will block until csound notifies that
                the table has been created and returns the table number
            sr: only needed if filling sample data. If given, it is used to fill
                metadata in csound, as if this table had been read via gen01
            timeout: how long to wait in blocking mode

        Returns:
            * if set to block (no callback), returns the table index
            * if a callback is given, returns -1 and the callback will be
              called when the value is ready
        """
        token = self._getToken()
        maketableInstrnum = self._builtinInstrs['maketable']
        delay = 0
        if data is None:
            assert size > 1
            # create an empty table of the given size
            empty = 1
            sr = 0
            numchannels = 1
            pargs = [maketableInstrnum, delay, 0.01, token, tabnum, size, empty,
                     sr, numchannels]
            tabnum = self._eventNotify(token, pargs, callback=callback, timeout=timeout)
            return int(tabnum)
        else:
            if not isinstance(data, np.ndarray):
                data = np.asarray(data)
            numchannels = tools.arrayNumChannels(data)
            numitems = len(data) * numchannels
            if numchannels > 1:
                data = data.ravel()

            if numitems < 1900:
                # create a table with the given data
                # if the table is small we can create it and fill it in one go
                empty = 0
                numchannels = tools.arrayNumChannels(data)
                if numchannels > 1:
                    data = data.flatten()
                pargs = [maketableInstrnum, delay, 0.01, token, tabnum, numitems, empty,
                         sr, numchannels]
                pargs.extend(data)
                tabnum = self._eventNotify(token, pargs, callback=callback, timeout=timeout)
                return int(tabnum)
            else:
                # create an empty table, fill it via a pointer
                empty = 1
                pargs = [maketableInstrnum, delay, 0.01, token, tabnum, numitems, empty,
                         sr, numchannels]
                # the next line blocks until the table is created
                tabnum = self._eventNotify(token, pargs)
                assert tabnum is not None

                self.fillTable(int(tabnum), data=data, method='pointer', block=False)
                return int(tabnum)

    def setChannel(self, channel:str, value:U[float, str, np.ndarray],
                   method:str=None, delay=0.) -> None:
        """
        Set the value of a software channel

        Args:
            channel: the name of the channel
            value: the new value, should math the type of the channel (a float for
                a control channel, a string for a string channel or a numpy array
                for an audio channel)
            method: one of 'api', 'score', 'udp'. None will choose the most appropriate
                method for the current engine/args
            delay: a delay to set the channel

        Example
        =======

        >>> from csoundengine import *
        >>> e = Engine()
        >>> e.initChannel("mastergain", 1.0)
        >>> e.defInstr(r'''
        ... instr 10
        ...   asig oscili 0.1, 1000
        ...   kmastergain = chnget:k("mastergain")
        ...   asig *= intrp(kmastergain)
        ... endin
        ... ''')
        >>> eventid = e.sched(10)
        >>> e.setChannel("mastergain", 0.5)
        """
        isaudio = isinstance(value, np.ndarray)
        if isaudio:
            method = "api"
        elif delay > 0:
            method = "score"
        elif method is None:
            if self.udpport and config['prefer_udp']:
                method = 'udp'
            else:
                method = 'api'

        if method == 'api':
            if isinstance(value, (int, float)):
                self.csound.setControlChannel(channel, value)
            elif isinstance(value, str):
                self.csound.setStringChannel(channel, value)
            else:
                self.csound.setAudioChannel(channel, value)
        elif method == 'score':
            if isinstance(value, (int, float)):
                instrnum = self._builtinInstrs['chnset']
                s = f'i {instrnum} {delay} 0.01 "{channel}" {value}'
                self._perfThread.inputMessage(s)
            else:
                instrnum = self._builtinInstrs['chnsets']
                s = f'i {instrnum} {delay} 0.01 "{channel}" "{value}"'
                self._perfThread.inputMessage(s)
        elif method == 'udp':
            if self.udpport is None:
                raise RuntimeError("This server has been started without udp support")
            self.udpSetChannel(channel, value)
        else:
            raise ValueError(f"method {method} not supported "
                             f"(choices: 'api', 'score', 'udp'")

    def initChannel(self, channel:str, value:U[float, str, np.ndarray]=0, kind:str=None,
                    mode="r") -> None:
        """
        Create a channel and set its initial value

        Args:
            channel (str): the name of the channel
            value (float|str|np.ndarray): the initial value of the channel,
                will also determine the type (k, a, S)
            kind (str): One of 'k', 'S', 'a'. Use None to auto determine the channel type.
            mode (str): r for read, w for write, rw for both.

        .. note::
                the `mode` is set from the perspective of csound. A read (input)
                channel is a channel which can be written to by the api and read
                from csound. An write channel (output) can be written by csound
                and read from the api

        Example
        =======

        >>> from csoundengine import *
        >>> e = Engine()
        >>> e.initChannel("mastergain", 1.0)
        >>> e.defInstr(r'''
        ... instr 10
        ...   asig oscili 0.1, 1000
        ...   kmastergain = chnget:k("mastergain")
        ...   asig *= intrp(kmastergain)
        ... endin
        ... ''')
        >>> eventid = e.sched(10)
        >>> e.setChannel("mastergain", 0.5)
        """
        modei = {
            "r": 1,
            "w": 2,
            "rw": 3
        }[mode]
        if kind is None:
            if isinstance(value, (int, float)):
                kind = 'k'
            elif isinstance(value, str):
                kind = 'S'
            elif isinstance(value, np.ndarray):
                kind = 'a'
        if kind == 'k':
            self.sendCode(f'chn_k "{channel}", {modei}\n')
            self.setChannel(channel, value, method="score")
        elif kind == 'a':
            self.sendCode(f'chn_a "{channel}", {modei}', block=True)
            if value:
                self.setChannel(channel, value)
        elif kind == 'S':
            self.sendCode(f'chn_S "{channel}", {modei}\n', block=True)
            self.setChannel(channel, value)
        else:
            raise TypeError("Expected an initial value of type float or string")

    def getControlChannel(self, channel:str) -> float:
        """
        Get the value of a channel

        Args:
            channel: the name of the channel

        Returns:
            the value of the channel. Raises KeyError if the channel
            does not exist.

        Example
        =======

        >>> from csoundengine import *
        >>> e = Engine()
        >>> e.initChannel("freq", mode="w")
        >>> e.defInstr('''
        ... instr pitchtrack
        ...   asig inch 1
        ...   afreq, alock plltrack asig, 0.25, 20, 0.33, 50, 800
        ...   kfreq = k(afreq)
        ...   chnset kfreq, "freq"
        ... endin
        ... ''')
        >>> eventid = e.sched("pitchtrack")
        >>> while True:
        ...     freq = e.getControlChannel("freq")
        ...     print(f"freq: {freq:.1f}")
        ...     time.sleep(0.1)
        """
        value, errorCode = self.csound.controlChannel(channel)
        if errorCode != 0:
            raise KeyError(f"control channel {channel} not found")
        return value

    def fillTable(self, tabnum:int, data, method='pointer', block=False) -> None:
        """
        Fill an existing table with data

        Args:
            data: the data to put into the table
            tabnum: the table number
            method: the method used, one of 'pointer', 'score' or 'api'
            block: this is only used with methods "score" and "api"
        """
        assert isinstance(tabnum, int) and tabnum > 0, \
            f"tabnum should be an int > 0, got {tabnum}"
        if method == 'pointer':
            # there is no async version
            numpyptr: np.array = self.csound.table(tabnum)
            if numpyptr is None:
                raise IndexError(f"Table {tabnum} does not exist")
            size = len(numpyptr)
            if size < len(data):
                numpyptr[:] = data[:size]
            else:
                numpyptr[:] = data
        elif method == 'score':
            return self._fillTableViaScore(data, tabnum=tabnum, block=block)
        elif method == 'api':
            return self._fillTableViaAPI(data, tabnum=tabnum, block=block)
        else:
            raise KeyError("method not supported. Must be one of pointer, score, api")

    def _fillTableViaScore(self, data, tabnum:int, block=False) -> None:
        """
        Fill a table through score messages.

        Args:
            data: the data to send to the table
            tabnum: the table index
            block: should we wait until everything is sent?
        """
        chunksize = 1800
        now = 0
        token = 0
        delayBetweenRows = 0
        instrnum= self._builtinInstrs['filltable']
        for idx, numitems in iterlib.chunks(0, len(data), chunksize):
            if block and numitems < chunksize:
                # last row
                token = self._getToken()
            pargs = [instrnum, token, now, 0.01, tabnum, idx, numitems]
            payload = data[idx: idx+numitems]
            pargs.extend(payload)
            self._perfThread.scoreEvent(0, "i", pargs)
            now += delayBetweenRows
        if block:
            self._perfThread.flushMessageQueue()

    def _fillTableViaAPI(self, data:np.ndarray, tabnum:int, block=True) -> None:
        """
        .. note::

            This method might have a long latency depending on the blocksize
            and number of buffers used

        Copy contents of a numpy array to a table. Table must exist.If data is 2D,
        it is flattened to 1D.

        Args:
            data: a numpy array (1D or 2D) of type float64
            tabnum: the table to copy data to. If not given, a table is created
            block: if True, data is copied synchronously

        """
        assert self.started

        if len(data.shape) == 2:
            data = data.flatten()
        else:
            raise ValueError("data should be a 1D or 2D array")

        if block:
            self.csound.tableCopyIn(tabnum, data)
        else:
            self.csound.tableCopyInAsync(tabnum, data)

    def readSoundfile(self, path:str, tabnum:int = None, chan=0,
                      callback=None, block=False) -> int:
        """
        Read a soundfile into a table, returns the table number

        Args:
            path: the path to the soundfile
            tabnum: if given, a table index. If None, an index is
                autoassigned
            chan: the channel to read. 0=read all channels
            block: if True, wait until soundfile is read, then return
            callback: if given, this function () -> None, will be called when
                soundfile has been read.

        Returns:
            the index of the created table

        >>> from csoundengine import *
        >>> e = Engine()
        >>> import sndfileio
        >>> tabnum = e.readSoundfile("stereo.wav", block=True)
        >>> eventid = e.playSample(tabnum)
        """
        if not block and not callback:
            return self._readSoundfileAsync(path=path, tabnum=tabnum, chan=chan)
        if tabnum is None:
            tabnum = self._assignTableNumber()
        token = self._getToken()
        ipath = self.strSet(path)
        pargs = [self._builtinInstrs['readSndfile'], 0, 0.01, token, ipath, tabnum, chan]
        self._eventNotify(token, pargs, callback=callback)

    def _readSoundfileAsync(self, path:str, tabnum:int=None, chan=0) -> int:
        assert self.started
        if tabnum is None:
            tabnum = self._assignTableNumber()
        s = f'f {tabnum} 0 0 -1 "{path}" 0 0 {chan}'
        self._perfThread.inputMessage(s)
        return tabnum

    def getUniqueInstrInstance(self, instr: U[int, str]) -> float:
        """
        Returns a truly unique instance number (a float p1) for the given instr

        Args:
            instr (int|str): an already defined csound instrument

        Returns:
            a unique p1.
        """
        if isinstance(instr, int):
            token = self._getToken()
            pargs = [self._builtinInstrs['uniqinstance'], 0, 0.01, token, instr]
            uniqinstr = self._eventNotify(token, pargs)
            return uniqinstr
        else:
            raise NotImplementedError("str instrs not implemented yet")

    def playSample(self, tabnum:int, delay=0., chan=1, speed=1., gain=1., fade=0.,
                   starttime=0., gaingroup=0, dur=-1.) -> float:
        """
        Play a sample already loaded into a table. Speed and gain can be modified
        via setp while playing

        Args:
            tabnum (int): the table where the sample data was loaded
            delay (float): when to start playback
            chan (int): the first channel to send output to (channels start with 1)
            speed (float): the playback speed
            gain (float): a gain applied to this sample
            fade (float): fadein/fadeout time in seconds
            starttime (float): playback can be started from anywhere within the table
            gaingroup (int): multiple instances can be gain-moulated via gaingroups
            dur (float): the duration of playback. Use -1 to play until the end

        Returns:
            the instance number of the playing instrument.
            dynamic pfields:

                * p4 = gain
                * p5 = speed

        Example
        =======

        >>> from csoundengine import *
        >>> e = Engine()
        >>> import sndfileio
        >>> sample, sr = sndfileio.sndread("stereo.wav")
        >>> # modify the sample in python
        >>> sample *= 0.5
        >>> tabnum = e.makeTable(sample, sr=sr, block=True)
        >>> eventid = e.playSample(tabnum)
        >>> # speed (p5) and gain (p4) can be modified while playing
        >>> e.setp(eventid, 5, 0.5)
        """
        return self.sched(self._builtinInstrs['playgen1'], delay=delay, dur=dur,
                          args=[gain, speed, tabnum, chan, fade, starttime, gaingroup])

    def playSoundFromDisc(self, path:str, delay=0., chan=0, speed=1., fade=0.01
                          ) -> float:
        """
        Play a soundfile from disc (via diskin2).

        Args:
            path: the path to the soundfile
            delay: time offset to start playing
            chan: first channel to output to
            speed: playback speed (2.0 will sound an octave higher)
            fade: fadein/out in seconds

        Returns:
            the instance number of the scheduled event
        """
        assert self.started
        p1 = self._assignEventId(self._builtinInstrs['playsndfile'])
        msg = f"i {p1} {delay} -1 \"{path}\" {chan} {speed} {fade}"
        self._perfThread.inputMessage(msg)
        return p1

    def setp(self, p1:float, *pairs, delay=0.) -> None:
        """
        Modify a parg of an active synth. Multiple pargs can be modified
        simultaneously. If only makes sense to modify a parg if a k-rate
        variable was assigned to this parg (see example)

        Args:
            p1 (float): the p1 of the instrument to automate
            *pairs: each pair consists of a parg index and a value
            delay (float): when to start the automation

        Example
        =======

        >>> engine = Engine(...)
        >>> engine.defInstr('''
        ... instr 10
        ...   kamp = p5
        ...   kfreq = p6
        ...   a0 oscili kamp, kfreq
        ...   outch 1, a0
        ... endin
        ... ''')
        >>> p1 = engine.sched(10, pargs=[0.1, 440])
        >>> engine.setp(p1, 5, 0.2, 6, 880, delay=0.5)
        """
        numpairs = len(pairs) // 2
        assert len(pairs) % 2 == 0 and numpairs <= 5
        # this limit is just the limit of the pwrite instr, not of the opcode
        args = [p1, numpairs]
        args.extend(pairs)
        self.sched(self._builtinInstrs['pwrite'], delay=delay, dur=-1, args=args)

    def automateTable(self, tabnum:int, idx:int, pairs: Seq[float],
                      mode='linear', delay=0.) -> float:
        """
        Automate a table slot

        Args:
            tabnum: the number of the table to modify
            idx: the slot index
            pairs: the automation data is given as a flat seq. of pairs (time,
              value). Times are relative to the start of the automation event
            mode: one of 'linear', 'cos', 'expon(xx)', 'smooth'. See the opcode
              `interp1d` for more information
            delay: the time delay to start the automation.

        Returns:
            the eventid of the instance performing the automation

        Example
        =======

        >>> e = Engine(...)
        >>> e.defInstr('''
        ... instr 10
        ...   itab = p4
        ...   kamp  table 0, itab
        ...   kfreq table 1, itab
        ...   outch 1, oscili:a(0.1, kfreq)
        ...   ftfree itab, 1  ; free the table when finished
        ... endin
        ... ''')
        >>> tabnum = e.makeTable([0.1, 1000])
        >>> eventid = e.sched(10, 0, 10, args=(tabnum,))
        # automate the frequency (slot 1)
        >>> e.automateTable(tabnum, 1, [0, 1000, 3, 200, 5, 200])
        """
        # tabpairs table will be freed by the instr itself
        tabpairs = self.makeTable(pairs, tabnum=0, block=True)
        args = [tabnum, idx, tabpairs, self.strSet(mode), 2, 1]
        dur = pairs[-2]+self.ksmps/self.sr
        return self.sched(self._builtinInstrs['automateTableViaTable'], delay=delay,
                          dur=dur, args=args)

    def automatep(self, p1: float, pidx: int, pairs:Seq[float], mode='linear', delay=0.
                  ) -> float:
        """
        Automate a pfield of a running event

        Args:
            p1: the fractional instr number of a running event, or an int number
                to modify all running instances of that instr
            pidx: the pfield index. If the pfield to modify if p4, pidx should be 4
            pairs: the automation data is given as a flat seq. of pairs (time, value).
                Times are relative to the start of the automation event
            mode: one of 'linear', 'cos', 'expon(xx)', 'smooth'. See the opcode `interp1d`
                for more information
            delay: the time delay to start the automation.
        Example
        =======

        >>> e = Engine(...)
        >>> e.defInstr('''
        ... instr 10
        ...   kfreq = p4
        ...   outch 1, oscili:a(0.1, kfreq)
        ... endin
        ... ''')
        >>> eventid = e.sched(10, 0, 10, args=(1000,))
        >>> e.automatep(eventid, 4, [0, 1000, 3, 200, 5, 200])
        """
        # table will be freed by the instr itself
        tabnum = self.makeTable(pairs, tabnum=0, block=True)
        args = [p1, pidx, tabnum, self.strSet(mode)]
        dur = pairs[-2]+self.ksmps/self.sr
        assert isinstance(dur, float)
        return self.sched(self._builtinInstrs['automatePargViaTable'], delay=delay,
                          dur=dur, args=args)

    def strSet(self, s:str) -> int:
        """
        Assign a numeric index to a string to be used inside csound
        """
        assert self.started
        stringIndex = self._strToIndex.get(s)
        if stringIndex:
            return stringIndex
        stringIndex = self._getStrIndex()
        instrnum = self._builtinInstrs['strset']
        msg = f'i {instrnum} 0 -1 "{s}" {stringIndex}'
        self._perfThread.inputMessage(msg)
        self._strToIndex[s] = stringIndex
        self._indexToStr[stringIndex] = s
        return stringIndex

    def strGet(self, index:int) -> Opt[str]:
        """
        Get the string previously set via strSet

        Example
        =======

        >>> e = Engine(...)
        >>> idx = e.strSet("foo")
        >>> e.strGet(idx)
        foo
        """
        return self._indexToStr.get(index)

    def _getStrIndex(self) -> int:
        out = self._strLastIndex
        self._strLastIndex += 1
        return out

    def _releaseTableNumber(self, tableindex:int) -> None:
        """
        Mark the given table as freed, so that it can be assigned
        again. It assumes that the table was deallocated already
        and the index can be assigned again.
        """
        instrnum = self._assignedTables.pop(tableindex, None)
        if instrnum is not None:
            logger.debug(f"Unassigning table {tableindex} for instr {instrnum}")
            self._tablePool.push(tableindex)

    def freeTable(self, tableindex:int, delay=0.) -> None:
        """
        Free the table with the given index
        """
        logger.debug(f"Freeing table {tableindex}")
        self._releaseTableNumber(tableindex)
        pargs = [self._builtinInstrs['freetable'], delay, 0.01, tableindex]
        self._perfThread.scoreEvent(0, "i", pargs)

    def testAudio(self, dur=4.) -> float:
        """
        Test this engine's output
        """
        assert self.started
        return self.sched(self._builtinInstrs['testaudio'], dur=dur)


    # ~~~~~~~~~~~~~~~ UDP ~~~~~~~~~~~~~~~~~~

    def _udpSend(self, code: str) -> None:
        if not self.udpport:
            logger.warning("This csound instance was started without udp")
            return
        msg = code.encode("ascii")
        self._udpsocket.sendto(msg, self._sendAddr)

    def udpSendOrc(self, code: str) -> None:
        """
        Send orchestra code via UDP. The code string can be of
        any size (if the code is too long for a UDP package, it is
        split into multiple packages)

        Args:
            code (str): the code to send
        """
        msg = code.encode("ascii")
        if len(msg) < 60000:
            self._udpsocket.sendto(msg, self._sendAddr)
            return
        msgs = iterlib.splitInChunks(msg, 60000)
        self._udpsocket.sendto(b"{{ " + msgs[0], self._sendAddr)
        for msg in msgs[1:-1]:
            self._udpsocket.sendto(msg, self._sendAddr)
        self._udpsocket.sendto(msgs[-1] + b" }}", self._sendAddr)

    def udpSendScoreline(self, scoreline:str) -> None:
        """ Send a score line to csound via udp

        Example
        =======

        >>> e = Engine(...)
        >>> e.defInstr('''
        ... instr 10
        ...   ifreq = p4
        ...   outch 1, oscili:a(0.1, ifreq)
        ... endin
        ... ''')
        >>> e.udpSendScoreline("i 10 0 4 440")
        """
        self._udpSend(f"& {scoreline}\n")

    def udpSetChannel(self, channel:str, value:U[float, str]) -> None:
        """
        Set a channel via UDP. The value will determine the kind of channel

        Args:
            channel (str): the channel name
            value (float|str): the new value
        """
        if isinstance(value, (int, float)):
            self._udpSend(f"@{channel} {value}")
        else:
            self._udpSend(f"%{channel} {value}")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    def setSubGain(self, idx: int, gain: float) -> None:
        """
        Sets one of the subgains to the given value

        Each engine has a set of subgains which can be used by instruments
        to set the gain as a group. These gains are stored in a table gi__subgains

        Args:
            idx: the subgain to set
            gain: the value of the subgain

        Example
        =======

        >>> # TODO
        """
        assert self.started
        self._subgainsTable[idx] = gain

    def assignBus(self) -> int:
        """
        Assign one audio bus, returns the bus number. This can be used
        together with the built-in opcodes `busout`, `busin` and `busmix`.
        From csound a bus can also be assigned by calling `busnew`

        Example
        =======

        >>> e = Engine(...)
        >>> e.defInstr('''
        ... instr 10
        ...   ibus = p4
        ...   asig vco2 0.1, kfreq
        ...   busout(ibus, asig)
        ... endin
        ... ''')
        >>> e.defInstr('''
        ... instr 20
        ...   ibus = p4
        ...   asig = busin(ibus)
        ...   ; do something with asig
        ...   asig *= 0.5
        ...   outch 1, asig
        ... ''')
        >>> busnum = e.assignBus()
        >>> s1 = e.sched(10, 0, 4, (busnum,))
        >>> s2 = e.sched(20, 0, 4, (busnum,))
        # When done with the bus, call `e.freeBus(busnum)`
        """
        busnum = numpyx.nearestidx(self._audioBusRefsTable, 0)
        if self._audioBusRefsTable[busnum] != 0:
            raise RuntimeError("Out of buses")
        self._audioBusRefsTable[busnum] = 1
        return busnum

    def freeBus(self, busnum:int) -> None:
        """
        Frees (releases) a previously assigned bus. Buses are reference counted
        along there use in csound also, so a bus is only actually released if
        all events which have access to it have stopped.
        (see :meth:`~Engine.assignBus` for more information)
        """
        if self._audioBusRefsTable[busnum] == 0:
            logger.warning(f"Bus {busnum} already released")
        else:
            self._audioBusRefsTable[busnum] -= 1


@_atexit.register
def _cleanup() -> None:
    engines = list(Engine.activeEngines.values())
    for engine in engines:
        engine.stop()


def activeEngines() -> KeysView[str]:
    """
    Returns a list with the names of the active engines
    """
    return Engine.activeEngines.keys()


def getEngine(name:str) -> Opt[Engine]:
    """
    Get an already created engine.
    """
    return Engine.activeEngines.get(name)


