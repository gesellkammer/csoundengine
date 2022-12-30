"""
Engine class
============

An :class:`Engine` implements a simple interface to run and control a csound process.

.. code::

    from csoundengine import Engine
    # create an engine with default options for the platform
    engine = Engine()
    engine.compile(r'''
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

    # start a synth with indefinite duration. This returns a unique (fractional)
    # instance number
    p1 = engine.sched("synth", args=[67, 0.1, 3000])

    # any parameter with k-rate can be modified while running:
    # change midinote
    engine.setp(p1, 4, 67)

    # modify cutoff
    engine.setp(p1, 6, 1000, delay=4)

    # stop the synth:
    engine.unsched(p1)

See also :class:`~csoundengine.session.Session` for a higher level interface:

.. code::

    from csoundengine import *
    session = Engine().session()
    session.defInstr('mysynth', r'''
        |kmidinote=60, kamp=0.1, kcutoff=3000|
        kfreq = mtof:k(kmidinote)
        asig = vco2:a(kamp, kfreq)
        asig = moogladder2(asig, kcutoff, 0.9)
        aenv = linsegr:a(0, 0.1, 1, 0.1, 0)
        asig *= aenv
        outs asig, asig
    ''')
    # Session sched returns a Synth object
    synth = session.sched('mysynth', kmidinote=67, kcutoff=2000)

    # Change the midinote after 2 seconds
    synth.setp(kmidinote=60, delay=2)


Configuration
-------------

Defaults for :class:`Engine` / :class:`~csoundengine.session.Session` can be
customized via::

    from csoundengine import *
    config.edit()

.. hint::

    For more information, see :ref:`Configuration<configuration>`

Interactive Use
---------------

**csoundengine** is optimized to be used interactively and particularly
within `Jupyter <https://jupyter.org/>`_. See :ref:`Csoundengine inside Jupyter<jupyternotebook>`

IPython Magic
~~~~~~~~~~~~~

**csoundengine** also defines a set of ipython/jupyter :doc:`magics <magics>`


.. figure:: assets/eventui.png


"""
from __future__ import annotations

import dataclasses
import tempfile
import sys as _sys

import ctypes as _ctypes
import atexit as _atexit
import queue as _queue
import fnmatch as _fnmatch
import math
import re as _re
import textwrap

import time

import emlib.textlib
import numpy as np
import pitchtools as pt

from emlib import iterlib, net
from emlib.containers import IntPool

from .config import config, logger
from . import csoundlib
from . import jacktools as jacktools
from . import internalTools
from . import engineorc
from . import state as _state
from . import termui as _termui
from .engineorc import CONSTS
from .errors import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *
    from . import session as _session
    import socket
    callback_t = Callable[[str, float], None]
elif 'sphinx' in _sys.modules:
    from typing import *

try:
    import ctcsound
    _MYFLTPTR = _ctypes.POINTER(ctcsound.MYFLT)
except (OSError, ImportError) as e:
    if 'sphinx' in _sys.modules:
        print("Called while building sphinx documentation?")
        print("Using mocked ctcsound, this should only happen when building"
              "the sphinx documentation")
        try:
            from sphinx.ext.autodoc.mock import _MockObject
            ctcsound = _MockObject()
        except ImportError:
            pass
    else:
        raise e


__all__ = [
    'Engine',
    'getEngine',
    'activeEngines',
    'config',
    'logger',
    'csoundlib'
]

_UNSET = float("-inf")


def _generateUniqueEngineName(prefix="engine") -> str:
    i = 0
    while True:
        name = f"{prefix}{i}"
        if name not in Engine.activeEngines:
            return name
        i += 1


def _asEngine(e: Union[str, Engine]) -> Engine:
    if isinstance(e, Engine):
        return e
    out = getEngine(e)
    if out is None:
        raise ValueError(f"No engine found with name {e}")
    return out


@dataclasses.dataclass
class TableInfo:
    """
    Information about a csound table
    """
    sr: float
    size: int
    numChannels: int = 1
    numFrames: int = -1
    path: str = ''
    hasGuard: bool = None

    def __post_init__(self):
        if self.hasGuard is None:
            self.hasGuard = self.size == self.numFrames * self.numChannels + 1
        if self.numFrames == -1:
            self.numFrames = self.size // self.numChannels


def _getSoundfileInfo(path) -> TableInfo:
    import sndfileio
    sndinfo = sndfileio.sndinfo(path)
    return TableInfo(sr=sndinfo.samplerate,
                     numChannels=sndinfo.channels,
                     numFrames=sndinfo.nframes,
                     size=sndinfo.channels*sndinfo.nframes,
                     path=path)


class _RefTimeContext:
    def __init__(self, engine: Engine):
        self.engine = engine

    def __enter__(self):
        self.engine.pushLock()

    def __exit__(self, *args, **kws):
        self.engine.popLock()


class Engine:
    """
    An :class:`Engine` implements a simple interface to run and control a csound
    process.



    Args:
        name: the name of the engine
        sr: sample rate
        ksmps: samples per k-cycle
        backend: passed to -+rtaudio (**"?" to select interactively**)
        outdev: the audio output device, passed to -o (**"?" to select interactively**)
        indev: the audio input device, passed to -i (**"?" to select interactively**)
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
        numControlBuses: number of control buses
        quiet: if True, suppress output of csound (-m 0)
        udpserver: if True, start a udp server for communication (see udpport)
        udpport: the udpport to use for real-time messages. 0=autoassign port
        commandlineOptions: extraOptions command line options passed verbatim to the
            csound process when started
        midiin: if given, use this device as midi input. Can be '?' to select
            from a list, or 'all' to use all devices. None indicates no midi input
        latency: an extra latency added when scheduling events to ensure synchronicity

    .. note::
        Any option with a default value of None has a corresponding slot in the
        config. Default values can be configured via `config.edit()`, see
        `Configuration <https://csoundengine.readthedocs.io/en/latest/config.html>`_


    Example
    ~~~~~~~

    .. code::

        from csoundengine import *
        # create an engine with default options for the platform
        engine = Engine()
        engine.compile(r'''
          instr synth
            kmidinote = p4
            kamp = p5
            kcutoff = p6
            asig = vco2:a(kamp, mtof:k(kmidinote))
            asig = moogladder2(asig, kcutoff, 0.9)
            asig *= linsegr:a(0, 0.1, 1, 0.1, 0)
            outs asig, asig
          endin
        ''')

        # start a synth with indefinite duration. This returns a unique (fractional)
        # instance number
        p1 = engine.sched("synth", args=[67, 0.1, 3000])

        # any parameter with k-rate can be modified while running
        engine.setp(p1, 4, 60)

        # modify cutoff
        engine.setp(p1, 6, 1000, delay=4)

        # stop the synth:
        engine.unsched(p1)

    """
    activeEngines: Dict[str, Engine] = {}
    "Active engines mapped by name (class variable)"

    _builtinTables = engineorc.BUILTIN_TABLES
    _channelMode = {'r': ctcsound.CSOUND_INPUT_CHANNEL,
                    'w': ctcsound.CSOUND_INPUT_CHANNEL,
                    'rw': ctcsound.CSOUND_INPUT_CHANNEL | ctcsound.CSOUND_OUTPUT_CHANNEL}

    def __init__(self,
                 name:str = None,
                 sr:int = None,
                 ksmps:int = None,
                 backend:str = 'default',
                 outdev:str=None,
                 indev:str=None,
                 a4:int = None,
                 nchnls:int = None,
                 nchnls_i:int=None,
                 realtime=False,
                 buffersize:int=None,
                 numbuffers:int=None,
                 globalcode:str = "",
                 numAudioBuses:int=None,
                 numControlBuses:int=None,
                 quiet:bool=None,
                 udpserver:bool=None,
                 udpport:int=0,
                 commandlineOptions:List[str]=None,
                 includes:List[str]=None,
                 midibackend:str = 'default',
                 midiin:str = None,
                 latency:float=None):
        if name is None:
            name = _generateUniqueEngineName()
        elif name in Engine.activeEngines:
            raise KeyError(f"engine {name} already exists")
        if backend == 'portaudio':
            backend = 'pa_cb'
        cfg = config
        availableBackends = csoundlib.getAudioBackendNames(available=True)
        if backend is None or backend == 'default':
            backend = cfg[f'{internalTools.platform}_backend']
        elif backend == '?':
            backend = internalTools.selectItem(availableBackends, title="Select Backend")
        elif backend not in availableBackends:
            knownBackends = csoundlib.getAudioBackendNames(available=False)
            if backend not in knownBackends:
                logger.error(f"Backend {backend} unknown. Available backends: "
                             f"{availableBackends}")
            else:
                logger.error(f"Backend {backend} not available. Available backends: "
                             f"{availableBackends}")


        cascadingBackends = [b.strip() for b in backend.split(",")]
        resolvedBackend = internalTools.resolveOption(cascadingBackends,
                                                      availableBackends)
        backendDef = csoundlib.getAudioBackend(resolvedBackend)
        if not resolvedBackend:
            logger.error(f'Could not find any available backends for {backend}')
            logger.error(f'    Available backends: {", ".join(availableBackends)}')
            logger.error('To configure the default backends, do:\n'
                         '    from csoundengine import config\n'
                         '    config.edit()\n'
                         f'And edit the "{internalTools.platform}.backend" key.')
            raise CsoundError(f'Backend "{resolvedBackend}" not available')

        indevs, outdevs = backendDef.audioDevices()
        defaultin, defaultout = backendDef.defaultAudioDevices()
        indevName, outdevName = "adc", "dac"
        if outdev is None:
            if not defaultout:
                raise RuntimeError(f"No output devices for backend {backendDef.name}")
            outdev, outdevName = defaultout.id, defaultout.name
            if not nchnls:
                nchnls = defaultout.numchannels
        elif outdev == '?':
            if len(outdevs) == 0:
                raise RuntimeError("No output audio devices")
            selected = internalTools.selectAudioDevice(outdevs, title="Select output device")
            if not selected:
                RuntimeError("No output audio device selected")
            outdev, outdevName = selected.id, selected.name
            if not nchnls:
                nchnls = selected.numchannels
        elif isinstance(outdev, int) or _re.search(r"\bdac[0-9]+\b", outdev):
            # dac1, dac8
            if resolvedBackend == 'jack':
                logger.waning(
                    "This way of setting the audio device is discouraged with jack"
                    ". Use a regex to select a specific client or None to connect"
                    "to the default client")
            if isinstance(outdev, int):
                outdev = f"dac{outdev}"
        else:
            if resolvedBackend == 'jack':
                outdevName = outdev
            else:
                selected = next((d for d in outdevs if _fnmatch.fnmatch(d.name, outdev)), None)
                if not selected:
                    raise ValueError(f"Output device {outdev} not known. Possible devices: "
                                     f"{outdevs}")
                outdev, outdevName = selected.id, selected.name

        if indev is None:
            indev, indevName = defaultin.id, defaultin.name
            if not nchnls_i:
                nchnls_i = defaultin.numchannels
        elif indev == '?':
            if len(indevs) == 0:
                raise RuntimeError("No input audio devices")
            selected = internalTools.selectAudioDevice(indevs, title="Select input device")
            if not selected:
                RuntimeError("No output audio device selected")
            indev, indevName = selected.id, selected.name
        elif isinstance(indev, int) or _re.search(r"\badc[0-9]+\b", indev):
            if resolvedBackend == 'jack':
                logger.waning(
                        "This way of setting the audio device is discouraged with jack"
                        ". Use a regex to select a specific client or None to connect"
                        "to the default client")
            if isinstance(indev, int):
                indev = f"adc{outdev}"
        else:
            if resolvedBackend == 'jack':
                indevName = indev
            else:
                selected = next((d for d in indevs if _fnmatch.fnmatch(d.name, indev)), None)
                if not selected:
                    raise ValueError(f"Output device {outdev} not known. Possible devices: "
                                     f"{outdevs}")
                indev, indevName = selected.id, selected.name

        if midibackend == 'default':
            midibackend = 'portmidi'

        commandlineOptions = commandlineOptions if commandlineOptions is not None else []
        sr = sr if sr is not None else cfg['sr']
        backendsr = csoundlib.getSamplerateForBackend(resolvedBackend)
        backendDef = csoundlib.getAudioBackend(resolvedBackend)
        if sr != 0 and backendDef.hasSystemSr and sr != backendsr:
            logger.warning(f"sr requested: {sr}, but backend has a fixed sr ({backendsr})"
                           f". Using backend's sr")
            sr = backendsr
        elif sr == 0:
            if backendDef.hasSystemSr:
                sr = backendsr
            else:
                sr = 44100
                logger.error(f"Asked for backend sr, but backend {resolvedBackend}, does not"
                             f"have a fixed sr. Using sr={sr}")

        if a4 is None: a4 = cfg['A4']
        if ksmps is None: ksmps = cfg['ksmps']
        if nchnls_i is None:
            nchnls_i = cfg['nchnls_i']
        if nchnls is None:
            nchnls = cfg['nchnls']
        if nchnls == 0 or nchnls_i == 0:
            inchnls, outchnls = csoundlib.getNchnls(resolvedBackend,
                                                    outpattern=outdev, inpattern=indev)
            nchnls = nchnls or outchnls
            nchnls_i = nchnls_i or inchnls
        assert nchnls > 0
        assert nchnls_i >= 0

        if quiet is None: quiet = cfg['suppress_output']
        if quiet:
            commandlineOptions.append('-m0')
            commandlineOptions.append('-d')
        self.name = name
        "Name of this Engine"

        self.sr = sr
        "Sample rate"

        self.backend = resolvedBackend
        "Name of the backend used (jack, portaudio, etc)"

        self.a4 = a4
        "Reference frequency for A4"

        self.ksmps = ksmps
        "Number of samples per cycle"

        self.outdev = outdev
        "Output device used"

        self.outdevName = outdevName
        "Long name of the output device"

        self.indev = indev
        "Input device"

        self.indevName = indevName
        "Long name of the input device"

        self.nchnls = nchnls
        "Number of output channels"

        self.nchnls_i = nchnls_i
        "Number of input channels"

        self.globalCode = globalcode
        "Global (init) code to execute at the start of the Engine"

        self.started = False
        "Is this engine started?"

        self.extraOptions = commandlineOptions
        "Extra options passed to csound"

        self.includes = includes
        "List of include files"

        self.extraLatency = latency if latency is not None else config['sched_latency']
        "Added latency for better synch"

        self.numAudioBuses = numAudioBuses if numAudioBuses is not None else config['num_audio_buses']
        "Number of audio buses"

        self.numControlBuses = numControlBuses if numControlBuses is not None else config['num_control_buses']
        "Number of control buses"

        self._realtime = realtime

        backendBufferSize, backendNumBuffers = backendDef.bufferSizeAndNum()
        buffersize = (buffersize or
                      backendBufferSize or
                      config['buffersize'] or
                      max(ksmps * 2, 256))
        self.bufferSize = max(ksmps*2, buffersize)
        "Buffer size"

        self.numBuffers = (numbuffers or
                           backendNumBuffers or
                           config['numbuffers'] or
                           internalTools.determineNumbuffers(self.backend or "portaudio",
                                                             buffersize=self.bufferSize))
        "Number of buffers to fill"

        self.midiBackend: Optional[str] = midibackend
        "Midi backend used"

        if midiin == 'all':
            midiindev = csoundlib.MidiDevice(deviceid='all', name='all')
        elif midiin == '?':
            midiindevs, midioutdevs = csoundlib.midiDevices(self.midiBackend)
            midiindevs.append(csoundlib.MidiDevice('all', 'all'))
            midiindev = internalTools.selectMidiDevice(midiindevs)
        elif midiin:
            midiindev = csoundlib.MidiDevice(deviceid=midiin, name='')
        else:
            midiindev = None
        self.midiin: Optional[csoundlib.MidiDevice] = midiindev
        "Midi input device"

        if udpserver is None: udpserver = config['start_udp_server']
        self._uddocket: Optional[socket.socket] = None
        self._sendAddr: Optional[Tuple[str, int]] = None
        self.udpPort = 0
        "UDP port used (0 if no udp port is active)"

        if udpserver:
            self.udpPort = udpport or net.findport()
            self._udpSocket = net.udpsocket()
            self._sendAddr = ("127.0.0.1", self.udpPort)

        self._perfThread: ctcsound.CsoundPerformanceThread
        self.csound: Optional[ctcsound.Csound] = None            # the csound object
        self._fracnumdigits = 4        # number of fractional digits used for unique instances
        self._exited = False           # are we still running?

        # counters to create unique instances for each instrument
        self._instanceCounters: Dict[int, int] = {}

        # Maps instrname/number: code
        self._instrRegistry: Dict[Union[str, int], str] = {}

        # a dict of callbacks, reacting to outvalue opcodes
        self._outvalueCallbacks: Dict[bytes, callback_t] = {}

        # Maps used for strSet / strGet
        self._indexToStr: Dict[int, str] = {}
        self._strToIndex: Dict[str, int] = {}
        self._strLastIndex = 20

        # global code added to this engine
        self._globalCode: Dict[str, str] = {}

        # this will be a numpy array pointing to a csound table of
        # NUMTOKENS size. When an instrument wants to return a value to the
        # host, the host sends a token, the instr sets table[token] = value
        # and calls 'outvale "__sync__", token' to signal that an answer is
        # ready
        # self._responsesTable: Optional[np.ndarray] = None
        self._responsesTable: np.ndarray

        # a table with sub-mix gains which can be used to group
        # synths, samples, etc.
        self._subgainsTable: Optional[np.ndarray] = None

        # tokens start at 1, leave token 0 to signal that no sync is needed
        # tokens are used as indices to _responsesTable, which is an alias of
        # gi__responses
        self._tokens = list(range(1, CONSTS['numtokens']))

        # a pool of reserved table numbers
        reservedTablesStart = CONSTS['reservedTablesStart']
        self._tablePool = IntPool(CONSTS['numReservedTables'], start=reservedTablesStart)

        # a dict of token:callback, used to register callbacks when asking for
        # feedback from csound
        self._responseCallbacks: Dict[int, Callable] = {}

        # a dict mapping tableindex to fractional instr number
        self._assignedTables: Dict[int, float] = {}

        self._tableCache: Dict[int, np.ndarray] = {}
        self._tableInfo: Dict[int, TableInfo] = {}

        self._channelPointers: Dict[str, np.ndarray] = {}

        self._instrNumCache: Dict[str, int] = {}

        self._session: Optional[_session.Session] = None
        self._busTokenCountPtr: Optional[np.ndarray] = None
        self._soundfontPresetCountPtr: Optional[np.ndarray] = None
        self._kbusTable: Optional[np.ndarray] = None
        self._busIndexes: Dict[int, int] = {}
        self._soundfontPresets: Dict[Tuple[str, int, int], int] = {}
        self._soundfontPresetCount = 0
        self._startTime = 0.
        self._lockedElapsedTime = 0.
        # A stack holding locked states
        self._clockStatesStack: list[bool] = []

        self.started = False
        self.commandlineOptions: List[str] = []
        self.builtinInstrs: Dict[str, int] = {}
        self._reservedInstrnums: Set[int] = set()
        self._reservedInstrnumRanges: list[tuple[str, int, int]] = [('builtinorc', CONSTS['reservedInstrsStart'], CONSTS['userInstrsStart']-1)]
        self.start()

    @property
    def blockSize(self) -> int:
        """
        The size of the processing block in samples.

        csound defines two variables to control its communication with the audio
        backend, a hardware buffer (-B option) and a software buffer (-b option).
        With each audio backend these values are interpreted somewhat differently.
        In general it can be said that ksmps must divide the software buffer (-b)
        and the software buffer itself must divide the hardware buffer (-B).
        Common values for these are: ksmps=64, software buffer=256, hardware
        buffer=512
        """
        return self.bufferSize * self.numBuffers

    def __repr__(self):
        outdev = self.outdev
        if _re.search(r'\bdac([0-9]+)\b', outdev):
            outdev += f" ({self.outdevName})"
        indev = self.indev
        if _re.search(r'\badc([0-9]+)\b', indev):
            indev += f" ({self.indevName})"
        parts = [f'name={self.name}, sr={self.sr}, backend={self.backend}, outdev={outdev}'
                 f', nchnls={self.nchnls}']
        if self.nchnls_i > 0:
            parts.append(f'indev={indev}, nchnls_i={self.nchnls_i}')
        parts.append(f'bufferSize={self.bufferSize}')
        return f"Engine({', '.join(parts)})"

    def __del__(self):
        self.stop()

    def _getSyncToken(self) -> int:
        """
        Get a unique token, to pass to csound for a sync response
        """
        assert self._responsesTable is not None
        token = self._tokens.pop()
        self._responsesTable[token] = _UNSET
        return token

    def _waitOnToken(self, token:int, sleepfunc=time.sleep, period=0.001, timeout:float=None
                     ) -> Optional[float]:
        if timeout is None:
            timeout = config['timeout']
        n = timeout // period
        table = self._responsesTable
        assert table is not None
        while n > 0:
            response = table[token]
            if response == _UNSET:
                return response
            n -= 1
            sleepfunc(period)
        return None

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

    def _assignEventId(self, instrnum: Union[int, str]) -> float:
        """
        Assign an eventid (fractional instr number) for this instr

        This is not really a unique instance, there might be conflicts
        with a previously scheduled event. To really generate a unique instance
        we would need to call uniqinstance, which creates a roundtrip to csound

        Args:
            instrnum (int): the instrument number

        """
        if isinstance(instrnum, str):
            instrnum = self.queryNamedInstr(instrnum)
        c = self._instanceCounters.get(instrnum, 0)
        c += 1
        self._instanceCounters[instrnum] = c
        instancenum = (c % int(10 ** self._fracnumdigits - 2)) + 1
        return self._makeEventId(instrnum, instancenum)

    def _makeEventId(self, num:int, instance:int) -> float:
        frac = (instance / (10**self._fracnumdigits)) % 1
        return num + frac
        
    def _startCsound(self) -> None:
        buffersize = self.bufferSize
        optB = buffersize * self.numBuffers
        if self.backend == 'jack':
            if not jacktools.isJackRunning():
                logger.error("Asked to use jack as backend, but jack is not running")
                raise RuntimeError("jack is not running")
            jackinfo = jacktools.getInfo()
            self.sr = jackinfo.samplerate
            minB = jackinfo.blocksize if jackinfo.onPipewire else jackinfo.blocksize*2
            if optB < minB:
                optB = minB
                self.numBuffers = optB // self.bufferSize
                logger.warning(f"Using -b {self.bufferSize}, -B {optB} "
                               f"(numBuffers: {self.numBuffers}, "
                               f"jack's blocksize: {jackinfo.blocksize})")
        options = ["-d",   # suppress all displays
                   f"-+rtaudio={self.backend}",
                   f"-o{self.outdev}", f"-i{self.indev}",
                   f"-b{buffersize}", f"-B{optB}",
                   ]

        if self.midiin is not None:
            if self.midiBackend:
                options.append(f"-+rtmidi={self.midiBackend}")
            options.append(f"-M{self.midiin.deviceid}")

        if self._realtime:
            options.append("--realtime")

        if self.extraOptions:
            options.extend(self.extraOptions)

        if self.backend == 'jack':
            if self.name is not None:
                clientname = self.name.strip().replace(" ", "_")
                options.append(f'-+jack_client=csoundengine.{clientname}')

        if self.udpPort:
            options.append(f"--port={self.udpPort}")

        # options.append(f"--opcode-dir={csoundlib.userPluginsFolder()}")

        cs = ctcsound.Csound()
        if cs.version() < 6160:
            ver = cs.version() / 1000
            raise RuntimeError(f"Csound's version should be >= 6.16, got {ver:.2f}")
        options = list(iterlib.unique(options))
        for opt in options:
            cs.setOption(opt)
        self.commandlineOptions = options
        if self.includes:
            includelines = [f'#include "{include}"' for include in self.includes]
            includestr = "\n".join(includelines)
        else:
            includestr = ""

        orc, instrmap = engineorc.makeOrc(sr=self.sr,
                                          ksmps=self.ksmps,
                                          nchnls=self.nchnls,
                                          nchnls_i=self.nchnls_i,
                                          a4=self.a4,
                                          globalcode=self.globalCode,
                                          includestr=includestr,
                                          numAudioBuses=self.numAudioBuses,
                                          numControlBuses=self.numControlBuses)
        self.builtinInstrs = instrmap
        self._reservedInstrnums = set(instrmap.values())
        logger.debug("--------------------------------------------------------------\n"
                     "  Starting performance thread. \n"
                     f"     Csound Options: {options}")
        logger.debug(orc)
        logger.debug("--------------------------------------------------------------")
        assert '\0' not in orc
        err = cs.compileOrc(orc)
        if err:
            tmporc = tempfile.mktemp(prefix="csoundengine-", suffix=".orc")
            open(tmporc, "w").write(orc)
            logger.error(f"Error compiling base orchestra. A copy of the orchestra"
                         f" has been saved to {tmporc}")

            logger.error(internalTools.addLineNumbers(orc))
            raise CsoundError(f"Error compiling base ochestra, error: {err}")
        logger.info(f"Starting csound with options: {options}")
        cs.start()
        pt = ctcsound.CsoundPerformanceThread(cs.csound())
        pt.play()
        self._startTime = time.time()
        self._orc = orc
        self.csound = cs
        self._perfThread = pt
        if config['set_sigint_handler']:
            internalTools.setSigintHandler()
        # time.sleep(0.05)
        self._setupCallbacks()
        self._subgainsTable = self.csound.table(self._builtinTables['subgains'])
        self._responsesTable = self.csound.table(self._builtinTables['responses'])


        chanptr, err = self.csound.channelPtr("_soundfontPresetCount",
                                              ctcsound.CSOUND_CONTROL_CHANNEL |
                                              ctcsound.CSOUND_INPUT_CHANNEL |
                                              ctcsound.CSOUND_OUTPUT_CHANNEL)
        assert chanptr is not None, f"_soundfontPresetCount channel is not set: {err}"
        self._soundfontPresetCountPtr = chanptr

        if self.hasBusSupport():
            chanptr, error = self.csound.channelPtr("_busTokenCount",
                                                    ctcsound.CSOUND_CONTROL_CHANNEL|
                                                    ctcsound.CSOUND_INPUT_CHANNEL|
                                                    ctcsound.CSOUND_OUTPUT_CHANNEL)
            assert chanptr is not None, f"_busTokenCount channel not set: {error}\n{orc}"
            self._busTokenCountPtr = chanptr
            kbustable = int(self.csound.evalCode("return gi__bustable"))
            self._kbusTable = self.csound.table(kbustable)
        else:
            logger.info("Server started without bus support")
        self.sync()

    def _setupGlobalInstrs(self):
        if self.hasBusSupport():
            self._perfThread.scoreEvent(0, "i", [self.builtinInstrs['clearbuses_post'], 0, -1])

    def stop(self):
        """
        Stop this Engine
        """
        if not hasattr(self, "name"):
            return
        logger.info(f"stopping Engine {self.name}")
        if not self.started or self._exited:
            logger.debug(f"Engine {self.name} was not running, so can't stop it")
            return
        logger.info("... stopping thread")
        self._perfThread.stop()
        time.sleep(0.1)
        logger.info("... stopping csound")
        self.csound.stop()
        time.sleep(0.1)
        logger.info("... cleaning up")
        self.csound.cleanup()
        self._exited = True
        self.csound = None
        self._instanceCounters = {}
        self._instrRegistry = {}
        self.activeEngines.pop(self.name, None)
        self.started = False

    def start(self):
        """
        Start this engine.

        The call to .start() is performed as part of the init process and
        only needs to be called explicitely if the engine was previously
        stopped. If the engine has already been started this method
        does nothing
        """
        if self.started:
            logger.debug(f"start:Engine {self.name} already started")
            return
        if priorengine := self.activeEngines.get(self.name):
            logger.debug("Stopping prior engine")
            priorengine.stop()
        logger.info(f"Starting engine {self.name}")
        self._startCsound()
        self.activeEngines[self.name] = self
        self._setupGlobalInstrs()

        self.started = True
        strsets = ["cos", "linear", "smooth", "smoother"]
        for s in strsets:
            self.strSet(s)
        self.sync()

    def restart(self, wait=1) -> None:
        """ Restart this engine. All defined instrs / tables are removed"""
        self.stop()
        if wait:
            _termui.waitWithAnimation(wait)
        self.start()
        
    def _outcallback(self, _, channelName, valptr, chantypeptr):
        func = self._outvalueCallbacks.get(channelName)
        if not func:
            logger.error(f"outvalue: callback not set for channel {channelName}")
            return
        val = _ctypes.cast(valptr, _MYFLTPTR).contents.value
        func(channelName, val)

    def _setupCallbacks(self) -> None:
        assert self.csound is not None

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

    def registerOutvalueCallback(self, chan:str, func: callback_t) -> None:
        """
        Set a callback to be fired when "outvalue" is used in csound

        Register a function ``func(channelname:str, newvalue:float) -> None``,
        which will be called whenever the given channel is modified via
        the "outvalue" opcode. Multiple functions per channel can be registered

        Args:
            chan: the name of a channel
            func: a function of the form ``func(chan:str, newvalue:float) -> None``

        """
        key = bytes(chan, "ascii")
        previousCallback = self._outvalueCallbacks.get(key)
        if chan.startswith("__"):
            if previousCallback:
                logger.warning("Attempting to set a reserved callback, but one "
                               "is already present. The new one will replace the old one")
            self._outvalueCallbacks[key] = func
        else:
            if previousCallback:
                logger.warning(f"Callback for channel {chan} already set, replacing it")
            self._outvalueCallbacks[key] = func

    def bufferLatency(self) -> float:
        """
        The latency of the communication to the csound process.

        ::

            bufferLatencySeconds = buffersize * numbuffers / sr

        This latency depends on the buffersize and number of buffers.
        """
        return self.bufferSize/self.sr * self.numBuffers

    def controlLatency(self) -> float:
        """
        Time latency between a scheduled action and its response.

        This is normally ``ksmps/sr * 2`` but the actual latency varies if
        the engine is being run in realtime (in that case init-pass is done
        async, which might result in longer latency).
        """
        return self.ksmps/self.sr * 2

    def sync(self, timeout: float = None) -> None:
        """
        Block until csound has processed its immediate events

        Args:
            timeout: a timeout in seconds; None = use default timeout as defined
                in the configuration (TODO: add link to configuration docs)

        Raises TimeoutError if the sync operation takes too long

        Example
        =======

            >>> from csoundengine import *
            >>> e = Engine(...)
            >>> tables = [e.makeEmptyTable(size=1000) for _ in range(10)]
            >>> e.sync()
            >>> # do something with the tables
        """
        assert self._perfThread
        # self._perfThread.flushMessageQueue()
        token = self._getSyncToken()
        pargs = [self.builtinInstrs['pingback'], 0, 0, token]
        self._eventWait(token, pargs, timeout=timeout)

    def _compileCode(self, code: str, block=False) -> None:
        if self.udpPort and config['prefer_udp']:
            logger.debug("Sengind code via udp: ")
            logger.debug(code)
            self._udpSend(code)
        else:
            assert self.csound is not None
            if block:
                logger.debug("Compiling csound code (blocking):")
                logger.debug(code)
                err = self.csound.compileOrc(code)
                if err:
                    logger.error("compileOrc error: ")
                    logger.error(internalTools.addLineNumbers(code))
                    raise CsoundError(f"Could not compile code")
            else:
                logger.debug("Compiling csound code (async):")
                logger.debug(code)
                err = self.csound.compileOrcAsync(code)
                if err:
                    logger.error("compileOrcAsync error: ")
                    logger.error(internalTools.addLineNumbers(code))
                    raise CsoundError(f"Could not compile async")

    def _compileInstr(self, instrname: str|int, code: str, block=False) -> None:
        self._instrRegistry[instrname] = code
        self._compileCode(code, block=block)
        if isinstance(instrname, str):
            # if named instrs are defined we sync in order to avoid assigning
            # the same number to different instrs. This should be taken
            # care by csound by locking but until this is in place, we
            # need to sync
            if not block and instrname not in self._instrNumCache:
                self.sync()
            self._queryNamedInstrs([instrname], timeout=0.1 if block else 0)

    def compile(self, code: str, block=False) -> None:
        """
        Send orchestra code to the running csound instance.

        The code sent can be any orchestra code

        Args:
            code (str): the code to compile
            block (bool): if True, this method will block until the code
                has been compiled
            client: who is compiling this code. Some clients have access to


        Raises :class:`CsoundError` if the compilation failed

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
            >>> e.compile("giMelody[] fillarray 60, 62, 64, 65, 67, 69, 71")
            >>> code = open("myopcodes.udo").read()
            >>> e.compile(code)

        """
        codeblocks = csoundlib.parseOrc(code)
        for block in codeblocks:
            if block.kind == 'instr' and block.name[0].isdigit():
                instrnum = int(block.name)
                for rangename, mininstr, maxinstr in self._reservedInstrnumRanges:
                    if mininstr <= instrnum < maxinstr:
                        logger.error(f"Instrument number {instrnum} is reserved. Code:")
                        logger.error("\n" + textwrap.indent(block.text, "    "))
                        raise ValueError(f"Cannot use instrument number {instrnum}, "
                                         f"the range {mininstr} - {maxinstr} is reserved for {rangename}")

                if instrnum in self._reservedInstrnums:
                    raise ValueError("Cannot compile instrument with number "
                                     f"{instrnum}: this is a reserved instr and "
                                     f"cannot be redefined. Reserved instrs: "
                                     f"{sorted(self._reservedInstrnums)}")

        instrs = [b for b in codeblocks if b.kind == 'instr']

        for instr in instrs:
            body = "\n".join(emlib.textlib.splitAndStripLines(instr.text)[1:-1])
            self._instrRegistry[instr.name] = body

        self._compileCode(code)

        names = [instr.name for instr in instrs
                 if instr.name[0].isalpha()]
        if names:
            # if named instrs are defined we sync in order to avoid assigning
            # the same number to different instrs. This should be taken
            # care by csound by locking but until this is in place, we
            # need to sync
            if not block and any(name not in self._instrNumCache for name in names):
                self.sync()
            namesToRegister = [n for n in names if n not in self._instrNumCache]
            if namesToRegister:
                self._queryNamedInstrs(namesToRegister, timeout=0.1 if block else 0, delay=self.bufferLatency())

    def evalCode(self, code:str) -> float:
        """
        Evaluate code, return the result of the evaluation

        Args:
            code (str): the code to evaluate. Usually an expression returning
                a float value (see example)

        Returns:
            the result of the evaluation

        NB: this operation is synchronous and has a latency of at least
        ``self.bufferLatency()``

        Example
        =======

            >>> e = Engine()
            >>> e.compile(r'''
            ... instr myinstr
            ...   prints "myinstr!"
            ...   turnoff
            ... ''')
            >>> e.compile(r'''
            ... opcode getinstrnum, i, S
            ...   Sinstr xin
            ...   inum nstrnum
            ...   xout inum
            ... endop''')
            >>> e.evalCode('getinstrnum("myinstr")')

        """
        assert self.started
        words = code.split()
        if words[0] != "return":
            code = "return " + code
        return self.csound.evalCode(code)

    def tableWrite(self, tabnum:int, idx:int, value:float, delay:float=0.) -> None:
        """
        Write to a specific index of a table

        Args:
            tabnum (int): the table number
            idx (int): the index to modify
            value (float): the new value
            delay (float): delay time in seconds. Use 0 to write synchronously

        .. seealso::

            :meth:`~Engine.getTableData`
            :meth:`~Engine.fillTable`

        """
        assert self.started
        if delay == 0:
            arr = self.getTableData(tabnum)
            if arr is None:
                raise ValueError(f"table {tabnum} not found")
            arr[idx] = value
        else:
            pargs = [self.builtinInstrs['tabwrite'], delay, 0, tabnum, idx, value]
            assert self._perfThread is not None
            self._perfThread.scoreEvent(0, "i", pargs)

    def getTableData(self, idx:int, flat=False) -> Optional[np.ndarray]:
        """
        Returns a numpy array pointing to the data of the table.

        Any modifications to this array will modify the table itself.

        .. note::

            Multichannel audio is loaded into a csound table as a flat
            array with samples interleaved.

        Args:
            idx (int): the table index
            flat: if True, the data will be returned as a flat (1D) array
                even if the table holds a multi-channel sample.

        Returns:
            a numpy array pointing to the data array of the table, or None
            if the table was not found

        """
        assert self.csound is not None
        arr = self._tableCache.get(idx)
        if arr is None:
            arr: np.ndarray = self.csound.table(idx)
            if not flat:
                tabinfo = self.tableInfo(idx)
                if tabinfo.numChannels > 1:
                    if tabinfo.size == tabinfo.numFrames*tabinfo.numChannels+1:
                        arr = arr[:-1]
                    arr.shape = (tabinfo.numFrames, tabinfo.numChannels)
            self._tableCache[idx] = arr
        return arr

    def elapsedTime(self) -> float:
        """
        Returns the elapsed time since start of the engine

        This time is used as a reference when scheduling events. Since scheduling
        itself takes a small but not negligible amount of time, when scheduling
        a great number of events, these will fall out of sync. For this reason
        the elapsed time can be used as a reference to schedule events in
        absolute time

        Example
        ~~~~~~~

            >>> from csoundengine import Engine
            >>> import numpy as np
            >>> e = Engine()
            >>> e.compile(r'''
            ... instr 100
            ...   ifreq = p4
            ...   outch 1, oscili:a(0.1, ifreq) * linseg:a(0, 0.01, 1, 0.1, 0)
            ... endin
            ... ''')
            >>> now = e.elapsedTime()
            # Schedule 5 events per second for 60 seconds. Without a time
            # reference the events would fall out of sync
            >>> for t in np.arange(0, 60, 0.2):
            ...     e.sched(100, t+now, 0.15, args=[1000], relative=False)
            ...     e.sched(100, t+now, 0.15, args=[800], relative=False)

        The same result can be achieved by locking the elapsed-time clock::

            >>> with e.lockedClock():
            ...     for t in np.arange(0, 10, 0.2):
            ...         e.sched(100, t, 0.15, args=[1000])
            ...         e.sched(100, t, 0.15, args=[800])

        """
        return self._lockedElapsedTime or self.csound.currentTimeSamples() / self.sr

    def lockClock(self, lock=True):
        """
        Lock the elapsed time clock

        This ensures that events scheduled while the clock is locked will run
        in sync. For this to work all events scheduled must have some latency (they
        must run in the future)

        Example
        ~~~~~~~

            >>> from csoundengine import Engine
            >>> import numpy as np
            >>> e = Engine()
            >>> e.compile(r'''
            ... instr 100
            ...   ifreq = p4
            ...   outch 1, oscili:a(0.1, ifreq) * linseg:a(0, 0.01, 1, 0.1, 0)
            ... endin
            ... ''')
            >>> e.lockClock()
            >>> for t in np.arange(0, 10, 0.2):
            ...     e.sched(100, t, 0.15, args=[1000])
            ...     e.sched(100, t, 0.15, args=[800])
            >>> e.lockClock(False)

        See Also
        ~~~~~~~~

        :meth:`Engine.lockedElapsedTime`, :meth:`Engine.elapsedTime`

        """
        if lock:
            if self.isClockLocked():
                logger.debug("The elapsed time clock is already locked")
            else:
                self._lockedElapsedTime = self.csound.currentTimeSamples()/self.sr
        else:
            if not self._lockedElapsedTime:
                logger.info("Asked to unlock the elapsed time clock, but it was not locked")
            self._lockedElapsedTime = 0

    def isClockLocked(self) -> bool:
        """Returns True if the clock is locked"""
        return self._lockedElapsedTime > 0

    def pushLock(self):
        """
        Lock the clock of this engine

        Allows for recursive locking, so users do not need to
        see if what the current state of the lock is

        .. seealso:: :meth:`Engine.popLock`
        """
        islocked = self.isClockLocked()
        self._clockStatesStack.append(islocked)
        if not islocked:
            self.lockClock(True)

    def popLock(self):
        """
        Reverts the action of pushLock, unlocking the clock

        .. seealso:: :meth:`Engine.pushLock`
        """
        waslocked = self._clockStatesStack.pop()
        if not waslocked:
            self.lockClock(False)

    def __enter__(self):
        self.pushLock()

    def __exit__(self, *args, **kws):
        self.popLock()

    def lockedClock(self) -> _RefTimeContext:
        """
        Context manager, locks and unlocks the reference time

        By locking the reference time it is possible to ensure that
        events which are supposed to be in sync are scheduled correctly
        into the future.

        .. note::

            A shortcut for this is to just use the engine as context manager::

                with engine:
                    engine.sched(...)
                    engine.sched(...)
                    engine.session().sched(...)
                    ...

        Example
        ~~~~~~~

            >>> from csoundengine import Engine
            >>> import numpy as np
            >>> e = Engine()
            >>> e.compile(r'''
            ... instr 100
            ...   ifreq = p4
            ...   outch 1, oscili:a(0.1, ifreq) * linseg:a(0, 0.01, 1, 0.1, 0)
            ... endin
            ... ''')
            >>> with e.lockedClock():
            ...     for t in np.arange(0, 10, 0.2):
            ...         e.sched(100, t, 0.15, args=[1000])
            ...         e.sched(100, t, 0.15, args=[800])

        """
        return _RefTimeContext(self)

    def sched(self, instr: int|float|str, delay=0., dur=-1.,
              args: np.ndarray|Sequence[float|str] = None,
              relative=True
              ) -> float:
        """
        Schedule an instrument

        Args:
            instr : the instrument number/name. If it is a fractional number,
                that value will be used as the instance number. An integer or a string
                will result in a unique instance assigned by csound. Named instruments
                with a fractional number can also be scheduled (for example,
                for an instrument named "myinstr" you canuse "myinstr.001")
            delay: time to wait before instrument is started. If relative is False,
                this represents the time since start of the engine (see examples)
            dur: duration of the event
            args: any other args expected by the instrument, starting with p4
                (as a list of floats/strings, or a numpy array). Any
                string arguments will be converted to a string index via strSet. These
                can be retrieved in csound via strget
            relative: if True, delay is relative to the scheduling time,
                otherwise it is relative to the start time of the engine.
                To get an absolute time since start of the engine, call
                `engine.elapsedTime()`

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
                kfreq = p4
                kcutoff = p5
                Smode strget p6
                asig vco2 0.1, kfreq
                if strcmp(Smode, "lowpass") == 0 then
                  asig moogladder2 asig, kcutoff, 0.95
                else
                  asig K35_hpf asig, kcutoff, 9.0
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
                e.sched(100, t+now, 0.2, relative=False)

        See Also
        ~~~~~~~~

        :meth:`~csoundengine.engine.Engine.unschedAll`
        """
        assert self.started
        if relative:
            t0 = self.elapsedTime()
            delay = t0 + delay + self.extraLatency
            relative = False

        if isinstance(instr, float):
            instrfrac = instr
        elif isinstance(instr, int):
            instrfrac = self._assignEventId(instr)
        elif isinstance(instr, str):
            if "." in instr:
                name, fractionstr = instr.split(".")
                instrnum = self._instrNumCache.get(name)
                if instrnum:
                    instrfrac = instrnum+float("."+fractionstr)
                else:
                    msg = f'i {instr} {delay} {dur}'
                    if args:
                        msg += ' '.join(map(str, args))
                    self._perfThread.inputMessage(msg)
                    return instr
            else:
                instrfrac = self._assignEventId(instr)
        else:
            raise TypeError(f"Expected a float, an int or a str as instr, "
                            f"got {instr} (type {type(instr)})")
        if not args:
            pargs = [instrfrac, delay, dur]
            self._perfThread.scoreEvent(0 if relative else 1, "i", pargs)
        elif isinstance(args, np.ndarray):
            pargsnp = np.empty((len(args)+3,), dtype=float)
            pargsnp[0] = instrfrac
            pargsnp[1] = delay
            pargsnp[2] = dur
            pargsnp[3:] = args
            # 1: we use always absolute time
            self._perfThread.scoreEvent(1, "i", pargsnp)
        else:
            needsSync = any(isinstance(a, str) and not a in self._strToIndex for a in args)
            pargs = [instrfrac, delay, dur]
            pargs.extend(float(a) if not isinstance(a, str) else self.strSet(a) for a in args)
            if needsSync:
                self.sync()
            self._perfThread.scoreEvent(1, "i", pargs)
        return instrfrac

    def _queryNamedInstrs(self, names:List[str], timeout=0.1, callback=None, delay=0.) -> None:
        """
        Query assigned instr numbers

        This operation is async if timeout is 0 or a callback is given.
        Otherwise it blocks for at most `timeout` time. If the operation
        timesout an exception is raised. The results are placed in the internal
        cache

        Args:
            names: the names to query
            timeout: if 0, the operation is async. Otherwise active polling is done
                for this amount of time
            callback: a func of the form `func(name2instr:Dict[str, int])` which will be
                called when all instrs have an assigned instr number
                called for
        """
        if not timeout or callback:
            # query async
            if not callback:
                for name in names:
                    self._queryNamedInstrAsync(name, delay=delay)
            else:
                results = {}
                def mycallback(name, instrnum, n=len(names), results=results, callback=callback):
                    results[name] = instrnum
                    if len(results) == n:
                        callback(results)
                for name in names:
                    self._queryNamedInstrAsync(name, delay=delay, callback=mycallback)
            return

        tokens = [self._getSyncToken() for _ in range(len(names))]
        instr = self.builtinInstrs['nstrnum']
        for name, token in zip(names, tokens):
            msg = f'i {instr} {delay} 0 {token} "{name}"'
            self._perfThread.inputMessage(msg)
        self.sync()
        token2name = dict(zip(tokens, names))
        # Active polling
        polltime = min(0.005, timeout*0.5)
        numtries = int(timeout / polltime)
        if delay:
            time.sleep(delay)
        for _ in range(numtries):
            if not token2name:
                break
            pairs = list(token2name.items())
            for token, name in pairs:
                instrnum = self._responsesTable[token]
                if not math.isinf(instrnum):
                    instrnum = int(self._responsesTable[token])
                    self._instrNumCache[name] = instrnum
                    if (body := self._instrRegistry.get(name)) is not None:
                        self._instrRegistry[instrnum] = body
                    del token2name[token]
            time.sleep(polltime)
        else:
            raise TimeoutError(f"operation timed out ater {timeout} secs")

    def _queryNamedInstrAsync(self, name:str, delay=0., callback=None) -> None:
        """
        Query the assigned instr number async

        The result is put in the cache and, if given, callback is called
        as `callback(name:str, instrnum:int)`
        """
        synctoken = self._getSyncToken()
        msg = f'i {self.builtinInstrs["nstrnumsync"]} {delay} 0 {synctoken} "{name}"'

        def _callback(synctoken, instrname=name, func=callback):
            instrnum = int(self._responsesTable[synctoken])
            self._instrNumCache[instrname] = instrnum
            body = self._instrRegistry.get(instrname)
            if body:
                self._instrRegistry[instrnum] = body
            if func:
                func(name, instrnum)

        self._inputMessageWithCallback(synctoken, msg, _callback)

    def queryNamedInstr(self, instrname: str, cached=True, callback=None) -> int:
        """ Find the instrument number corresponding to instrument name

        Args:
            instrname: the name of the instrument
            cached: if True, results are cached
            callback: if given, the operation is async and the callback will
                be called when the result is available. Callback must a of
                the form ``func(instrname:str, instrnum:int) -> None``

        Returns:
            the instr number if called without callback, 0 otherwise. If the instrument was
            not found (either because it was never compiled or the compilation is not ready yet)
            -1 will be returned
        """
        if cached and (instrnum := self._instrNumCache.get(instrname)) > 0:
            if callback:
                callback(instrname, instrnum)
            return instrnum
        if callback:
            self._queryNamedInstrAsync(instrname, delay=0, callback=callback)
            return 0
        token = self._getSyncToken()
        msg = f'i {self.builtinInstrs["nstrnumsync"]} 0 0 {token} "{instrname}"'
        out = self._inputMessageWait(token, msg)
        assert out is not None
        out = int(out)
        if out > 0:
            self._instrNumCache[instrname] = out
        return out

    def unsched(self, p1:Union[float, str], delay:float = 0) -> None:
        """
        Stop a playing event

        If p1 is a round number, all events with the given number
        are unscheduled. Otherwise only an exact matching event
        is unscheduled, if it exists

        Args:
            p1: the instrument number/name to stop
            delay: if 0, remove the instance as soon as possible

        Example
        =======

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

        See Also
        ~~~~~~~~

        :meth:`~Engine.unschedAll`

        """
        if isinstance(p1, str):
            p1 = self.queryNamedInstr(p1)
        if int(p1) == p1:
            mode = 0   # all instances
        else:
            mode = 0   # exact matching
        pfields = [self.builtinInstrs['turnoff'], delay, 0, p1, mode]
        self._perfThread.scoreEvent(0, "i", pfields)

    def unschedFuture(self, p1:Union[float, str]) -> None:
        """
        Stop a future event

        Args:
            p1: the instrument number/name to stop

        See Also
        ~~~~~~~~

        :meth:`~csoundengine.engine.Engine.unschedAll`,
        :meth:`~csoundengine.engine.Engine.unsched`

        """
        if isinstance(p1, str):
            p1 = self.queryNamedInstr(p1)
        dur = self.ksmps/self.sr * 2
        pfields = [self.builtinInstrs['turnoff_future'], 0, dur, p1]
        self._perfThread.scoreEvent(0, "i", pfields)

    def unschedAll(self) -> None:
        """
        Remove all playing and future events

        See Also
        ~~~~~~~~

        :meth:`~csoundengine.engine.Engine.unsched`,
        :meth:`~csoundengine.engine.Engine.unschedFuture`
        """
        assert self.csound is not None
        self.csound.rewindScore()
        self._setupGlobalInstrs()

    def session(self):
        """
        Return the Session corresponding to this Engine

        Returns:
            the corresponding Session

        Example
        =======

        >>> from csoundengine import *
        >>> session = Engine().session()
        >>> session.defInstr("synth", r'''
        ... kamp  = p5    ; notice that p4 is reserved
        ... kmidi = p6
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
        if self._session is None:
            from .session import Session
            self._session = session = Session(self.name)
        return self._session

    def reserveInstrRange(self, name: str, mininstrnum: int, maxinstrnum: int) -> None:
        """
        Declares the instrument numbers in the given range as reserved

        Instrument numbers within this range will not be allocated when using
        named instruments.

        Args:
            name: the name of the reserved block
            mininstrnum: lowest instrument number to reserve
            maxinstrnum: highest instrument number to reserve (not included in the range)

        """
        self._reservedInstrnumRanges.append((name, mininstrnum, maxinstrnum))

    def makeEmptyTable(self, size, numchannels=1, sr=0, instrnum=-1) -> int:
        """
        Create an empty table, returns the index of the created table

        Example
        =======

        >>> from csoundengine import *
        >>> e = Engine()
        >>> source = e.makeEmptyTable(128)
        >>> e.compile(r'''
        ... instr 100
        ...   imidi = p4
        ...   iamptab = p5
        ...   islot = p6
        ...   kamp table islot, iamptab
        ...   asig = oscili:a(interp(kamp), mtof(imidi))
        ...   outch 1, asig
        ... endin
        ... ''')
        >>> tabarray = e.getTableData(source)
        >>> tabarray[0] = 0.5
        >>> eventid = e.sched(100, args=[67, source, 0])
        >>> # fade out
        >>> e.automateTable(source=source, idx=0, pairs=[1, 0.5, 5, 0.])

        .. seealso::

            :meth:`~Engine.makeTable`
            :meth:`~Engine.fillTable`
            :meth:`~Engine.automateTable`

        """
        tabnum = self._assignTableNumber(p1=instrnum)
        token = 0
        empty = 1
        pargs = [self.builtinInstrs['maketable'], 0, 0., token, tabnum, size, empty, sr, numchannels]
        # pargs = [self.builtinInstrs['ftnewmeta'], 0, 0., tabnum, size, numchannels, sr]
        # pargs = [tabnum, 0, size, -2, 0]
        # self._perfThread.scoreEvent(0, "f", pargs)
        # self._perfThread.flushMessageQueue()
        # self.setTableMetadata(tabnum, numchannels=numchannels, sr=sr)
        self._perfThread.scoreEvent(0, "i", pargs)
        self._tableInfo[tabnum] = TableInfo(sr=sr, size=size, numChannels=numchannels)
        return tabnum

    def makeTable(self,
                  data: list[float] | np.ndarray,
                  tabnum: int = 0,
                  sr: int = 0,
                  block=True,
                  callback=None,
                  _instrnum=-1.
                  ) -> int:
        """
        Create a new table and fill it with data.

        Args:
            data: the data used to fill the table
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

        .. code-block:: python

            from csoundengine import *
            e = Engine()
            import sndfileio
            sample, sr = sndfileio.sndread("stereo.wav")
            # modify the sample in python
            sample *= 0.5
            source = e.makeTable(sample, sr=sr, block=True)
            e.playSample(source)

        See Also
        ~~~~~~~~

        :meth:`~csoundengine.engine.Engine.readSoundfile`
        :meth:`~csoundengine.engine.Engine.fillTable`


        """
        if tabnum == -1:
            tabnum = self._assignTableNumber(p1=_instrnum)
        elif tabnum == 0 and not callback:
            block = True
        if block or callback or len(data) >= 1900:
            if not block and not callback:
                # User didn't ask for blocking operation, but there is no way
                # to do this totally async, since we need to first create the
                # table in order to fill it with data.
                logger.info(f"Creating table {tabnum}. This operation will block"
                            f" anyway, since the data is too big to be sent inline")
            assignedTabnum = self._makeTableNotify(data=data, sr=sr, tabnum=tabnum,
                                                   callback=callback)
            assert assignedTabnum > 0
            # Remove cached table data array, if any
            self._tableCache.pop(assignedTabnum, None)
            return assignedTabnum

        # Create a table asynchronously
        assert tabnum > 0
        self._tableCache.pop(int(tabnum), None)

        # data can be passed as p-args directly (non-blocking)
        size = len(data)
        numchannels = 1 if len(data.shape) == 1 else data.shape[1]
        arr = np.zeros((len(data)+4,), dtype=float)
        arr[0:4] = [tabnum, 0, size, -2]
        arr[4:] = data
        self._perfThread.scoreEvent(0, "f", arr)
        self._tableInfo[tabnum] = TableInfo(sr=sr, size=size, numChannels=numchannels)
        return int(tabnum)

    def tableExists(self, tabnum: int) -> bool:
        """
        Returns True if a table with the given number exists
        """

        try:
            tabinfo = self.tableInfo(tabnum)
        except TableNotFoundError:
            return False
        return True

    def setTableMetadata(self, tabnum:int, sr:int, numchannels:int = 1,
                         check=True) -> None:
        """
        Set metadata for a table holding sound samples.

        When csound reads a soundfile into a table, it stores some additional data,
        like samplerate and number of channels. A table created by other means and
        then filled with samples normally does not have this information. In most
        of the times this is ok, but there are some opcodes which need this information
        (loscil, for example). This method allows to set this information for such
        tables.

        Args:
            tabnum: the table number
            sr: the sample rate
            numchannels: number of channels of data
            check: if True, it will check if the table exists in the case where
                the table wass not created via the engine
        """
        logger.info(f"Setting table metadata. {tabnum=}, {sr=}, {numchannels=}")
        pargs = [self.builtinInstrs['ftsetparams'], 0, 0., tabnum, sr, numchannels]
        tabinfo = self._tableInfo.get(tabnum)
        if tabinfo:
            tabinfo.sr = sr
            tabinfo.numChannels = numchannels
        else:
            if check and not self.tableExists(tabnum):
                raise ValueError(f"Table {tabnum} does not exist")
            else:
                logger.debug(f"setTableMetadata: table {tabnum} was not created by this Engine")
        self._perfThread.scoreEvent(0, "i", pargs)

    def _registerSync(self, token:int) -> _queue.Queue:
        table = self._responsesTable
        q = _queue.Queue()
        self._responseCallbacks[token] = lambda token, q=q, t=table: q.put(t[token])
        return q

    def schedCallback(self, delay:float, callback:Callable) -> None:
        """
        Call callback after delay, triggered by csound scheduler

        Args:
            delay (float): the delay time, in seconds
            callback (callable): the callback, a function of the sort () -> None

        The callback will be called after the given delay, plus some jitter
        (~ 2/3 k-cycles after, never before)

        Example
        =======

        >>> from csoundengine import *
        >>> import time
        >>> e = Engine()
        >>> startTime = time.time()
        >>> e.schedCallback(2, lambda:print(f"Elapsed time: {time.time() - startTime}"))
        """
        token = self._getSyncToken()
        pargs = [self.builtinInstrs['pingback'], delay, 0.01, token]
        self._eventWithCallback(token, pargs, lambda token: callback())

    def _eventWait(self, token:int, pargs:Sequence[float], timeout: float = None
                   ) -> Optional[float]:
        if timeout is None:
            timeout = config['timeout']
        assert timeout > 0
        q = self._registerSync(token)
        self._perfThread.scoreEvent(0, "i", pargs)
        try:
            outvalue = q.get(block=True, timeout=timeout)
            return outvalue if outvalue != _UNSET else None
        except _queue.Empty:
            raise TimeoutError(f"{token=}, {pargs=}")

    def plotTableSpectrogram(self, tabnum: int, fftsize=2048, mindb=-90,
                             maxfreq:int=None, overlap:int=4, minfreq:int=0,
                             sr:int=44100, chan=0
                             ) -> None:
        """
        Plot a spectrogram of the audio data in the given table

        Requires that the samplerate is set, either because it was read via
        gen01 (or using .readSoundfile), or it was manually set via setTableMetadata

        Args:
            source: the table to plot
            fftsize (int): the size of the fft
            mindb (int): the min. dB to plot
            maxfreq (int): the max. frequency to plot
            overlap (int): the number of overlaps per window
            minfreq (int): the min. frequency to plot
            sr: the fallback samplerate, used when the table has no samplerate
                information of its own
            chan: which channel to plot if the table is multichannel

        Example
        -------

            >>> from csoundengine import *
            >>> e = Engine()
            >>> source = e.readSoundfile("mono.wav", block=True)
            >>> e.plotTableSpectrogram(source)

        .. image:: assets/tableproxy-plotspectrogram.png

        """
        from . import plotting
        data = self.getTableData(tabnum)
        if internalTools.arrayNumChannels(data) > 1:
            data = data[:, chan]
        tabinfo = self.tableInfo(tabnum)
        if tabinfo.sr > 0:
            sr = tabinfo.sr
        plotting.plotSpectrogram(data, sr, fftsize=fftsize, mindb=mindb,
                                 maxfreq=maxfreq, minfreq=minfreq, overlap=overlap,
                                 show=True)

    def plotTable(self, tabnum: int, sr: int=0) -> None:
        """
        Plot the content of the table via matplotlib.pyplot

        If the sr is known the table is plotted as a waveform, with time as the x-coord.
        Otherwise the table's raw data is plotted, with the index as x the x-coord.
        The samplerate will be known if the table was created via
        :meth:`Engine.readSoundfile` or read via GEN1. The sr can also be passed explicitely
        as a parameter.

        Args:
            tabnum: the table to plot
            sr: the samplerate of the data. Needed to plot as a waveform if the table was
                not loaded via GEN1 (or via :meth:`Engine.readSoundfile`).

        .. code::

            from csoundengine import *
            e = Engine()
            source = e.readSoundfile("mono.wav", block=True)
            # no sr needed here since the output was rea via readSoundfile
            e.plotTable(source)

        .. figure:: assets/tableproxy-plot.png

        .. code::

            import sndfileio
            data, sr = sndfileio.sndread("stereo.wav")
            tabnum2 = e.makeTable(data, sr=sr)
            e.plotTable(tabnum2)

        .. figure:: assets/tableplot-stereo.png

        .. code::

            e = Engine()
            xs = np.linspace(0, 6.28, 1000)
            ys = np.sin(xs)
            source = e.makeEmptyTable(len(ys))
            e.fillTable(source, data=ys)
            e.plotTable(source)

        .. image:: assets/tableplot-sine.png

        See Also
        ~~~~~~~~

        :meth:`~csoundengine.engine.Engine.readSoundfile`
        :meth:`~csoundengine.engine.Engine.makeTable`
        """
        from csoundengine import plotting
        assert isinstance(tabnum, int) and tabnum > 0
        data = self.getTableData(tabnum, flat=False)
        tabinfo = self.tableInfo(tabnum)

        if not sr and tabinfo.sr > 0:
            sr = tabinfo.sr

        if data is None:
            raise ValueError(f"Table {tabnum} is invalid")
        if sr:
            plotting.plotSamples(data, samplerate=sr, show=True)
        else:
            plotting.plt.plot(data)
            if not plotting.matplotlibIsInline():
                plotting.plt.show()

    def schedSync(self, instr:Union[int, float, str], delay:float = 0, dur:float = -1,
                  args:Union[np.ndarray, Sequence[Union[float, str]]] = None,
                  timeout=-1):
        """
        Schedule an instr, wait for a sync message

        Similar to :meth:`~Engine.sched` but waits until the instrument sends a
        sync message.

        .. note::

            In this case, args should start with p5 since the sync token
            is sent as p4

        Args:
            instr: the instrument number/name. If it is a fractional number,
                that value will be used as the instance number.
            delay: time to wait before instrument is started
            dur: duration of the event
            args: any other args expected by the instrument, starting with p4
                (as a list of floats/strings, or as a numpy float array). Any
                string arguments will be converted to a string index via strSet. These
                can be retrieved via strget in the csound instrument
            timeout: if a non-negative number is given, this function will block
                at most this time and then raise a TimeoutError

        Returns:
            a fractional p1

        Example
        =======

        TODO
        """
        assert self.started
        instrfrac = instr if isinstance(instr, float) else self._assignEventId(instr)
        token = self._getSyncToken()
        q = self._registerSync(token)
        if not args:
            pargs = [instrfrac, delay, dur, token]
            self._perfThread.scoreEvent(0, "i", pargs)
        elif isinstance(args, np.ndarray):
            pargsnp = np.empty((len(args)+4,), dtype=float)
            pargsnp[0] = instrfrac
            pargsnp[1] = delay
            pargsnp[2] = dur
            pargsnp[3] = token
            pargsnp[4:] = args
            self._perfThread.scoreEvent(0, "i", pargsnp)
        else:
            pargs = [instrfrac, delay, dur, token]
            pargs.extend(a if not isinstance(a, str) else self.strSet(a) for a in args)
            self._perfThread.scoreEvent(0, "i", pargs)
        try:
            outvalue = q.get(block=True, timeout=timeout)
            return outvalue if outvalue != _UNSET else None
        except _queue.Empty:
            raise TimeoutError(f"{token=}, {instr=}")


    def _eventWithCallback(self, token:int, pargs, callback) -> None:
        """
        Create a csound "i" event with the given pargs with the possibility
        of receiving a notification from the instrument

        The event is passed a token as p4 and can set a return value by:

        .. code-block:: csound

            itoken = p4
            tabw kreturnValue, itoken, gi__responses
            ; or tabw_i ireturnValue, itoken, gi__responses
            outvalue "__sync__", itoken

        Args:
            token: a token as returned by self._getToken()
            pargs: the pfields passed to the event (beginning by p1)
            callback: A function (returnValue) -> None. It will be called when the instr
                outvalues a "__sync__" message.
        """
        assert token == pargs[3]
        assert isinstance(token, int)
        table = self._responsesTable
        self._responseCallbacks[token] = lambda token, t=table, c=callback: c(t[token])
        self._perfThread.scoreEvent(0, "i", pargs)
        return None

    def _inputMessageWait(self, token:int, inputMessage:str,
                          timeout:float=None) -> Optional[float]:
        """
        This function passes the str inputMessage to csound and waits for
        the instr to notify us with a "__sync__" outvalue

        If the instr returned a value via gi__responses, this value
        is returned. Otherwise None is returned

        The input message should pass the token:

        .. code-block:: csound

            itoken = p4
            tabw kreturnValue, itoken, gi__responses
            ; or tabw_i ireturnValue, itoken, gi__responses
            outvalue "__sync__", itoken

        Args:
            token: a sync token, assigned via _getSyncToken
            inputMessage: the input message passed to csound
            timeout: a timeout for blocking

        Return:
            a float response, or None if the instrument did not set a response value
        """
        if timeout is None:
            timeout = config['timeout']
        q = self._registerSync(token)
        self._perfThread.inputMessage(inputMessage)
        try:
            value = q.get(block=True, timeout=timeout)
            return value if value != _UNSET else None
        except _queue.Empty:
            raise TimeoutError(f"{token=}, {inputMessage=}")

    def _inputMessageWithCallback(self, token:int, inputMessage:str, callback) -> None:
        """
        This function passes the str inputMessage to csound and before that
        sets a callback waiting for an outvalue notification. If no callback
        is passed the function will block until the instrument notifies us
        via outvalue "__sync__"

        Args:
            token: a sync token, assigned via _getSyncToken
            inputMessage: the input message passed to csound
            callback: if given, this function will be called when the instrument
                notifies us via `outvalue "__sync__", token`. The callback should
                be of kind `(token:int) -> None`
        """
        self._responseCallbacks[token] = callback
        self._perfThread.inputMessage(inputMessage)
        return None

    def _makeTableNotify(self,
                         data: list[float] | np.ndarray | None = None,
                         size=0,
                         tabnum=0,
                         callback=None,
                         sr: int = 0,
                         numchannels=1) -> int:
        """
        Create a table with data (or an empty table of the given size).

        Let csound generate a table index if needed.

        Args:
            data: the data to put in the table
            size: if no data is given, size must be set
            tabnum: the table number to create, 0 to let csound generate
                a table number
            callback: a callback func(source) -> None
                If no callback is given this method will block until csound notifies
                that the table has been created and returns the table number
            sr: only needed if filling sample data. If given, it is used to fill
                metadata in csound, as if this table had been read via gen01
            numchannels: only needed if no data is given (only size). Size always
                determines the size of the table, not the number of frames

        Returns:
            returns the table number
        """
        token = self._getSyncToken()
        maketableInstrnum = self.builtinInstrs['maketable']
        delay = 0
        assert tabnum >= 0
        if data is None:
            assert size > 1
            # create an empty table of the given size
            empty = 1
            sr = 0
            pargs = [maketableInstrnum, delay, 0, token, tabnum, size, empty,
                     sr, numchannels]
        else:
            if not isinstance(data, np.ndarray):
                data = np.asarray(data)
            numchannels = internalTools.arrayNumChannels(data)
            numitems = len(data) * numchannels
            if numchannels > 1:
                data = data.flatten()
                # data = data.ravel()
            if numitems < 1900:
                # create a table with the given data
                # if the table is small we can create it and fill it in one go
                empty = 0
                numchannels = internalTools.arrayNumChannels(data)
                if numchannels > 1:
                    data = data.flatten()
                pargs = [maketableInstrnum, delay, 0., token, tabnum, numitems, empty,
                         sr, numchannels]
                pargs.extend(data)
            else:
                # create an empty table (blocking), fill it via a pointer
                empty = 1
                pargs = [maketableInstrnum, delay, 0., token, tabnum, numitems, empty,
                         sr, numchannels]
                if not callback:
                    # the next line blocks until the table is created
                    tabnum = int(self._eventWait(token, pargs))
                    self.fillTable(tabnum, data=data, method='pointer', block=False)
                    self._tableInfo[tabnum] = TableInfo(sr=sr, size=numitems,
                                                        numChannels=numchannels)
                else:
                    def callback2(tabnum, self=self, data=data, callback=callback):
                        self.fillTable(tabnum, data=data, method='pointer', block=False)
                        callback()
                    self._eventWithCallback(token, pargs, callback2)
                return tabnum
        if callback:
            self._eventWithCallback(token, pargs, callback)
        else:
            tabnum = int(self._eventWait(token, pargs))
            assert tabnum is not None and tabnum > 0
        self._tableInfo[tabnum] = TableInfo(sr=sr, size=size, numChannels=numchannels)
        return tabnum

    def channelPointer(self, channel: str, kind='control', mode='rw') -> np.ndarray:
        """
        Returns a numpy array aliasing the memory of a control or audio channel

        If the channel does not exist, it will be created with the given `kind` and set to
        the given mode.
        The returned numpy arrays are internally cached and are valid as long as this Engine is active.
        Accessing the channel through the pointer is not thread-safe.

        Args:
            channel: the name of the channel
            kind: one of 'control' or 'audio' (string channels are not supported yet)

        Returns:
            a numpy array of either 1 or ksmps size

        .. seealso:: :meth:`Engine.setChannel`
        """
        if kind == 'string':
            raise NotImplementedError("Only kind 'control' and 'audio' are implemented at the moment")
        ptr = self._channelPointers.get(channel)
        if ptr is None:
            kindint = ctcsound.CSOUND_CONTROL_CHANNEL if kind == 'control' else ctcsound.CSOUND_AUDIO_CHANNEL
            ptr, err = self.csound.channelPtr(channel, kindint | self._channelMode[mode])
            if err:
                raise RuntimeError(f"Error while trying to retrieve/create a channel pointer: {err}")
            self._channelPointers[channel] = ptr
        assert ptr is not None
        return ptr

    def setChannel(self, channel:str, value:Union[float, str, np.ndarray],
                   method:str=None, delay=0.) -> None:
        """
        Set the value of a software channel

        Args:
            channel: the name of the channel
            value: the new value, should math the type of the channel (a float for
                a control channel, a string for a string channel or a numpy array
                for an audio channel)
            method: one of ``'api'``, ``'score'``, ``'udp'``. None will choose the most appropriate
                method for the current engine/args
            delay: a delay to set the channel

        Example
        =======

        >>> from csoundengine import *
        >>> e = Engine()
        >>> e.initChannel("mastergain", 1.0)
        >>> e.compile(r'''
        ... instr 100
        ...   asig oscili 0.1, 1000
        ...   kmastergain = chnget:k("mastergain")
        ...   asig *= intrp(kmastergain)
        ... endin
        ... ''')
        >>> eventid = e.sched(100)
        >>> e.setChannel("mastergain", 0.5)
        """
        assert self.csound is not None
        isaudio = isinstance(value, np.ndarray)
        if isaudio:
            method = "api"
        elif delay > 0:
            method = "score"
        elif method is None:
            if self.udpPort and config['prefer_udp']:
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
                instrnum = self.builtinInstrs['chnset']
                s = f'i {instrnum} {delay} 0 "{channel}" {value}'
                self._perfThread.inputMessage(s)
            else:
                instrnum = self.builtinInstrs['chnsets']
                s = f'i {instrnum} {delay} 0 "{channel}" "{value}"'
                self._perfThread.inputMessage(s)
        elif method == 'udp':
            if not self.udpPort:
                raise RuntimeError("This server has been started without udp support")
            assert isinstance(value, (float, str))
            self.udpSetChannel(channel, value)
        elif method == 'pointer':
            if isinstance(value, str):
                raise ValueError("Method 'pointer' not available for string channels")
            ptr = self.channelPointer(channel)
            if isinstance(value, float):
                ptr[0] = value
            elif isinstance(value, np.ndarray):
                assert len(value) == self.ksmps
                ptr[:] = value
        else:
            raise ValueError(f"method {method} not supported "
                             f"(choices: 'api', 'score', 'udp'")

    def initChannel(self, channel:str, value:Union[float, str, np.ndarray]=0, kind:str=None,
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
        >>> e.compile(r'''
        ... instr 100
        ...   asig oscili 0.1, 1000
        ...   kmastergain = chnget:k("mastergain")
        ...   asig *= intrp(kmastergain)
        ... endin
        ... ''')
        >>> eventid = e.sched(100)
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
            self.compile(f'chn_k "{channel}", {modei}\n')
            self.setChannel(channel, value, method="score")
        elif kind == 'a':
            self.compile(f'chn_a "{channel}", {modei}', block=True)
            if value:
                self.setChannel(channel, value)
        elif kind == 'S':
            self.compile(f'chn_S "{channel}", {modei}\n', block=True)
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
        >>> e.compile('''
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
        assert self.csound is not None
        value, errorCode = self.csound.controlChannel(channel)
        if errorCode != 0:
            raise KeyError(f"control channel {channel} not found")
        return value

    def fillTable(self, tabnum:int, data, method='pointer', block=False) -> None:
        """
        Fill an existing table with data

        Args:
            data: the data to put into the table
            source: the table number
            method: the method used, one of 'pointer' or 'api'.
            block: this is only used for the api method

        Example
        -------

            >>> from csoundengine import *
            >>> import numpy as np
            >>> e = Engine()
            >>> xs = np.linspace(0, 6.28, 1000)
            >>> ys = np.sin(xs)
            >>> source = e.makeEmptyTable(len(ys))
            >>> e.fillTable(source, data=ys)
            >>> e.plotTable(source)

        .. figure:: assets/tableplot-sine.png

        See Also
        ~~~~~~~~

        :meth:`~Engine.makeTable`
        :meth:`~Engine.plotTable`
        :meth:`~Engine.readSoundfile`
        """
        assert self.csound is not None
        if len(data.shape) == 2:
            data = data.flatten()
        elif len(data.shape) > 2:
            raise ValueError(f"data should be a 1D or 2D array, got shape {data.shape}")

        assert isinstance(tabnum, int) and tabnum > 0, \
            f"source should be an int > 0, got {tabnum}"

        if method == 'pointer':
            # this is always blocking
            numpyptr: np.ndarray = self.csound.table(tabnum)
            if numpyptr is None:
                raise IndexError(f"Table {tabnum} does not exist")
            size = len(numpyptr)
            if size < len(data):
                numpyptr[:] = data[:size]
            else:
                numpyptr[:] = data
        elif method == 'api':
            if block:
                self.csound.tableCopyIn(tabnum, data)
            else:
                self.csound.tableCopyInAsync(tabnum, data)
        else:
            raise KeyError("Method not supported. Must be pointer or score")

    def tableInfo(self, tabnum: int, cache=True) -> TableInfo:
        """
        Retrieve information about the given table

        Args:
            source: the table number
            cache: if True, query the cache to see if info for this table
                has already been requested

        Returns:
            a TableInfo with fields `tableNumber`, `sr` (``ftsr``),
            `numChannels` (``ftchnls``), `numFrames` (``nsamps``), 
            `size` (``ftlen``). 
            
        .. note::
        
            Some information, like *sr*, is only available for tables
            allocated via ``GEN01`` (for example, using :meth:`~Engine.readSoundfile`).
            This data can also be set explicitely via :meth:`~Engine.setTableMetadata`

        Example
        -------

            >>> from csoundengine import *
            >>> e = Engine()
            >>> source = e.readSoundfile("stereo.wav", block=True)
            >>> e.tableInfo(source)
            TableInfo(tableNumber=200, sr=44100.0, numChannels=2, numFrames=88200, size=176401)

        See Also
        ~~~~~~~~

        :meth:`~Engine.readSoundfile`
        :meth:`~Engine.plotTable`
        :meth:`~Engine.getTableData`

        """
        info = self._tableInfo.get(tabnum)
        if info and cache:
            return info
        toks = [self._getSyncToken() for _ in range(4)]
        pargs = [self.builtinInstrs['tableInfo'], 0, 0., tabnum]
        pargs.extend(toks)
        q = _queue.Queue()

        def callback(tok0, q=q, t=self._responsesTable, toks=toks):
            values = [t[tok] for tok in toks]
            q.put(values)

        self._responseCallbacks[toks[0]] = callback
        self._perfThread.scoreEvent(0, "i", pargs)
        vals = q.get(block=True)
        for tok in toks:
            self._releaseToken(tok)
        sr = vals[0]
        if sr <= 0:
            raise TableNotFoundError(f"Table {tabnum} does not exist!")
        return TableInfo(sr=vals[0], numChannels=int(vals[1]),
                         numFrames=int(vals[2]), size=int(vals[3]))

    def readSoundfile(self, path:str="?", tabnum:int = None, chan=0,
                      callback=None, block=False) -> int:
        """
        Read a soundfile into a table (via GEN1), returns the table number

        Args:
            path: the path to the output -- **"?" to open file interactively**
            tabnum: if given, a table index. If None, an index is
                autoassigned
            chan: the channel to read. 0=read all channels
            block: if True, wait until output is read, then return
            callback: if given, this function () -> None, will be called when
                output has been read.

        Returns:
            the index of the created table

        >>> from csoundengine import *
        >>> e = Engine()
        >>> source = e.readSoundfile("stereo.wav", block=True)
        >>> eventid = e.playSample(source)
        >>> # Reduce the gain to 0.8 and playback speed to 0.5 after 2 seconds
        >>> e.setp(eventid, 4, 0.8, 5, 0.5, delay=2)

        See Also
        ~~~~~~~~

        :meth:`~Engine.playSample`
        :meth:`~Engine.makeTable`
        :meth:`~Engine.fillTable`
        :meth:`~Engine.getTableData`
        """
        if block and callback:
            raise ValueError("blocking mode not supported when a callback is given")
        if path == "?":
            path = _state.openSoundfile(ensureSelection=True)
        if not block and not callback:
            return self._readSoundfileAsync(path=path, tabnum=tabnum, chan=chan)

        if tabnum is None:
            tabnum = self._assignTableNumber()
        elif tabnum == 0 and not callback and not block:
            logger.info("readSoundfile: tabnum==0 indicates that csound must assign"
                        "a table number. This operation will block until the soundfile"
                        "is read. To avoid this, set tabnum to None; in this case"
                        "csoundengine will assign a table number itself and the"
                        "operation can be non-blocking")
            block = True

        self._tableInfo[tabnum] = _getSoundfileInfo(path)

        if block or callback:
            token = self._getSyncToken()
        else:
            # if token is set to 0, no notification takes place
            token = 0
        p1 = self.builtinInstrs['readSndfile']
        msg = f'i {p1} 0 0. {token} "{path}" {tabnum} {chan}'
        if callback:
            self._inputMessageWithCallback(token, msg, lambda *args: callback())
        elif block:
            self._inputMessageWait(token, msg)
        else:
            self._perfThread.inputMessage(msg)
        return tabnum

    def soundfontPlay(self, index: int, pitch:float, amp:float=0.7, delay=0.,
                      dur=-1., vel:int=None, chan=1
                      ) -> float:
        """
        Play a note of a previously loaded soundfont

        The soundfont's program (bank, preset) must have been read before
        via :meth:`Engine.soundfontPreparePreset`

        Args:
            index (int): as returned via :meth:`~Engine.soundfontPrearePreset`
            pitch (float): the pitch of the played note, as a midinote (can
                be fractional)
            amp (float): the amplitude. If vel (velocity) is left as None, this
                is used to determine the velocity. Otherwise set the velocity
                (this might  be used by the soundfont to play the correct sample)
                and the amplitude is used to scale the output
            vel (int): the velocity of the played note, used internally to determine
                which sample/layer to play
            chan (int): the first channel to send output to (channels start with 1)
            delay (float): when to start playback
            dur (float): the duration of playback. Use -1 to play until the end
                (the note will be stopped when the soundfont playback detects the
                end of the sample)

        Returns:
            the instance number of the playing instrument.

        .. important::

            **Dynamic Fields**

            - **p4**: `pitch`
            - **p5**: `amp`

        Example
        =======

        .. code::

            from csoundengine import *
            e = Engine()
            # Since the preset is not specified, this will launch a gui dialog
            # to select a preset from a list of available presets
            idx = e.soundfontPreparePreset('Orgue-de-salon.sf2')
            event = e.soundfontPlay(idx, 60)

            # Automate a major 3rd glissando from the current pitch,
            offset, glissdur = 2, 8
            pitch = e.getp(event, 4)
            event.automatep(event, 4, [offset, pitch, offset+glissdur, pitch+4])

        .. figure:: assets/select-preset.png

        See Also
        ~~~~~~~~

        :meth:`~Engine.soundfontPreparePreset`
        :meth:`~Engine.playSample`

        """
        assert index in self._soundfontPresets.values()
        if vel is None:
            vel = amp/127
        args = [pitch, amp, index, vel, chan]
        return self.sched(self.builtinInstrs['soundfontPlay'], delay=delay, dur=dur,
                          args=args)


    def soundfontPreparePreset(self, sf2path:str, preset:Tuple[int, int]=None) -> int:
        """
        Prepare a soundfont's preset to be used

        Assigns an index to a soundfont bank:preset to be used with sfplay or via
        :meth:`~Engine.soundfontPlay`

        The soundfont is loaded if it was not loaded before

        .. figure:: assets/select-preset.png

        Args:
            sf2path: the path to a sf2 file -- **Use "?" to select a file interactively**
            preset: a tuple (bank, presetnum), where both bank and presetnum
                are ints in the range (0-127). None to select a preset interactively

        Returns:
            an index assigned to this preset, which can be used with
            sfplay/sfplay3 or with :meth:``~Engine.soundfontPlay``

        See Also
        ~~~~~~~~

        :meth:`~Engine.soundfontPlay`
        :meth:`~Engine.playSample`
        """
        if sf2path == "?":
            sf2path = _state.openSoundfont(ensureSelection=True)
        if preset is None:
            item = csoundlib.soundfontSelectPreset(sf2path)
            bank, presetnum, _ = item
        else:
            bank, presetnum = preset
        tup = (sf2path, bank, presetnum)
        idxnum = self._soundfontPresets.get(tup)
        if idxnum is not None:
            return idxnum
        idx = self._soundfontPresetCountPtr[0]
        self._soundfontPresetCountPtr[0] += 1
        self._soundfontPresets[tup] = idx
        instrnum = self.builtinInstrs['sfPresetAssignIndex']
        s = f'i {instrnum} 0 0 "{sf2path}" {bank} {presetnum} {idx}'
        self._perfThread.inputMessage(s)
        return idx

    def _readSoundfileAsync(self, path:str, tabnum:int=None, chan=0) -> int:
        assert self.started
        if tabnum is None:
            tabnum = self._assignTableNumber()
        s = f'f {tabnum} 0 0 -1 "{path}" 0 0 {chan}'
        self._perfThread.inputMessage(s)
        return tabnum

    def getUniqueInstrInstance(self, instr: Union[int, str]) -> float:
        """
        Returns a unique instance number (a float p1) for `instr`

        Args:
            instr (int|str): an already defined csound instrument

        Returns:
            a unique p1.
        """
        if isinstance(instr, int):
            token = self._getSyncToken()
            pargs = [self.builtinInstrs['uniqinstance'], 0, 0.01, token, instr]
            uniqinstr = self._eventWait(token, pargs)
            if uniqinstr is None:
                raise RuntimeError("failed to get unique instance")
            return uniqinstr
        else:
            raise NotImplementedError("str instrs not implemented yet")

    def playSample(self, tabnum:int, delay=0., chan=1, speed=1., gain=1., fade=0.,
                   starttime=0., gaingroup=0, lagtime=0.01, dur=-1.) -> float:
        """
        Play a sample already loaded into a table.

        Speed and gain can be modified via setp while playing

        Args:
            source (int): the table where the sample data was loaded
            delay (float): when to start playback
            chan (int): the first channel to send output to (channels start with 1)
            speed (float): the playback speed
            gain (float): a gain applied to this sample
            fade (float): fadein/fadeout time in seconds
            starttime (float): playback can be started from anywhere within the table
            gaingroup (int): multiple instances can be gain-moulated via gaingroups
            lagtime (float): a lag value for dynamic pfields (see below)
            dur (float): the duration of playback. Use -1 to play until the end

        Returns:
            the instance number of the playing instrument.

        .. admonition:: Dynamic Fields
            :class: important

            - **p4**: `gain`
            - **p5**: `speed`

        Example
        =======

            >>> from csoundengine import *
            >>> e = Engine()
            >>> import sndfileio
            >>> sample, sr = sndfileio.sndread("stereo.wav")
            >>> # modify the sample in python
            >>> sample *= 0.5
            >>> source = e.makeTable(sample, sr=sr, block=True)
            >>> eventid = e.playSample(source)
            ... # gain (p4) and speed (p5) can be modified while playing
            ... # Play at half speed
            >>> e.setp(eventid, 5, 0.5)

        See Also
        ~~~~~~~~

        :meth:`~Engine.playSoundFromDics`
        :meth:`~Engine.makeTable`
        :meth:`~Engine.readSoundfile`
        :meth:`~Engine.soundfontPlay`

        """
        args = [gain, speed, tabnum, chan, fade, starttime, gaingroup, lagtime]
        return self.sched(self.builtinInstrs['playgen1'], delay=delay, dur=dur,
                          args=args)

    def playSoundFromDisc(self, path:str, delay=0., chan=0, speed=1., fade=0.01
                          ) -> float:
        """
        Play a soundfile from disc (via diskin2).

        Args:
            path: the path to the output
            delay: time offset to start playing
            chan: first channel to output to
            speed: playback speed (2.0 will sound an octave higher)
            fade: fadein/out in seconds

        Returns:
            the instance number of the scheduled event

        See Also
        ~~~~~~~~

        :meth:`~Engine.readSoundfile`
        :meth:`~Engine.playSample`

        """
        assert self.started
        p1 = self._assignEventId(self.builtinInstrs['playsndfile'])
        #pargs = [p1, delay, -1, self.strSet(path), chan, speed, fade]
        #self._perfThread.scoreEvent(0, "i", pargs)
        msg = f"i {p1} {delay} -1 \"{path}\" {chan} {speed} {fade}"
        self._perfThread.inputMessage(msg)
        return p1

    def setp(self, p1:float, *pairs, delay=0.) -> None:
        """
        Modify a parg of an active synth.

        Multiple pargs can be modified simultaneously. If only makes sense to
        modify a parg if a k-rate variable was assigned to this parg (see example)

        Args:
            p1 (float): the p1 of the instrument to automate
            *pairs: each pair consists of a parg index and a value
            delay (float): when to start the automation

        Example
        =======

        >>> engine = Engine(...)
        >>> engine.compile(r'''
        ... instr 100
        ...   kamp = p5
        ...   kfreq = p6
        ...   a0 oscili kamp, kfreq
        ...   outch 1, a0
        ... endin
        ... ''')
        >>> p1 = engine.sched(100, args=[0.1, 440])
        >>> engine.setp(p1, 5, 0.2, 6, 880, delay=0.5)

        See Also
        ~~~~~~~~

        :meth:`~Engine.getp`
        :meth:`~Engine.automatep`
        """
        numpairs = len(pairs) // 2
        assert len(pairs) % 2 == 0 and numpairs <= 5
        # this limit is just the limit of the pwrite instr, not of the opcode
        args = [p1, numpairs]
        args.extend(pairs)
        self.sched(self.builtinInstrs['pwrite'], delay=delay, dur=0, args=args)

    def getp(self, eventid: float, idx: int) -> float:
        """
        Get the current pfield value of an active synth.

        .. note::

            This action has always a certain latency, since it implies
            scheduling an internal event to read the value and send it
            back to python. The action is blocking

        Args:
            eventid: the (fractional) id (a.k.a p1) of the event
            idx: the index of the p-field, starting with 1 (4=p4)

        Returns:
            the current value of the given pfield

        Example
        -------

        TODO

        .. seealso::

            :meth:`~Engine.setp`
        """
        token = self._getSyncToken()
        notify = 1
        pargs = [self.builtinInstrs['pread'], 0, 0, token, eventid, idx, notify]
        value = self._eventWait(token, pargs)
        return value

    def automateTable(self, tabnum:int, idx:int, pairs: Sequence[float],
                      mode='linear', delay=0., overtake=False) -> float:
        """
        Automate a table slot

        Args:
            tabnum: the number of the table to modify
            idx: the slot index
            pairs: the automation data is given as a flat sequence of pairs (time,
              value). Times are relative to the start of the automation event.
            mode: one of 'linear', 'cos', 'expon(xx)', 'smooth'. See the opcode
              `interp1d` for more information
            delay: the time delay to start the automation.
            overtake: if True, the first value of pairs is replaced with
                the current value in the param table of the running instance

        Returns:
            the eventid of the instance performing the automation

        Example
        =======

        >>> e = Engine(...)
        >>> e.compile(r'''
        ... instr 100
        ...   itab = p4
        ...   kamp  table 0, itab
        ...   kfreq table 1, itab
        ...   outch 1, oscili:a(0.1, kfreq)
        ...   ftfree itab, 1  ; free the table when finished
        ... endin
        ... ''')
        >>> source = e.makeTable([0.1, 1000])
        >>> eventid = e.sched(100, 0, 10, args=(source,))
        # automate the frequency (slot 1)
        >>> e.automateTable(source, 1, [0, 1000, 3, 200, 5, 200])

        >>> # Automate from the current value, will produce a fade-out
        >>> e.automateTable(source, 0, [0, -1, 2, 0], overtake=True, delay=5)

        See Also
        ~~~~~~~~

        :meth:`~Engine.setp`
        :meth:`~Engine.automatep`
        """
        # tabpairs table will be freed by the instr itself
        tabpairs = self.makeTable(pairs, tabnum=0, block=False)
        args = [tabnum, idx, tabpairs, self.strSet(mode), 2, 1, int(overtake)]
        dur = pairs[-2]+self.ksmps/self.sr
        return self.sched(self.builtinInstrs['automateTableViaTable'], delay=delay,
                          dur=dur, args=args)

    def automatep(self, p1: float, pidx: int, pairs:Sequence[float], mode='linear',
                  delay=0., overtake=False
                  ) -> float:
        """
        Automate a pfield of a running event

        The automation is done by another csound event, so it happens within the
        "csound" realm and thus is assured to stay in sync

        Args:
            p1: the fractional instr number of a running event, or an int number
                to modify all running instances of that instr
            pidx: the pfield index. If the pfield to modify if p4, pidx should be 4
            pairs: the automation data is given as a flat data. of pairs (time, value).
                Times are relative to the start of the automation event
            mode: one of 'linear', 'cos', 'expon(xx)', 'smooth'. See the csound opcode
                `interp1d` for more information
                (https://csound-plugins.github.io/csound-plugins/opcodes/interp1d.html)
            delay: the time delay to start the automation.
            overtake: if True, the first value of pairs is replaced with
                the current value in the running instance

        Returns:
            the p1 associated with the automation synth

        Example
        =======

        >>> e = Engine()
        >>> e.compile(r'''
        ... instr 100
        ...   kfreq = p4
        ...   outch 1, oscili:a(0.1, kfreq)
        ... endin
        ... ''')
        >>> eventid = e.sched(100, 0, 10, args=(1000,))
        >>> e.automatep(eventid, 4, [0, 1000, 3, 200, 5, 200])

        See Also
        ~~~~~~~~

        :meth:`~Engine.setp`
        :meth:`~Engine.automateTable`
        """
        # table will be freed by the instr itself
        if len(pairs) < 1900:
            args = [p1, pidx, self.strSet(mode), int(overtake), len(pairs)]
            args.extend(pairs)
            dur = pairs[-2] + self.ksmps / self.sr
            assert isinstance(dur, float)
            return self.sched(self.builtinInstrs['automatePargViaPargs'], delay=delay,
                              dur=dur, args=args)

        else:
            tabnum = self.makeTable(pairs, tabnum=0, block=False)
            args = [p1, pidx, tabnum, self.strSet(mode), int(overtake)]
            dur = pairs[-2]+self.ksmps/self.sr
            assert isinstance(dur, float)
            return self.sched(self.builtinInstrs['automatePargViaTable'], delay=delay,
                              dur=dur, args=args)

    def strSet(self, s:str, sync=False) -> int:
        """
        Assign a numeric index to a string to be used inside csound

        Args:
            s: the string to set
            sync: if True, block until the csound process receives the message

        Returns:
            the index associated with *s*. When passed to a csound instrument
            it can be used to retrieve the original string via
            ``Sstr = strget(idx)``

        See Also
        ~~~~~~~~

        :meth:`~Engine.strGet`

        """
        assert self.started and s
        stringIndex = self._strToIndex.get(s)
        if stringIndex:
            return stringIndex
        stringIndex = self._getStrIndex()
        self._strToIndex[s] = stringIndex
        self._indexToStr[stringIndex] = s
        # self.csound.compileOrcAsync(f'strset {stringIndex}, "{s}"\n')
        instrnum = self.builtinInstrs['strset']
        msg = f'i {instrnum} 0 0 "{s}" {stringIndex}'
        self._perfThread.inputMessage(msg)
        if sync:
            self.sync()
        return stringIndex

    def strGet(self, index:int) -> Optional[str]:
        """
        Get the string previously set via strSet.

        This method will not retrieve any string set internally via the
        `strset` opcode, only strings set via :meth:`~Engine.strSet`

        Example
        =======

        >>> e = Engine(...)
        >>> idx = e.strSet("foo")
        >>> e.strGet(idx)
        foo


        See Also
        ~~~~~~~~

        :meth:`~Engine.strSet`

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

        .. seealso::

            :meth:`~Engine.makeTable`

        """
        logger.debug(f"Freeing table {tableindex}")
        self._releaseTableNumber(tableindex)
        pargs = [self.builtinInstrs['freetable'], delay, 0., tableindex]
        self._perfThread.scoreEvent(0, "i", pargs)

    def testAudio(self, dur=4., delay=0.5, period=1, mode='pink',
                  gaindb=-6) -> float:
        """
        Test this engine's output

        Args:
            dur: the duration of the test
            delay: when to start the test
            period: the duration of sound output on each channel
            mode: the test mode, one of 'pink', 'sine'
            gaindb: the gain of the output, in dB

        Returns:
            the p1 of the scheduled event
        """
        assert self.started
        modeid = {
            'pink': 0,
            'sine': 1
        }.get(mode)

        if modeid is None:
            raise ValueError(f"mode must be one of 'pink', 'sine', got {mode}")

        return self.sched(self.builtinInstrs['testaudio'], dur=dur, delay=delay,
                          args=[modeid, period, pt.db2amp(gaindb)])


    # ~~~~~~~~~~~~~~~ UDP ~~~~~~~~~~~~~~~~~~

    def _udpSend(self, code: str) -> None:
        if not self.udpPort:
            logger.warning("This csound instance was started without udp")
            return
        msg = code.encode("ascii")
        logger.debug(f"_udpSend: {code}")
        self._udpSocket.sendto(msg, self._sendAddr)

    def udpSendOrc(self, code: str) -> None:
        """
        Send orchestra code via UDP.

        The code string can be of any size (if the code is too long for a UDP
        package, it is split into multiple packages)

        Args:
            code (str): the code to send

        .. seealso::

            :meth:`~Engine.udpSendScoreline`
            :meth:`~Engine.udpSetChannel`

        """
        msg = code.encode("ascii")
        if len(msg) < 60000:
            self._udpSocket.sendto(msg, self._sendAddr)
            return
        msgs = emlib.textlib.splitInChunks(msg, 60000)
        self._udpSocket.sendto(b"{{ " + msgs[0], self._sendAddr)
        for msg in msgs[1:-1]:
            self._udpSocket.sendto(msg, self._sendAddr)
        self._udpSocket.sendto(msgs[-1] + b" }}", self._sendAddr)

    def udpSendScoreline(self, scoreline:str) -> None:
        """
        Send a score line to csound via udp

        Example
        =======

        >>> e = Engine(udpserver=True)
        >>> e.compile(r'''
        ... instr 100
        ...   ifreq = p4
        ...   outch 1, oscili:a(0.1, ifreq)
        ... endin
        ... ''')
        >>> e.udpSendScoreline("i 100 0 4 440")

        .. seealso::

            :meth:`~Engine.udpSetChannel`
            :meth:`~Engine.udpSendOrc`

        """
        self._udpSend(f"& {scoreline}\n")

    def udpSetChannel(self, channel:str, value:Union[float, str]) -> None:
        """
        Set a channel via UDP. The value will determine the kind of channel

        Args:
            channel (str): the channel name
            value (float|str): the new value

        .. seealso::

            :meth:`~Engine.udpSendScoreline`
            :meth:`~Engine.udpSendOrc`
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
        assert self._subgainsTable is not None
        self._subgainsTable[idx] = gain

    def writeBus(self, bus:int, value:float, delay=0.) -> None:
        """
        Set the value of a control bus

        Normally a control bus is set via another running instrument,
        but it is possible to set it directly from python. The first
        time a bus is set or queried there is short delay, all
        subsequent operations on the bus are very fast.

        Args:
            bus: the bus token, as returned via assignBus
            value: the new value
            delay: if given, the modification is scheduled in the future

        .. seealso::

            :meth:`~Engine.readBus`
            :meth:`~Engine.assignBus`

        """
        if not self.hasBusSupport():
            raise RuntimeError("This engine does not have bus support")
        if delay > 0:
            dur = self.ksmps / self.sr
            msg = f'i "_busoutk" {delay} {dur} {bus} {value}'
            self._perfThread.inputMessage(msg)
        else:
            busidx = self._busIndex(bus, create=True)
            self._kbusTable[busidx] = value

    def readBus(self, bus:int, default:float=0.) -> float:
        """
        Read the current value of a control bus

        Normally a control bus is modulated by another running instrument,
        but it is possible to set/read it directly from python. The first
        time a bus is set or queried there is short delay, all
        subsequent operations on the bus are very fast.

        Args:
            bus: the bus number, as returned by assignBus
            default: the value returned if the bus does not exist

        Returns:
            the current value of the bus, or `default` if the bus does not exist

        .. seealso::

            :meth:`~Engine.assignBus`
            :meth:`~Engine.writeBus`
        """
        if not self.hasBusSupport():
            raise RuntimeError("This engine does not have bus support")
        busidx = self._busIndex(bus)
        if busidx < 0:
            return default
        return self._kbusTable[busidx]

    def _busIndex(self, bus:int, create=False) -> int:
        """
        Find the bus index corresponding to `bus` token. This is only needed for
        the case where a bus is written/read from python
        """
        index = self._busIndexes.get(bus)
        if index is not None:
            return index
        synctoken = self._getSyncToken()
        pargs = [self.builtinInstrs['busindex'], 0, 0, synctoken, bus, int(create)]
        out = self._eventWait(synctoken, pargs)
        index = int(out)
        self._busIndexes[bus] = index
        return index

    def releaseBus(self, bus: int) -> None:
        """
        Release a persistent bus

        The bus must have been created in python with the *persistent* flag

        .. seealso:: :meth:`~Engine.assignBus`
        """
        # bus is the bustoken
        if not self.hasBusSupport():
            raise RuntimeError("This Engine was created without bus support")
        pargs = [self.builtinInstrs['busrelease'], 0, 0, bus]
        self._perfThread.scoreEvent(0, "i", pargs)


    def assignBus(self, kind='audio', persist=False) -> int:
        """
        Assign one audio/control bus, returns the bus number.

        Audio buses are always mono.

        Args:
            kind: the kind of bus, "audio" or "control"
            persist: if True the bus created is keps alive until the user
                calls :meth:`~Engine.releaseBus`

        Returns:
            the bus id, can be passed to any instrument expecting a bus
            to be used with the built-in opcodes :ref:`busin`, :ref:`busout`, etc.

        A bus created here can be used together with the built-in opcodes :ref:`busout`,
        :ref:`busin` and :ref:`busmix`. A bus can also be created directly in csound by
        calling :ref:`busassign`

        A bus is reference counted and is collected when there are no more clients
        using it. At creation the bus is "parked", waiting to be used by any client.
        As long as no clients use it, the bus stays in this state and is ready to
        be used.
        Multiple clients can use a bus and the bus is kept alive as long as there
        are clients using it or if the bus was created as *persistent*.
        When each client starts using the bus via any of the bus opcodes, like :ref:`busin`,
        the reference count of the bus is increased. After a client has finished
        using it the reference count is automatically decreased and if it reaches
        0 the bus is collected.

        Order of evaluation is important: **audio buses are cleared at the end of each
        performance cycle** and can only be used to communicate from a low
        priority to a high priority instrument.

        For more information, see :ref:`Bus Opcodes<busopcodes>`

        Example
        =======

        Pass audio from one instrument to another. The bus will be released after the events
        are finished.

        >>> e = Engine(...)
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


        Modulate one instr with another, at k-rate. **NB: control buses act like global
        variables, the are not cleared at the end of each cycle**.

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
        bustoken = int(self._busTokenCountPtr[0])
        ikind = 0 if kind == 'audio' else 1
        assert isinstance(bustoken, int)
        self._busTokenCountPtr[0] = bustoken+1

        # before returning the bustoken we schedule a query
        # so that we can return immediately but update the actual
        # bus index assigned by csound for any future query

        synctoken = self._getSyncToken()
        # Assigns a bus to the given token
        pfields = [self.builtinInstrs['busindex'], 0, 0, synctoken, bustoken, ikind]

        def callback(synctoken, bustoken=bustoken, self=self):
            self._busIndexes[bustoken] = int(self._responsesTable[int(synctoken)])

        self._eventWithCallback(synctoken, pfields, callback)

        if persist:
            pfields = [self.builtinInstrs['busaddref'], 0, 0, bustoken, ikind]
            self._perfThread.scoreEvent(0, "i", pfields)

        return bustoken

    def hasBusSupport(self) -> bool:
        """
        Returns True if this Engine was started with bus support

        .. seealso::

            :meth:`Engine.assignBus`
            :meth:`Engine.writeBus`
            :meth:`Engine.readBus`
        """
        return (self.numAudioBuses > 0 or self.numControlBuses > 0)

    def eventUI(self, eventid: float, **pargs: Dict[str, Tuple[float, float]]) -> None:
        """
        Modify pfields through an interactive user-interface

        If run inside a jupyter notebook, this method will create embedded widgets
        to control the values of the dynamic pfields of an event

        Args:
            eventid: p1 of the event to modify
            **pfields: a dict mapping pfield to a tuple (minvalue, maxvalue)

        Example
        =======

        .. code::

            from csoundengine import *
            e = Engine()
            e.compile(r'''
            instr synth
              kmidinote = p4
              kampdb = p5
              kcutoff = p6
              kres = p7
              kfreq = mtof:k(kmidinote)
              asig = vco2:a(ampdb(kampdb), kfreq)
              asig = moogladder2(asig, kcutoff, kres)
              asig *= linsegr:a(0, 0.1, 1, 0.1, 0)
              outs asig, asig
            endin
            ''')
            ev = e.sched('synth', args=[67, -12, 3000, 0.9])
            e.eventUI(ev, p4=(0, 127), p5=(-48, 0), kcutoff=(200, 5000), kres=(0.1, 1))

        .. figure:: assets/eventui.png
        """
        from . import interact
        specs = {}
        instr = internalTools.instrNameFromP1(eventid)
        body = self._instrRegistry.get(instr)
        pfieldsNameToIndex = csoundlib.instrParseBody(body).pfieldsNameToIndex if body else None
        for pfield, spec in pargs.items():
            minval, maxval = spec
            idx = internalTools.resolvePfieldIndex(pfield, pfieldsNameToIndex)
            if not idx:
                raise KeyError(f"pfield {pfield} not understood")
            value = self.getp(eventid, idx)
            specs[idx] = interact.ParamSpec(pfield, minvalue=minval, maxvalue=maxval,
                                            startvalue=value, widgetHint='slider')
        return interact.interactPargs(self, eventid, specs=specs)


@_atexit.register
def _cleanup() -> None:
    engines = list(Engine.activeEngines.values())
    if engines:
        print("Exiting python, closing all active engines")
        for engine in engines:
            print(f"... stopping {engine.name}")
            engine.stop()


def activeEngines() -> KeysView[str]:
    """
    Returns the names of the active engines

    Example
    ~~~~~~~

        >>> import csoundengine as ce
        >>> ce.Engine(nchnls=2)   # Will receive a generic name
        >>> ce.Engine(name='multichannel', nchnls=8)
        >>> ce.activeEngines()
        dict_keys(['engine0', 'multichannel'])
    """
    return Engine.activeEngines.keys()


def getEngine(name: str) -> Engine | None:
    """
    Get an already created engine by name

    Example
    ~~~~~~~

        >>> import csoundengine as ce
        >>> ce.Engine(name='old', a4=435)
        >>> getEngine('old')
        Engine(name=old, sr=44100, backend=jack, outdev=dac, nchnls=2, indev=adc, nchnls_i=2,
               bufferSize=256)
        >>> getEngine('foo') is None
        True

    """
    return Engine.activeEngines.get(name)


