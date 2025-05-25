"""
Engine class
============

An :class:`Engine` implements a simple interface to run and control a realtime
csound process.

For offline processing, see :class:`~csoundengine.offlineengine.OfflineEngine`

Example
-------

.. code::

    from csoundengine import Engine

    # create an engine with default options for the platform
    engine = Engine()

    # Compile an instrument
    engine.compile(r'''
      instr synth
        ; assign preset values for parameters
        ;    p1 p2 p3 p4  p5 p6
        pset 0, 0, 0, 60, 1, 4000   ; the first three values (p1, p2, p3) are not considered
        kmidinote = p4
        kamp = p5
        kcutoff = p6
        kfreq = mtof:k(kmidinote)
        asig = vco2:a(kamp, kfreq)
        asig = moogladder2(asig, kcutoff, 0.9)
        asig *= linsegr:a(0, 0.1, 1, 0.1, 0)
        outs asig, asig
      endin
    ''')

    # start an event with indefinite duration. This returns a unique (fractional)
    # instance number, which can be used to control its parameters
    # The `args` parameter expects any arguments starting with p4
    p1 = engine.sched("synth", args=[67, 0.1, 3000], unique=True)

    # any parameter with k-rate can be modified while running:
    # change midinote
    engine.setp(p1, 4, 67)

    # modify cutoff
    engine.setp(p1, 6, 1000, delay=4)

    # stop the synth:
    engine.unsched(p1)

    # Parameters can also be set using their assigned names. Defaults
    # can be set via the pset opcode. In this case the cutoff parameter
    # will be assigned the default 4000
    p1 = engine.sched("synth", kmidinote=60, kamp=0.1)

    # Change the pitch
    engine.setp(p1, 'kmidinote', 62, delay=2)

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


Instrument Numbers
------------------

An :class:`Engine` defines internal instruments to perform some of its
tasks (reading tables, sample playback, etc). To avoid clashes between these
internal instruments and user instruments, there are some reserved instrument
numbers: all instrument numbers from 1 to 99 are reserved for internal use, so
the first available instrument number is 100.


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

import atexit as _atexit
import ctypes as _ctypes
import fnmatch as _fnmatch
import os.path
import queue as _queue
import re as _re
import sys as _sys
import tempfile
import textwrap as _textwrap
import threading as _threading
import time

import numpy as np
import pitchtools as pt
from emlib import iterlib
from emlib.containers import IntPool

from . import csoundparse
from . import engineorc
from . import internal
from . import state as _state
from .config import config, logger
from .enginebase import TableInfo, _EngineBase
from .engineorc import BUSKIND_AUDIO, BUSKIND_CONTROL, CONSTS
from .errors import CsoundError, TableNotFoundError

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from matplotlib.figure import Figure
    import socket
    from typing import Callable, Sequence
    from . import csoundlib
    from . import session as _session
    callback_t = Callable[[str, float], None]
    import libcsound as lcs

elif 'sphinx' in _sys.modules:
    import socket
    from typing import Callable, Sequence
    callback_t = Callable[[str, float], None]



__all__ = [
    'Engine',
    'getEngine',
]


_UNSET = float("-inf")


def _generateUniqueEngineName(prefix="engine") -> str:
    for i in range(9999):
        name = f"{prefix}{i}"
        if name not in Engine.activeEngines:
            return name

    import uuid
    return str(uuid.uuid4())


def _asEngine(e: str | Engine) -> Engine:
    if isinstance(e, Engine):
        return e
    out = getEngine(e)
    if out is None:
        raise ValueError(f"No engine found with name {e}")
    return out


class Engine(_EngineBase):
    """
    Implements a simple interface to run and control a realtime csound process.

    For offline rendering, see :class:`~csoundengine.offline.OfflineSession` and
    :class:`~csoundengine.offlineengine.OfflineEngine`

    Args:
        name: the name of the engine
        sr: sample rate. If not given, the sr of the backend will be used, if possible
        ksmps: samples per k-cycle
        backend: passed to -+rtaudio (**"?" to select interactively**). If not given, the most
            appropriate backend will be used.
        outdev: the audio output device, passed to -o (**"?" to select interactively**). Leave
            unset to use the default
        indev: the audio input device, passed to -i (**"?" to select interactively**). Leave
            unset to use the default
        a4: freq of a4
        nchnls: number of output channels (passed to nchnls). Leave unset to use the number of
            channels defined by the backend (if known)
        nchnls_i: number of input channels. Similar to nchnls. If not given it will either
            fallback to the number of input channels provided by the backend, or to nchnls
        buffersize: samples per buffer, corresponds to csound's -b option
        numbuffers: the number of buffers to fill. Together with the buffersize determines
            the latency of csound and any communication between csound and the python
            host
        globalcode: code to evaluate as instr0 (global variables, etc.)
        includes: a list of files to include. Can be added later via :meth:`Engine.includeFile`
        numAudioBuses: number of audio buses (see :ref:`Bus Opcodes<busopcodes>`)
        numControlBuses: number of control buses (see :ref:`Bus Opcodes<busopcodes>`)
        quiet: if True, suppress output of csound (-m 0)
        udpserver: if True, start a udp server for communication (see udpport)
        udpport: the udpport to use for real-time messages. 0=autoassign port
        commandlineOptions: command line options passed verbatim to the
            csound process when started
        midiin: if given, use this device as midi input. Can be '?' to select
            from a list, or 'all' to use all devices. None indicates no midi input
        latency: an extra latency added when scheduling events to ensure synchronicity.
            See also :meth:`Engine.lockClock` and :meth:`Engine.pushClock`
        sampleAccurate: use sample-accurate scheduling
        useProcessQueue: only valid for csound 6. If True, will set the process
            callback to handle async tasks

    .. note::
        Any option with a default value of None has a corresponding slot in the
        config. Default values can be configured via `config.edit()`, see
        `Configuration <https://csoundengine.readthedocs.io/en/latest/config.html>`_

    Example
    -------

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
    activeEngines: dict[str, Engine] = {}
    "Active engines mapped by name (class variable)"

    _builtinTables = engineorc.BUILTIN_TABLES

    def __init__(self,
                 name: str = '',
                 sr=0,
                 ksmps: int = 0,
                 backend: str = '',
                 outdev: str = '',
                 indev: str = '',
                 a4: int = 0,
                 nchnls: int | None = None,
                 nchnls_i: int | None = None,
                 realtime=False,
                 buffersize: int = 0,
                 numbuffers: int = 0,
                 globalcode: str = "",
                 numAudioBuses: int | None = None,
                 numControlBuses: int | None = None,
                 quiet: bool | None = None,
                 udpserver=False,
                 udpport: int = 0,
                 commandlineOptions: list[str] | None = None,
                 includes: list[str] | None = None,
                 midibackend: str = 'default',
                 midiin: str | None = None,
                 autosync=False,
                 latency: float | None = None,
                 sampleAccurate: bool = False,
                 numthreads: int = 0,
                 withBusSupport=False,
                 nosound=False,
                 useProcessQueue=False,
    ):
        import libcsound as lcs
        from . import csoundlib

        if not name:
            name = _generateUniqueEngineName()
        elif name in Engine.activeEngines:
            raise KeyError(f"Engine '{name}' already exists")

        if backend == 'portaudio':
            backend = 'pa_cb'
        cfg = config
        availableBackends = csoundlib.getAudioBackendNames()
        if not backend or backend == 'default':
            backend = csoundlib.getDefaultBackend().name
        elif backend == '?':
            selection = internal.selectItem(availableBackends, title="Select Backend")
            if not selection:
                raise ValueError("No backend selected")
            backend = selection
        elif backend not in availableBackends:
            logger.error(f"Backend {backend} unknown. Available backends: "
                            f"{availableBackends}")

        cascadingBackends = [b.strip() for b in backend.split(",")]
        resolvedBackend = internal.resolveOption(cascadingBackends, availableBackends)
        logger.debug(f"Resolved backend: {resolvedBackend}")
        if resolvedBackend is None:
            raise RuntimeError("No audio backends available")

        backendDef = csoundlib.getAudioBackend(resolvedBackend)

        if not backendDef:
            logger.error(f'Could not find any available backends for {backend}')
            logger.error(f'    Available backends: {", ".join(availableBackends)}')
            logger.error('To configure the default backends, do:\n'
                         '    from csoundengine import config\n'
                         '    config.edit()\n'
                         f'And edit the "{internal.platform}.backend" key.')
            raise CsoundError(f'Backend "{resolvedBackend}" not available')

        indevs, outdevs = backendDef.audioDevices()
        defaultin, defaultout = backendDef.defaultAudioDevices()
        indevName, outdevName = "adc", "dac"
        if nosound:
            outdev, outdevName = '', ''
        elif not outdev:
            if not defaultout:
                raise RuntimeError(f"No output devices for backend {backendDef.name}")
            outdev, outdevName = defaultout.id, defaultout.name
            logger.debug(f"No output device given for backend {resolvedBackend}, "
                         f"using default: {outdevName}, id: {outdev}")
            if not nchnls:
                nchnls = defaultout.numChannels
        elif outdev == '?':
            if len(outdevs) == 0:
                raise RuntimeError("No output audio devices")
            logger.debug(f"Selecting output device for backend {resolvedBackend}, devices: {outdevs}")
            selected = internal.selectAudioDevice(outdevs, title="Select output device")
            if selected is None:
                raise RuntimeError("No output audio device selected")
            outdev, outdevName = selected.id, selected.name
            if not nchnls:
                nchnls = selected.numChannels
        elif isinstance(outdev, int) or _re.search(r"\bdac[0-9]+\b", outdev):
            # dac1, dac8
            if backendDef.name == 'jack':
                logger.warning(
                    "This way of setting the audio device is discouraged with jack"
                    ". Use a regex to select a specific client or None to connect"
                    "to the default client")
            if isinstance(outdev, int):
                outdev = f"dac{outdev}"
        else:
            if backendDef.name == 'jack':
                outdevName = outdev
            else:
                selected = next((d for d in outdevs if _fnmatch.fnmatch(d.name, outdev)), None)
                if not selected:
                    raise ValueError(f"Output device {outdev} not known. Possible devices: "
                                     f"{outdevs}")
                outdev, outdevName = selected.id, selected.name

        if not indev:
            if defaultin is None:
                raise RuntimeError(f"No default device for backend {backend}")
            indev, indevName = defaultin.id, defaultin.name
            if not nchnls_i:
                nchnls_i = defaultin.numChannels
        elif indev == '?':
            if len(indevs) == 0:
                raise RuntimeError("No input audio devices")
            selected = internal.selectAudioDevice(indevs, title="Select input device")
            if selected is None:
                raise RuntimeError("No output audio device selected")
            indev, indevName = selected.id, selected.name
        elif isinstance(indev, int) or _re.search(r"\badc[0-9]+\b", indev):
            if resolvedBackend == 'jack':
                logger.warning(
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
        sr = sr if sr else cfg['sr']
        backendsr = csoundlib.getSamplerateForBackend(resolvedBackend)
        backendDef = csoundlib.getAudioBackend(resolvedBackend)
        assert backendDef is not None
        if sr and backendDef.hasSystemSr and sr != backendsr:
            logger.warning(f"sr requested: {sr}, but backend has a fixed sr ({backendsr})"
                           f". Using backend's sr")
            sr = backendsr
        elif not sr:
            if backendDef.hasSystemSr:
                sr = backendsr
            else:
                sr = 44100
                logger.info(f"Asked for system sr, but backend '{resolvedBackend}', does not"
                            f"have a fixed sr. Using sr={sr}")

        if not a4:
            a4 = cfg['A4']
        if numthreads == 0:
            numthreads = config['numthreads']
        if not ksmps:
            ksmps = cfg['ksmps']
        if nchnls_i is None:
            nchnls_i = cfg['nchnls_i']
        if nchnls is None:
            nchnls = cfg['nchnls']

        inchnls, outchnls = csoundlib.getNchnls(resolvedBackend,
                                                outpattern=outdev, inpattern=indev)
        if nchnls == -1:
            nchnls = outchnls
        if nchnls_i == -1:
            nchnls_i = inchnls

        assert nchnls is not None and nchnls > 0
        assert nchnls_i is not None and nchnls_i >= 0

        if nosound and '--nosound' not in commandlineOptions:
            commandlineOptions.append('--nosound')


        if quiet is None:
            quiet = cfg['suppress_output']

        if quiet:
            commandlineOptions.append('-m0')
            commandlineOptions.append('-d')

        if sampleAccurate:
            commandlineOptions.append('--sample-accurate')

        self.name = name
        "Name of this Engine"

        assert sr is not None and sr > 0
        assert isinstance(ksmps, int) and ksmps > 0

        if numAudioBuses is None:
            numAudioBuses = int(config['num_audio_buses'])
        if numControlBuses is None:
            numControlBuses = int(config['num_control_buses'])

        if withBusSupport:
            if not numAudioBuses and not numControlBuses:
                raise ValueError("At least one audio or control bus must be enabled")

        super().__init__(sr=sr,
                         ksmps=ksmps,
                         nchnls=nchnls,
                         numAudioBuses=numAudioBuses,
                         numControlBuses=numControlBuses,
                         sampleAccurate=sampleAccurate,
                         a4=a4)

        self.backend = resolvedBackend
        "Name of the backend used (jack, portaudio, etc)"

        self.onecycle: float = ksmps / sr
        "Duration of one performance cycle (ksmps/sr)"

        self.outdev = outdev
        "Output device used"

        self.outdevName = outdevName
        "Long name of the output device"

        self.indev = indev
        "Input device"

        self.indevName = indevName
        "Long name of the input device"

        self.nchnls_i: int = nchnls_i
        "Number of input channels"

        self.globalCode: str = globalcode
        "Global (init) code to execute at the start of the Engine"

        self.started = False
        "Is this engine started?"

        self.extraOptions: list[str] = commandlineOptions
        "Extra options passed to csound"

        self.commandlineOptions: list[str] = []
        """All command line options used to start the engine"""

        self.includes: list[str] = includes if includes is not None else []
        "List of include files"

        self.extraLatency: float = latency if latency is not None else config['sched_latency']
        "Added latency for better synch"

        self.udpPort = 0
        "UDP port used (0 if no udp port is active)"

        self.csound: lcs.Csound
        "The csound object"

        self.autosync: bool = autosync
        """If True, call .sync whenever is needed"""

        assert isinstance(withBusSupport, bool)
        self._hasBusSupport: bool = withBusSupport

        backendBufferSize, backendNumBuffers = backendDef.bufferSizeAndNum()
        buffersize = (buffersize or backendBufferSize or config['buffersize'] or 256)
        buffersize = max(self.ksmps * 2, buffersize)

        numbuffers = (numbuffers or backendNumBuffers or config['numbuffers'] or
                      internal.determineNumbuffers(self.backend or "portaudio", buffersize=buffersize))

        self.bufferSize: int = buffersize
        "Buffer size"

        self.numBuffers: int = numbuffers
        "Number of buffers to fill"

        self.midiBackend: None | str = midibackend
        "Midi backend used"

        self.started = False
        """Has this engine already started?"""

        self.numthreads: int = numthreads
        """Number of threads to use in performance (corresponds to csound -j N)"""

        if midiin == 'all':
            midiindev = csoundlib.MidiDevice(deviceid='all', name='all')
        elif midiin == '?':
            midiindevs, midioutdevs = csoundlib.midiDevices(self.midiBackend)
            midiindevs.append(csoundlib.MidiDevice('all', 'all'))
            selecteddev = internal.selectMidiDevice(midiindevs)
            if selecteddev is not None:
                midiindev = selecteddev
            else:
                raise RuntimeError("No MIDI device selected")
        elif midiin:
            midiindev = csoundlib.MidiDevice(deviceid=midiin, name='')
        else:
            midiindev = None

        self.midiin: csoundlib.MidiDevice | None = midiindev
        "Midi input device"

        self._uddocket: None | socket.socket = None
        self._sendAddr: None | tuple[str, int] = None

        if udpserver or udpport > 0:
            import emlib.net
            self.udpPort = udpport or emlib.net.findport()
            self._udpSocket = emlib.net.udpsocket()
            self._sendAddr = ("127.0.0.1", self.udpPort)

        self._perfThread: lcs.PerformanceThread
        self._fracnumdigits = 4        # number of fractional digits used for unique instances
        self._exited = False           # are we still running?
        self._realtime = realtime
        self._useProcessQueue = useProcessQueue if lcs.VERSION < 7000 else False

        # counters to create unique instances for each instrument
        self._instanceCounters: dict[int, int] = {}

        # # Maps instrname/number: code
        self._instrRegistry: dict[str | int, str] = {}

        # Keeps parsed instr bodies, used to analyze pfields, defaults, etc.
        self._parsedInstrs: dict[str, csoundparse.ParsedInstrBody] = {}

        # a dict of callbacks, reacting to outvalue opcodes
        self._outvalueCallbacks: dict[bytes, callback_t] = {}

        # Maps used for strSet / strGet
        self._indexToStr: dict[int, str] = {}
        self._strToIndex: dict[str, int] = {}
        self._strLastIndex = 20

        # Marks the last modification to the state of the engine, to track sync
        self._lastModification = 0.

        # global code added to this engine
        self._globalCode: dict[str, str] = {}

        # This will be a numpy array pointing to a csound table of
        # numTokens size. When an instrument wants to return a value to the
        # host, the host sends a token, the instr sets table[token] = value
        # and calls 'outvale "__sync__", token' to signal that an answer is
        # ready
        self._responsesTable: np.ndarray

        # tokens start at 1, leave token 0 to signal that no sync is needed
        # tokens are used as indices to _responsesTable, which is an alias of
        # gi__responses
        self._tokens = list(range(1, CONSTS['numTokens']))

        # a pool of reserved table numbers
        reservedTablesStart = CONSTS['reservedTablesStart']
        self._tablePool = IntPool(CONSTS['numReservedTables'], start=reservedTablesStart)

        # a dict of token:callback, used to register callbacks when asking for
        # feedback from csound
        self._responseCallbacks: dict[int, Callable] = {}

        self._tableCache: dict[int, np.ndarray] = {}
        self._tableInfo: dict[int, TableInfo] = {}

        self._channelPointers: dict[str, np.ndarray] = {}

        self._instrNumCache: dict[str, int] = {}

        self._session: None | _session.Session = None
        self._busTokenCountPtr: np.ndarray = np.empty((1,), dtype=float)
        self._soundfontPresetCountPtr: np.ndarray = np.empty((1,), dtype=float)
        self._kbusTable: np.ndarray | None = None
        self._busIndexes: dict[int, int] = {}
        self._busTokenToKind: dict[int, str] = {}
        self._soundfontPresets: dict[tuple[str, int, int], int] = {}
        self._soundfontPresetCount = 0
        self._startTime = 0.
        self._lockedElapsedTime = 0.
        self._realElapsedTime = (0., -float('inf'))

        # A stack holding locked states
        self._clockStatesStack: list[tuple[bool, float]] = []

        self._reservedInstrnums: set[int] = set()
        self._reservedInstrnumRanges: list[tuple[str, int, int]] = [('builtinorc', CONSTS['reservedInstrsStart'], CONSTS['userInstrsStart']-1)]

        self._minCyclesForAbsoluteMode = 4
        self._fltptr = _ctypes.POINTER(lcs.MYFLT)
        self.version = lcs.VERSION
        """Csound version as integer (6.18 = 6180)"""

        self.start()

    def reservedInstrRanges(self) -> list[tuple[str, int, int]]:
        """
        A dict containing reserved instrument number ranges

        An Engine has some internal instruments for performing tasks like
        automation, bus support, etc. Moreover, if an Engine has an attached
        Session, the session will declare a range of instrument numbers
        as reserved.

        This method returns all those reserved ranges in the form of a
        list of tuples, where each tuple represents a reserved range.
        Each tuple has the form ``(rangename: str, minInstrNumber: int, maxInstrNumber: int)``,
        where ``rangename`` is the name of the range, ``minInstrNumber`` and ``maxInstrNumber``
        represent the instr numbers reserved

        Any instr number outside of this range can be used.
        """
        return self._reservedInstrnumRanges

    def userInstrumentsRange(self) -> tuple[int, int]:
        """
        Returns the range of available instrument numbers

        Notice that the first instrument numbers are reserved for internal instruments.
        If this Engine has an attached Session, the session itself will reserve
        a range of numbers for its events

        Returns:
            a tuple (mininstr: int, maxinstr: int) defining a range of available
            instrument numbers for user instruments.
        """
        maxinstr = CONSTS['maxNumInstrs']
        if len(self._reservedInstrnumRanges) > 1:
            maxinstr = self._reservedInstrnumRanges[1][1]
        return CONSTS['userInstrsStart'], maxinstr

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
        token = self._tokens.pop()
        self._responsesTable[token] = _UNSET
        return token

    def _releaseToken(self, token: int) -> None:
        """ Release token back to pool when done """
        self._tokens.append(token)

    def _assignTableNumber(self) -> int:
        """
        Return a free table number and mark that as being used.
        To release the table, call unassignTable

        Returns:
            the table number (an integer)
        """
        if len(self._tablePool) == 0:
            raise RuntimeError("Table pool is empty")
        tabnum = self._tablePool.pop()
        self._tableCache.pop(tabnum, None)
        return tabnum

    def _assignEventId(self, instrnum: int | str) -> float:
        """
        Assign an eventid (fractional instr number) for this instr

        This is not really a unique instance, there might be conflicts
        with a previously scheduled event. To really generate a unique instance
        we would need to call uniqinstance, which creates a roundtrip to csound

        Args:
            instrnum (int): the instrument number

        Returns:
            the eventid assigned, a float

        """
        if isinstance(instrnum, str):
            instrnum = self.queryNamedInstr(instrnum)
        c = self._instanceCounters.get(instrnum, 0)
        c += 1
        self._instanceCounters[instrnum] = c
        instancenum = (c % int(10 ** self._fracnumdigits - 2)) + 1
        return self._makeEventId(instrnum, instancenum)

    def _makeEventId(self, num: int, instance: int) -> float:
        frac = (instance / (10**self._fracnumdigits)) % 1
        return num + frac

    def _setupGlobalInstrs(self):
        if self.hasBusSupport():
            self._perfThread.scoreEvent(False, "i", [self._builtinInstrs['clearbuses_post'], 0, -1])
        self._modified()

    def stop(self):
        """
        Stop this Engine
        """
        if not hasattr(self, "name"):
            return
        if self.csound is None or not self.started or self._exited:
            logger.debug(f"Engine {self.name} was not running, so can't stop it")
            return
        logger.info(f"stopping Engine {self.name}")
        if self._session is not None:
            self._session.stop()
        else:
            self._perfThread.stop()
            # time.sleep(0.1)
            logger.info("... stopping csound")
            self.csound.stop()
            # time.sleep(0.1)
            logger.info("... cleaning up")
            self.csound.cleanup()
            self._exited = True
            self._instanceCounters.clear()
            self._instrRegistry.clear()
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
            logger.debug(f"Stopping prior engine with same name ('{self.name}')")
            priorengine.stop()

        logger.info(f"Starting engine {self.name}")
        import libcsound as lcs
        buffersize = self.bufferSize
        optB = buffersize * self.numBuffers
        if self.backend == 'jack':
            from . import jacktools
            jackinfo = jacktools.getInfo()
            if jackinfo is None:
                logger.error("Asked to use jack as backend, but jack is not running")
                raise RuntimeError("jack is not running")
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
                    f"-b{buffersize}",
                    f"-B{optB}"]
        options.append(f'-o"{self.outdev}"')
        options.append(f'-i"{self.indev}"')

        if self.numthreads > 1:
            options.append(f'-j {self.numthreads}')

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

        if config['disable_signals']:
            lcs.csoundInitialize(atExitHandler=False, signalHandler=False)
        else:
            logger.debug("Starting a csound instance without disabling signals...")
        cs = lcs.Csound()
        csversion = cs.version()
        if csversion < 6180:
            raise RuntimeError(f"Csound's version should be >= 6.18, got {csversion:.2f}")
        options = list(iterlib.unique(options))
        for opt in options:
            cs.setOption(opt)
        self.commandlineOptions = options
        if self.includes:
            includelines = [f'#include "{include}"' for include in self.includes]
            includestr = "\n".join(includelines)
        else:
            includestr = ""

        assert self.sr > 0
        orcheader = engineorc.makeOrcHeader(sr=self.sr, ksmps=self.ksmps, nchnls=self.nchnls, nchnls_i=self.nchnls_i, a4=self.a4)
        orc, instrmap = engineorc.makeOrc(globalcode=self.globalCode,
                                          includestr=includestr)
        self._builtinInstrs = instrmap
        self._reservedInstrnums = set(instrmap.values())
        cs.compileOrc(orcheader)
        err = cs.compileOrc(orc)
        if err:
            tmporc = tempfile.mktemp(prefix="csoundengine-", suffix=".orc")
            with open(tmporc, "w") as f:
                f.write(orcheader)
                f.write(orc)
            logger.error(f"Error compiling base orchestra. A copy of the orchestra"
                         f" has been saved to {tmporc}")

            logger.error(internal.addLineNumbers(orc))
            raise CsoundError(f"Error compiling base ochestra, error: {err}")
        logger.info(f"Starting csound with options: {options}")
        cs.start()
        responsesTable = cs.table(self._builtinTables['responses'])
        if responsesTable is None:
            raise RuntimeError("Could not create responses table")
        self._responsesTable = responsesTable
        self._startTime = time.time()
        self._orc = orc
        if config['set_sigint_handler']:
            internal.setSigintHandler()

        # Bidirectional channel to keep track of soundfont presets
        chanptr, err = cs.channelPtr("_soundfontPresetCount", kind='control', mode='rw')

        assert isinstance(chanptr, np.ndarray), f"_soundfontPresetCount channel is not set: {err}"
        self._soundfontPresetCountPtr = chanptr

        self.csound = cs

        if self._hasBusSupport:
            self._hasBusSupport = False
            self.addBusSupport()

        if csversion < 7000:
            pt = cs.performanceThread()
            if self._useProcessQueue:
                pt.setProcessQueue()
        else:
            pt = cs.performanceThread()

        pt.play()
        self._perfThread = pt
        self.started = True
        self._setupCallbacks()
        self._setupGlobalInstrs()
        self.activeEngines[self.name] = self
        strsets = ["cos", "linear", "smooth", "smoother"]
        for s in strsets:
            self.strSet(s)
        self._modified()
        self.sync()

    def restart(self, wait=0.) -> None:
        """ Restart this engine. All defined instrs / tables are removed"""
        self.stop()
        if wait:
            import time
            time.sleep(wait)
        self.start()

    def _outvalueCallback(self, _, channelName, valptr, chantypeptr):
        func = self._outvalueCallbacks.get(channelName)
        if not func:
            logger.error(f"outvalue: callback not set for channel {channelName}")
            return
        if valptr is not None:
            val = _ctypes.cast(valptr, self._fltptr).contents.value
            func(channelName, val)
        else:
            logger.warning(f"outvalueCallback: {channelName=} called with null pointer, skipping")

    def _setupCallbacks(self) -> None:
        assert self.started

        def _syncCallback(_, token):
            """ Called with outvalue __sync__, the value is put
            in gi__responses at token idx, then __sync__ is
            called with token to signal that a response is
            waiting. The value can be retrieved via self._responsesTable[token]
            """
            token = int(token)
            callback = self._responseCallbacks.get(token)
            if callback:
                callback(token)
                self._releaseToken(token)
                del self._responseCallbacks[token]

            else:
                logger.error(f"Unknown sync token: {token}")

        self.registerOutvalueCallback("__sync__", _syncCallback)
        self.csound.setOutputChannelCallback(self._outvalueCallback)

    def registerOutvalueCallback(self, chan: str, func: callback_t) -> None:
        """
        Set a callback to be fired when "outvalue" is used in csound

        Register a function ``func(channelname:str, newvalue: float) -> None``,
        which will be called whenever the given channel is modified via
        the "outvalue" opcode. Multiple functions per channel can be registered

        Args:
            chan: the name of a channel
            func: a function of the form ``func(chan:str, newvalue: float) -> None``

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

    def sync(self, timeout=0., force=False, threshold=2.) -> bool:
        """
        Block until csound has processed its immediate events

        Args:
            timeout: a timeout in seconds; 0. = use default timeout as defined
                in the configuration (TODO: add link to configuration docs)
            force: if True, sync even if not needed
            threshold: if time since last modification is longuer than this
                threshold assume that sync is not needed

        Returns:
            True if it synced, False if no sync was performed

        Raises TimeoutError if the sync operation takes too long

        Example
        ~~~~~~~

            >>> from csoundengine import *
            >>> e = Engine(...)
            >>> tables = [e.makeEmptyTable(size=1000) for _ in range(10)]
            >>> e.sync()
            >>> # do something with the tables
        """
        if not force and not self.needsSync():
            return False

        if self.version >= 7000:
            self._perfThread.flushMessageQueue()
        elif self._perfThread._processQueue:
            self._perfThread.flushProcessQueue()
        else:
            self.pingback(timeout=timeout)
        self._lastModification = 0
        return True

    def pingback(self, timeout: float = 0.):
        token = self._getSyncToken()
        self._eventWait(token, [self._builtinInstrs['pingback'], 0, 0, token], timeout=timeout)

    def _compileCode(self, code: str, block=False) -> None:
        if not block and self.udpPort and config['prefer_udp']:
            logger.debug("Sengind code via udp: \n{code}")
            self._udpSend(code)
            self._modified()
            return

        assert self.csound is not None
        if self.version >= 7000:
            self._perfThread.compileOrc(code)
            if block:
                self._perfThread.flushMessageQueue()
        elif self._perfThread._processQueue is None:
            self.csound.compileOrc(code, block=block)
        else:
            if not block:
                self._perfThread.processQueueTask(lambda cs: cs.compileOrc(code))
            else:
                q = _queue.SimpleQueue()
                self._perfThread.processQueueTask(lambda cs, q=q: q.put(cs.compileOrc(code)))
                err = q.get()
                if err:
                    logger.error("compileOrc error: ")
                    logger.error(internal.addLineNumbers(code))
                    raise CsoundError("Could not compile code")
        self._modified()

    def _modified(self, status=True) -> None:
        self._lastModification = time.time() if status else 0

    def needsSync(self, threshold=1.) -> bool:
        """
        True if ths engine has been modified

        Actions that modify an engine: code compilation, table allocation, ...
        """
        if self._lastModification == 0:
            return False
        if time.time() - self._lastModification > threshold:
            self._lastModification = 0
            return False
        return True

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
            code: the code to compile
            block: if True, this method will block until the code
                has been compiled

        Raises :class:`CsoundError` if the compilation failed

        .. note::

            If this instance has been started with a UDP port and
            the config option 'prefer_udp' is true, the code will be sent
            via udp. Otherwise the API is used. This might have an impact in
            the resulting latency of the operation, since using the API when
            running a performance thread can cause delays under certain
            circumstances

        Example
        ~~~~~~~

            >>> e = Engine()
            >>> e.compile("giMelody[] fillarray 60, 62, 64, 65, 67, 69, 71")
            >>> code = open("myopcodes.udo").read()
            >>> e.compile(code)

        """
        codeblocks = csoundparse.parseOrc(code, keepComments=False)
        for codeblock in codeblocks:
            if codeblock.kind == 'include':
                includepath = csoundparse.splitInclude(codeblock.text)
                if not os.path.exists(includepath):
                    logger.warning(f"Include path not found '{includepath}'")
                self.includes.append(includepath)
            elif codeblock.kind == 'instr':
                parsedbody = csoundparse.instrParseBody(csoundparse.instrGetBody(codeblock.lines))
                self._parsedInstrs[codeblock.name] = parsedbody
                self._instrRegistry[codeblock.name] = codeblock.text
                if codeblock.name[0].isdigit():
                    instrnum = int(codeblock.name)
                    for rangename, mininstr, maxinstr in self._reservedInstrnumRanges:
                        if mininstr <= instrnum < maxinstr:
                            logger.error(f"Instrument number {instrnum} is reserved. Code:")
                            logger.error("\n" + _textwrap.indent(codeblock.text, "    "))
                            raise ValueError(f"Cannot use instrument number {instrnum}, "
                                            f"the range {mininstr} - {maxinstr} is reserved for '{rangename}'")

                    if instrnum in self._reservedInstrnums:
                        raise ValueError("Cannot compile instrument with number "
                                        f"{instrnum}: this is a reserved instr and "
                                        f"cannot be redefined. Reserved instrs: "
                                        f"{sorted(self._reservedInstrnums)}")

        self._compileCode(code)

        names = [b.name for b in codeblocks if b.kind == 'instr' and b.name[0].isalpha()]
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

    def evalCode(self, code: str) -> float:
        """
        Evaluate code, return the result of the evaluation

        The expression to evaluate must be prefixed by ``return``

        Args:
            code (str): the code to evaluate. Usually an expression returning
                a float value (see example)

        Returns:
            the result of the evaluation

        Example
        ~~~~~~~

            >>> e = Engine()
            >>> e.compile(r'''
            ... instr foo
            ...   outch 1, oscili(0.1, p4)
            ... endin''')
            >>> e.evalCode('return nstrnum("foo")')

        """
        assert self.started and self.csound is not None
        if self.version >= 7000:
            out = self._perfThread.evalCode(code)
        elif self._useProcessQueue:
            q = _queue.SimpleQueue()
            self._perfThread.processQueueTask(lambda cs, q=q: q.put(cs.evalCode(code)))
            out = q.get()
        else:
            out = self.csound.evalCode(code)
        self._modified(False)
        return out

    def tableWrite(self, tabnum: int, idx: int, value: float, delay=0.) -> None:
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
            pargs = [self._builtinInstrs['tabwrite'], delay, 0, tabnum, idx, value]
            self._perfThread.scoreEvent(False, "i", pargs)

    def getTableData(self, idx: int, flat=False) -> np.ndarray:
        """
        Returns a numpy array pointing to the data of the table.

        Any modifications to this array will modify the table itself.

        .. note::

            Multichannel audio is loaded into a csound table as a flat
            array with samples interleaved.

        Args:
            idx: the table index
            flat: if True, the data will be returned as a flat (1D) array
                even if the table holds a multi-channel sample.

        Returns:
            a numpy array pointing to the data array of the table. Raises IndexError
            if the table was not found

        """
        arr: np.ndarray | None = self._tableCache.get(idx)
        if arr is not None:
            return arr

        if self.version >= 7000:
            # Accessing the table directly is faster than using requestCallback :-)
            arr = self.csound.table(idx)
            # q = _queue.SimpleQueue()
            # self._perfThread.requestCallback(lambda cs, q=q, idx=idx: q.put(cs.table(idx)))
            # arr = q.get()
        elif self._perfThread._processQueue is None:
            arr = self.csound.table(idx)
        else:
            q = _queue.SimpleQueue()
            self._perfThread.processQueueTask(lambda cs, q=q, idx=idx: q.put(cs.table(idx)))
            arr = q.get()
        if arr is None:
            raise ValueError(f"Table {idx} does not exist")

        if not flat:
            tabinfo = self.tableInfo(idx)
            if not tabinfo:
                raise IndexError(f"Table {idx} not found")
            if tabinfo.nchnls > 1:
                if tabinfo.size == tabinfo.numFrames*tabinfo.nchnls+1:
                    arr = arr[:-1]
                arr.shape = (tabinfo.numFrames, tabinfo.nchnls)
        self._tableCache[idx] = arr
        return arr

    def realElapsedTime(self, threshold=0.1) -> float:
        """
        Reports the elapsed time of the engine, independent of any locking

        Args:
            threshold: the reporting threshold. If this method is called multiple times
                during this time interval the engine time is extrapolated from the
                time reported by python and no call to csound is made

        Returns:
            the time elapsed since start of the engine.

        .. seealso:: :meth:`Engine.elapsedTime`
        """
        # assert self.csound is not None
        # return self.csound.currentTimeSamples() / self.sr

        reportedTime, lastTime = self._realElapsedTime
        now = time.time()
        if now - lastTime > threshold:
            reportedTime = self.csound.currentTimeSamples() / self.sr
            self._realElapsedTime = (reportedTime, now)
        else:
            reportedTime += now - lastTime
        return reportedTime

    def elapsedTime(self) -> float:
        """
        Returns the elapsed time since start of the engine

        This time is used as a reference when scheduling events. Since scheduling
        itself takes a small but not negligible amount of time, when scheduling
        a great number of events, these will fall out of sync. For this reason
        the elapsed time can be used as a reference to schedule events in
        absolute time. Moreover, the elapsed time stays unmodified
        as long as the engine's clock is locked for scheduling (see example)

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
            >>> for t in np.arange(0, 60, 0.2):
            ...     e.sched(100, t+now, 0.15, args=[1000], relative=False)
            ...     e.sched(100, t+now, 0.15, args=[800], relative=False)
            >>> # The same result can be achieved by locking the elapsed-time clock:
            >>> with e.lockedClock():
            ...     for t in np.arange(0, 10, 0.2):
            ...         e.sched(100, t, 0.15, args=[1000])
            ...         e.sched(100, t, 0.15, args=[800])

        """
        # _lockedELapsedTime will be a value if this engine is locked, or None otherwise
        return self._lockedElapsedTime or self.realElapsedTime()

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

        .. seealso:: :meth:`Engine.elapsedTime`, :meth:`Engine.lockedClock`

        """
        if lock:
            if self.isClockLocked():
                logger.debug("The elapsed time clock is already locked")
            else:
                assert self.csound is not None
                self._lockedElapsedTime = self.csound.currentTimeSamples()/self.sr
        else:
            if not self._lockedElapsedTime:
                logger.info("Asked to unlock the elapsed time clock, but it was not locked")
            self._lockedElapsedTime = 0

    def isClockLocked(self) -> bool:
        """Returns True if the clock is locked"""
        return self._lockedElapsedTime > 0

    def pushLock(self, latency: float | None = None):
        """
        Lock the clock of this engine

        Allows for recursive locking, so users do not need to
        see if what the current state of the lock is

        .. seealso:: :meth:`Engine.popLock`
        """
        islocked = self.isClockLocked()
        oldlatency = self.extraLatency
        self._clockStatesStack.append((islocked, oldlatency))

        if not islocked:
            self.lockClock(True)
        if latency is not None:
            self.extraLatency = latency
        logger.debug(f"Locking this engine's clock with latency {self.extraLatency}")

    def popLock(self):
        """
        Reverts the action of pushLock, unlocking the clock

        .. seealso:: :meth:`Engine.pushLock`
        """
        if not self._clockStatesStack:
            logger.warning("Clock stack is empty, ignoring")
            return

        waslocked, latency = self._clockStatesStack.pop()
        if not waslocked:
            self.lockClock(False)
        if latency is not None:
            self.extraLatency = latency

    def __enter__(self):
        self.pushLock()
        return self

    def __exit__(self, *args, **kws):
        self.popLock()

    def lockedClock(self, latency: float | None = None) -> Engine:
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
        return self

    def _presched(self, delay: float, relative: bool) -> tuple[bool, float]:
        """
        Args:
            delay: time offset
            relative: is this offset relative?

        Returns:
            tuple (isabsolute: bool, delay: float)
        """
        if relative:
            if (delay > self.onecycle * self._minCyclesForAbsoluteMode) or self._lockedElapsedTime:
                t0 = self.elapsedTime()
                delay = t0 + delay + self.extraLatency
                relative = False

        # 1 if absolute, 0 if relative
        return not relative, delay

    def sched(self,
              instr: int | float | str,
              delay=0.,
              dur=-1.,
              *pfields,
              args: np.ndarray | Sequence[float | str] = (),
              relative=True,
              unique=False,
              **namedpfields
              ) -> float:
        """
        Schedule an instrument

        Args:
            instr : the instrument number/name. If it is a fractional number,
                that value will be used as the instance number.
                An integer or a string will result in a unique instance assigned
                by csound if unique is True. Named instruments
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

        .. seealso:: :meth:`~csoundengine.engine.Engine.unschedAll`
        """
        assert self.started

        isabsolute, delay = self._presched(delay=delay, relative=relative)

        if self.autosync and self.needsSync():
            self.sync()

        if pfields and args:
            raise ValueError("Either pfields as positional arguments or args can be given, "
                             "got both")
        elif pfields:
            args = pfields

        if namedpfields:
            instrdef = self._parsedInstrs.get(instr if isinstance(instr, str) else str(int(instr)))
            if instrdef:
                kwargs = csoundparse.normalizeNamedPfields(namedpfields, instrdef.pfieldNameToIndex)
            else:
                assert all(csoundparse.isPfield(key) for key in namedpfields)
                kwargs = {int(key[1:]):value for key, value in namedpfields.items()}
            args = csoundparse.fillPfields(args, kwargs, defaults=instrdef.pfieldIndexToValue if instrdef else None)

        if isinstance(instr, int):
            instrfrac = self._assignEventId(instr) if unique else instr
        elif isinstance(instr, float):
            if unique and int(instr) == instr:
                instrfrac = self._assignEventId(int(instr))
            else:
                instrfrac = instr
        elif isinstance(instr, str):
            if not unique and "." not in instr:
                instrnum = self._instrNumCache.get(instr)
                if not instrnum:
                    msg = f'i {instr} {delay} {dur}'
                    if args:
                        msg += ' '.join(map(str, args))
                    self._perfThread.inputMessage(msg)
                    return 0
                instrfrac = instrnum
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
                    return 0
            else:
                instrfrac = self._assignEventId(instr)
        else:
            raise TypeError(f"Expected a float, an int or a str as instr, "
                            f"got {instr} (type {type(instr)})")
        if isinstance(args, np.ndarray):
            pargsnp = np.empty((len(args)+3,), dtype=float)
            pargsnp[0] = instrfrac
            pargsnp[1] = delay
            pargsnp[2] = dur
            pargsnp[3:] = args
            # 1: we use always absolute time
            self._perfThread.scoreEvent(isabsolute, "i", pargsnp)
        elif not args:
            pargs = [instrfrac, delay, dur]
            self._perfThread.scoreEvent(isabsolute, "i", pargs)
        elif isinstance(args, (list, tuple)):
            pargs = [instrfrac, delay, dur]
            pargs.extend(float(a) if not isinstance(a, str) else self.strSet(a) for a in args)
            #if any(isinstance(a, str) and a not in self._strToIndex for a in args):
            #    self.sync()
            self._perfThread.scoreEvent(isabsolute, "i", pargs)
        else:
            raise TypeError(f"Expected a sequence or array, got {args}")
        return instrfrac

    def _queryNamedInstrs(self, names: list[str], timeout=0.1, callback=None, delay=0.
                          ) -> None:
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
            callback: a func of the form `func(name2instr:dict[str, int])` which will be
                called when all instrs have an assigned instr number
                called for
        """
        if not timeout or callback:
            # query async
            if not callback:
                for name in names:
                    self._queryNamedInstrAsync(name, delay=delay)
            else:
                results: dict[str, int] = {}
                def mycallback(name, instrnum, n=len(names), results=results, callback=callback):
                    results[name] = instrnum
                    if len(results) == n:
                        callback(results)
                for name in names:
                    self._queryNamedInstrAsync(name, delay=delay, callback=mycallback)
        else:
            # blocking
            q = _queue.SimpleQueue()
            def _func(csound, names=names, q=q):
                nums = []
                for name in names:
                    num = csound.evalCode(f'return nametoinstrnum "{name}"')
                    nums.append(num)
                q.put(nums)
            self._perfThread.requestCallback(_func)
            nums = q.get()
            for num, name in zip(nums, names):
                if num > 0:
                    self._instrNumCache[name] = int(num)
                    if (body := self._instrRegistry.get(name)) is not None:
                        self._instrRegistry[num] = body

    def _queryNamedInstrAsync(self, name: str, delay=0., callback=None) -> None:
        """
        Query the assigned instr number async

        The result is put in the cache and, if given, callback is called
        as `callback(name: str, instrnum: int)`
        """
        if self.version >= 7000:
            def func7(cs, f=callback, name=name):
                num = int(cs.evalCode(f'return nametoinstrnum:i("{name}")'))
                self._instrNumCache[name] = num
                if f:
                    f(name, num)
            self._perfThread.requestCallback(func7)
        else:
            synctoken = self._getSyncToken()
            msg = f'i {self._builtinInstrs["nstrnum"]} {delay} 0 {synctoken} "{name}"'
            def _callback(synctoken, instrname=name, func=callback):
                instrnum = int(self._responsesTable[synctoken])
                self._instrNumCache[instrname] = instrnum
                if (body := self._instrRegistry.get(instrname)):
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
                be called when the result is available. Callback is of
                the form ``func(instrname: str, instrnum: int) -> None``. If no
                callback is given this method blocks until the instrument number
                is returned

        Returns:
            the instr number if called without callback, 0 otherwise. If the instrument was
            not found (either because it was never compiled or the compilation is not ready yet)
            -1 will be returned
        """
        if cached and (instrnum := self._instrNumCache.get(instrname, 0)) > 0:
            if callback:
                callback(instrname, instrnum)
            return instrnum
        if callback:
            self._queryNamedInstrAsync(instrname, delay=0, callback=callback)
            return 0
        # Block!
        if self.version >= 7000 or self._perfThread._processQueue is not None:
            instrnum = int(self._perfThread.evalCode(f'return nametoinstrnum:i("{instrname}")'))
        else:
            token = self._getSyncToken()
            msg = f'i {self._builtinInstrs["nstrnum"]} 0 0 {token} "{instrname}"'
            out = self._inputMessageWait(token, msg)
            if out is None or out <= 0:
                raise RuntimeError(f"Could not query the instrument number for '{instrname}'")
            instrnum = int(out)
        self._instrNumCache[instrname] = instrnum
        return instrnum

    def print(self, msg: str, delay=0.) -> None:
        """
        Print a message via csound

        Args:
            msg: the message to print
            delay: when to print it
        """
        instrnum = self._builtinInstrs['print']
        self._perfThread.inputMessage(f'i {instrnum} {delay} 0. "{msg}"')

    def unsched(self, p1: float | str, delay: float = 0, future=False) -> None:
        """
        Stop a playing event

        If p1 is a round number, all events with the given number
        are unscheduled. Otherwise only an exact matching event
        is unscheduled, if it exists

        Args:
            p1: the instrument number/name to stop
            delay: if 0, remove the instance as soon as possible
            future: if True, unsched an event in the future

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
        if future:
            mode = -1
        elif (isinstance(p1, float) and int(p1) != p1) or (isinstance(p1, str) and "." in p1):
            mode = 4
        else:
            mode = 0
        pfields = [self._builtinInstrs['turnoff'], delay, 0, p1, mode]
        self._perfThread.scoreEvent(False, "i", pfields)

    def unschedAll(self) -> None:
        """
        Remove all playing and future events

        .. seealso:: :meth:`~csoundengine.engine.Engine.unsched`, :meth:`~csoundengine.engine.Engine.unschedFuture`
        """
        assert self.csound is not None
        self.csound.rewindScore()
        self._setupGlobalInstrs()

    def session(self,
                priorities: int | None = None,
                maxControlsPerInstr: int | None = None,
                numControlSlots: int | None = None
                ) -> _session.Session:
        """
        Return the Session corresponding to this Engine

        Since each Engine can have only one associated Session,
        the parameters passed are only valid for the creation of
        the Session. Any subsequent call to this method returns the
        already created Session, and the arguments passed are not
        taken into consideration.

        Args:
            priorities: the max. number of priorities for scheduled instrs
            numControlSlots: the total number of slots allocated for
                dynamic args. The default is determined by the config
                'dynamic_args_num_slots'
            maxControlsPerInstr: the max. number of dynamic args per instr
                (the default is set in the config 'max_dynamic_args_per_instr')

        Returns:
            the corresponding Session

        Example
        ~~~~~~~

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
            self._session = Session(engine=self,
                                    priorities=priorities,
                                    numControlSlots=numControlSlots,
                                    maxControlsPerInstr=maxControlsPerInstr)
        else:
            if maxControlsPerInstr is not None and maxControlsPerInstr != self._session.maxDynamicArgs:
                logger.info(f"Asking to create a session with dynamicArgsPerInstr={maxControlsPerInstr}, "
                            f"which differs from the value of the current session "
                            f"({self._session.maxDynamicArgs}). The old value will be kept")
            if numControlSlots is not None and numControlSlots != self._session._dynargsNumSlots:
                logger.info(f"Asking to create a session with dynamicArgsSlices={numControlSlots}, "
                            f"which differs from the value of the current session "
                            f"({self._session._dynargsNumSlots}). The old value will be kept")
            if priorities is not None and priorities != self._session.numPriorities:
                logger.info(f"Asking to create a session with priorites={priorities}, "
                            f"which differs from the value of the current session "
                            f"({self._session.numPriorities}). The old value will be kept")
        return self._session

    def _reserveInstrRange(self, name: str, mininstrnum: int, maxinstrnum: int) -> None:
        """
        Declares the instrument numbers in the given range as reserved

        Instrument numbers within this range will not be allocated when using
        named instruments. This is used internally by a :class:`~csoundengine.session.Session`
        to reserve instrument numbers to be assigned to the different priorities.


        Args:
            name: the name of the reserved block
            mininstrnum: lowest instrument number to reserve
            maxinstrnum: highest instrument number to reserve (not included in the range)

        """
        self._reservedInstrnumRanges.append((name, mininstrnum, maxinstrnum))

    def makeEmptyTable(self, size, numchannels=1, sr=0, delay=0., block=False
                       ) -> int:
        """
        Create an empty table, returns the index of the created table

        Args:
            size: the size of the table
            numchannels: if the table will be used to hold audio, the
                number of channels of the audio
            sr: the samplerate of the audio, if the table is used to hold audio
            delay: when to create the table


        Example
        ~~~~~~~

        Use a table as an array of buses

        >>> from csoundengine import *
        >>> engine = Engine()
        >>> source = engine.makeEmptyTable(128)
        >>> engine.compile(r'''
        ... instr 100
        ...   imidi = p4
        ...   iamptab = p5
        ...   islot = p6
        ...   kamp table islot, iamptab
        ...   asig = oscili:a(interp(kamp), mtof(imidi))
        ...   outch 1, asig
        ... endin
        ... ''')
        >>> tabarray = engine.getTableData(source)
        >>> tabarray[0] = 0.5
        >>> eventid = engine.sched(100, args=[67, source, 0])

        .. seealso::

            :meth:`~Engine.makeTable`
            :meth:`~Engine.fillTable`
            :meth:`~Engine.automateTable`

        """
        if block and delay > 0:
            raise ValueError(f"Either block=True or delay > 0, got {block=}, {delay=}")
        tabnum = self._assignTableNumber()
        self._tableInfo[tabnum] = TableInfo(sr=sr, size=size, nchnls=numchannels)
        if block or self.version >= 7000:
            if sr == 0 and numchannels == 1:
                self._perfThread.scoreEvent(False, "f", [tabnum, delay, -size, -2, 0])
                if block:
                    self._perfThread.flushMessageQueue()
            else:
                self._compileCode(fr'''
                    i__tabnum__ ftgen {tabnum}, 0, {size}, -2, 0
                    ftsetparams {tabnum}, {sr}, {numchannels}
                    ''', block=block)
        else:
            if sr == 0:
                self._perfThread.scoreEvent(False, "f", [tabnum, delay, -size, -2, 0])
            else:
                itoken = 0  # we don't need a notification
                iempty = 1
                self._perfThread.scoreEvent(False, "i", [self._builtinInstrs['maketable'], delay,
                                            0, itoken, tabnum, size, iempty, sr, numchannels])
            self._modified()
        return tabnum

    def makeTable(self,
                  data: Sequence[float] | np.ndarray,
                  sr: int = 0,
                  tabnum: int = -1,
                  block=True,
                  callback=None,
                  ) -> int:
        """
        Create a new table and fill it with data.

        Args:
            data: the data used to fill the table
            tabnum: the table number. If -1, a number is assigned by the engine.
            block: wait until the table is actually created
            callback: call this function when ready - f(token, tablenumber) -> None
            sr: only needed if filling sample data. If given, it is used to fill the
                table metadata in csound, as if this table had been read via gen01

        Returns:
            the index of the new table, if wait is True

        Example
        ~~~~~~~

        .. code-block:: python

            from csoundengine import *
            e = Engine()
            import sndfileio
            sample, sr = sndfileio.sndread("stereo.wav")
            # modify the sample in python
            sample *= 0.5
            source = e.makeTable(sample, sr=sr, block=True)
            e.playSample(source)

        .. seealso:: :meth:`~csoundengine.engine.Engine.readSoundfile`, :meth:`~csoundengine.engine.Engine.fillTable`
        """
        dataarr = np.asarray(data)
        nchnls = internal.arrayNumChannels(dataarr)
        numitems = len(dataarr) * nchnls
        flatdata = dataarr if nchnls == 1 else dataarr.flatten()
        if tabnum == -1:
            tabnum = self._assignTableNumber()
        elif tabnum == 0:
            tabnum = self._makeTableNotify(data=data, tabnum=0, sr=sr, numchannels=nchnls)
            self._tableInfo[tabnum] = TableInfo(sr=sr, size=len(data), nchnls=nchnls)
            self._modified()
            return tabnum

        self._tableCache.pop(int(tabnum), None)
        if self.version < 7000 and not self._perfThread._processQueue:
            if callback:
                self._makeTableNotify(data=data, sr=sr, tabnum=tabnum, callback=callback)
            elif not block:
                self._makeTableAsync(tabnum=tabnum, data=flatdata, sr=sr, numchannels=nchnls)
            else:
                # block
                q = _queue.SimpleQueue()
                self._makeTableNotify(data=data, sr=sr, tabnum=tabnum, callback=lambda q=q: q.put(True))
                _ = q.get()
        else:
            ev = _threading.Event() if block else None
            def task(cs, func=callback, ev=ev):
                lines = [f'i__tabnum ftgen {tabnum}, 0, {numitems}, -2, 0']
                if sr > 0 or nchnls > 1:
                    # ftsetparams ift, isamplerate, inumchannels, iloopstart=0, ibasenote=60
                    lines.append(f'ftsetparams {tabnum}, {sr}, {nchnls}')
                cs.compileOrc('\n'.join(lines))
                tabptr = cs.table(tabnum)
                if tabptr is None:
                    raise RuntimeError(f"Could not create csound table {tabnum}")
                self._tableCache[tabnum] = tabptr
                tabptr[:] = flatdata
                if func:
                    func(tabnum)
                elif ev:
                    ev.set()
            if lcs.VERSION >= 7000:
                self._perfThread.requestCallback(task)
            else:
                self._perfThread.processQueueTask(task)
            if ev:
                ev.wait()
        self._tableInfo[tabnum] = TableInfo(sr=sr, size=len(data), nchnls=nchnls)
        self._modified()
        assert tabnum > 0
        return tabnum

    def _makeTableAsync(self, tabnum: int, data: np.ndarray, sr=0, numchannels=1, delay=0.):
        """
        Create a table async, without process queue or csound 7's requestCallback
        """
        assert len(data.shape) == 1, "Only 1D arrays accepted here"
        arr = np.zeros((len(data)+4,), dtype=float)
        arr[0:4] = [tabnum, delay, len(data), -2]
        arr[4:] = data
        self._perfThread.scoreEvent(False, "f", arr)
        if sr > 0 or numchannels > 1:
            pargs = [self._builtinInstrs['ftsetparams'], delay, 0., tabnum, sr, numchannels]
            self._perfThread.scoreEvent(False, "i", pargs)

    def tableExists(self, tabnum: int) -> bool:
        """
        Returns True if a table with the given number exists
        """
        # TODO: really check if the table exists
        try:
            _ = self.tableInfo(tabnum)
        except TableNotFoundError:
            return False
        return True

    def setTableMetadata(self, tabnum: int, sr: int, numchannels=1, delay=0.) -> None:
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
                the table was not created via the engine
            delay: when to perform the operation. This is useful if setting the metadata
                for a table not created yet
        """
        logger.debug(f"Setting table metadata. {tabnum=}, {sr=}, {numchannels=}")
        if delay == 0:
            self._compileCode(f"ftsetparams {tabnum}, {sr}, {numchannels}")
        else:
            pargs = [self._builtinInstrs['ftsetparams'], delay, 0., tabnum, sr, numchannels]
            self._perfThread.scoreEvent(False, "i", pargs)
            # self._modified()
        tabinfo = self._tableInfo.get(tabnum)
        if tabinfo:
            tabinfo.sr = sr
            tabinfo.nchnls = numchannels
        else:
            logger.debug(f"setTableMetadata: table {tabnum} was not created via python")

    def _registerSync(self, token: int) -> _queue.SimpleQueue:
        """
        Register a token for a __sync__ callback

        Args:
            token: an int token, as returned via _getSyncToken

        Returns:
            a Queue. It can be used to wait on, the value obtained
            will be the returned value
        """
        q: _queue.SimpleQueue[float] = _queue.SimpleQueue()
        self._responseCallbacks[token] = lambda token, q=q, table=self._responsesTable: q.put(float(table[token]))
        return q

    def callLater(self, delay: float, callback: Callable) -> None:
        """
        Call callback after delay, triggered by csound scheduler

        Args:
            delay: the delay time, in seconds
            callback: the callback, a function of the sort ``() -> None``

        The callback will be called after the given delay, plus some jitter
        (~ 2/3 k-cycles after, never before)

        Example
        ~~~~~~~

        >>> from csoundengine import *
        >>> import time
        >>> e = Engine()
        >>> start = time.time()
        >>> e.callLater(2, lambda:print(f"Elapsed time: {time.time() - start:.4f}"))
        2.0018
        """
        if delay == 0:
            callback()
        else:
            token = self._getSyncToken()
            pargs = [self._builtinInstrs['pingback'], delay, 0.01, token]
            self._eventWithCallback(token, pargs, lambda token: callback())

    def _eventWait(self, token: int, pargs: Sequence[float], timeout: float = 0.
                   ) -> float | None:
        if not timeout:
            timeout = config['timeout']
        else:
            assert timeout > 0
        q = self._registerSync(token)
        self._perfThread.scoreEvent(False, "i", pargs)
        try:
            outvalue = q.get(block=True, timeout=timeout)
            self._modified(False)
            return outvalue if outvalue != _UNSET else None
        except _queue.Empty:
            raise TimeoutError(f"{token=}, {pargs=}")

    def plotTableSpectrogram(self,
                             tabnum: int,
                             fftsize=2048,
                             mindb=-90,
                             maxfreq: int = 0,
                             overlap: int = 4,
                             minfreq: int = 0,
                             sr: int = 44100,
                             chan=0
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
        if internal.arrayNumChannels(data) > 1:
            data = data[:, chan]
        tabinfo = self.tableInfo(tabnum)
        assert tabinfo is not None
        if tabinfo.sr > 0:
            sr = tabinfo.sr
        plotting.plotSpectrogram(data, sr, fftsize=fftsize, mindb=mindb,
                                 maxfreq=maxfreq, minfreq=minfreq, overlap=overlap,
                                 show=True)

    def plotTable(self, tabnum: int, sr: int = 0) -> Figure:
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
            axes: if given, the plot is done to this axes

        .. code::

            from csoundengine import *
            e = Engine()
            source = e.readSoundfile("mono.wav", block=True)
            # no sr needed here since the output was rea via readSoundfile
            e.plotTable(source)

        .. figure:: assets/tableproxy-plot.png

        .. code::

            import matplotlib.pyplot as plt
            import sndfileio
            fig, ax = plt.subplots()
            data, sr = sndfileio.sndread("stereo.wav")
            tabnum2 = e.makeTable(data, sr=sr)
            e.plotTable(tabnum2, axes=ax)

        .. figure:: assets/tableplot-stereo.png

        .. code::

            e = Engine()
            xs = np.linspace(0, 6.28, 1000)
            ys = np.sin(xs)
            source = e.makeEmptyTable(len(ys))
            e.fillTable(source, data=ys)
            e.plotTable(source)

        .. image:: assets/tableplot-sine.png

        .. seealso::

            * :meth:`~csoundengine.engine.Engine.readSoundfile`
            * :meth:`~csoundengine.engine.Engine.makeTable`
        """
        from csoundengine import plotting
        assert isinstance(tabnum, int) and tabnum > 0
        data = self.getTableData(tabnum, flat=False)
        tabinfo = self.tableInfo(tabnum)
        if not sr and tabinfo is not None and tabinfo.sr > 0:
            sr = tabinfo.sr

        if data is None:
            raise ValueError(f"Table {tabnum} is invalid")
        if sr:
            return plotting.plotSamples(data, samplerate=sr, show=True)
        else:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots()
            axes.plot(data)
            fig.tight_layout(pad=0.1)
            if not plotting.matplotlibIsInline():
                plt.show()
            return fig

    def schedSync(self,
                  instr: int | float | str,
                  delay: float = 0,
                  dur: float = -1,
                  args: np.ndarray | Sequence[float | str] | None = None,
                  timeout=-1
                  ) -> tuple[float, float | None]:
        """
        Schedule an instr, wait for a sync message

        Similar to :meth:`~Engine.sched` but waits until the instrument sends a
        sync message. The instrument should expect a sync token at p4 (see example)
        and send an answer via 'sendsync' (``sendsync itoken, ivalue=0``)

        .. note::

            args should start with p5 since the sync token is sent as p4

        Args:
            instr: the instrument number/name. If it is a fractional number,
                that value will be used as the instance number.
            delay: time to wait before instrument is started
            dur: duration of the event
            args: any other args expected by the instrument, starting with p5
                (as a list of floats/strings, or as a numpy float array). Any
                string arguments will be converted to a string index via strSet. These
                can be retrieved via strget in the csound instrument
            timeout: if a non-negative number is given, this function will block
                at most this time and then raise a TimeoutError

        Returns:
            the fractional p1 of the scheduled note, the sync return value (see example) or
            None if the instrument does not return any value. Raises TimeoutError
            if the operation times out

        Example
        ~~~~~~~

            >>> from csoundengine import *
            >>> e = Engine()
            >>> e.compile(r'''
            ... instr readsound
            ...   itoken = p4
            ...   Spath strget p5
            ...   itab ftgen ftgen 0, 0, 0, -1, Spath, 0, 0, 0
            ...   sendsync(itoken, itab)
            ...   turnoff
            ... endin
            ... ''')
            >>> eventid, tabnum = e.schedSync('readsound', args=['/path/to/sound.wav'])

        """
        assert self.started
        instrfrac = instr if isinstance(instr, float) else self._assignEventId(instr)
        token = self._getSyncToken()
        q = self._registerSync(token)
        if not args:
            pargs = [instrfrac, delay, dur, token]
            self._perfThread.scoreEvent(False, "i", pargs)
        elif isinstance(args, np.ndarray):
            pargsnp = np.empty((len(args)+4,), dtype=float)
            pargsnp[0] = instrfrac
            pargsnp[1] = delay
            pargsnp[2] = dur
            pargsnp[3] = token
            pargsnp[4:] = args
            self._perfThread.scoreEvent(False, "i", pargsnp)
        else:
            pargs = [instrfrac, delay, dur, token]
            pargs.extend(a if not isinstance(a, str) else self.strSet(a) for a in args)
            self._perfThread.scoreEvent(False, "i", pargs)

        try:
            outvalue = q.get(block=True, timeout=timeout)
            if outvalue == _UNSET:
                outvalue = None
            return instrfrac, outvalue
        except _queue.Empty:
            raise TimeoutError(f"{token=}, {instr=}")

    def _eventWithCallback(self, token: int, pargs, callback) -> None:
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
        assert token == pargs[3] and isinstance(token, int)
        self._responseCallbacks[token] = lambda tok, t=self._responsesTable, c=callback: c(t[tok])
        self._perfThread.scoreEvent(False, "i", pargs)
        return None

    def _inputMessageWait(self,
                          token: int,
                          inputMessage: str,
                          timeout: float | None = None
                          ) -> float | None:
        """
        This function passes the str `inputMessage` to csound and waits for
        the instr to notify back via a "__sync__" outvalue

        If the instr returned a value via gi__responses, this value
        is returned. Otherwise, None is returned

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
            self._modified(False)
            return value if value != _UNSET else None
        except _queue.Empty:
            raise TimeoutError(f"{token=}, {inputMessage=}")

    def _inputMessageWithCallback(self, token:int, inputMessage:str, callback) -> None:
        """
        This function passes the str inputMessage to csound and before that
        sets a callback waiting for an outvalue notification. If no callback
        is passed the function will block until the instrument notifies us

        Args:
            token: a sync token, assigned via _getSyncToken
            inputMessage: the input message passed to csound
            callback: if given, this function will be called when the instrument
                notifies us via `outvalue "__sync__", token`. The callback should
                be of kind `(token:int) -> None`
        """
        self._responseCallbacks[token] = callback
        self._perfThread.inputMessage(inputMessage)

    def _makeTableNotify(self,
                         data: Sequence[float] | np.ndarray | None = None,
                         size=0,
                         tabnum=0,
                         callback=None,
                         sr: int = 0,
                         numchannels=1) -> int:
        """
        Create a table with data (or an empty table of the given size).

        Lets csound generate a table index if needed.

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
            the table number
        """
        token = self._getSyncToken()
        maketableInstrnum = self._builtinInstrs['maketable']
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
            # Create table with data
            if not isinstance(data, np.ndarray):
                data = np.asarray(data)
            numchannels = internal.arrayNumChannels(data)
            numitems = len(data) * numchannels
            if numchannels > 1:
                data = data.flatten()
                # data = data.ravel()
            if numitems < 1900:
                # create a table with the given data
                # if the table is small we can create it and fill it in one go
                empty = 0
                numchannels = internal.arrayNumChannels(data)
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
                    response = self._eventWait(token, pargs)
                    if response is None:
                        raise RuntimeError(f"Failed to make table with args: {pargs}")
                    tabnum = int(response)
                    self.fillTable(tabnum, data=data)
                    self._tableInfo[tabnum] = TableInfo(sr=sr, size=numitems,
                                                        nchnls=numchannels)
                else:
                    def callback2(tabnum, self=self, data=data, callback=callback):
                        self.fillTable(tabnum, data=data)
                        callback()
                    self._eventWithCallback(token, pargs, callback2)
                return tabnum
        if callback:
            self._eventWithCallback(token, pargs, callback)
        else:
            response = self._eventWait(token, pargs)
            if response is None:
                raise RuntimeError(f"Failed to create table with args {pargs}")
            tabnum = int(response)
            assert tabnum > 0
        self._tableInfo[tabnum] = TableInfo(sr=sr, size=size, nchnls=numchannels)
        return tabnum

    def channelPointer(self, channel: str, kind='control', mode='rw') -> np.ndarray:
        """
        Get a pointer to a csound channel

        Args:
            channel: the name of the channel
            kind: one of 'control' or 'audio'
            mode: one of 'r' (input), 'w' (output), 'rw' (input/output)

        Returns:
            the pointer to the data as a numpy array
        """
        if kind != 'control' and kind != 'audio':
            raise NotImplementedError("Only kind 'control' and 'audio' are implemented at the moment")
        assert self.csound is not None
        ptr = self._channelPointers.get(channel)
        if ptr is None:
            ptr, err = self.csound.channelPtr(channel, kind=kind, mode=mode)
            if err:
                raise RuntimeError(f"Error while trying to retrieve/create a channel pointer: {err}")
            assert isinstance(ptr, np.ndarray)
            self._channelPointers[channel] = ptr
        return ptr

    def setChannel(self, channel: str, value: float | str | np.ndarray,
                   method='', delay=0.
                   ) -> None:
        """
        Set the value of a software channel (threadsafe)

        Args:
            channel: the name of the channel
            value: the new value, should match the type of the channel (a float for
                a control channel, a string for a string channel or a numpy array
                for an audio channel)
            method: one of ``'api'``, ``'score'``, ``'udp'``. An empty str will choose the most appropriate
                method for the current engine/args
            delay: a delay to set the channel

        Example
        ~~~~~~~

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
        if delay > 0:
            method = "score"
        elif isaudio or not method:
            method = "api"

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
                s = f'i {instrnum} {delay} 0 "{channel}" {value}'
                self._perfThread.inputMessage(s)
            else:
                instrnum = self._builtinInstrs['chnsets']
                s = f'i {instrnum} {delay} 0 "{channel}" "{value}"'
                self._perfThread.inputMessage(s)
            self._modified()
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
                             f"(choices: 'api', 'score', 'udp')")

    def initChannel(self,
                    channel: str,
                    value: float | str | np.ndarray = 0,
                    kind='',
                    mode="r") -> None:
        """
        Create a channel and set its initial value

        Args:
            channel: the name of the channel
            value: the initial value of the channel,
                will also determine the type (k, a, S)
            kind: One of 'k', 'S', 'a'. Use None to auto determine the channel type.
            mode: r for read, w for write, rw for both.

        .. note::
                the `mode` is set from the perspective of csound. A read (input)
                channel is a channel which can be written to by the api and read
                from csound. An write channel (output) can be written by csound
                and read from the api

        Example
        ~~~~~~~

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
        if not kind:
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

    def getControlChannel(self, channel: str) -> float:
        """
        Get the value of a channel

        Args:
            channel: the name of the channel

        Returns:
            the value of the channel. Raises KeyError if the channel
            does not exist.

        Example
        ~~~~~~~

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
            raise KeyError(f"control channel {channel} not found, error: {errorCode}, value: {value}")
        return value

    def fillTable(self, tabnum: int, data: Sequence[float] | np.ndarray) -> None:
        """
        Fill an existing table with data

        This is a blocking operation

        Args:
            tabnum: the table number
            data: the data to put into the table
            block: if True, block while performing the operation

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

        .. seealso::

            * :meth:`~Engine.makeTable`
            * :meth:`~Engine.plotTable`
            * :meth:`~Engine.readSoundfile`
        """
        assert self.csound is not None
        if not isinstance(tabnum, int) or tabnum <= 0:
            raise ValueError(f"tabnum should be an int > 0, got {tabnum}")

        arr = data if isinstance(data, np.ndarray) else np.asarray(data)
        if len(arr.shape) == 2:
            arr = arr.flatten()
        elif len(arr.shape) > 2:
            raise ValueError(f"data should be a 1D or 2D array, got shape {arr.shape}")

        numpyptr = self.getTableData(tabnum)
        if numpyptr is None:
            raise IndexError(f"Table {tabnum} does not exist")
        size = len(numpyptr)
        numpyptr[:] = data if size < len(data) else data[:size]

    def tableInfo(self, tabnum: int, cache=True) -> TableInfo | None:
        """
        Retrieve information about the given table

        Args:
            tabnum: the table number
            cache: if True, query the cache to see if info for this table
                has already been requested

        Returns:
            a TableInfo with fields `tableNumber`, `sr` (``ftsr``),
            `numChannels` (``ftchnls``), `numFrames` (``nsamps``),
            `size` (``ftlen``). Returns None if the table was not found

        .. note::

            Some information, like *sr*, is only available for tables
            allocated via ``GEN01`` (for example, using :meth:`~Engine.readSoundfile`).
            This data can also be set explicitely via :meth:`~Engine.setTableMetadata`

        .. rubric:: Example

        .. code-block:: python

            >>> from csoundengine import *
            >>> e = Engine()
            >>> source = e.readSoundfile("stereo.wav", block=True)
            >>> e.tableInfo(source)
            TableInfo(tableNumber=200, sr=44100.0, numChannels=2, numFrames=88200, size=176401)

        .. seealso::

            * :meth:`~Engine.readSoundfile`
            * :meth:`~Engine.plotTable`
            * :meth:`~Engine.getTableData`

        """
        info = self._tableInfo.get(tabnum)
        if info and cache:
            if info.path and info.sr == 0 and info.size == 0:
                # Table created via .readSoundFile, info only has the
                # path of the soundfile
                info = TableInfo.get(info.path)
                self._tableInfo[tabnum] = info
            return info
        toks = [self._getSyncToken() for _ in range(4)]
        pargs = [self._builtinInstrs['tableInfo'], 0, 0., tabnum]
        pargs.extend(toks)
        q: _queue.SimpleQueue[list[float]] = _queue.SimpleQueue()

        def callback(tok0, _q=q, t=self._responsesTable, _toks=toks):
            values = [float(t[_tok]) for _tok in _toks]
            _q.put(values)

        self._responseCallbacks[toks[0]] = callback
        self._perfThread.scoreEvent(False, "i", pargs)
        vals = q.get(block=True)
        for tok in toks:
            self._releaseToken(tok)
        sr = vals[0]
        if sr <= 0:
            return None
        tabinfo = TableInfo(sr=int(vals[0]), nchnls=int(vals[1]),size=int(vals[3]))
        self._tableInfo[tabnum] = tabinfo
        return tabinfo

    def includeFile(self, include: str) -> None:
        """
        Add an #include file to this Engine

        Args:
            include: the path to the include file
        """
        abspath = os.path.abspath(include)
        for f in self.includes:
            if abspath == f:
                return
        self.includes.append(abspath)
        code = open(abspath).read()
        self.compile(code)

    def readSoundfile(self, path="?", tabnum: int | None = None, chan=0,
                      callback=None, block=False, skiptime=0.
                      ) -> int:
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
            skiptime: time to skip at the beginning of the soundfile.

        Returns:
            the index of the created table

        >>> from csoundengine import *
        >>> engine = Engine()
        >>> source = engine.readSoundfile("stereo.wav", block=True)
        >>> eventid = engine.playSample(source)
        >>> # Reduce the gain to 0.8 after 2 seconds
        >>> engine.setp(eventid, 4, 0.8, delay=2)

        .. seealso::

            * :meth:`~Engine.playSample`
            * :meth:`~Engine.makeTable`
            * :meth:`~Engine.fillTable`
            * :meth:`~Engine.getTableData`

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

        if block and tabnum:
            self._compileCode(f'i__tab__ ftgen {tabnum},0,0,-1, "{path}", {skiptime}, 0, {chan}', block=True)
        else:
            token = self._getSyncToken() if (block or callback) else 0
            p1 = self._builtinInstrs['readSndfile']
            msg = f'i {p1} 0 0.01 {token} "{path}" {tabnum} {chan} {skiptime}'
            if callback:
                self._inputMessageWithCallback(token, msg, lambda *args: callback())
            elif block:
                self._inputMessageWait(token, msg)
            else:
                self._perfThread.inputMessage(msg)
        self._tableInfo[tabnum] = TableInfo(sr=0, size=0, nchnls=1, path=path)
        return tabnum

    def soundfontPlay(self, index: int, pitch: float, amp=0.7, delay=0.,
                      dur=-1., vel: int = 0, chan=1
                      ) -> float:
        """
        Play a note of a previously loaded soundfont

        The soundfont's program (bank, preset) must have been read before
        via :meth:`Engine.soundfontPreparePreset`

        Args:
            index: as returned via :meth:`~Engine.soundfontPrearePreset`
            pitch: the pitch of the played note, as a midinote (can
                be fractional)
            amp: the amplitude. If vel (velocity) is left as None, this
                is used to determine the velocity. Otherwise, set the velocity
                (this might  be used by the soundfont to play the correct sample)
                and the amplitude is used to scale the output
            vel: the velocity of the played note, used internally to determine
                which sample/layer to play. If not given, a velocity is calculated
                from the given amplitude
            chan: the first channel to send output to (channels start with 1)
            delay: when to start playback
            dur: the duration of playback. Use -1 to play until the end
                (the note will be stopped when the soundfont playback detects the
                end of the sample)

        Returns:
            the instance number of the playing instrument.

        .. important::

            **Dynamic Fields**

            - **p4**: `kpitch`
            - **p5**: `kamp`

        Example
        ~~~~~~~

        .. code::

            from csoundengine import *
            e = Engine()

            # Since the preset is not specified, this will launch a gui dialog
            # to select a preset from a list of available presets
            idx = e.soundfontPreparePreset('Orgue-de-salon.sf2')
            event = e.soundfontPlay(idx, 60)

            # Automate kpitch (p4) a major 3rd glissando from the current pitch,
            offset, glissdur = 2, 8
            e.automatep(event, 4, [offset, 60, offset+glissdur, 64])

        .. figure:: assets/select-preset.png

        .. seealso::

            * :meth:`~Engine.soundfontPreparePreset`
            * :meth:`~Engine.playSample`

        """
        assert index in self._soundfontPresets.values()
        if not vel:
            vel = int(amp * 127)
        args = [pitch, amp, index, vel, chan]
        return self.sched(self._builtinInstrs['soundfontPlay'], delay=delay, dur=dur,
                          args=args)

    def soundfontPreparePreset(self,
                               sf2path: str,
                               preset: tuple[int, int] | None = None
                               ) -> int:
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

        .. seealso::

            * :meth:`~Engine.soundfontPlay`
            * :meth:`~Engine.playSample`
        """
        if sf2path == "?":
            sf2path = _state.openSoundfont(ensureSelection=True)
        if preset is None:
            from . import sftools
            item = sftools.soundfontSelectPreset(sf2path)
            if item is None:
                return 0
            presetname, bank, presetnum = item
        else:
            bank, presetnum = preset
        tup = (sf2path, bank, presetnum)
        idxnum = self._soundfontPresets.get(tup)
        if idxnum is not None:
            return idxnum
        idx = self._soundfontPresetCountPtr[0]
        self._soundfontPresetCountPtr[0] += 1
        self._soundfontPresets[tup] = idx
        instrnum = self._builtinInstrs['sfPresetAssignIndex']
        s = f'i {instrnum} 0 0 "{sf2path}" {bank} {presetnum} {idx}'
        self._perfThread.inputMessage(s)
        return idx

    def _readSoundfileAsync(self,
                            path: str,
                            tabnum: int | None = None,
                            chan=0) -> int:
        assert self.started
        if tabnum is None:
            tabnum = self._assignTableNumber()
        else:
            assert tabnum > 0
        s = f'f {tabnum} 0 0 -1 "{path}" 0 0 {chan}'
        self._perfThread.inputMessage(s)
        self._tableInfo[tabnum] = TableInfo(sr=0, size=0, nchnls=1, path=path)
        return tabnum

    def getUniqueInstrInstance(self, instr: int | str) -> float:
        """
        Returns a unique instance number (a float p1) for `instr`

        Args:
            instr (int|str): an already defined csound instrument

        Returns:
            a unique p1.
        """
        if isinstance(instr, int):
            token = self._getSyncToken()
            pargs = [self._builtinInstrs['uniqinstance'], 0, 0.01, token, instr]
            uniqinstr = self._eventWait(token, pargs)
            if uniqinstr is None:
                raise RuntimeError("failed to get unique instance")
            return uniqinstr
        else:
            raise NotImplementedError("str instrs not implemented yet")

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


        .. seealso::

            * :meth:`~Engine.playSoundFromDisk`
            * :meth:`~Engine.makeTable`
            * :meth:`~Engine.readSoundfile`
            * :meth:`~Engine.soundfontPlay`

        """
        if not isinstance(tabnum, int):
            raise TypeError(f"Expected a table number as integer, got {tabnum}")
        args = [gain, speed, tabnum, chan, fade, starttime, lagtime]
        return self.sched(self._builtinInstrs['playgen1'], delay=delay, dur=dur,
                          args=args)

    def playSoundFromDisk(self, path: str, delay=0., dur=-1, chan=1, speed=1., fade=0.01,
                          gain=1.
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

        **Dynamic parameters**
            * p5: kgain - Modifies the gain
            * p6: kspeed - Modifies the playback speed

        .. seealso:: :py:meth:`readSoundfile`, :py:meth:`playSample`

        """
        assert self.started
        p1 = self._assignEventId(self._builtinInstrs['playsndfile'])
        if chan <= 0:
            raise ValueError("chan must be >= 1")
        msg = f'i {p1} {delay} {dur} "{path}" {gain} {speed} {chan} {fade}'
        self._perfThread.inputMessage(msg)
        return p1

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

        Example
        ~~~~~~~

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
        >>> engine.setp(p1, 5, 0.2, delay=0.5)

        .. seealso:: :meth:`~Engine.automatep`
        """
        if isinstance(p1, str):
            raise TypeError("named instrs not supported yet")
        numpairs = len(pairs) // 2
        assert len(pairs) % 2 == 0 and numpairs <= 5
        # this limit is just the limit of the pwrite instr, not of the opcode
        args = [0, p1, numpairs]
        args.extend(pairs)
        self.sched(self._builtinInstrs['pwrite'], delay=delay, dur=0, args=args)

    def _getp(self, eventid: float, idx: int) -> float | None:
        """
        Get the current pfield value of an active note.

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
        ~~~~~~~

        TODO

        .. seealso::

            :meth:`~Engine.setp`
        """
        if self.version >= 7000:
            value = self.csound.evalCode(f'return pread({eventid}, {idx})')
        else:
            token = self._getSyncToken()
            notify = 1
            pargs = [self._builtinInstrs['pread'], 0, 0, token, eventid, idx, notify]
            value = self._eventWait(token, pargs)
        return value

    def automateTable(self,
                      tabnum: int,
                      idx: int,
                      pairs: Sequence[float] | np.ndarray,
                      mode='linear',
                      delay=0.,
                      overtake=False) -> float:
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
        ~~~~~~~

        >>> engine = Engine(...)
        >>> engine.compile(r'''
        ... instr 100
        ...   itab = p4
        ...   kamp  table 0, itab
        ...   kfreq table 1, itab
        ...   outch 1, oscili:a(0.1, kfreq)
        ...   ftfree itab, 1  ; free the table when finished
        ... endin
        ... ''')
        >>> source = engine.makeTable([0.1, 1000])
        >>> eventid = engine.sched(100, 0, 10, args=(source,))
        >>> # automate the frequency (slot 1)
        >>> engine.automateTable(source, 1, [0, 1000, 3, 200, 5, 200])
        >>> # Automate from the current value, will produce a fade-out
        >>> engine.automateTable(source, 0, [0, -1, 2, 0], overtake=True, delay=5)

        .. seealso::

            * :meth:`~Engine.setp`
            * :meth:`~Engine.automatep`


        """
        if self.version >= 7000 or len(pairs) <= 1900:
            # iargtab = p4, iargidx = p5, imode = p6, iovertake = p7, ilenpairs = p8
            args: list[float|int] = [tabnum, idx, self.strSet(mode), int(overtake), len(pairs)]
            if isinstance(pairs, np.ndarray):
                args.extend(pairs.tolist())
            else:
                args.extend(pairs)
            return self.sched(self._builtinInstrs['automateTableViaPargs'],
                              delay=delay,
                              dur=args[-2] + self.ksmps / self.sr,
                              args=args)
        else:
            events = [self.automateTable(tabnum=tabnum, idx=idx, pairs=subgroup,
                                         mode=mode, delay=delay+subdelay,
                                         overtake=overtake)
                      for subdelay, subgroup in internal.splitAutomation(pairs, 1900//2)]
            return events[0]

    def automatep(self,
                  p1: float | str,
                  pidx: int,
                  pairs: Sequence[float] | np.ndarray,
                  mode='linear',
                  delay=0.,
                  overtake=False
                  ) -> float:
        """
        Automate a pfield of a running event

        The automation is done by another csound event, so it happens within the
        "csound" realm and thus is assured to stay in sync

        Args:
            p1: the fractional instr number of a running event, or an int number
                to modify all running instances of that instr. A named instr with
                an optional fractional part is also accepted
            pidx: the pfield index. For example, if the pfield to modify if p4,
                pidx should be 4. Values of 1, 2, and 3 are not allowed.
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
        ~~~~~~~

        >>> e = Engine()
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
        maxDataSize618 = 1900
        if self.version >= 7000 or len(pairs) <= maxDataSize618:
            if isinstance(p1, str):
                args = [float(self.strSet(p1)), float(pidx), float(self.strSet(mode)), float(overtake), float(len(pairs)), 1.]
            else:
                args = [p1, float(pidx), float(self.strSet(mode)), float(overtake), float(len(pairs)), 0.]

            if isinstance(pairs, np.ndarray):
                args.extend(pairs.tolist())
            else:
                args.extend(pairs)
            return self.sched(self._builtinInstrs['automatePargViaPargs'],
                              delay=delay,
                              dur=pairs[-2] + self.ksmps / self.sr,
                              args=args)
        else:
            events = [self.automatep(p1=p1, pidx=pidx, pairs=subgroup, mode=mode, delay=delay+subdelay,
                                     overtake=overtake)
                      for subdelay, subgroup in internal.splitAutomation(pairs, maxDataSize618 // 2)]
            return events[0]

    def strSet(self, s: str) -> int:
        stridx = self._strToIndex.get(s)
        if stridx:
            return stridx
        stridx = self._getStrIndex()
        self._strToIndex[s] = stridx
        self._indexToStr[stridx] = s
        cmd = f'strset {stridx}, "{s}"'
        if self.version >= 7000:
            self._perfThread.compileOrc(cmd)
        elif self._useProcessQueue:
            def task(cs):
                cs.compileOrc(cmd)
            self._perfThread.processQueueTask(task)
        else:
            self.csound.compileOrc(cmd)
        self._modified()
        return stridx

    def _strSet(self, s: str, sync=False) -> int:
        """
        Assign a numeric index to a string to be used inside csound

        Args:
            s: the string to set
            sync: if True, block if needed until the csound process receives the message

        Returns:
            the index associated with *s*. When passed to a csound instrument
            it can be used to retrieve the original string via
            ``Sstr = strget(idx)``

        .. seealso:: :meth:`~Engine.strGet`

        """
        assert self.started and s
        stringIndex = self._strToIndex.get(s)
        if stringIndex:
            return stringIndex
        stringIndex = self._getStrIndex()
        self._strToIndex[s] = stringIndex
        self._indexToStr[stringIndex] = s
        instrnum = self._builtinInstrs['strset']
        msg = f'i {instrnum} 0 0 "{s}" {stringIndex}'
        self._perfThread.inputMessage(msg)
        if sync:
            self.sync()
        else:
            self._modified()
        return stringIndex

    def definedStrings(self) -> dict[str, int]:
        """
        Returns a dict mapping defined strings to their integer id

        These are strings defined via :meth:`~Engine.strSet` by the Engine,
        not internally using csound itself

        .. warning::

            Using strset within an instrument or as global code will probably
            result in conflicts with the strings defined via the Engine
            using :meth:`Engine.setStr`

        Returns:
            a dict mapping defined strings to their corresponding index
        """
        return self._strToIndex

    def strGet(self, index: int) -> str | None:
        """
        Get a string previously set via strSet.

        This method will not retrieve any string set internally via the
        `strset` opcode, only strings set via :meth:`~Engine.strSet`

        Example
        ~~~~~~~

        >>> e = Engine(...)
        >>> idx = e.strSet("foo")
        >>> e.strGet(idx)
        foo

        .. seealso:: :meth:`~Engine.strSet`

        """
        return self._indexToStr.get(index)

    def _getStrIndex(self) -> int:
        out = self._strLastIndex
        self._strLastIndex += 1
        return out

    def _releaseTableNumber(self, tableindex: int) -> None:
        """
        Mark the given table as freed, so that it can be assigned again.

        It assumes that the table was deallocated already and the index
        can be assigned again.
        """
        if tableindex not in self._tablePool:
            self._tablePool.push(tableindex)
        else:
            logger.warning(f"Table number {tableindex} was not assigned by csoundengine")

    def freeTable(self, tableindex: int, delay=0.) -> None:
        """
        Free the table with the given index

        Args:
            tableindex: the index of the table to free
            delay: when to free it (0=right now)

        .. seealso:: :meth:`~Engine.makeTable`

        """
        logger.debug(f"Freeing table {tableindex}")
        self._releaseTableNumber(tableindex)
        pargs = [self._builtinInstrs['freetable'], delay, 0., tableindex]
        self._perfThread.scoreEvent(False, "i", pargs)

    def testAudio(self, dur=4., delay=0., period=1., mode='pink',
                  gaindb=-6., echo=True) -> float:
        """
        Test this engine's output

        Args:
            dur: the duration of the test
            delay: when to start the test
            period: the duration of sound output on each channel
            mode: the test mode, one of 'pink', 'sine'
            gaindb: the gain of the output, in dB
            echo: if True, csound prints the channel to which it is outputting audio

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

        return self.sched(self._builtinInstrs['testaudio'], dur=dur, delay=delay,
                          args=[modeid, period, pt.db2amp(gaindb), int(echo)])

    def _udpSend(self, code: str) -> None:
        if not self.udpPort:
            logger.warning("This csound instance was started without udp")
            return
        assert self._sendAddr is not None
        msg = code.encode("ascii")
        logger.debug(f"_udpSend: {code}")
        self._udpSocket.sendto(msg, self._sendAddr)

    def udpSendOrc(self, code: str) -> None:
        """
        Send orchestra code via UDP.

        Args:
            code (str): the code to send

        .. seealso::

            :meth:`~Engine.udpSendScoreline`
            :meth:`~Engine.udpSetChannel`

        """
        if self.udpPort is None:
            raise RuntimeError("This engine has no udp port assigned")
        assert self._sendAddr is not None
        msg = code.encode("ascii")
        if len(msg) < 60000:
            self._udpSocket.sendto(msg, self._sendAddr)
            return
        msgs = internal.splitBytes(msg, 60000)
        self._udpSocket.sendto(b"{{ " + msgs[0], self._sendAddr)
        for msg in msgs[1:-1]:
            self._udpSocket.sendto(msg, self._sendAddr)
        self._udpSocket.sendto(msgs[-1] + b" }}", self._sendAddr)
        self._modified()

    def udpSendScoreline(self, scoreline:str) -> None:
        """
        Send a score line to csound via udp

        Example
        ~~~~~~~

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

    def udpSetChannel(self, channel: str, value: float | str) -> None:
        """
        Set a channel via UDP. The value will determine the kind of channel

        Args:
            channel: the channel name
            value: the new value

        .. seealso::

            :meth:`~Engine.udpSendScoreline`
            :meth:`~Engine.udpSendOrc`
        """
        if isinstance(value, (int, float)):
            self._udpSend(f"@{channel} {value}")
        else:
            self._udpSend(f"%{channel} {value}")

    def writeBus(self, bus: int, value: float, delay=0.) -> None:
        """
        Set the value of a control bus

        Normally a control bus is set via another running instrument,
        but it is possible to set it directly from python. The first
        time a bus is set or queried there is short delay, all
        subsequent operations on the bus are very fast.

        Args:
            bus: the bus token, as returned via :meth:`Engine.assignBus`
            value: the new value
            delay: if given, the modification is scheduled in the future

        .. seealso::

            :meth:`~Engine.readBus`
            :meth:`~Engine.assignBus`
            :meth:`~Engine.automateBus`

        Example
        ~~~~~~~

        >>> e = Engine(...)
        >>> e.compile(r'''
        ... instr 100
        ...   ifreqbus = p4
        ...   kfreq = busin:k(ifreqbus)
        ...   outch 1, vco2:a(0.1, kfreq)
        ... endin
        ... ''')
        >>> freqbus = e.assignBus(value=1000)
        >>> e.sched(100, 0, 4, args=[freqbus])
        >>> e.writeBus(freqbus, 500, delay=0.5)

        """
        if not self.hasBusSupport():
            raise RuntimeError("This engine does not have bus support")

        bus = int(bus)
        kind = self._busTokenToKind.get(bus)
        if not kind:
            logger.warning(f"Bus token {bus} not known")
        elif kind != 'control':
            raise ValueError(f"Only control buses can be written to, got {kind}")

        if delay <= self.onecycle:
            busindex = self._busIndexes.get(bus)
            if busindex is not None:
                assert self._kbusTable is not None
                self._kbusTable[busindex] = value
            else:
                self.sched(self._builtinInstrs['busoutk'], delay=delay, dur=self.onecycle*8, args=(int(bus), value))
                self._getBusIndex(bus, blocking=False)
        else:
            self.sched(self._builtinInstrs['busoutk'], delay=delay, dur=self.onecycle*8, args=(int(bus), value))

    def readBus(self, bus: int, default: float | None = None) -> float | None:
        """
        Read the current value of a control bus

        Buses can be used to allow communication between instruments, or
        between a running csound instrument and python. Buses are useful
        for continuous streams; when using buses to communicate discrete
        values with python an opcode like trighold might be necessary.
        In general for discrete events it might be better to use other
        communication means, like OSC, which provide buffering.

        Args:
            bus: the bus number, as returned by assignBus
            default: the value returned if the bus does not exist

        Returns:
            the current value of the bus, or `default` if the bus does not exist

        .. seealso::

            :meth:`~Engine.assignBus`
            :meth:`~Engine.writeBus`

        Example
        ~~~~~~~

        >>> e = Engine()
        >>> e.compile(r'''
        ... instr 100
        ...   irmsbus = p4
        ...   asig inch 1
        ...   krms = rms:k(asig)
        ...   busout irmsbus, krms
        ... endin
        ... ''')
        >>> rmsbus = e.assignBus(kind='control')
        >>> event = e.sched(100, 0, args=[rmsbus])
        >>> while True:
        ...     rmsvalue = e.readBus(rmsbus)
        ...     print(f"Rms value: {rmsvalue}")
        ...     time.sleep(0.1)
        """
        if not self.hasBusSupport():
            raise RuntimeError("This engine does not have bus support")

        # The int conversion makes it possible to pass a Bus object created
        # via Session.assignBus, see busproxy.Bus.__int__
        busidx = self._getBusIndex(int(bus), blocking=True)
        if busidx is None:
            raise RuntimeError(f"Could not get the actual csound bus for token {bus}")
        if busidx < 0:
            return default
        assert self._kbusTable is not None
        return self._kbusTable[busidx]

    def _getBusIndex(self, bus: int, blocking=True, whendone: Callable | None = None
                     ) -> int | None:
        """
        Find the bus index corresponding to `bus` token.

        Args:
            bus: the bus token, as returned by assignBus
            blocking: if True, block execution until csound has responded. Returns
                the bus index
            whendone: a callable of the form (busindex) -> None, will be called with
                the bus index corresponding to the bus token. Passing a callback
                will make this method async

        Returns:
            if blocking, will return the bus index as int. Otherwise, returns None
        """
        assert isinstance(bus, int)
        kind = self._busTokenToKind.get(bus)
        if not kind:
            raise ValueError(f"Bus not found, token: {bus}")

        if kind != 'control':
            raise ValueError("Only control buses are supported here")

        bus = int(bus)
        index = self._busIndexes.get(bus)
        if index is not None:
            if whendone:
                whendone(index)
            return index

        if whendone:
            blocking = False

        synctoken = self._getSyncToken()
        ikind = BUSKIND_CONTROL   # control bus
        #          1                                 2  3              4          5    6      7  8           8
        pfields = [self._builtinInstrs['busassign'], 0, self.onecycle, synctoken, bus, ikind, 0, 0]
        if blocking:
            out = self._eventWait(synctoken, pfields)
            if out is None:
                raise RuntimeError(f"Could not fetch the bus index for token {bus}")
            index = int(out)
            self._busIndexes[bus] = index
            return index
        else:
            def callback(busnum, bustoken=bus, self=self, callback=whendone):
                busnum = int(busnum)
                self._busIndexes[bustoken] = busnum
                if callback:
                    callback(busnum)

            self._eventWithCallback(token=synctoken, pargs=pfields, callback=callback)
            return None

    def releaseBus(self, bus: int, delay: float | None = None) -> None:
        """
        Release a persistent bus

        Args:
            bus: the bus to release, as returned by :py:meth:`assignBus`
            delay: when to release the bus (relative time). None means now

        .. seealso:: :meth:`~Engine.assignBus`
        """
        # bus is the bustoken
        if not self.hasBusSupport():
            raise RuntimeError("This Engine was created without bus support")

        self._busIndexes.pop(bus, None)
        pargs = [self._builtinInstrs['busrelease'], delay or 0., 0, int(bus)]
        self._perfThread.scoreEvent(False, "i", pargs)

    def hasBusSupport(self) -> bool:
        return self._hasBusSupport and (self.numAudioBuses > 0 or self.numControlBuses > 0)

    def addBusSupport(self, numAudioBuses: int|None = None, numControlBuses: int|None = None) -> None:
        if self.hasBusSupport():
            return

        if numAudioBuses is None:
            numAudioBuses = self.numAudioBuses or config['num_audio_buses']

        if numControlBuses is None:
            numControlBuses = self.numControlBuses or config['num_control_buses']
        logger.debug(f"Adding bus support, {numAudioBuses=}, {numControlBuses=}")

        startInstr = max(instrnum for instrnum in self._builtinInstrs.values() if instrnum < CONSTS['postProcInstrnum']) + 1
        postInstrnum = 1 + max(max(self._builtinInstrs.values()), CONSTS['postProcInstrnum'])
        busorc, businstrs = engineorc.makeBusOrc(numAudioBuses=numAudioBuses,
                                                 numControlBuses=numControlBuses,
                                                 startInstr=startInstr,
                                                 postInstr=postInstrnum)
        self._builtinInstrs.update(businstrs)
        self._reservedInstrnumRanges.append(('busorc', min(businstrs.values()), max(businstrs.values())))
        self._hasBusSupport = True
        self.numAudioBuses = numAudioBuses
        self.numControlBuses = numControlBuses
        chanptr, error = self.csound.channelPtr("_busTokenCount", kind='control', mode='rw')
        assert isinstance(chanptr, np.ndarray)
        self._busTokenCountPtr = chanptr
        if self.started:
            self._compileCode(busorc, block=True)
            kbustable = int(self.evalCode("return gi__bustable"))
            self._kbusTable = self.getTableData(kbustable)
        else:
            self.csound.compileOrc(busorc)
            kbustable = int(self.csound.evalCode("return gi__bustable"))
            self._kbusTable = self.csound.table(kbustable)
        assert self._kbusTable is not None

    def assignBus(self, kind='', value: float | None = None, persist=False
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
        ~~~~~~~

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

        >>> e = Engine(...)
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
        assert isinstance(bustoken, int)
        assert kind == 'audio' or kind == 'control'
        self._busTokenToKind[bustoken] = kind
        ikind = BUSKIND_AUDIO if kind == 'audio' else BUSKIND_CONTROL
        ivalue = value if value is not None else 0.

        self._busTokenCountPtr[0] = bustoken + 1

        # We call busassign to assign a new bus. We are not interested, at this point,
        # to get the actual bus index for the given token. Getting the actual index
        # is only relevant if we want to read/write to it.
        synctoken = 0
        #          1                                 2  3  4          5         6      7             8
        pfields = [self._builtinInstrs['busassign'], 0, 0, synctoken, bustoken, ikind, int(persist), ivalue]
        self._perfThread.scoreEvent(False, "i", pfields)
        return bustoken

    def busSystemStatus(self) -> dict:
        """
        Get debugging information about the status of the bus system

        This is only provided for debugging

        Returns:
            a dict containing information about the status of the bus system
            (used buses, free buses, etc)
        """
        if not self.hasBusSupport():
            raise RuntimeError("This Engine has no bus support")

        audioBusesFree = self.evalCode('return pool_size:i(gi__buspool)')
        controlBusesFree = self.evalCode('return pool_size:i(gi__buspoolk)')
        return {'audioBusesFree': audioBusesFree,
                'controlBusesFree': controlBusesFree,
                'numAudioBuses': self.numAudioBuses,
                'numControlBuses': self.numControlBuses}

    def eventUI(self, eventid: float, **pargs: tuple[float, float]) -> None:
        """
        Modify pfields through an interactive user-interface

        If run inside a jupyter notebook, this method will create embedded widgets
        to control the values of the dynamic pfields of an event

        Args:
            eventid: p1 of the event to modify
            **pfields: a dict mapping pfield to a tuple (minvalue, maxvalue)

        Example
        ~~~~~~~

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
        specs: dict[int|str, interact.ParamSpec] = {}
        body = self._instrRegistry.get(int(eventid)) or self._instrRegistry.get(internal.instrNameFromP1(eventid))
        pfieldsNameToIndex = csoundparse.instrParseBody(body).pfieldNameToIndex if body else None
        for pfield, spec in pargs.items():
            minval, maxval = spec
            idx = internal.resolvePfieldIndex(pfield, pfieldsNameToIndex)
            if not idx:
                raise KeyError(f"pfield {pfield} not understood")
            value = self._getp(eventid, idx)
            specs[idx] = interact.ParamSpec(pfield,
                                            minvalue=minval, maxvalue=maxval,
                                            startvalue=value if value is not None else 0.,
                                            widgetHint='slider')
        return interact.interactPargs(self, eventid, specs=specs)


@_atexit.register
def _cleanup() -> None:
    engines = list(Engine.activeEngines.values())
    if engines:
        logger.debug("Exiting python, closing all active engines")
        for engine in engines:
            logger.debug(f"... stopping {engine.name}")
            engine.stop()


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
