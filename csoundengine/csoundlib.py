"""
This module provides miscellaneous functionality for working with csound.

This functionality includes:

* parse csound code
* generate a ``.csd`` file
* inspect the audio environment
* query different paths used by csound
* etc.

"""
from __future__ import annotations

import dataclasses
import sys
import functools as _functools
import logging as _logging
import math as _math
import os as _os
import re as _re
import shutil as _shutil
import subprocess as _subprocess
import tempfile as _tempfile
import textwrap as _textwrap

import cachetools as _cachetools
import emlib.textlib
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Callable, Sequence
    Curve = Callable[[float], float]
    from . import jacktools


# try:
#     import libcsound
# except Exception as e:
#     if 'sphinx' in sys.modules:
#         print("Called while building sphinx documentation?")
#         from sphinx.ext.autodoc.mock import _MockObject
#         libcsound = _MockObject()
#     else:
#         print("Error importing libcsound")
#         raise e

logger = _logging.getLogger("csoundengine")


_cache: dict[str, Any] = {
    'opcodes': None,
    'versionTriplet': None
}


_audioDeviceRegex = r"(\d+):\s((?:adc|dac)\d+)\s*\((.*)\)(?:\s+\[ch:(\d+)\])?"


def midiDevices(backend='portmidi') -> tuple[list[MidiDevice], list[MidiDevice]]:
    """
    Returns input and output midi devices for the given backend

    Args:
        backend: the backend used for realtime midi (as passed to
            csound via -+rtmidi={backend}

    Returns:
        a tuple (inputdevices, outputdevices), which each of these
        is a list of MidiDevice with attributes ``deviceid`` (the value
        passed to -M), ``name`` (the name of the device) and ``kind``
        (one of 'input' or 'output')

    ========   ===========================
    Platform   Possible Backends
    ========   ===========================
    linux      portmidi, alsaseq, alsaraw
    macos      portmidi
    windows    portmidi
    ========   ===========================
    """
    import libcsound
    csound = libcsound.Csound()
    csound.setOption(f"-+rtmidi={backend}")
    csound.setOption("-odac")
    csound.start()
    inputdevs = csound.midiDevList(False)
    outputdevs = csound.midiDevList(True)
    logger.debug(f"MIDI Inputs:  {inputdevs}")
    logger.debug(f"MIDI Outputs: {outputdevs}")
    midiins = [MidiDevice(deviceid=d.deviceId, kind='input',
                          name=f"{d.interfaceName}:{d.deviceName}")
               for d in inputdevs]
    midiouts = [MidiDevice(deviceid=d.deviceId, kind='output',
                           name=f"{d.interfaceName}:{d.deviceName}")
                for d in outputdevs]
    return midiins, midiouts


def compressionBitrateToQuality(bitrate: int, fmt='ogg') -> float:
    """
    Convert a bitrate to a compression quality between 0-1, as passed to --vbr-quality

    Args:
        bitrate: the bitrate in kb/s, oneof 64, 80, 96, 128, 160, 192, 224, 256, 320, 500
        fmt: the encoding format (ogg at the moment)
    """
    if fmt == 'ogg':
        bitrates = [64, 80, 96, 128, 128, 160, 192, 224, 256, 320, 500]
        import emlib.misc
        idx = emlib.misc.nearest_index(bitrate, bitrates)
        return idx / 10
    else:
        raise ValueError(f"Format {fmt} not supported")


def compressionQualityToBitrate(quality: float, fmt='ogg') -> int:
    """
    Convert compression quality to bitrate

    Args:
        quality: the compression quality (0-1) as passed to --vbr-quality
        fmt: the encoding format (ogg at the moment)

    Returns:
        the resulting bit rate


    =======   =======
    quality   bitrate
    =======   =======
    0.0       64
    0.1       80
    0.2       96
    0.3       112
    0.4       128
    0.5       160
    0.6       192
    0.7       224
    0.8       256
    0.9       320
    1.0       500
    =======   =======
    """
    if fmt == 'ogg':
        idx = int(quality * 10 + 0.5)
        if idx > 10:
            idx = 10
        return (64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 500)[idx]
    else:
        raise ValueError(f"Format {fmt} not supported")


@dataclasses.dataclass(unsafe_hash=True)
class AudioBackend:
    """
    Holds information about a csound audio backend

    Attributes:
        name: the name of this backend
        alwaysAvailable: is this backend always available?
        hasSystemSr: does this backend have a system samplerate?
        needsRealtime: the backend needs to be run in realtime
        platforms: a list of platform for which this backend is available
        longname: an alternative name for the backend
        defaultBufferSize: the default buffer size for this backend (-b)
        defaultNumBuffers: the number of buffers to fill a block (determines -B)
        audioDeviceRegex: a regex to grep the audio devices from csound's output
        acceptsDeviceIndex: can this backend accept a device in the form -dac0 or -adc1?
    """
    name: str
    alwaysAvailable: bool = False
    needsRealtime: bool = False
    platforms: tuple[str, ...] = ('linux', 'darwin', 'win32')
    hasSystemSr: bool = False
    longname: str = ""
    defaultBufferSize: int = 1024
    defaultNumBuffers: int = 2
    audioDeviceRegex: str = ''
    acceptsDeviceIndex: bool = True
    priority: int = 0

    def __post_init__(self):
        if not self.longname:
            self.longname = self.name
        if not self.audioDeviceRegex:
            self.audioDeviceRegex = _audioDeviceRegex

    def searchAudioDevice(self, pattern: str, kind: str) -> AudioDevice | None:
        """
        Search a certain audio device from the devices presented by this backend
        """
        # we get the devices via getAudioDevices to enable caching
        indevs, outdevs = self.audioDevices()
        indevs, outdevs = getAudioDevices(self.name)
        devs = indevs if kind == 'input' else outdevs
        match = next((d for d in devs if d.id == pattern), None)
        if match:
            return match
        return next((d for d in devs
                    if _re.search(pattern, d.name)), None)

    def isAvailable(self) -> bool:
        """ Is this backend available? """
        if sys.platform not in self.platforms:
            return False
        if self.alwaysAvailable:
            return True
        indevices, outdevices = getAudioDevices(backend=self.name)
        return bool(indevices or outdevices)

    def getSystemSr(self) -> int | None:
        """
        Get the system samplerate for this backend, if available

        We use the default output device.
        """
        if not self.hasSystemSr:
            logger.debug(f"Backend {self.name} does not have a system sr, returning default")
            return 44100
        import libcsound
        cs = libcsound.Csound()
        cs.setOption("-odac")
        cs.setOption(f"-+rtaudio={self.name}")
        ok = cs.start()
        if ok == -1:
            logger.error(f"Backend {self.name} not available")
            return None
        sr = cs.systemSr(0)
        cs.stop()
        return int(sr) if sr > 0 else None

    def bufferSizeAndNum(self) -> tuple[int, int]:
        """
        The buffer size and number of buffers needed for this backend
        """
        return (self.defaultBufferSize, self.defaultNumBuffers)

    def audioDevices(self) -> tuple[list[AudioDevice], list[AudioDevice]]:
        return self.audioDevicesViaAPI()

    def audioDevicesViaAPI(self) -> tuple[list[AudioDevice], list[AudioDevice]]:
        """
        Query csound for audio devices for this backend

        Returns:
            a tuple (inputDevices: list[AudioDevice], outputDevices: list[AudioDevice])
        """
        logger.info(f"Querying csound's audio devices for backend {self.name}")
        import libcsound
        cs = libcsound.Csound()
        cs.createMessageBuffer()
        for opt in ['-+rtaudio='+self.name, "-m16", "-odac", "--use-system-sr"]:
            cs.setOption(opt)
        cs.start()
        csoutdevs = cs.audioDevList(isOutput=True)
        csindevs = cs.audioDevList(isOutput=False)
        outdevs = [AudioDevice(id=d.deviceId,
                               name=d.deviceName,
                               backend=self.name,
                               kind='output',
                               index=i,
                               numChannels=d.maxNchnls)
                   for i, d in enumerate(csoutdevs)]
        indevs = [AudioDevice(id=d.deviceId,
                              name=d.deviceName,
                              backend=self.name,
                              kind='input',
                              index=i,
                              numChannels=d.maxNchnls)
                  for i, d in enumerate(csindevs)]
        cs.stop()
        cs.destroyMessageBuffer()
        return indevs, outdevs

    def defaultAudioDevices(self) -> tuple[AudioDevice | None, AudioDevice | None]:
        """
        Returns the default audio devices for this backend

        Returns:
            a tuple ``(inputDevice: AudioDevice, outputDevice: AudioDevice)`` for this backend
        """
        indevs, outdevs = getAudioDevices(self.name)  # self.audioDevices()
        return indevs[0] if indevs else None, outdevs[0] if outdevs else None


@_cachetools.cached(cache=_cachetools.TTLCache(1, 20))
def _jackdata() -> tuple[jacktools.JackInfo, list[jacktools.JackClient]] | None:
    from . import jacktools
    info = jacktools.getInfo()
    if info is None:
        return None
    return info, jacktools.getClients()


class _JackAudioBackend(AudioBackend):
    def __init__(self):
        super().__init__(name='jack',
                         priority=10,
                         alwaysAvailable=False,
                         platforms=('linux', 'darwin', 'win32'),
                         hasSystemSr=True)

    def getSystemSr(self) -> int:
        if (data := _jackdata()) is not None:
            return data[0].samplerate
        raise RuntimeError("Jack is not available")

    def bufferSizeAndNum(self) -> tuple[int, int]:
        data = _jackdata()
        if not data:
            raise RuntimeError("Jack is not available")
        blocksize = data[0].blocksize
        import emlib.mathlib
        if not emlib.mathlib.ispowerof2(blocksize):
            logger.warning(f"Jack's blocksize is not a power of 2: {blocksize}!")
        # jack buf: 512 -> -B 1024 -b 256
        periodsize = blocksize // 2
        numbuffers = 4
        return periodsize, numbuffers

    def isAvailable(self) -> bool:
        return _jackdata() is not None

    def audioDevices(self) -> tuple[list[AudioDevice], list[AudioDevice]]:
        data = _jackdata()
        if data is None:
            raise RuntimeError("Jack is not available")
        info, clients = data
        indevs = [AudioDevice(id=f'adc:{c.regex}', name=c.name, backend=self.name, kind='input',
                              numChannels=len(c.ports), index=c.firstIndex, isPhysical=c.isPhysical)
                  for c in clients if c.kind == 'output']
        outdevs = [AudioDevice(id=f'dac:{c.regex}', name=c.name, backend=self.name, kind='output',
                               numChannels=len(c.ports), index=c.firstIndex, isPhysical=c.isPhysical)
                   for i, c in enumerate(clients) if c.kind == 'input']
        return indevs, outdevs

    def defaultAudioDevices(self) -> tuple[AudioDevice|None, AudioDevice|None]:
        indevs, outdevs = self.audioDevices()
        defaultin = next((dev for dev in indevs if dev.isPhysical), None)
        defaultout = next((dev for dev in outdevs if dev.isPhysical), None)
        return defaultin, defaultout


class _PulseAudioBackend(AudioBackend):
    def __init__(self):
        super().__init__(name='pulse',
                         alwaysAvailable=False,
                         hasSystemSr=True,
                         defaultBufferSize=1024,
                         defaultNumBuffers=2,
                         platforms=('linux',),
                         priority=0)

    def getSystemSr(self) -> int:
        from . import linuxaudio
        info = linuxaudio.pulseaudioInfo()
        return info.sr if info else 0

    def isAvailable(self) -> bool:
        return self.getSystemSr() > 0

    def audioDevices(self) -> tuple[list[AudioDevice], list[AudioDevice]]:
        from . import linuxaudio
        pulseinfo = linuxaudio.pulseaudioInfo()
        if pulseinfo is None:
            raise RuntimeError("PulseAudio not available")
        indevs = [AudioDevice(id="adc", name="adc", backend=self.name, kind='input', index=0,
                              numChannels=pulseinfo.numchannels)]
        outdevs = [AudioDevice(id="dac", name="dac", backend=self.name, kind='output', index=0,
                               numChannels=pulseinfo.numchannels)]
        return indevs, outdevs


class _PortaudioBackend(AudioBackend):
    def __init__(self, kind='callback'):
        shortname = "pa_cb" if kind == 'callback' else 'pa_bl'
        longname = f"portaudio-{kind}"
        priority = 2 if kind == 'callback' else 0
        if sys.platform == 'linux':
            from . import linuxaudio
            hasSystemSr = linuxaudio.isPipewireRunning()
        else:
            hasSystemSr = False
        super().__init__(name=shortname,
                         alwaysAvailable=True,
                         longname=longname,
                         hasSystemSr=hasSystemSr,
                         priority=priority)

    def getSystemSr(self) -> int | None:
        if sys.platform == 'linux' and self.hasSystemSr:
            from . import linuxaudio
            info = linuxaudio.pipewireInfo()
            return info.sr if info is not None else 44100
        return super().getSystemSr()

    @_functools.cache
    def defaultAudioDevices(self) -> tuple[AudioDevice | None, AudioDevice | None]:
        logger.debug("Querying default device via portaudio/sounddevice")
        try:
            import sounddevice as sd
        except ImportError:
            logger.warning('Could not initialize the sounddevice library. Falling back'
                           ' to querying via csound')
            return self._defaultAudioDevicesViaCsound()

        devices = sd.query_devices()
        defaultoutdev, defaultindev = None, None
        indevs, outdevs = self.audioDevices()
        if indevs and sd.default.device[0] is not None:
            defaultName = devices[sd.default.device[0]]['name']
            defaultindev = next((dev for dev in indevs
                                 if dev.name.split("[")[0].strip() == defaultName), None)

        if outdevs and sd.default.device[1] is not None:
            defaultName = devices[sd.default.device[1]]['name']
            defaultoutdev = next((dev for dev in outdevs
                                  if dev.name.split("[")[0].strip() == defaultName), None)

        return defaultindev, defaultoutdev

    def _defaultAudioDevicesViaCsound(self) -> tuple[AudioDevice | None, AudioDevice | None]:
        indevs, outdevs = self.audioDevices()
        indev = next((d for d in indevs if _re.search(r"\bdefault\b", d.name)), None)
        outdev = next((d for d in outdevs if _re.search(r"\bdefault\b", d.name)), None)
        if indev is None:
            if not indevs:
                logger.warning(f"No input devices for backend {self.name}")
            else:
                indev = indevs[0]
        if outdev is None:
            if not outdevs:
                logger.warning(f"No output devices for backend {self.name}")
            else:
                outdev = outdevs[0]
        return indev, outdev


# linux priorities: jack(10), portaudio-callback(2), alsa(1), pulse(0), portaudio-blocking(0)

class _AlsaBackend(AudioBackend):
    def __init__(self):
        super().__init__(name="alsa",
                         alwaysAvailable=True,
                         platforms=('linux',),
                         audioDeviceRegex=r"([0-9]+):\s((?:adc|dac):.*)\((.*)\)",
                         acceptsDeviceIndex=False,
                         priority=1)

    def getSystemSr(self) -> int | None:
        if (jackdata := _jackdata()) is not None:
            return jackdata[0].samplerate
        elif sys.platform == 'linux':
            from . import linuxaudio
            if linuxaudio.isPipewireRunning():
                info = linuxaudio.pipewireInfo()
                return info.sr if info else None
            else:
                return 44100
        else:
            return 44100


_backendPortaudioBlocking = _PortaudioBackend('blocking')
_backendPortaudioCallback = _PortaudioBackend('callback')


@_functools.cache
def _getAvailableAudioBackends() -> dict[str, AudioBackend]:
    backends: dict[str, AudioBackend] = {
        'portaudio': _backendPortaudioCallback,
        'pa_cb': _backendPortaudioCallback,
        'pa_bl': _backendPortaudioBlocking,
    }

    if _jackdata() is not None:
        backends['jack'] = _JackAudioBackend()


    if sys.platform == 'linux':
        backends['alsa'] = _AlsaBackend()
        backends['pulseaudio'] = _PulseAudioBackend()
    elif sys.platform == 'darwin':
        backends['coreaudio'] = AudioBackend('auhal', alwaysAvailable=True, hasSystemSr=True,
                                             needsRealtime=False, longname="coreaudio",
                                             platforms=('darwin',))
        backends['auhal'] = backends['coreaudio']
    return backends


def nextpow2(n:int) -> int:
    """ Returns the power of 2 higher or equal than n"""
    return int(2 ** _math.ceil(_math.log(n, 2)))


def findCsound() -> str | None:
    """
    Find the csound binary or None if not found
    """
    csound = _shutil.which("csound")
    if not csound:
        logger.error("csound is not in the path!")
    return csound


def _getVersionViaApi() -> tuple[int, int, int]:
    """
    Returns the csound version as tuple (major, minor, patch)
    """
    if (version := _cache.get('versionTriplet')) is not None:
        return version
    return _csoundGetInfoViaAPI()['versionTriplet']


@_functools.cache
def getVersion(useApi=True) -> tuple[int, int, int | str]:
    """
    Returns the csound version as tuple (major, minor, patch)

    Args:
        useApi: if True, the API is used to query the version. Otherwise
            the output of "csound --version" is parsed. Both versions might
            differ

    Returns:
        the versions as a tuple (major:int, minor:int, patch:int|str)

    Raises RuntimeError if csound is not present or its version
    can't be parsed
    """
    if useApi:
        return _getVersionViaApi()

    csound = findCsound()
    if not csound:
        raise IOError("Csound not found")
    cmd = '{csound} --version'.format(csound=csound).split()
    proc = _subprocess.Popen(cmd, stderr=_subprocess.PIPE)
    proc.wait()
    if proc.stderr is None:
        return (0, 0, 0)
    outputbytes = proc.stderr.read()
    if not outputbytes:
        raise RuntimeError("Could not read csounds output")
    output = outputbytes.decode('utf8')
    lines = output.splitlines()
    for line in lines:
        if match := _re.search(r"Csound\s+version\s+(\d+)\.(\d+)(\.\w+)?", line):
            major = int(match.group(1))
            minor = int(match.group(2))
            patch = match.group(3)
            if patch is None:
                patch = 0
            elif patch.isdigit():
                patch = int(patch)
            return (major, minor, patch)
    else:
        raise RuntimeError(f"Did not find a csound version, csound output: '{output}'")


def csoundSubproc(args: list[str], piped=True, wait=False) -> _subprocess.Popen:
    """
    Calls csound with given args in a subprocess, returns a subprocess.Popen object.

    Args:
        args: the args passed to csound (each argument is a string)
        piped: if True, stdout and stderr are piped to the Popen object
        wait: if True, wait until csound exits

    Returns:
        the subprocess.Popen object

    Raises RuntimeError if csound is not found

    Example
    -------

        >>> from csoundengine import csoundlib
        >>> proc = csoundlib.csoundSubproc(["-+rtaudio=jack", "-odac", "myfile.csd"])

    See Also
    ~~~~~~~~

    * :func:`runCsd`

    """
    csound = findCsound()
    if not csound:
        raise RuntimeError("Csound not found")
    p = _subprocess.PIPE if piped else None
    callargs = [csound]
    callargs.extend(args)
    proc = _subprocess.Popen(callargs, stderr=p, stdout=p)
    if wait:
        proc.wait()
    return proc


def getSystemSr(backend: str) -> float | None:
    """
    Get the system samplerate for a given backend

    None is returned if the backend does not support a system sr. At the
    moment only **jack** and **coreaudio** (auhal) report a system-sr

    Args:
        backend: the name of the backend (jack, pa_cb, auhal, etc)

    Returns:
        the system sr if the backend reports this information, or None

    See Also
    ~~~~~~~~

    * :func:`getAudioBackend`
    * :func:`getDefaultBackend`

    """
    b = getAudioBackend(backend)
    if not b:
        raise ValueError(f"backend {backend} not known")
    return b.getSystemSr()


def _getJackSrViaClient() -> float:
    import jack
    c = jack.Client("query")
    sr = c.samplerate
    c.close()
    return sr


def _getCsoundSystemSr(backend: str) -> float:
    if backend not in {'jack', 'auhal'}:
        raise ValueError(f"backend {backend} does not support system sr")
    import libcsound
    csound = libcsound.Csound()
    csound.setOption(f"-+rtaudio={backend}")
    csound.setOption("-odac")
    csound.setOption("--use-system-sr")
    csound.start()
    sr = csound.sr()
    csound.stop()
    return sr


def getDefaultBackend() -> AudioBackend:
    """
    Get the default active backend for platform

    Discard any backend which is not available at the moment

    ==============  =================================
    Platform        Backends (in order of priority)
    ==============  =================================
    Windows         portaudio
    macOS           auhal (coreaudio), portaudio
    linux           jack (if running), portaudio
    ==============  =================================

    """
    backends = audioBackends()
    if not backends:
        raise RuntimeError("No available backends")
    return max(backends, key=lambda b: b.priority)


_pluginsFolders = {
    '6.0': {
        'linux': '$HOME/.local/lib/csound/6.0/plugins64',
        'darwin': '$HOME/Library/csound/6.0/plugins64',
        'win32': '%LOCALAPPDATA%/csound/6.0/plugins64'
    },
    '7.0': {
        'linux': '$HOME/.local/lib/csound/7.0/plugins64',
        'darwin': '$HOME/Library/csound/7.0/plugins64',
        'win32': '%LOCALAPPDATA%/csound/7.0/plugins64'

    },
    'float32': {
        'linux': '$HOME/.local/lib/csound/6.0/plugins',
        'darwin': '$HOME/Library/csound/6.0/plugins',
        'win32': '%LOCALAPPDATA%/csound/6.0/plugins'
    }
}


def userPluginsFolder(float64=True, apiversion='6.0') -> str:
    """
    Returns the user plugins folder for this platform

    This is the folder where csound will search for user-installed
    plugins. The returned folder is always an absolute path. It is not
    checked if the folder actually exists.

    Args:
        float64: if True, report the folder for 64-bit plugins
        apiversion: 6.0 or 7.0

    Returns:
        the user plugins folder for this platform

    **Folders for 64-bit plugins**:

    ======== ===== =================================================
     OS       api  Plugins folder
    ======== ===== =================================================
     Linux    6.0  ``~/.local/lib/csound/6.0/plugins64``
              7.0  ``~/.local/lib/csound/7.0/plugins64``
     macOS    6.0  ``~/Library/csound/6.0/plugins64``
              7.0  ``~/Library/csound/7.0/plugins64``
     windows  6.0  ``C:/Users/<User>/AppData/Local/csound/6.0/plugins64``
              6.0  ``C:/Users/<User>/AppData/Local/csound/7.0/plugins64``
    ======== ===== =================================================

    For 32-bit plugins the folder is the same, without the '64' ending (``.../plugins``)
    """
    key = apiversion if float64 else 'float32'
    folders = _pluginsFolders[key]
    if sys.platform not in folders:
        raise RuntimeError(f"Platform {sys.platform} not known")
    folder = folders[sys.platform]
    return _os.path.abspath(_os.path.expandvars(folder))


def runCsd(csdfile: str,
           outdev='',
           indev='',
           backend='',
           nodisplay=False,
           nomessages=False,
           comment='',
           piped=False,
           extra: list[str] | None = None
           ) -> _subprocess.Popen:
    """
    Run the given .csd as a csound subprocess

    Args:
        csdfile: the path to a .csd file
        outdev: "dac" to output to the default device, the label of the
            device (dac0, dac1, ...), or a filename to render offline
            (-o option)
        indev: The input to use (for realtime) (-i option)
        backend: The name of the backend to use. If no backend is given,
            the default for the platform is used (this is only meaningful
            if running in realtime)
        nodisplay: if True, eliminates debugging info from output
        nomessages: if True, suppress debugging messages
        piped: if True, the output of the csound process is piped and can be accessed
            through the Popen object (.stdout, .stderr)
        extra: a list of extraOptions arguments to be passed to csound
        comment: if given, will be added to the generated output
            as comment metadata (when running offline)

    Returns:
        the `subprocess.Popen` object. In order to wait until
        rendering is finished in offline mode, call .wait on the
        returned process

    .. seealso:: :func:`csoundSubproc`
    """
    args = []
    offline = True
    if outdev is not None and outdev:
        args.extend(["-o", outdev])
        if outdev.startswith("dac"):
            offline = False
    if not offline and not backend:
        backend = getDefaultBackend().name
    if backend:
        args.append(f"-+rtaudio={backend}")
    if indev:
        args.append(f"-i {indev}")
    if nodisplay:
        args.append('-d')
    if nomessages:
        args.append('-m16')
    if comment and offline:
        args.append(f'-+id_comment="{comment}"'
                    )
    if extra:
        args.extend(extra)
    args.append(csdfile)
    return csoundSubproc(args, piped=piped)


def joinCsd(orc: str, sco='', options: list[str] | None = None) -> str:
    """
    Joins an orc and a score (both as str), returns a csd as string

    Args:
        orc: the text of the orchestra
        sco: the text of the score
        options: any command line options to be included

    Returns:
        the text of the csd
    """
    optionstr = "" if options is None else "\n".join(options)
    csd = r"""
<CsoundSynthesizer>
<CsOptions>
{optionstr}
</CsOptions>
<CsInstruments>

{orc}

</CsInstruments>
<CsScore>

{sco}

</CsScore>
</CsoundSynthesizer>
    """.format(optionstr=optionstr, orc=orc, sco=sco)
    csd = _textwrap.dedent(csd)
    return csd


@dataclasses.dataclass
class CsoundProc:
    """
    A CsoundProc wraps a running csound subprocess

    Attributes:
        proc: the running csound subprocess
        backend: the backend used
        outdev: the output device
        sr: the sample rate of the running process
        nchnls: the number of channels
        csdstr: the csd being run, as a str
    """
    proc: _subprocess.Popen
    backend: str
    outdev: str
    sr: int
    nchnls: int
    csdstr: str = ""


def testCsound(dur=8., nchnls=2, backend='', device="dac", sr=0,
               verbose=True
               ) -> CsoundProc:
    """
    Test the current csound installation for realtime output

    Args:
        dur: the duration of the test
        nchnls: the number of output channels
        backend: which backend to use.
        device: which device to use
        sr: the sample rate. Use 0 to use system sample rate if applicable,
            or a default sample rate otherwise
        verbose: if True, make csound display debugging and status information

    Returns:
        a :class:`CsoundProc`
    """
    backend = backend or getDefaultBackend().name
    if not sr:
        sr = getSamplerateForBackend(backend) or 44100
    printchan = "printk2 kchn" if verbose else ""
    orc = f"""
sr = {sr}
ksmps = 128
nchnls = {nchnls}

instr 1
    iperiod = 1
    kchn init -1
    ktrig metro 1/iperiod
    kchn = (kchn + ktrig) % nchnls
    anoise pinker
    outch kchn+1, anoise
    {printchan}
endin
    """
    sco = f"i1 0 {dur}"
    orc = _textwrap.dedent(orc)
    logger.debug(orc)
    csd = joinCsd(orc, sco=sco)
    tmp = _tempfile.mktemp(suffix=".csd")
    open(tmp, "w").write(csd)
    proc = runCsd(tmp, outdev=device, backend=backend)
    return CsoundProc(proc=proc, backend=backend, outdev=device, sr=sr,
                      nchnls=nchnls, csdstr=csd)


def installedOpcodes(cached=True, opcodedir: str = '') -> set[str]:
    """
    Return a list of the opcodes present

    Args:
        cached: if True, results are remembered between calls
        opcodedir: if given, plugin libraries will be loaded from
            this path (option --opcode-dir in csound). In this case
            the cache is not used

    Returns:
        a list of all available opcodes
    """
    if opcodedir:
        cached = False
    if cached and _cache.get('opcodes') is not None:
        return _cache['opcodes']
    return _csoundGetInfoViaAPI(opcodedir=opcodedir)['opcodes']


def _csoundGetInfoViaAPI(opcodedir='') -> dict:
    global _cache
    import libcsound
    cs = libcsound.Csound()
    cs.setOption("-d")
    cs.setOption("--nosound")
    cs.createMessageBuffer(echo=False)
    if opcodedir:
        cs.setOption(f'--opcode-dir={opcodedir}')
    version = cs.version()
    vs = str(version)
    patch = int(vs[-1])
    minor = int(vs[-3:-1])
    major = int(vs[:-3])
    versionTriplet = (major, minor, patch)
    opcodes = cs.getOpcodes()
    opcodenames = set(opc.name for opc in opcodes)
    _cache['versionTriplet'] = versionTriplet
    _cache['opcodes'] = opcodenames
    _cache['opcodedefs'] = opcodes
    # opcodes, n = cs.newOpcodeList()
    # assert opcodes is not None
    # opcodeNames = [opc.opname.decode('utf-8') for opc in opcodes]
    # cs.disposeOpcodeList(opcodes)
    # opcodes = list(set(opcodeNames))
    # _cache['opcodes'] = opcodes
    # cs.stop()
    return {'opcodedefs': opcodes,
            'opcodes': opcodenames,
            'versionTriplet': versionTriplet}


def _opcodesList(opcodedir='') -> list[str]:
    options = ["-z"]
    if opcodedir:
        options.append(f'--opcode-dir={opcodedir}')
    s = csoundSubproc(options)
    assert s.stderr is not None
    lines = s.stderr.readlines()
    allopcodes = []
    for line in lines:
        if line.startswith(b"end of score"):
            break
        opcodes = line.decode('utf8').split()
        if opcodes:
            allopcodes.extend(opcodes)
    return allopcodes


def saveAsGen23(data: Sequence[float] | np.ndarray,
                outfile: str,
                fmt="%.12f",
                header=''
                ) -> None:
    """
    Saves the data to a gen23 table

    .. note::
        gen23 is a 1D list of numbers in text format, sepparated by a space

    Args:
        data: A 1D sequence (list or array) of floats
        outfile: The path to save the data to. Recommended extension: '.gen23'
        fmt: If saving frequency tables, fmt can be "%.1f" and save space, for
            amplitude the default if "%.12f" is best
        header: If specified it is included as a comment as the first line
            (csound will skip it). It is there just to document what is in the table


    Example
    ~~~~~~~

    .. code-block:: python

        >>> import bpf4
        >>> from csoundengine import csoundlib
        >>> a = bpf.linear(0, 0, 1, 10, 2, 300)
        >>> dt = 0.01
        >>> csoundlib.saveAsGen23(a[::dt].ys, "out.gen23", header=f"dt={dt}")


    In csound:

    .. code-block:: csound

        gi_tab ftgen 0, 0, 0, -23, "out.gen23"

        instr 1
          itotaldur = ftlen(gi_tab) * 0.01
          ay poscil 1, 1/itotaldur, gi_tab
        endin
    """
    if header:
        np.savetxt(outfile, data, fmt=fmt, header="# " + header)
    else:
        np.savetxt(outfile, data, fmt=fmt)


def _metadataAsComment(d: dict[str, Any], maxSignificantDigits=10,
                       sep=", ") -> str:
    fmt = f"%.{maxSignificantDigits}g"
    parts = []
    for key, val in d.items():
        if isinstance(val, int):
            valstr = str(val)
        elif isinstance(val, float):
            valstr = fmt % val
        elif isinstance(val, str):
            valstr = f'"{val}"'
        else:
            raise TypeError(f"Value should be int, float or str, got {val}")
        parts.append(f"{key}: {valstr}")
    return sep.join(parts)


def saveMatrixAsMtx(outfile: str,
                    data: np.ndarray,
                    metadata: dict[str, str | float] | None = None,
                    encoding="float32",
                    title='',
                    sr: int = 44100) -> None:
    """
    Save `data` in wav format using the mtx extension

    This is not a real output. It is used to transfer the data in
    binary form to be read by another program. To distinguish this from a
    normal wav file an extension `.mtx` is recommended. Data is saved
    always flat, and a header with the shape of `data` is included
    before the data.

    **Header Format**::

        headerlength, numRows, numColumns, ...

    The description of each metadata value is included as wav metadata
    at the comment key with the format::

        "headerSize: xx, numRows: xx, numColumns: xx, columns: 'headerSize numRows numColumns ...'"

    This metadata can be retrieved in csound via:

    .. code-block:: csound

        itabnum ftgen 0, 0, 0, -1, "sndfile.mtx", 0, 0, 1
        Scomment = filereadmeta("sndfile.mtx", "comment")
        imeta = dict_loadstr(Scomment)
        ScolumnNames = dict_get(imeta, "columns")
        idatastart = tab_i(0, itabnum)
        inumrows = dict_get(imeta, "numRows")
        ; inumrows can also be retrieved by reading the table at index 1
        ; inumrows = tab_i(1, itabnum)
        inumcols = tab_i(2, itabnum)
        ; The data at (krow, kcol) can be read via
        kvalue = tab(idatastart + krow*inumcols + kcol, itabnum)

        ; Alternatively an array can be created as a view:
        kArr[] memview itabnum, idatastart
        reshapearray kArr, inumrows, inumcols
        kvalue = kArr[krow][kcol]

    Args:
        outfile (str): The path where the data is written to
        data (numpy array): a numpy array of shape (numcols, numsamples). A 2D matrix
            representing a series of streams sampled at a regular period (dt)
        metadata: Any float values here are included in the header, and the description
            of this data is included as metadata in the wav file
        encoding: the data can be encoded in float32 or float64
        title: if given will be included in the output metadata
        sr: sample rate. I
    """
    assert isinstance(outfile, str)
    assert encoding == 'float32' or encoding == 'float64'
    if _os.path.splitext(outfile)[1] != ".mtx":
        logger.warning(f"The extension should be .mtx, but asked to save"
                       f"the matrix as {outfile}")

    import sndfileio
    header: list[int|float] = [3, data.shape[0], data.shape[1]]
    allmeta: dict[str, Any] = {
        'headerSize': 3,
        'numRows': data.shape[0],
        'numColumns': data.shape[1]
    }
    columns = ['headerSize', 'numRows', 'numColumns']

    if metadata:
        for k, v in metadata.items():
            if isinstance(v, (int, float)):
                header.append(v)
                allmeta[k] = v
                columns.append(k)
        headersize = len(allmeta)
        allmeta['HeaderSize'] = headersize
        header[0] = headersize

        for k, v in metadata.items():
            if isinstance(v, str):
                allmeta[k] = v

    allmeta['columns'] = " ".join(columns)
    wavmeta = {'comment': _metadataAsComment(allmeta),
               'software': 'MTX1'}
    if title:
        wavmeta['title'] = title
    sndwriter = sndfileio.sndwrite_chunked(outfile=outfile, sr=sr,
                                           encoding=encoding, metadata=wavmeta,
                                           fileformat='wav')
    sndwriter.write(np.array(header, dtype=float))
    sndwriter.write(data.ravel())
    sndwriter.close()


def saveMatrixAsGen23(outfile: str,
                      mtx: np.ndarray,
                      extradata: list[float] | None = None,
                      header=True
                      ) -> None:
    """
    Save a numpy 2D array as gen23

    Args:
        outfile (str): the path to save the data to. Suggestion: use '.gen23' as ext
        mtx (np.ndarray): a 2D array of floats
        extradata: if given, this data will be prependedto the data in `mtx`.
            Implies `include_header=True`
        header: if True, a header of the form [headersize, numrows, numcolumns]
            is prepended to the data.

    .. note::
        The format used by gen23 is a text format with numbers separated by any space.
        When read inside csound the table is of course 1D but can be interpreted as
        2D with the provided header metadata.

    """
    numrows, numcols = mtx.shape
    mtx = mtx.round(6)
    with open(outfile, "w") as f:
        if header or extradata:
            headerrow = [3., numrows, numcols]
            if extradata is not None:
                headerrow.extend(extradata)
            headerrow[0] = len(headerrow)
            f.write(" ".join(np.array(headerrow).astype(str)))
            f.write("\n")
        for row in mtx:
            rowstr = " ".join(row.astype(str))
            f.write(rowstr)
            f.write("\n")


@dataclasses.dataclass
class MidiDevice:
    """
    A MidiDevice holds information about a midi device for a given backend

    Attributes:
        index: the index as listed by csound and passed to -M
        name: the name of the device
        kind: the kind of the device ('input', 'output')
    """
    deviceid: str
    name: str
    kind: str = 'input'


@dataclasses.dataclass
class AudioDevice:
    """
    An AudioDevice holds information about a an audio device for a given backend

    Attributes:
        id: the device identification (dac3, adc2, etc)
        name: the name of the device
        index: the index of this audio device, as passed to adcXX or dacXX
        kind: 'output' or 'input'
        numChannels: the number of channels
        isPhysical: True if this is a physical (hardware) device. Used for jack
    """
    id: str
    name: str
    kind: str
    backend: str
    index: int = -1
    numChannels: int | None = 0
    isPhysical: bool | None = None

    def info(self) -> str:
        """
        Returns a summary of this device in one line
        """
        s = f"{self.id}:{self.name}"
        if self.kind == 'output':
            s += f":{self.numChannels}outs"
        else:
            s += f":{self.numChannels}ins"
        return s


@_cachetools.cached(cache=_cachetools.TTLCache(10, 20))
def getDefaultAudioDevices(backend='') -> tuple[AudioDevice | None, AudioDevice | None]:
    """
    Returns the default audio devices for a given backend

    Args:
        backend: the backend to use (None to get the default backend)

    Returns:
        a tuple (input devices, output devices)

    .. note:: Results are cached for a period of time

    """
    backendDef = getAudioBackend(backend)
    if backendDef is None:
        raise ValueError(f"Backend {backend} not known")
    return backendDef.defaultAudioDevices()


@_cachetools.cached(cache=_cachetools.TTLCache(10, 20))
def getAudioDevices(backend) -> tuple[list[AudioDevice], list[AudioDevice]]:
    """
    Returns (indevices, outdevices), where each of these lists is an AudioDevice.

    Args:
        backend: specify a backend supported by your installation of csound.
            None to use a default for you OS

    Returns:
        a tuple of (input devices, output devices)

    .. note::

        For jack an audio device is a client

    Each returned device is an AudioDevice instance with attributes:

    * index: The device index
    * label: adc{index} for input devices, dac{index} for output devices.
        The label can be passed to csound directly with either the -i or the -o flag
        (``-i{label}`` or ``-o{label}``)
    * name: A description of the device
    * ins: number of input channels
    * outs: number of output channels

    ======== ==== =====  ==== ==================  =======================
    Backend  OSX  Linux  Win   Multiple-Devices    Description
    ======== ==== =====  ==== ==================  =======================
    jack      x      x    -     -                  Jack
    auhal     x      -    -     x                  CoreAudio
    pa_cb     x      x    x     x                  PortAudio (Callback)
    pa_bl     x      x    x     x                  PortAudio (blocking)
    ======== ==== =====  ==== ==================  =======================
    """
    backendDef = getAudioBackend(backend)
    if backendDef is None:
        raise ValueError(f"Backend '{backend}' not supported, known backends: {list(_getAvailableAudioBackends().keys())}")
    return backendDef.audioDevices()


def getSamplerateForBackend(backend='') -> int | None:
    """
    Returns the samplerate reported by the given backend

    Args:
        backend: the backend to query. None to use the default backend

    Returns:
        the samplerate for the given backend or None if failed
    """
    backendDef = getAudioBackend(backend)
    if backendDef is None:
        possible = [name for name, backend in _getAvailableAudioBackends().items()
                    if sys.platform in backend.platforms]
        raise ValueError(f"Backend '{backend}' not supported. Possible backends for this platform: {possible}")
    if not backendDef.isAvailable():
        raise RuntimeError(f"Audiobackend {backendDef.name} is not available")
    return backendDef.getSystemSr()


def _csoundTestJackRunning():
    proc = csoundSubproc(['-+rtaudio=jack', '-odac', '--get-system-sr'], wait=True, piped=True)
    assert proc.stderr is not None
    return b'could not connect to JACK server' not in proc.stderr.read()


def audioBackends() -> list[AudioBackend]:
    """
    Return a list of available audio backends

    Returns:
        a list of AudioBackend

    If available is True, only those backends supported for the current
    platform and currently available are returned. For example, jack will
    not be returned in linux if the jack server is not running.

    Example
    ~~~~~~~

        >>> from csoundengine import *
        >>> [backend.name for backend in audioBackends()]
        ['jack', 'pa_cb', 'pa_bl', 'alsa']
    """
    return list(_getAvailableAudioBackends().values())


def dumpAudioBackends() -> None:
    """
    Prints all **available** backends and their properties as a table
    """
    rows = []
    headers = "backend longname sr".split()
    backends = audioBackends()
    backends.sort(key=lambda backend:backend.name)
    from emlib.misc import print_table

    for b in backends:
        if b.hasSystemSr:
            sr = str(b.getSystemSr())
        else:
            sr = "-"
        rows.append((b.name, b.longname, sr))
    print_table(rows, headers=headers, showindex=False)


def getAudioBackend(name='') -> AudioBackend | None:
    """
    Given the name of the backend, return the AudioBackend structure

    Only available backends are considered. Some backends listed
    for a given platform might not be running and thus will
    not be listed

    Args:
        name: the name of the backend

    Returns:
        the AudioBackend structure, or None if the audio backend
        is not available or unknown

    ========== =================== ======= ========= =======
    Name       Description         Linux   Windows   MacOS
    ========== =================== ======= ========= =======
    pa_cb      portaudio-callback     ✓        ✓        ✓
    pa_bl      portaudio-blocking     ✓        ✓        ✓
    portaudio  alias to pa_cb         ✓        ✓        ✓
    jack       jack                   ✓        ?        ✓
    pulse      pulseaudio             ✓        ✗        ✗
    pulseaudio alias to pulse         ✓        ✗        ✗
    auhal      coreaudio              ✗        ✗        ✓
    coreaudio  alias to auhal         ✗        ✗        ✓
    ========== =================== ======= ========= =======
    """
    if not name:
        return getDefaultBackend()
    return _getAvailableAudioBackends().get(name)


def getAudioBackendNames() -> list[str]:
    """
    Returns a list with the names of the available audio backends for this

    Returns:
        a list with the names of all available backends for the
        given platform

    Example
    -------

        >>> from csoundengine.csoundlib import *
        >>> getAudioBackendNames()   # in Linux
        ['jack', 'pa_cb', 'pa_bl', 'alsa', 'pulse']
        >>> getAudioBackendNames(platform='macos')
        ['pa_cb', 'pa_bl', 'auhal']
        # In linux with pulseaudio disabled
        >>> getAudioBackendNames(available=True)
        ['jack', 'pa_cb', 'pa_bl', 'alsa']
    """
    return [b.name for b in audioBackends()]


def _quoteIfNeeded(arg: float | int | str) -> float | int | str:
    if isinstance(arg, str):
        return emlib.textlib.quoteIfNeeded(arg)
    else:
        return arg


_normalizer = emlib.textlib.makeReplacer({".":"_", ":":"_", " ":"_"})


def normalizeInstrumentName(name: str) -> str:
    """
    Transform name so that it can be accepted as an instrument name
    """
    return _normalizer(name)


_fmtoptions = {
    'pcm16': '',
    'pcm24': '--format=24bit',
    'float32': '--format=float',  # also -f
    'float64': '--format=double',
    'vorbis': '--format=vorbis'
}


_optionForSampleFormat = {
    'wav': '--format=wav',   # could also be --wave
    'aif': '--format=aiff',
    'aiff': '--format=aiff',
    'flac': '--format=flac',
    'ogg': '--format=ogg'
}


_csoundFormatOptions = {'-3', '-f', '--format=24bit', '--format=float',
                        '--format=double', '--format=long', '--format=vorbis',
                        '--format=short'}


_defaultEncodingForFormat = {
    'wav': 'float32',
    'flac': 'pcm24',
    'aif': 'float32',
    'aiff': 'float32',
    'ogg': 'vorbis'
}


def _normalizeArgs(args, quote=True) -> list[float | str]:
    out = []
    for arg in args:
        if isinstance(arg, str):
            if quote:
                arg = emlib.textlib.quoteIfNeeded(arg)
            out.append(arg)
        elif isinstance(arg, (int, float)):
            out.append(arg)
        else:
            try:
                out.append(float(arg))
            except ValueError as e:
                raise ValueError(f"Could not interpret {arg} as float: {e}")
    return out


def csoundOptionsForOutputFormat(fmt='wav',
                                 encoding=''
                                 ) -> list[str]:
    """
    Returns the command-line options for the given format+encoding

    Args:
        fmt: the format of the output file ('wav', 'flac', 'aif', etc)
        encoding: the encoding ('pcm16', 'pcm24', 'float32', etc). If not given,
            the best encoding for the given format is chosen

    Returns:
        a tuple of two strings holding the command-line options for the given
        sample format/encoding

    Example
    -------

        >>> csoundOptionsForOutputFormat('flac')
        ('--format=flac', '--format=24bit')
        >>> csoundOptionsForOutputFormat('wav', 'float32')
        ('--format=wav', '--format=float')
        >>> csoundOptionsForOutputFormat('aif', 'pcm16')
        ('--format=aiff', '--format=short')

    .. seealso:: :func:`csoundOptionForSampleEncoding`
    """
    if fmt.startswith("."):
        fmt = fmt[1:]
    assert fmt in _defaultEncodingForFormat, f"Unknown format: {fmt}, possible formats are: " \
                                             f"{_defaultEncodingForFormat.keys()}"
    if not encoding:
        encoding = _defaultEncodingForFormat.get(fmt)
        if not encoding:
            raise ValueError(f"Default encoding unknown for format {fmt}")
    encodingOption = csoundOptionForSampleEncoding(encoding)
    fmtOption = _optionForSampleFormat[fmt]
    options = [fmtOption]
    if encodingOption:
        options.append(encodingOption)
    return options


def csoundOptionForSampleEncoding(encoding: str) -> str:
    """
    Returns the command-line option for the given sample encoding.

    Given a sample encoding of the form pcmXX or floatXX, where
    XX is the bit-rate, returns the corresponding command-line option
    for csound

    Args:
        fmt (str): the desired sample format. Either pcmXX, floatXX, vorbis
          where XX stands for the number of bits per sample (pcm24,
          float32, etc)

    Returns:
        the csound command line option corresponding to the given format

    Example
    -------

        >>> csoundOptionForSampleEncoding("pcm24")
        --format=24bit
        >>> csoundOptionForSampleEncoding("float64")
        --format=double

    .. seealso:: :func:`csoundOptionsForOutputFormat`

    """
    if encoding not in _fmtoptions:
        raise ValueError(f'format {encoding} not known. Possible values: '
                         f'{_fmtoptions.keys()}')
    return _fmtoptions[encoding]


def mincer(sndfile: str,
           outfile: str,
           timecurve: float | Callable[[float], float] = 1.,
           pitchcurve: float | Callable[[float], float] = 1.,
           lock=False, fftsize=2048, ksmps=128, debug=False
           ) -> dict:
    """
    Stretch/Pitchshift a output using csound's mincer opcode (offline)

    Args:
        sndfile: the path to a soundfile
        timecurve: a func mapping time to playback time or a scalar indicating
            a timeratio (2 means twice as fast, 1 to leave unmodified)
        pitchcurve: a func time to pitchscale, or a scalar indicating a freqratio
        outfile: the path to a resulting outfile. The resulting file is always a
            32-bit float .wav file. The samplerate and number of channels match those
            of the input file
        dt: the sampling period to sample the curves
        lock: should mincer be run with phase-locking?
        fftsize: the size of the fft
        ksmps: the ksmps to pass to the csound process
        debug: run csound with debug information

    Returns:
        a dict with information about the process (keys: outfile,
        csdstr, csd)

    .. note::

        If the mapped time excedes the bounds of the sndfile, silence is generated.
        For example, a negative time or a time exceding the duration of the sndfile

    Examples
    --------

        # Example 1: stretch a output 2x

        >>> from csoundengine import csoundlib
        >>> import bpf4
        >>> import sndfileio
        >>> snddur = sndfileio.sndinfo("mono.wav").duration
        >>> timecurve = bpf4.linear(0, 0, snddur*2, snddur)
        >>> mincer(sndfile, "mono2.wav", timecurve=timecurve, pitchcurve=1)
    """
    import sndfileio

    info = sndfileio.sndinfo(sndfile)
    sr = info.samplerate
    nchnls = info.channels
    t0 = 0
    dt = 0.002
    if isinstance(timecurve, (int, float)):
        t1 = info.duration / timecurve
        ts = np.arange(0, t1 + dt, dt)
        times = ts * (1./timecurve)
    elif callable(timecurve):
        t1 = info.duration
        ts = np.arange(0, t1 + dt, dt)
        times = [timecurve(float(t)) for t in ts]
    else:
        raise TypeError("timecurve should be either a scalar or a bpf")

    if isinstance(pitchcurve, (int, float)):
        pitches = np.ones_like(ts) * pitchcurve
    else:
        pitches = [pitchcurve(float(t)) for t in ts]

    ts = np.arange(t0, t1+dt, dt)
    fmt = "%.12f"
    _, time_gen23 = _tempfile.mkstemp(prefix='time-', suffix='.gen23')
    np.savetxt(time_gen23, times, fmt=fmt, header=str(dt), comments='')
    _, pitch_gen23 = _tempfile.mkstemp(prefix='pitch-', suffix='.gen23')
    np.savetxt(pitch_gen23, pitches, fmt=fmt, header=str(dt), comments='')
    ext = _os.path.splitext(outfile)[1][1:]
    extraoptions = []
    extraoptions.extend(csoundOptionsForOutputFormat(fmt=ext))
    optionsstr = '\n'.join(extraoptions)
    csd = f"""
    <CsoundSynthesizer>
    <CsOptions>
    -o {outfile}

    {optionsstr}

    </CsOptions>
    <CsInstruments>

    sr = {sr}
    ksmps = {ksmps}
    nchnls = {nchnls}
    0dbfs = 1.0

    gi_snd   ftgen 0, 0, 0, -1,  "{sndfile}", 0, 0, 0
    gi_time  ftgen 0, 0, 0, -23, "{time_gen23}"
    gi_pitch ftgen 0, 0, 0, -23, "{pitch_gen23}"

    instr vartimepitch
        idt tab_i 0, gi_time
        ilock = {int(lock)}
        ifftsize = {fftsize}
        ikperiod = ksmps/sr
        isndfiledur = ftlen(gi_snd) / ftsr(gi_snd)
        isndchnls = ftchnls(gi_snd)
        ifade = ikperiod*2
        inumsamps = ftlen(gi_time)
        it1 = (inumsamps-2) * idt           ; account for idt and last value
        kt timeinsts
        aidx    linseg 1, it1, inumsamps-1
        at1     tablei aidx, gi_time, 0, 0, 0
        kpitch  tablei k(aidx), gi_pitch, 0, 0, 0
        kat1 = k(at1)
        kgate = (kat1 >= 0 && kat1 <= isndfiledur) ? 1 : 0
        agate = interp(kgate)
        aenv linseg 0, ifade, 1, it1 - (ifade*2), 1, ifade, 0
        aenv *= agate
        if isndchnls == 1 then
            a0  mincer at1, 1, kpitch, gi_snd, ilock, ifftsize, 8
            outch 1, a0*aenv
        else
            a0, a1   mincer at1, 1, kpitch, gi_snd, ilock, ifftsize, 8
            outs a0*aenv, a1*aenv
        endif

      if (kt >= it1 + ikperiod) then
        event "i", "exit", 0.1, 1
            turnoff
        endif
    endin

    instr exit
        puts "exiting!", 1
        exitnow
    endin

    </CsInstruments>
    <CsScore>
    i "vartimepitch" 0 -1
    f 0 36000

    </CsScore>
    </CsoundSynthesizer>
    """
    _, csdfile = _tempfile.mkstemp(suffix=".csd")
    with open(csdfile, "w") as f:
        f.write(csd)
    _subprocess.call(["csound", "-f", csdfile])
    if not debug:
        _os.remove(time_gen23)
        _os.remove(pitch_gen23)
        _os.remove(csdfile)
    return {'outfile': outfile, 'csdstr': csd, 'csd': csdfile}


def _instr_as_orc(instrid, body, initstr, sr, ksmps, nchnls):
    orc = """
sr = {sr}
ksmps = {ksmps}
nchnls = {nchnls}
0dbfs = 1

{initstr}

instr {instrid}
    {body}
endin

    """.format(sr=sr, ksmps=ksmps, instrid=instrid, body=body, nchnls=nchnls, initstr=initstr)
    return orc


def recInstr(body: str, events: list, init='', outfile='',
             sr=44100, ksmps=64, nchnls=2, a4=442, samplefmt='float',
             dur=None
             ) -> tuple[str, _subprocess.Popen]:
    """
    Record one instrument for a given duration

    Args:
        dur: the duration of the recording
        body: the body of the instrument
        init: the initialization code (ftgens, global vars, etc)
        outfile: the generated output, or None to generate a temporary file
        events: a list of events, where each event is a list of pargs passed
            to the instrument, beginning with p2: delay, dur, [p4, p5, ...]
        sr: the samplerate
        a4: A4 frequency
        ksmps: block size
        nchnls: number of output channels
        samplefmt: defines the sample format used for outfile, one of (16, 24, 32, 'float')

    Returns:
        a tuple (outfile to be generated, _subprocess.Popen running csound)
    """
    if not isinstance(events, list) or not all(isinstance(event, (tuple, list)) for event in events):
        raise ValueError("events is a data., where each item is a data. of pargs passed to"
                         "the instrument, beginning with p2: [delay, dur, ...]"
                         f"Got {events} instead")

    from .csd import Csd
    csd = Csd(sr=sr, ksmps=ksmps, nchnls=nchnls, a4=a4)
    if not outfile:
        outfile = _tempfile.mktemp(suffix='.wav', prefix='csdengine-rec-')

    if init:
        csd.addGlobalCode(init)

    instrnum = 100

    csd.addInstr(instrnum, body)
    for event in events:
        start, dur = event[0], event[1]
        csd.addEvent(instrnum, start, dur, event[2:])

    if dur is not None:
        csd.setEndMarker(dur)

    fmtoption = {16: '', 24: '-3', 32: '-f', 'float': '-f'}.get(samplefmt)
    if fmtoption is None:
        raise ValueError("samplefmt should be one of 16, 24, 32, or 'float'")
    csd.addOptions(fmtoption)

    renderjob = csd.run(output=outfile)
    assert renderjob.process is not None
    return outfile, renderjob.process


def _ftsaveReadText(path: str) -> list[np.ndarray]:
    # a file can have multiple tables saved
    lines = iter(open(path))
    tables = []
    while True:
        tablength = -1
        try:
            headerstart = next(lines)
            if not headerstart.startswith("===="):
                raise IOError(f"Expecting header start, got {headerstart}")
        except StopIteration:
            # no more tables
            break
        # Read header
        for line in lines:
            if line.startswith("flen:"):
                tablength = int(line[5:])
            if 'END OF HEADER' in line:
                break
        if tablength < 0:
            raise IOError("Could not read table length")
        values = np.zeros((tablength+1,), dtype=float)
        # Read data
        for i, line in enumerate(lines):
            if line.startswith("---"):
                break
            values[i] = float(line)
        tables.append(values)
    return tables


def ftsaveRead(path, mode="text") -> list[np.ndarray]:
    """
    Read a file saved by ftsave, returns a list of tables
    """
    if mode == "text":
        return _ftsaveReadText(path)
    else:
        raise ValueError(f"mode {mode} not supported")


def getNchnls(backend='',
              outpattern='',
              inpattern='',
              defaultin=2,
              defaultout=2
              ) -> tuple[int, int]:
    """
    Get the default number of channels for a given device

    Args:
        backend: the backend, one of 'jack', 'portaudio', etc. None to use default
        outpattern: the output device. Use None for default device. Otherwise either the
            device id ("dac0") or a regex pattern matching the long name of the device
        inpattern: the input device. Use None for default device. Otherwise either the
            device id ("dac0") or a regex pattern matching the long name of the device
        defaultin: default value returned if it is not possible to determine
            the number of channels for given backend+device
        defaultout: default value returned if it is not possible to determine
            the number of channels for given backend/device

    Returns:
        a tuple (nchnls_i, nchnls) for that backend+device combination

    """
    backendDef = getAudioBackend(backend)
    if not backendDef:
        raise RuntimeError(f"Backend '{backend}' not found")
    adc, dac = backendDef.defaultAudioDevices()
    if not outpattern:
        outdev = dac
    else:
        outdev = backendDef.searchAudioDevice(outpattern, kind='output')
        if not outdev:
            indevs, outdevs = backendDef.audioDevices()
            outdevids = [d.id for d in outdevs]
            raise ValueError(f"Output device '{outpattern}' not found. Possible devices "
                             f"are: {outdevids}")
    nchnls = outdev.numChannels if (outdev and outdev.numChannels is not None) else defaultout
    if not inpattern:
        indev = adc
    else:
        indev = backendDef.searchAudioDevice(inpattern, kind='input')
        if not indev:
            raise ValueError(f"Input device {inpattern} not found")
    nchnlsi = indev.numChannels if (indev and indev.numChannels is not None) else defaultin
    return nchnlsi, nchnls


def _getNchnlsJackViaJackclient(indevice: str, outdevice: str
                                ) -> tuple[int, int]:
    """
    Get the number of ports for the given clients using JACK-Client
    This is faster than csound and should give the same results

    Args:
        indevice (str): A regex pattern matching the input client, or "adc" or
            None to query the physical ports
        outdevice (str): A regex pattern matching the output client
            Use "dac" or None to query the physical ports
    Returns:
        a tuple (number of inputs, number of outputs)
    """
    import jack
    c = jack.Client("query")
    if indevice == "adc" or not indevice:
        inports = [p for p in c.get_ports(is_audio=True, is_output=True, is_physical=True)]
    else:
        inports = c.get_ports(indevice, is_audio=True, is_output=True)
    if outdevice == "dac" or not outdevice:
        outports = [p for p in c.get_ports(is_audio=True, is_input=True, is_physical=True)]
    else:
        outports = c.get_ports(outdevice, is_audio=True, is_input=True)
    c.close()
    return len(inports), len(outports)


def _parsePortaudioDeviceName(name: str) -> tuple[str, str, int, int]:
    """
    Parses a string like "HDA Intel PCH: CX20590 Analog (hw:0,0) [ALSA, 2 in, 4 out]"

    Args:
        name: the name of the device

    Returns:
        a tuple: ("HDA Intel PCH: CX20590 Analog (hw:0,0)", "ALSA", 2, 4)
    """
    devname, rest = name.split("[")
    rest = rest.replace("]", "")
    if "," in rest:
        api, inchstr, outchstr = rest.split(",")
        inch = int(inchstr.split()[0])
        outch = int(outchstr.split()[0])
    else:
        api = rest
        inch = -1
        outch = -1
    return devname.strip(), api, inch, outch


def dumpAudioInfo(backend=''):
    """
    Dump information about audio backends and audio devices for the selected backend
    """
    if not backend:
        dumpAudioBackends()
        print()
    dumpAudioDevices(backend=backend)


def dumpAudioDevices(backend=''):
    """
    Print a list of audio devices for the given backend.

    If backend is not given, the default backend (of all available backends
    for the current platform) is chosen
    """

    backendDef = getAudioBackend(backend)
    if backendDef is None:
        raise ValueError(f"Backend '{backend}' not supported")

    from emlib.misc import print_table

    print(f"Backend: {backendDef.name}")
    indevs, outdevs = getAudioDevices(backend=backend)
    fields = [field.name for field in dataclasses.fields(AudioDevice)]
    inputrows = [dataclasses.astuple(dev) for dev in indevs]
    outputrows = [dataclasses.astuple(dev) for dev in outdevs]
    print("\nInput Devices:")
    if inputrows:
        print_table(inputrows, headers=fields, showindex=False)
    else:
        print("-- No input devices")

    print("\nOutput Devices:")
    if outputrows:
        print_table(outputrows, headers=fields, showindex=False)
    else:
        print("-- No output devices")


def instrNames(instrdef: str) -> list[int | str]:
    """
    Returns the list of names/instrument numbers in the instrument definition.

    Most of the time this list will have one single element, either an instrument
    number or a name

    Args:
        instrdef: the code defining an instrument

    Returns:
        a list of names/instrument numbers. An empty list is returned if
        this is not a valid instr definition

    Example
    -------

        >>> instr = r'''
        ... instr 10, foo
        ...     outch 1, oscili:a(0.1, 440)
        ... endin
        ... '''
        >>> instrNames(instr)
        [10, "foo"]

    """
    lines = instrdef.splitlines()
    matches = [line for line in lines if _re.match(r"^[\ \t]*\binstr\b", line)]
    if len(matches) > 1:
        raise ValueError(f"Expected only one instrument definition, got {matches}")
    elif len(matches) == 0:
        return []
    line = matches[0].strip()
    names = [name.strip() for name in line[6:].split(",")]
    return [int(name) if name.isdigit() else name
            for name in names]


def _hashdict(d: dict) -> int:
    return hash((frozenset(d.keys()), frozenset(d.values())))


def bestSampleEncodingForExtension(ext: str) -> str:
    """
    Given an extension, return the best sample encoding.

    .. note::

        float64 is not considered necessary for holding sound information

    Args:
        ext (str): the extension of the file will determine the format

    Returns:
        a sample format of the form "pcmXX" or "floatXX", where XX determines
        the bit rate ("pcm16", "float32", etc)

    ========== ================
    Extension  Sample Format
    ========== ================
    wav        float32
    aif        float32
    flac       pcm24
    mp3        pcm16
    ogg        vorbis
    ========== ================

    """
    if ext[0] == ".":
        ext = ext[1:]

    if ext in {"wav", "aif", "aiff"}:
        return "float32"
    elif ext == "flac":
        return "pcm24"
    elif ext == 'ogg':
        return 'vorbis'
    else:
        raise ValueError(f"Format {ext} not supported. Formats supported: wav, aiff, flac and ogg")


def _parsePresetSflistprograms(line: str) -> tuple[str, int, int] | None:
    # 012345678
    # xxx:yyy zzzzzzzzzz
    bank = int(line[:3])
    num = int(line[4:7])
    name = line[8:].strip()
    return (name, bank, num)


def _parsePreset(line: str) -> tuple[str, int, int] | None:
    match = _re.search(r">> Bank: (\d+)\s+Preset:\s+(\d+)\s+Name:\s*(.+)", line)
    if not match:
        return None
    name = match.group(3).strip()
    bank = int(match.group(1))
    presetnum = int(match.group(2))
    return (name, bank, presetnum)


def channelTypeFromValue(value: int | float | str) -> str:
    """
    Channel type (k, S, a) from value
    """
    if isinstance(value, (int, float)):
        return 'k'
    elif isinstance(value, str):
        return 'S'
    elif isinstance(value, np.ndarray):
        return 'a'
    else:
        raise TypeError(f"Value of type {type(value)} not supported")
