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

import math as _math
import os as _os
import sys
import subprocess as _subprocess
import re as _re
import shutil as _shutil
import logging as _logging
import textwrap as _textwrap
import io as _io
from pathlib import Path as _Path
import tempfile as _tempfile
import cachetools as _cachetools
import dataclasses
from ._common import *
from csoundengine import jacktools
from csoundengine import linuxaudio
from csoundengine import state as _state
from csoundengine.config import config
from csoundengine.internalTools import normalizePlatform
import functools as _functools
import emlib.misc
import emlib.textlib
import emlib.dialogs
import emlib.mathlib
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Callable, Sequence, Iterator, Any, Set
    Curve = Callable[[float], float]


try:
    import ctcsound7 as ctcsound
except Exception as e:
    if 'sphinx' in sys.modules:
        print("Called while building sphinx documentation?")
        from sphinx.ext.autodoc.mock import _MockObject
        ctcsound = _MockObject()
    else:
        print("ctcsound (ctcsound7) not found! Install it via 'pip install ctcsound7'")
        raise e

logger = _logging.getLogger("csoundengine")


@_cachetools.cached(cache=_cachetools.TTLCache(1, 10))
def _isPulseaudioRunning() -> bool:
    """ Return True if Pulseaudio is running """
    if not _shutil.which("pactl"):
        return False
    output = _subprocess.check_output(['pactl', 'info']).decode('utf-8')
    if _re.search(r'\bConnection\ failure', output) is None:
        return True
    return False


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
    csound = ctcsound.Csound()
    csound.setOption(f"-+rtmidi={backend}")
    csound.setOption("-odac")
    csound.start()
    inputdevs = csound.midiDevList(False)
    outputdevs = csound.midiDevList(True)
    logger.debug(f"MIDI Inputs:  {inputdevs}")
    logger.debug(f"MIDI Outputs: {outputdevs}")
    midiins = [MidiDevice(deviceid=d['device_id'], kind='input',
                          name=f"{d['interface_name']}:{d['device_name']}")
               for d in inputdevs]
    midiouts = [MidiDevice(deviceid=d['device_id'], kind='output',
                           name=f"{d['interface_name']}:{d['device_name']}")
               for d in outputdevs]
    return midiins, midiouts


def compressionBitrateToQuality(bitrate: int, format='ogg') -> float:
    """
    Convert a bitrate to a compression quality between 0-1, as passed to --vbr-quality

    Args:
        bitrate: the bitrate in kb/s, oneof 64, 80, 96, 128, 160, 192, 224, 256, 320, 500
        format: the encoding format (ogg at the moment)
    """
    if format == 'ogg':
        bitrates = [64, 80, 96, 128, 128, 160, 192, 224, 256, 320, 500]
        idx = emlib.misc.nearest_index(bitrate, bitrates)
        return idx / 10
    else:
        raise ValueError(f"Format {format} not supported")


def compressionQualityToBitrate(quality: float, format='ogg') -> int:
    """
    Convert compression quality to bitrate

    Args:
        quality: the compression quality (0-1) as passed to --vbr-quality
        format: the encoding format (ogg at the moment)

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
    if format == 'ogg':
        idx = int(quality * 10 + 0.5)
        if idx > 10:
            idx = 10
        return (64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 500)[idx]
    else:
        raise ValueError(f"Format {format} not supported")


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

    def __post_init__(self):
        if not self.longname:
            self.longname = self.name
        if not self.audioDeviceRegex:
            self.audioDeviceRegex = _audioDeviceRegex

    def searchAudioDevice(self, pattern:str, kind:str) -> AudioDevice | None:
        """
        Search a certain audio device from the devices presented by this backend
        """
        # we get the devices via getAudioDevices to enable caching
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
        """Get the system samplerate for this backend, if available"""
        if not self.hasSystemSr:
            logger.debug(f"Backend {self.name} does not have a system sr, returning default")
            return 44100
        cs = ctcsound.Csound()
        cs.setOption("-odac")
        cs.setOption(f"-+rtaudio={self.name}")
        ok = cs.start()
        if ok == -1:
            logger.error(f"Backend {self.name} not available")
            return None
        sr = cs.systemSr(0)
        cs.stop()
        return int(sr) if sr > 0 else None

    def _getSystemSr(self) -> int | None:
        """Get the system samplerate for this backend, if available"""
        if not self.hasSystemSr:
            return 44100
        proc = csoundSubproc(["-odac", f"-+rtaudio={self.name}", "--get-system-sr"], wait=True)
        if not proc.stdout:
            return None
        for line in proc.stdout.readlines():
            if line.startswith(b"system sr:"):
                uline = line.decode('utf-8')
                sr = int(float(uline.split(":")[1].strip()))
                return sr if sr > 0 else None
        logger.error(f"Failed to get sr with backend {self.name}")
        return None

    def bufferSizeAndNum(self) -> tuple[int, int]:
        """
        The buffer size and number of buffers needed for this backend
        """
        return (self.defaultBufferSize, self.defaultNumBuffers)

    def audioDevices(self) -> tuple[list[AudioDevice], list[AudioDevice]]:
        """
        Query csound for audio devices for this backend

        Returns:
            a tuple (inputDevices: list[AudioDevice], outputDevices: list[AudioDevice])
        """
        logger.info(f"Querying csound's audio devices for backend {self.name}")
        cs = ctcsound.Csound()
        for opt in ['-+rtaudio='+self.name, "-m16", "-odac"]:
            cs.setOption(opt)
        cs.start()
        csoutdevs = cs.audioDevList(True)
        csindevs = cs.audioDevList(False)
        outdevs = [AudioDevice(id=d['device_id'],
                               name=d['device_name'],
                               kind='output',
                               index=i,
                               numchannels=d['max_nchnls'])
                   for i, d in enumerate(csoutdevs)]
        indevs = [AudioDevice(id=d['device_id'],
                              name=d['device_name'],
                              kind='input',
                              index=i,
                              numchannels=d['max_nchnls'])
                  for i, d in enumerate(csindevs)]
        cs.stop()
        return indevs, outdevs

    def _audioDevices(self) -> tuple[list[AudioDevice], list[AudioDevice]]:
        """
        Returns a tuple (input devices, output devices)
        """
        indevices: list[AudioDevice] = []
        outdevices: list[AudioDevice] = []
        proc = csoundSubproc(['-+rtaudio=%s' % self.name, '--devices'])
        proc.wait()
        assert proc.stderr is not None
        lines = proc.stderr.readlines()
        for line in lines:
            line = line.decode("utf-8")
            match = _re.search(self.audioDeviceRegex, line)
            if not match:
                continue
            groups = match.groups()
            if len(groups) == 3:
                idxstr, devid, devname = groups
                numchannels = None
            else:
                idxstr, devid, devname, numchannels_ = groups
                numchannels = int(numchannels_) if numchannels_ is not None else 2
            kind = 'input' if devid.startswith("adc") else 'output'
            dev = AudioDevice(index=int(idxstr), id=devid.strip(), name=devname,
                              kind=kind, numchannels=numchannels)
            (indevices if kind == 'input' else outdevices).append(dev)
        return indevices, outdevices

    def defaultAudioDevices(self) -> tuple[AudioDevice | None, AudioDevice | None]:
        """
        Returns the default audio devices for this backend

        Returns:
            a tuple ``(inputDevice: AudioDevice, outputDevice: AudioDevice)`` for this backend
        """
        indevs, outdevs = getAudioDevices(self.name)  # self.audioDevices()
        return indevs[0], outdevs[0]


class _JackAudioBackend(AudioBackend):
    def __init__(self):
        super().__init__(name='jack',
                         alwaysAvailable=False,
                         platforms=('linux', 'darwin', 'win32'),
                         hasSystemSr=True)

    def getSystemSr(self) -> int:
        info = jacktools.getInfo()
        return info.samplerate if info else 0

    def bufferSizeAndNum(self) -> tuple[int, int]:
        info = jacktools.getInfo()
        return (info.blocksize, 2) if info else (0, 0)

    def isAvailable(self) -> bool:
        return jacktools.isJackRunning()

    def audioDevices(self) -> tuple[list[AudioDevice], list[AudioDevice]]:
        clients = jacktools.getClients()
        indevs = [AudioDevice(id=f'adc:{c.regex}', name=c.name, kind='input',
                              numchannels=len(c.ports))
                  for c in clients if c.kind == 'output']
        outdevs = [AudioDevice(id=f'dac:{c.regex}', name=c.name, kind='output',
                               numchannels=len(c.ports))
                   for c in clients if c.kind == 'input']
        return indevs, outdevs

    def defaultAudioDevices(self) -> tuple[AudioDevice|None, AudioDevice|None]:
        outclient, inclient = jacktools.getSystemClients()
        # NB: notice the inversion: a jack 'output' client is an input for csound
        if outclient is None:
            outdev = None
        else:
            outdev = AudioDevice(id=f"dac:{outclient.regex}", name=outclient.name,
                                 kind='output', numchannels=len(outclient.ports))
        if inclient is None:
            indev = None
        else:
            indev = AudioDevice(id=f"adc:{inclient.regex}", name=inclient.name,
                                kind='input', numchannels=len(inclient.ports))
        return indev, outdev


class _PulseAudioBackend(AudioBackend):
    def __init__(self):
        super().__init__(name='pulse',
                         alwaysAvailable=False,
                         hasSystemSr=True,
                         defaultBufferSize=1024,
                         defaultNumBuffers=2)

    def getSystemSr(self) -> int:
        info = linuxaudio.pulseaudioInfo()
        return info.sr if info else 0

    def isAvailable(self) -> bool:
        if linuxaudio.isPulseaudioRunning():
            return True
        pwinfo = linuxaudio.pipewireInfo()
        return pwinfo is not None and pwinfo.isPulseServer

    def audioDevices(self) -> tuple[list[AudioDevice], list[AudioDevice]]:
        pulseinfo = linuxaudio.pulseaudioInfo()
        assert pulseinfo is not None
        indevs = [AudioDevice(id="adc", name="adc", kind='input', index=0,
                              numchannels=pulseinfo.numchannels)]
        outdevs = [AudioDevice(id="dac", name="dac", kind='output', index=0,
                               numchannels=pulseinfo.numchannels)]
        return indevs, outdevs


class _PortaudioBackend(AudioBackend):
    def __init__(self, kind='callback'):
        shortname = "pa_cb" if kind == 'callback' else 'pa_bl'
        longname = f"portaudio-{kind}"
        if sys.platform == 'linux' and linuxaudio.isPipewireRunning():
            hasSystemSr = True
        else:
            hasSystemSr = False
        super().__init__(name=shortname,
                         alwaysAvailable=True,
                         longname=longname,
                         hasSystemSr=hasSystemSr)

    def getSystemSr(self) -> int | None:
        if sys.platform == 'linux' and linuxaudio.isPipewireRunning():
            info = linuxaudio.pipewireInfo()
            assert info is not None
            return info.sr
        return super().getSystemSr()

    def defaultAudioDevices(self) -> tuple[AudioDevice | None, AudioDevice | None]:
        indevs, outdevs = getAudioDevices(self.name)
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

    def _audioDevices(self) -> tuple[list[AudioDevice], list[AudioDevice]]:
        indevices: list[AudioDevice] = []
        outdevices: list[AudioDevice] = []
        proc = csoundSubproc(['-+rtaudio=pa_cb', '-odac', '--devices'], wait=True)
        if sys.platform == 'win32':
            assert proc.stdout is not None
            lines = proc.stdout.readlines()
        else:
            assert proc.stderr is not None
            lines = proc.stderr.readlines()
        for line in lines:
            line = line.decode("utf-8")
            if match := _re.search(self.audioDeviceRegex, line):
                groups = match.groups()
                if len(groups) == 3:
                    idxstr, devid, devname = groups
                    numchannels = None
                else:
                    idxstr, devid, devname, numchannels_ = groups
                    numchannels = int(numchannels_) if numchannels_ is not None else 2
                kind = 'input' if devid.startswith("adc") else 'output'
                dev = AudioDevice(index=int(idxstr), id=devid, name=devname, kind=kind,
                                  numchannels=numchannels)
                (indevices if kind == 'input' else outdevices).append(dev)
            elif match := _re.search(r"(\d+):\s*(dac\d+)\s\((.+)", line):
                idxstr = match.group(1)
                devid = match.group(2)
                devname = match.group(3)
                kind = 'input' if devid.startswith("adc") else 'output'
                dev = AudioDevice(index=int(idxstr), id=devid, name=devname, kind=kind,
                                  numchannels=None)
                (indevices if kind == 'input' else outdevices).append(dev)
        return indevices, outdevices


class _AlsaBackend(AudioBackend):
    def __init__(self):
        super().__init__(name="alsa",
                         alwaysAvailable=True,
                         platforms=('linux',),
                         audioDeviceRegex=r"([0-9]+):\s((?:adc|dac):.*)\((.*)\)")

    def getSystemSr(self) -> int | None:
        if (jackinfo := jacktools.getInfo()) is not None:
            return jackinfo.samplerate
        if linuxaudio.isPipewireRunning():
            info = linuxaudio.pipewireInfo()
            return info.sr if info else None
        return 44100

_backendJack = _JackAudioBackend()
_backendPulseaudio = _PulseAudioBackend()
_backendPortaudioBlocking = _PortaudioBackend('blocking')
_backendPortaudioCallback = _PortaudioBackend('callback')
_backendAlsa = _AlsaBackend()
_backendCoreaudio = AudioBackend('auhal',
                                 alwaysAvailable=True,
                                 hasSystemSr=True,
                                 needsRealtime=False,
                                 longname="coreaudio",
                                 platforms=('darwin',))


_allAudioBackends: dict[str, AudioBackend] = {
    'jack' : _backendJack,
    'auhal': _backendCoreaudio,
    'coreaudio': _backendCoreaudio,
    'pa_cb': _backendPortaudioCallback,
    'portaudio': _backendPortaudioCallback,
    'pa_bl': _backendPortaudioBlocking,
    'pulse': _backendPulseaudio,
    'pulseaudio': _backendPulseaudio,
    'alsa' : _backendAlsa
}


_backendsByPlatform: dict[str, list[AudioBackend]] = {
    'linux': [_backendJack, _backendPortaudioCallback, _backendAlsa,
              _backendPortaudioBlocking, _backendPulseaudio],
    'darwin': [_backendJack, _backendCoreaudio, _backendPortaudioCallback],
    'win32': [_backendPortaudioCallback, _backendPortaudioBlocking]
}


_cache: dict[str, Any] = {
    'opcodes': None,
    'versionTriplet': None
}


def nextpow2(n:int) -> int:
    """ Returns the power of 2 higher or equal than n"""
    return int(2 ** _math.ceil(_math.log(n, 2)))
    

@emlib.misc.runonce
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
    info = _csoundGetInfoViaAPI()
    return info['versionTriplet']


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


def _getCsoundSystemSr(backend:str) -> float:
    if backend not in {'jack', 'auhal'}:
        raise ValueError(f"backend {backend} does not support system sr")
    csound = ctcsound.Csound()
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
    backends = audioBackends(available=True)
    if not backends:
        raise RuntimeError("No available backends")
    return backends[0]


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

    ======== ======================================================
     OS       Plugins folder
    ======== ======================================================
     Linux    ``~/.local/lib/csound/6.0/plugins64``
     macOS    ``~/Library/csound/6.0/plugins64``
     windows  ``C:/Users/<User>/AppData/Local/csound/6.0/plugins64``
    ======== ======================================================

    For 32-bit plugins the folder is the same, without the '64' ending (``.../plugins``)
    """
    key = apiversion if float64 else 'float32'
    folders = _pluginsFolders[key]
    if not sys.platform in folders:
        raise RuntimeError(f"Platform {sys.platform} not known")
    folder = folders[sys.platform]
    return _os.path.abspath(_os.path.expandvars(folder))


def runCsd(csdfile:str,
           outdev = "",
           indev = "",
           backend = "",
           nodisplay = False,
           nomessages = False,
           comment:str = None,
           piped = False,
           extra:list[str] = None) -> _subprocess.Popen:
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

    See Also
    ~~~~~~~~

    * :func:`csoundSubproc`
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
        args.append(f'-+id_comment="{comment}"')
    if extra:
        args.extend(extra)
    args.append(csdfile)
    return csoundSubproc(args, piped=piped)
    

def joinCsd(orc: str, sco="", options: list[str] | None = None) -> str:
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
    """
    backend = backend or getDefaultBackend().name
    sr = sr or getSamplerateForBackend(backend)
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


@dataclasses.dataclass
class ScoreLine:
    """
    An event line in the score (an instrument, a table declaration, etc.)

    Attributes:
        kind: 'i' for instrument event, 'f' for table definition
        p1: the p1 of the event
        start: the start time of the event
        dur: the duration of the event
        args: any other args of the event (starting with p4)
    """
    kind: str
    p1: str | int | float
    start: float
    dur: float
    args: list[float | str]


@dataclasses.dataclass
class TableDataFile:
    """
    A table holding either the data or a file to the data

    Attributes:
        tabnum: the f-table number
        data: the data itself or a path to a file
        fmt: the format of the file
        start: start time to define the table
        size: the size of the table
    """
    tabnum: int
    """The assigned table number"""

    data: Sequence[float] | np.ndarray | str
    """the data itself or a path to a file"""

    fmt: str   # One of 'wav', 'flac', 'gen23', etc
    """The format of the data, one of 'wav', flac', 'gen23', etc"""

    start: float = 0
    """Allocation time of the table (p2)"""

    size: int = 0
    """Size of the data"""

    chan: int = 0
    """Which channel to read, if applicable. 0=all"""

    def __post_init__(self):
        assert self.fmt in {'gen23', 'wav', 'aif', 'aiff', 'flac'}, \
            f"Format not supported: {self.fmt}"
        if self.fmt == 'gen23' and isinstance(self.data, np.ndarray):
            assert len(self.data.shape) == 1 or self.data.shape[1] == 1

    def write(self, outfile:str) -> None:
        if isinstance(self.data, str):
            # just copy the file
            assert _os.path.exists(self.data)
            _shutil.copy(self.data, outfile)
            return

        base, ext = _os.path.splitext(outfile)
        if self.fmt == 'gen23':
            if ext != '.gen23':
                raise ValueError(f"Wrong extension: it should be .gen23, got {outfile}")
            saveAsGen23(self.data, outfile=outfile)
        elif self.fmt in ('wav', 'aif', 'aiff', 'flac'):
            import sndfileio
            dataarr = np.asarray(self.data, dtype=float)
            sndfileio.sndwrite(outfile, dataarr, sr=44100,
                               metadata={'comment': 'Datafile'})

    def scoreLine(self, outfile: str) -> str:
        if self.fmt == 'gen23':
            return f'f {self.tabnum} {self.start} {self.size} -23 "{outfile}"'
        elif self.fmt == 'wav':
            # time  size  1  filcod  skiptime  format  channel
            return f'f {self.tabnum} {self.start} {self.size} -1 "{outfile}" 0 0 0'
        raise ValueError(f"Unknown format {self.fmt}")

    def orchestraLine(self, outfile: str) -> str:
        if self.fmt == 'gen23':
            return f'ftgen {self.tabnum}, {self.start}, {self.size}, -23, "{outfile}"'
        elif self.fmt in ('wav', 'aif', 'aiff', 'flac'):
            return f'ftgen {self.tabnum}, {self.start}, {self.size}, -1, "{outfile}", 0, 0, 0'
        raise ValueError(f"Unknown format {self.fmt}")


def parseScore(sco: str) -> Iterator[ScoreLine]:
    """
    Parse a score given as string, returns a data. of :class:`ScoreLine`
    
    Args:
        sco: the score to parse, as string
        
    Returns:
        a generator of ScoreLines
    """
    p1: str | int
    for line in sco.splitlines():
        words = line.split()
        w0 = words[0]
        if w0 in {'i', 'f'}:
            kind = w0
            p1 = words[1]
            if namenum := emlib.misc.asnumber(p1) is not None:
                p1 = namenum
            t0 = float(words[2])
            dur = float(words[3])
            rest = words[4:]
        elif w0[0] in {'i', 'f'}:
            kind = w0[0]
            p1 = w0[1:]
            t0 = float(words[1])
            dur = float(words[2])
            rest = words[3:]
        else:
            continue
        args: list[float | str] = []
        for w in rest:
            if w.startswith('"'):
                args.append(w)
            else:
                arg = emlib.misc.asnumber(w)
                assert isinstance(arg, (int, float))
                args.append(arg)
        yield ScoreLine(kind, p1, t0, dur, args)


def opcodesList(cached=True, opcodedir: str = '') -> list[str]:
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
    cs = ctcsound.Csound()
    cs.setOption("-d")  # supress displays
    if opcodedir:
        cs.setOption(f'--opcode-dir={opcodedir}')
    opcodes, n = cs.newOpcodeList()
    opcodeNames = [opc.opname.decode('utf-8') for opc in opcodes]
    cs.disposeOpcodeList(opcodes)
    version = cs.version()
    vs = str(version)
    patch = int(vs[-1])
    minor = int(vs[-3:-1])
    major = int(vs[:-3])
    opcodes = list(set(opcodeNames))
    versionTriplet = (major, minor, patch)
    _cache['opcodes'] = opcodes
    _cache['versionTriplet'] = versionTriplet
    return {'opcodes': opcodes,
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
                header=""
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

        import bpf4
        from csoundengine import csoundlib
        a = bpf.linear(0, 0, 1, 10, 2, 300)
        dt = 0.01
        csoundlib.saveAsGen23(a[::dt].ys, "out.gen23", header=f"dt={dt}")


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
    if len(data.shape) > 1 and data.shape[1] > 1023:
        raise ValueError("Only matrices with less than 1024 rows can be saved "
                         "via this method")

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
            header = [3, numrows, numcols]
            if extradata:
                header.extend(extradata)
            header[0] = len(header)
            f.write(" ".join(np.array(header).astype(str)))
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
        numchannels: the number of channels
    """
    id: str
    name: str
    kind: str
    index: int = -1
    numchannels: int | None = 0

    def info(self) -> str:
        """
        Returns a summary of this device in one line
        """
        s = f"{self.id}:{self.name}"
        if self.kind == 'output':
            s += f":{self.numchannels}outs"
        else:
            s += f":{self.numchannels}ins"
        return s


@_cachetools.cached(cache=_cachetools.TTLCache(10, 20))
def getDefaultAudioDevices(backend='') -> tuple[AudioDevice | None, AudioDevice | None]:
    """
    Returns the default audio devices for a given backend

    .. note:: Results are cached for a period of time

    Args:
        backend: the backend to use (None to get the default backend)

    Returns:
        a tuple (input devices, output devices)
    """
    backendDef = getAudioBackend(backend)
    if backendDef is None:
        raise ValueError(f"Backend {backend} not known")
    return backendDef.defaultAudioDevices()


@_cachetools.cached(cache=_cachetools.TTLCache(10, 20))
def getAudioDevices(backend='') -> tuple[list[AudioDevice], list[AudioDevice]]:
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
        raise ValueError(f"Backend '{backend}' not supported")
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
        raise ValueError(f"Backend {backend} not supported")
    if not backendDef.isAvailable():
        raise RuntimeError(f"Audiobackend {backendDef.name} is not available")
    return backendDef.getSystemSr()


def _csoundTestJackRunning():
    proc = csoundSubproc(['-+rtaudio=jack', '-odac', '--get-system-sr'], wait=True)
    return b'could not connect to JACK server' not in proc.stderr.read()


def audioBackends(available=False, platform='') -> list[AudioBackend]:
    """
    Return a list of audio backends for the given platform
    
    Args:
        available: if True, only available backends are returned. This
            is only possible if querying backends for the current platform
        platform: defaults to the current platform. Possible values: 'linux',
            'macos', 'windows', but also any value returned by sys.platform

    Returns:
        a list of AudioBackend
        
    If available is True, only those backends supported for the current 
    platform and currently available are returned. For example, jack will 
    not be returned in linux if the jack server is not running.

    Example
    ~~~~~~~

        >>> from csoundengine import *
        >>> [backend.name for backend in audioBackends(available=True)]
        ['jack', 'pa_cb', 'pa_bl', 'alsa']
    """
    if platform:
        platform = normalizePlatform(platform)
    if available:
        platform = sys.platform
    elif not platform:
        platform = sys.platform
    if available and platform != sys.platform:
        available = False
    backends = _backendsByPlatform[platform]
    if available:
        backends = [b for b in backends if b.isAvailable()]
    return backends


def dumpAudioBackends() -> None:
    """
    Prints all **available** backends and their properties as a table
    """
    rows = []
    headers = "name longname sr".split()
    backends = audioBackends(available=True)
    backends.sort(key=lambda backend:backend.name)
    for b in backends:
        if b.hasSystemSr:
            sr = str(b.getSystemSr())
        else:
            sr = "-"
        rows.append((b.name, b.longname, sr))
    from emlib.misc import print_table
    print_table(rows, headers=headers, showindex=False)


def getAudioBackend(name='') -> AudioBackend | None:
    """ Given the name of the backend, return the AudioBackend structure

    Args:
        name: the name of the backend

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
    return _allAudioBackends.get(name)


def getAudioBackendNames(available=False, platform='') -> list[str]:
    """
    Returns a list with the names of the audio backends for a given platform

    Args:
        available: if True, return the names for only those backends which are
            currently available
        platform: if given, return only names for those backends present in the
            given platform

    Returns:
        a list with the names of all available backends for the
        given platform

    Example
    =======

        >>> from csoundengine.csoundlib import *
        >>> getAudioBackendNames()   # in Linux
        ['jack', 'pa_cb', 'pa_bl', 'alsa', 'pulse']
        >>> getAudioBackendNames(platform='macos')
        ['pa_cb', 'pa_bl', 'auhal']
        # In linux with pulseaudio disabled
        >>> getAudioBackendNames(available=True)
        ['jack', 'pa_cb', 'pa_bl', 'alsa']
    """
    backends = audioBackends(available=available, platform=platform)
    return [b.name for b in backends]


def _quoteIfNeeded(arg: float | int | str) -> float | int | str:
    if isinstance(arg, str):
        return emlib.textlib.quoteIfNeeded(arg)
    else:
        return arg


def _eventStartTime(event: Sequence) -> float:
    kind = event[0]
    if kind == "e":           # end
        return event[1]
    elif kind == "C":         # carry
        return 0.
    else:
        assert len(event) >= 3
        return event[2]

_normalizer = emlib.textlib.makeReplacer({".":"_", ":":"_", " ":"_"})


def normalizeInstrumentName(name:str) -> str:
    """
    Transform name so that it can be accepted as an instrument name
    """
    return _normalizer(name)


_fmtoptions = {
    'pcm16'    : '',
    'pcm24'    : '--format=24bit',
    'float32'  : '--format=float',  # also -f
    'float64'  : '--format=double',
    'vorbis'   : '--format=vorbis'
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
    ~~~~~~~

        >>> csoundOptionsForOutputFormat('flac')
        ('--format=flac', '--format=24bit')
        >>> csoundOptionsForOutputFormat('wav', 'float32')
        ('--format=wav', '--format=float')
        >>> csoundOptionsForOutputFormat('aif', 'pcm16')
        ('--format=aiff', '--format=short')

    .. seealso:: :func:`csoundOptionForSampleEncoding`
    """
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
    =======

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

_builtinInstrs = {
    '_playgen1': r'''
      kgain  = p4
      kspeed = p5
      ; 6      7      8      9
      itabnum, ichan, ifade, ioffset passign 6
      ifade = max(ifade, 0.005)
      ksampsplayed = 0
      inumsamples = nsamp(itabnum)
      itabsr = ftsr(itabnum)
      istartframe = ioffset * itabsr
      ksampsplayed += ksmps * kspeed
      aouts[] loscilx kgain, kspeed, itabnum, 4, 1, istartframe
      aenv linsegr 0, ifade, 1, ifade, 0
      aouts = aouts * aenv
      inumouts = lenarray(aouts)
      kchan = 0
      while kchan < inumouts do
        outch kchan+ichan, aouts[kchan]
        kchan += 1
      od
      if ksampsplayed >= inumsamples then
        turnoff
      endif
    ''',
    '_ftnew': r'''
      itabnum = p4
      isize = p5
      isr = p6
      inumchannels = p7
      ift ftgen itabnum, 0, -isize, -2, 0
      if isr > 0 || inumchannels > 0 then
        ftsetparams itabnum, isr, inumchannels
      endif
    ''',
    '_ftfree': r'''
        itabnum = p4
        ftfree itabnum, 0
        turnoff
    '''
}


@dataclasses.dataclass
class _InstrDef:
    p1: int | str
    body: str
    samelineComment: str = ''
    preComment: str = ''
    postComment: str = ''


class Csd:
    """
    Build a csound script by adding global code, instruments, score events, etc.

    Args:
        sr: the sample rate of the generated audio
        ksmps: the samples per cycle to use
        nchnls: the number of output channels
        nchnls_i: if given, the number of input channels
        a4: the reference frequency
        options (list[str]): any number of command-line options passed to csound
        nodisplay: if True, avoid outputting debug information
        carry: should carry be enabled in the score?
        reservedTables: when creating tables, table numbers are autoassigned from
            python. There can be conflicts of any code uses ``ftgen``

    Example
    =======

    .. code::

        >>> from csoundengine.csoundlib import *
        >>> csd = Csd(ksmps=32, nchnls=4)
        >>> csd.addInstr('sine', r'''
        ...   ifreq = p4
        ...   outch 1, oscili:a(0.1, ifreq)
        ... ''')
        >>> source = csd.addSndfile("sounds/sound1.wav")
        >>> csd.playTable(source)
        >>> csd.addEvent('sine', 0, 2, [1000])
        >>> csd.write('out.csd')
    """

    def __init__(self,
                 sr: int = 44100,
                 ksmps=64,
                 nchnls=2,
                 a4=442.,
                 options: list[str] | None = None,
                 nodisplay=False,
                 carry=False,
                 nchnls_i: int | None = None,
                 numthreads=0,
                 reservedTables=0):
        self.score: list[Sequence[int | float | str]] = []
        """The score, a list of events of the form (p1, p2, p3, ...)"""

        self.instrs: dict[str | int, _InstrDef] = {}
        """The orchestra"""

        self.globalcodes: list[str] = []
        """Code to evaluate at the instr0 level"""

        self.options: list[str] = []
        """Command line options"""

        self._sr = sr
        """Samplerate"""

        self.ksmps = ksmps
        """Samples per cycle"""

        self.nchnls = nchnls
        """Number of output channels"""

        self.nchnls_i = nchnls_i
        """Number of input channels"""

        self.a4 = a4
        """Reference frequency"""

        self.nodisplay = nodisplay
        """Disable display opcodes"""

        self.enableCarry = carry
        """Enable carry in the score"""

        self.numthreads = numthreads
        """Number of threads used for rendering"""

        self.datafiles: dict[int, TableDataFile] = {}
        """Maps assigned table numbers to their metadata"""

        self._datafileIndex: dict[str, TableDataFile] = {}
        """Maps soundfiles read to their assigned table number"""

        self._strLastIndex = 20
        self._str2index: dict[str, int] = {}

        if options:
            self.addOptions(*options)

        self._outfileFormat = ''
        self._outfileEncoding = ''
        self._compressionQuality = ''

        self._definedTables: Set[int] = set()
        self._minTableIndex = 1
        self._endMarker: float = 0
        self._numReservedTables = reservedTables
        self._maxTableNumber = reservedTables
        self.score.append(("C", 0, "    ; Disable carry"))

    @property
    def sr(self) -> int:
        return self._sr

    @sr.setter
    def sr(self, value: int):
        if not isinstance(value, int):
            raise TypeError(f"Samplerate must be an int, got {value}")
        self._sr = value

    def copy(self) -> Csd:
        """
        Copy this csd
        """
        out = Csd(sr=self.sr,
                  ksmps=self.ksmps,
                  nchnls=self.nchnls,
                  a4=self.a4,
                  options=self.options.copy(),
                  nodisplay=self.nodisplay,
                  carry=self.enableCarry,
                  nchnls_i=self.nchnls_i,
                  numthreads=self.numthreads)

        out.instrs = self.instrs.copy()
        out.score = self.score.copy()
        out._str2index = self._str2index.copy()
        out._strLastIndex = self._strLastIndex
        if self.globalcodes:
            for code in self.globalcodes:
                out.addGlobalCode(code)

        out._outfileEncoding = self._outfileEncoding
        out._outfileFormat = self._outfileFormat
        out._compressionQuality = self._compressionQuality

        out._definedTables = self._definedTables
        out._minTableIndex = self._minTableIndex
        out._maxTableNumber = self._maxTableNumber

        if self.datafiles:
            out.datafiles = self.datafiles.copy()

        if self._outfileEncoding:
            out.setSampleEncoding(self._outfileEncoding)

        return out

    def cropScore(self, start=0., end=0.) -> None:
        """
        Crop the score at the given boundaries

        Any event starting earlier or ending after the given times will
        be cropped, any event ending before start or starting before
        end will be removed
        """
        score = _cropScore(self.score, start, end)
        self.score = score

    def dumpScore(self) -> None:
        from emlib.misc import print_table
        maxp = max(len(event) for event in self.score)
        headers = ["#"] + [f'p{n}' for n in range(maxp)]
        print_table(self.score, headers=headers, floatfmt=".3f")

    def addEvent(self,
                 instr: int | float | str,
                 start: float,
                 dur: float,
                 args: Sequence[float | str] | None = None,
                 comment='') -> None:
        """
        Add an instrument ("i") event to the score

        Args:
            instr: the instr number or name, as passed to addInstr
            start: the start time
            dur: the duration of the event
            args: pargs beginning at p4
            comment: if given, the text is attached as a comment to the event
                line in the score
        """
        start = round(start, 8)
        dur = round(dur, 8)
        event = ["i", _quoteIfNeeded(instr), start, dur]
        if args:
            if any(not isinstance(arg, (int, float, str)) for arg in args):
                badargs = [f"{a} ({type(a)})" for a in args if not isinstance(a, (int, float, str))]
                raise TypeError(f"pargs must be int, float or str, got {', '.join(badargs)} "
                                f"({instr=}, {args=}, {start=}, {dur=})")
            event.extend(_quoteIfNeeded(arg) for arg in args)
        if comment:
            event.append(f"     ; {comment}")
        self.score.append(event)

    def strset(self, s: str, index: int | None) -> int:
        """
        Add a strset to this csd

        If ``s`` has already been passed, the same index is returned
        """
        if s in self._str2index:
            if index is not None and index != self._str2index[s]:
                raise KeyError(f"String '{s}' already set with different index "
                               f"(old: {self._str2index[s]}, new: {index})")
            return self._str2index[s]

        if index is None:
            index = self._strLastIndex
        else:
            self._strLastIndex = max(self._strLastIndex, index)
        self._strLastIndex += 1
        self._str2index[s] = index
        return index

    def _assignTableIndex(self, tabnum=0) -> int:
        if tabnum == 0:
            tabnum = self._maxTableNumber + 1
        else:
            if tabnum in self._definedTables:
                raise ValueError(f"ftable {tabnum} already defined")
        if tabnum > self._maxTableNumber:
            self._maxTableNumber = tabnum
        self._definedTables.add(tabnum)
        return tabnum

    def _addTable(self, pargs, comment='') -> int:
        """
        Adds a ftable to the score

        Args:
            pargs: as passed to csound (without the "f")
                p1 can be 0, in which case a table number
                is assigned

        Returns:
            The index of the new ftable
        """
        tabnum = pargs[0]
        if tabnum == 0:
            tabnum = self._assignTableIndex()
        else:
            assert tabnum in self._definedTables
        pargs = [_quoteIfNeeded(p) for p in pargs[1:]]
        scoreline = ["f", tabnum] + pargs
        if comment:
            scoreline.append(f'    ; {comment}')
        self.score.append(scoreline)
        return tabnum

    def addTableFromData(self,
                         data: Sequence[float] | np.ndarray,
                         tabnum: int = 0,
                         start=0,
                         filefmt='',
                         sr=0,
                         ) -> int:
        """
        Add a table definition with the data

        Args:
            data: a sequence of floats to fill the table. The size of the
                table is determined by the size of the seq.
            tabnum: 0 to auto-assign an index
            start: allocation time of the table
            filefmt: format to use when saving the table as a datafile. If not given,
                the default is used. Possible values: 'gen23', 'wav'
            sr: if given and data is a numpy array, it is saved as a soundfile
                and loaded via gen1

        Returns:
            the table number

        .. note::

            The data is either included in the table definition (if it is
            small enough) or saved as an external file. All external files are
            saved relative to the generated .csd file when writing. Table data
            is saved as 32 bit floats, so it might loose some precission from
            the original.
        """
        sizeThreshold = config['offline_score_table_size_limit']

        if isinstance(data, np.ndarray) and sr:
            sndfile = _tempfile.mktemp(suffix=".wav")
            import sndfileio
            sndfileio.sndwrite(sndfile, samples=data, sr=sr, encoding='float32')
            tabnum = self.addSndfile(sndfile, tabnum=tabnum, asProjectFile=True,
                                     start=start)
        else:
            if not filefmt:
                filefmt = config['datafile_format']

            tabnum = self._assignTableIndex(tabnum)

            if len(data) > sizeThreshold:
                # If the data is big, we save the data. We will write
                # it to a file when rendering
                datafile = TableDataFile(tabnum, data, start=start, fmt=filefmt)
                self._addProjectFile(datafile)
            else:
                pargs = [tabnum, start, -len(data), -2]
                pargs.extend(data)
                tabnum = self._addTable(pargs)

        assert tabnum > 0
        return tabnum

    def _addProjectFile(self, datafile: TableDataFile) -> None:
        self.datafiles[datafile.tabnum] = datafile
        if isinstance(datafile.data, str):
            self._datafileIndex[datafile.data] = datafile
        assert datafile.tabnum in self._definedTables

    def addEmptyTable(self, size: int, tabnum: int = 0, sr: int = 0,
                      numchannels=1, time=0.
                      ) -> int:
        """
        Add an empty table to this Csd

        A table remains valid until the end of the csound process or until
        the table is explicitely freed (see :meth:`~Csd.freeTable`)

        Args:
            tabnum: use 0 to autoassign an index
            size: the size of the empty table
            sr: if given, set the sr of the empty table to the given sr
            numchannels: the number of channels in the table
            time: when to do the allocation.

        Returns:
            The index of the created table
        """
        if sr == 0:
            pargs = (tabnum, 0, -size, -2, 0)
            return self._addTable(pargs)
        else:
            tabnum = self._assignTableIndex(tabnum)
            self._ensureBuiltinInstr('_ftnew')
            args = [tabnum, size, sr, numchannels]
            self.addEvent('_ftnew', start=time, dur=0, args=args)
            return tabnum

    def freeTable(self, tabnum: int, time: float):
        """
        Free a table

        Args:
            tabnum: the table number
            time: when to free it
        """
        self._ensureBuiltinInstr('_ftfree')
        self.addEvent('_ftfree', start=time, dur=0, args=[tabnum])

    def _ensureBuiltinInstr(self, name: str):
        if self.instrs.get(name) is None:
            self.addInstr(name, _builtinInstrs[name])

    def addSndfile(self, sndfile: str, tabnum=0, start=0., skiptime=0, chan=0,
                   asProjectFile=False) -> int:
        """
        Add a table which will load this sndfile

        Args:
            sndfile: the output to load
            tabnum: fix the table number or use 0 to generate a unique table number
            start: when to load this output (normally this should be left 0)
            skiptime: begin reading at `skiptime` seconds into the file.
            chan: channel number to read. 0 denotes read all channels.
            asProjectFile: if True, the sndfile is included as a project file and
                copied to a path relative to the .csd when writing

        Returns:
            the table number
        """
        sndfmt = _os.path.splitext(sndfile)[1][1:].lower()
        supportedFormats = ('wav', 'aif', 'aiff', 'flac')
        if sndfmt not in supportedFormats:
            raise ValueError(f"Format '{sndfmt}' not supported, "
                             f"supported formats: {supportedFormats}")

        if datafile := self._datafileIndex.get(sndfile):
            return datafile.tabnum

        tabnum = self._assignTableIndex(tabnum)
        datafile = TableDataFile(tabnum, data=sndfile, start=start, fmt=sndfmt)

        if not asProjectFile:
            pargs = [tabnum, start, 0, -1, sndfile, skiptime, 0, chan]
            self._datafileIndex[sndfile] = datafile
            self._addTable(pargs)
        else:
            self._addProjectFile(datafile)
        assert tabnum > 0
        return tabnum

    def destroyTable(self, tabnum: int, time: float) -> None:
        """
        Schedule ftable with index `source` to be destroyed at time `time`

        Args:
            tabnum: the index of the table to be destroyed
            time: the time to destroy it
        """
        pargs = ("f", -tabnum, time)
        self.score.append(pargs)

    def setEndMarker(self, time: float) -> None:
        """
        Add an end marker to the score

        This is needed if, for example, all events are endless
        events (with dur == -1).

        If an end marker has been already set, setting it later will remove
        the previous endmarker (there can be only one)
        """
        if time == 0 or self._endMarker > 0:
            self.removeEndMarker()
        self._endMarker = time
        # We don't add the marker to the score because this needs to go at the end
        # of the score. Any score line after the end marker will not be read

    def removeEndMarker(self) -> None:
        """
        Remove the end-of-score marker
        """
        self._endMarker = 0

    def setComment(self, comment: str) -> None:
        """ Add a comment to the renderer output soundfile"""
        self.addOptions(f'-+id_comment="{comment}"')

    def setOutfileFormat(self, fmt: str) -> None:
        """
        Sets the format for the output soundfile

        If this is not explicitely set it will be induced from
        the output soundfile set when running the csd

        Args:
            fmt: the format to use ('wav', 'aif', 'flac', etc)
        """
        assert fmt in {'wav', 'aif', 'aiff', 'flac', 'ogg'}
        self._outfileFormat = fmt

    def setSampleEncoding(self, encoding: str) -> None:
        """
        Set the sample encoding for recording

        If not set, csound's own default for encoding will be used

        Args:
            encoding: one of 'pcm16', 'pcm24', 'pcm32', 'float32', 'float64'

        """
        assert encoding in {'pcm16', 'pcm24', 'pcm32', 'float32', 'float64', 'vorbis'}
        self._outfileEncoding = encoding

    def setCompressionQuality(self, quality=0.4) -> None:
        """
        Set the compression quality

        Args:
            quality: a value between 0 and 1
        """
        self._compressionQuality = quality

    def setCompressionBitrate(self, bitrate=128, format='ogg') -> None:
        """
        Set the compression quality by defining a bitrate

        Args:
            bitrate: the bitrate in kB/s
            format: the format used (only 'ogg' at the moment)
        """
        self.setCompressionQuality(compressionBitrateToQuality(bitrate, format))

    def _writeScore(self, stream, datadir='.', dataprefix='') -> None:
        """
        Write the score to `stream`

        Args:
            stream (file-like): the open stream to write to
            datadir: the folder to save data files
        """
        self.score.sort(key=_eventStartTime)
        for event in self.score:
            line = " ".join(str(arg) for arg in event)
            stream.write(line)
            stream.write("\n")
        for tabnum, datafile in self.datafiles.items():
            assert tabnum > 0
            outfilebase = f'table-{tabnum:04d}.{datafile.fmt}'
            if dataprefix:
                outfilebase = f'{dataprefix}-{outfilebase}'
            datadirpath = _Path(datadir)
            outfile = datadirpath / outfilebase
            datafile.write(outfile.as_posix())
            relpath = outfile.relative_to(datadirpath.parent)
            stream.write(datafile.scoreLine(relpath.as_posix()))
            stream.write('\n')
        if self._endMarker:
            stream.write(f'e {self._endMarker}    ; end marker')

    def scoreDuration(self) -> float:
        if self._endMarker:
            return self._endMarker

        endtime = 0.
        for ev in self.score:
            evstart = ev[2]
            evdur = ev[3]
            assert isinstance(evdur, (int, float)) and isinstance(evstart, (int, float))
            if evdur < 0:
                endtime = float('inf')
                break
            else:
                endtime = max(endtime, evstart + evdur)
        return endtime

    def addInstr(self, instr: int | str, body: str, instrComment='') -> None:
        """
        Add an instrument definition to this csd

        Args:
            instr: the instrument number of name
            body: the body of the instrument (the part between 'instr' / 'endin')
            instrComment: if given, it will be added at the end of the 'instr' line
        """
        if _re.search(r"^\s*instr", body):
            raise ValueError(f"The body should only include the instrument definition, "
                             f"the part between 'instr' / 'endin', got: {body}")

        instrdef = _InstrDef(p1=instr, body=body, samelineComment=instrComment)
        self.instrs[instr] = instrdef

    def addGlobalCode(self, code: str, acceptDuplicates=True) -> None:
        """ Add code to the instr 0 """
        if not acceptDuplicates and code in self.globalcodes:
            return
        self.globalcodes.append(code)

    def addOptions(self, *options: str) -> None:
        """
        Adds options to this csd

        Options are any command-line options passed to csound itself or which could
        be used within a <CsOptions> tag. They are not checked for correctness
        """
        self.options.extend(options)

    def dump(self) -> str:
        """ Returns a string with the .csd """
        stream = _io.StringIO()
        self._writeCsd(stream)
        return stream.getvalue()

    def playTable(self, tabnum: int, start: float, dur: float = -1,
                  gain=1., speed=1., chan=1, fade=0.05,
                  skip=0.) -> None:
        """
        Add an event to play the given table

        Args:
            tabnum: the table number to play
            start: schedule time (p2)
            dur: duration of the event (leave -1 to play until the end)
            gain: a gain factor applied to the table samples
            chan: ??
            fade: fade time (both fade-in and fade-out
            skip: time to skip from playback (enables playback to crop a fragment at the beginning)

        Example
        =======

        >>> csd = Csd()
        >>> source = csd.addSndfile("stereo.wav")
        >>> csd.playTable(source, source, start=1, fade=0.1, speed=0.5)
        >>> csd.write("out.csd")
        """
        if self.instrs.get('_playgen1') is None:
            self.addInstr('_playgen1', _builtinInstrs['_playgen1'])
        assert tabnum > 0
        args = [gain, speed, tabnum, chan, fade, skip]
        self.addEvent('_playgen1', start=start, dur=dur, args=args)

    def write(self, csdfile: str) -> None:
        """
        Write this as a .csd

        Any data files added are written to a folder <csdfile>.assets besides the
        generated .csd file.

        Example
        -------

        >>> from csoundengine.csoundlib import Csd
        >>> csd = Csd(...)
        >>> csd.write("myscript.csd")

        This will generate a ``myscript.csd`` file and a folder ``myscript.assets`` holding
        any data file needed. If no data files are used, no ``.assets`` folder is created

        """
        csdfile = _os.path.expanduser(csdfile)
        base = _os.path.splitext(csdfile)[0]
        stream = open(csdfile, "w")
        if self.datafiles:
            datadir = base + ".assets"
            _os.makedirs(datadir, exist_ok=True)
        else:
            datadir = ''
        self._writeCsd(stream, datadir=datadir)

    def _writeCsd(self, stream, datadir='') -> None:
        """
        Write this as a csd

        Args:
            stream: the stream to write to. Either a path, an open file or
                a io.StringIO
            datadir: the folder where all datafiles are written. Datafiles are
                used whenever the user defines tables with data too large to
                include 'inline' (as gen2) or when adding soundfiles.
        """
        if isinstance(stream, str):
            outfile = stream
            stream = open(outfile, "w")
        write = stream.write
        write("<CsoundSynthesizer>\n<CsOptions>\n")
        options = self.options.copy()
        if self.nodisplay:
            options.append("-m0")

        if self.numthreads > 1:
            options.append(f"-j {self.numthreads}")

        if self._outfileFormat:
            options.extend(csoundOptionsForOutputFormat(self._outfileFormat, self._outfileEncoding))
        elif self._outfileEncoding:
            options.append(csoundOptionForSampleEncoding(self._outfileEncoding))

        for option in options:
            write(option)
            write("\n")
        write("</CsOptions>\n")

        srstr = f"sr     = {self.sr}" if self.sr is not None else ""
        
        txt = rf"""
            <CsInstruments>

            {srstr}
            ksmps  = {self.ksmps}
            0dbfs  = 1
            A4     = {self.a4}
            nchnls = {self.nchnls}
            """
        txt = _textwrap.dedent(txt)
        write(txt)
        if self.nchnls_i is not None:
            write(f'nchnls_i = {self.nchnls_i}\n')
        tab = "  "

        if self._str2index:
            for s, idx in self._str2index.items():
                write(f'strset {idx}, "{s}"\n')
            write("\n")

        if self.globalcodes:
            write("; ----- global code\n")
            for globalcode in self.globalcodes:
                write(globalcode)
                write("\n")
            write("; ----- end global code\n\n")

        for instr, instrdef in self.instrs.items():
            if instrdef.preComment:
                for line in instrdef.preComment.splitlines():
                    write(f";;  {line}\n")
            instrline = f"instr {instr}"
            if instrdef.samelineComment:
                instrline += f"  ; {instrdef.samelineComment}\n"
            else:
                instrline += "\n"
            write(instrline)
            if instrdef.postComment:
                if instrdef.preComment:
                    for line in instrdef.preComment.splitlines():
                        write(f"{tab};;  {line}\n")
            body = _textwrap.dedent(instrdef.body)
            body = _textwrap.indent(body, tab)
            write(body)
            write("\nendin\n")
        
        write("\n</CsInstruments>\n")
        write("\n<CsScore>\n\n")
        
        self._writeScore(stream, datadir=datadir)
        
        write("\n</CsScore>\n")
        write("</CsoundSynthesizer>")

    def run(self,
            output: str,
            csdfile: str = None,
            inputdev: str = None,
            backend: str = None,
            suppressdisplay = True,
            nomessages = False,
            piped = False,
            extraOptions: list[str] = None) -> _subprocess.Popen:
        """
        Run this csd. 
        
        Args:
            output: the output of the csd. This will be passed
                as the -o argument to csound. If an empty string or None is given,
                no sound is produced (adds the '--nosound' flag).
            inputdev: the input device to use when running in realtime
            csdfile: if given, the csd file will be saved to this path and run
                from it. Otherwise a temp file is created and run.
            backend: the backend to use
            suppressdisplay: if True, display (table plots, etc.) is supressed
            nomessages: if True, debugging scheduling information is suppressed
            piped: if True, stdout and stderr are piped through
                the Popen object, accessible through .stdout and .stderr
                streams
            extraOptions: any extra args passed to the csound binary

        Returns:
            the _subprocess.Popen object

        """
        options = self.options.copy()
        outfileFormat = ''
        outfileEncoding = ''
        if not output:
            options.append('--nosound')
        elif not output.startswith('dac'):
            outfileFormat = self._outfileFormat or _os.path.splitext(output)[1][1:]
            outfileEncoding = self._outfileEncoding or bestSampleEncodingForExtension(outfileFormat)
            if self._compressionQuality:
                options.append(f'--vbr-quality={self._compressionQuality}')

        if not csdfile:
            csdfile = _tempfile.mktemp(suffix=".csd")
            logger.debug(f"Runnings Csd from tempfile {csdfile}")

        if outfileFormat:
            options.extend(csoundOptionsForOutputFormat(outfileFormat, outfileEncoding))

        if extraOptions:
            options.extend(extraOptions)

        options = emlib.misc.remove_duplicates(options)

        self.write(csdfile)
        return runCsd(csdfile, outdev=output, indev=inputdev,
                      backend=backend, nodisplay=suppressdisplay,
                      nomessages=nomessages,
                      piped=piped, extra=options)


def mincer(sndfile:str,
           outfile:str,
           timecurve: Curve | float,
           pitchcurve: Curve | float,
           dt=0.002, lock=False, fftsize=2048, ksmps=128, debug=False
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
    ========

        # Example 1: stretch a output 2x

        >>> from csoundengine import csoundlib
        >>> import bpf4
        >>> import sndfileio
        >>> snddur = sndfileio.sndinfo("mono.wav").duration
        >>> timecurve = bpf4.linear(0, 0, snddur*2, snddur)
        >>> mincer(sndfile, "mono2.wav", timecurve=timecurve, pitchcurve=1)
    """
    import bpf4 as bpf
    import sndfileio

    info = sndfileio.sndinfo(sndfile)
    sr = info.samplerate
    nchnls = info.channels
    pitchbpf = bpf.asbpf(pitchcurve)
    
    if isinstance(timecurve, (int, float)):
        t0, t1 = 0, info.duration / timecurve
        timebpf = bpf.linear(0, 0, t1, info.duration)
    elif isinstance(timecurve, bpf.core.BpfInterface):
        t0, t1 = timecurve.bounds()
        timebpf = timecurve
    else:
        raise TypeError("timecurve should be either a scalar or a bpf")
    
    assert isinstance(pitchcurve, (int, float, bpf.core.BpfInterface))
    ts = np.arange(t0, t1+dt, dt)
    fmt = "%.12f"
    _, time_gen23 = _tempfile.mkstemp(prefix='time-', suffix='.gen23')
    np.savetxt(time_gen23, timebpf.map(ts), fmt=fmt, header=str(dt), comments="")
    _, pitch_gen23 = _tempfile.mkstemp(prefix='pitch-', suffix='.gen23')
    np.savetxt(pitch_gen23, pitchbpf.map(ts), fmt=fmt, header=str(dt), comments="")
    csd = f"""
    <CsoundSynthesizer>
    <CsOptions>
    -o {outfile}
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


def recInstr(body:str, events:list, init="", outfile:str=None,
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

    proc = csd.run(output=outfile)
    return outfile, proc


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
    nchnls = outdev.numchannels if (outdev and outdev.numchannels is not None) else defaultout
    if not inpattern:
        indev = adc
    else:
        indev = backendDef.searchAudioDevice(inpattern, kind='input')
        if not indev:
            raise ValueError(f"Input device {inpattern} not found")
    nchnlsi = indev.numchannels if (indev and indev.numchannels is not None) else defaultin
    return nchnlsi, nchnls


def _getNchnlsJackViaJackclient(indevice:str, outdevice:str
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


def _parsePortaudioDeviceName(name:str) -> tuple[str, str, int, int]:
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
    print_table(inputrows, headers=fields, showindex=False)
    print("\nOutput Devices:")
    print_table(outputrows, headers=fields, showindex=False)


def instrNames(instrdef: str) -> list[int | str]:
    """
    Returns the list of names/instrument numbers in the instrument definition.

    Most of the time this list will have one single element, either an instrument
    number or a name

    Args:
        code (str): the code defining an instrument

    Returns:
        a list of names/instrument numbers. An empty list is returned if
        this is not a valid instr definition

    Example
    ~~~~~~~

        >>> instr = r'''
        ... instr 10, foo
        ...     outch 1, oscili:a(0.1, 440)
        ... endin
        ... '''
        >>> instrNames(instr)
        [10, "foo"]

    """
    lines  = instrdef.splitlines()
    matches = [line for line in lines if _re.match(r"^[\ \t]*\binstr\b", line)]
    if len(matches) > 1:
        raise ValueError(f"Expected only one instrument definition, got {matches}")
    elif len(matches) == 0:
        return []
    line = matches[0].strip()
    names = [name.strip() for name in line[6:].split(",")]
    return [int(name) if name.isdigit() else name
            for name in names]


@dataclasses.dataclass
class ParsedBlock:
    """
    A ParsedBlock represents a block (an instr, an opcode, etc) in an orchestra

    Used by :func:`parseOrc` to split an orchestra in individual blocks

    Attributes:
        kind: the kind of block ('instr', 'opcode', 'header', 'include', 'instr0')
        text: thet text of the block
        startLine: where does this block start within the parsed orchestra
        endLine: where does this block end
        name: name of the block
        attrs: some blocks need extraOptions information. Opcodes define attrs 'outargs' and
            'inargs' (corresponding to the xin and xout opcodes), header blocks have
            a 'value' attr
    """
    kind: str
    text: str
    startLine: int
    endLine: int = -1
    name: str = ''
    attrs: dict[str, str] | None = None

    def __post_init__(self):
        assert self.kind in ('instr', 'opcode', 'header', 'include', 'instr0')
        if self.endLine == -1:
            self.endLine = self.startLine


@dataclasses.dataclass
class _OrcBlock:
    name: str
    startLine: int
    lines: list[str]
    endLine: int = 0
    outargs: str = ""
    inargs: str = ""

def parseOrc(code: str, keepComments=True) -> list[ParsedBlock]:
    """
    Parse orchestra code into blocks

    Each block is either an instr, an opcode, a header line, a comment
    or an instr0 line

    Example
    -------

    .. code-block:: python

        >>> from csoundengine import csoundlib
        >>> orc = r'''
        ... sr = 44100
        ... nchnls = 2
        ... ksmps = 32
        ... 0dbfs = 1
        ... seed 0
        ...
        ... opcode AddSynth,a,i[]i[]iooo
        ...  /* iFqs[], iAmps[]: arrays with frequency ratios and amplitude multipliers
        ...  iBasFreq: base frequency (hz)
        ...  iPtlIndex: partial index (first partial = index 0)
        ...  iFreqDev, iAmpDev: maximum frequency (cent) and amplitude (db) deviation */
        ...  iFqs[], iAmps[], iBasFreq, iPtlIndx, iFreqDev, iAmpDev xin
        ...  iFreq = iBasFreq * iFqs[iPtlIndx] * cent(rnd31:i(iFreqDev,0))
        ...  iAmp = iAmps[iPtlIndx] * ampdb(rnd31:i(iAmpDev,0))
        ...  aPartial poscil iAmp, iFreq
        ...  if iPtlIndx < lenarray(iFqs)-1 then
        ...   aPartial += AddSynth(iFqs,iAmps,iBasFreq,iPtlIndx+1,iFreqDev,iAmpDev)
        ...  endif
        ...  xout aPartial
        ... endop
        ...
        ... ;frequency and amplitude multipliers for 11 partials of Risset's bell
        ... giFqs[] fillarray  .56, .563, .92, .923, 1.19, 1.7, 2, 2.74, 3, 3.74, 4.07
        ... giAmps[] fillarray 1, 2/3, 1, 1.8, 8/3, 5/3, 1.46, 4/3, 4/3, 1, 4/3
        ...
        ... instr Risset_Bell
        ...  ibasfreq = p4
        ...  iamp = ampdb(p5)
        ...  ifqdev = p6 ;maximum freq deviation in cents
        ...  iampdev = p7 ;maximum amp deviation in dB
        ...  aRisset AddSynth giFqs, giAmps, ibasfreq, 0, ifqdev, iampdev
        ...  aRisset *= transeg:a(0, .01, 0, iamp/10, p3-.01, -10, 0)
        ...  out aRisset, aRisset
        ... endin
        ... ''')
        >>> csoundlib.parseOrc(orc)
        [ParsedBlock(kind='header'P, text='sr = 44100', startLine=1, endLine=1, name='sr',
                     attrs={'value': '44100'}),
         ParsedBlock(kind='header', text='ksmps = 32', startLine=2, endLine=2, name='ksmps', attrs={'value': '32'}),
         ParsedBlock(kind='header', text='nchnls = 2', startLine=3, endLine=3, name='nchnls', attrs={'value': '2'}),
         ParsedBlock(kind='header', text='0dbfs = 1', startLine=4, endLine=4, name='0dbfs', attrs={'value': '1'}),
         ParsedBlock(kind='instr0', text='seed 0', startLine=6, endLine=6, name='', attrs=None),
         ParsedBlock(kind='opcode', text='opcode AddSynth,a,i[]i[]iooo\\n iFqs[], iAmps[], iBasFreq, iPtlIndx, <...>',
                     name='AddSynth', attrs={'outargs': 'a', 'inargs': 'i[]i[]iooo'}),
         ParsedBlock(kind='comment', text=";frequency and amplitude multipliers for 11 partials of Risset's bell",
                     startLine=19, endLine=19, name='', attrs=None),
         ParsedBlock(kind='instr0', text='giFqs[] fillarray  .56, .563, .92, .923, 1.19, 1.7, 2, 2.74, 3, 3.74, 4.07', startLine=20, endLine=20, name='', attrs=None),
         ParsedBlock(kind='instr0', text='giAmps[] fillarray 1, 2/3, 1, 1.8, 8/3, 5/3, 1.46, 4/3, 4/3, 1, 4/3', startLine=21, endLine=21, name='', attrs=None),
         ParsedBlock(kind='instr', text='instr Risset_Bell\\n ibasfreq = p4\\n iamp = ampdb(p5)\\n <...>'
                     startLine=23, endLine=31, name='Risset_Bell', attrs=None)]

    """
    context = []
    blocks: list[ParsedBlock] = []
    block = _OrcBlock("", 0, [])
    for i, line in enumerate(code.splitlines()):
        strippedline = line.strip()
        if not strippedline:
            continue
        if match := _re.search(r"\binstr\s+(\d+|[a-zA-Z_]\w+)", line):
            context.append('instr')
            block = _OrcBlock(name=match.group(1),
                              startLine=i,
                              lines=[line])
        elif strippedline == "endin":
            assert context[-1] == "instr"
            context.pop()
            assert block.name
            block.endLine = i
            block.lines.append(line)
            blocks.append(ParsedBlock(kind='instr',
                                      startLine=block.startLine,
                                      endLine=block.endLine,
                                      text='\n'.join(block.lines),
                                      name=block.name))
        elif strippedline == 'endop':
            assert context[-1] == "opcode"
            context.pop()
            block.endLine = i
            block.lines.append(line)
            blocks.append(ParsedBlock(kind='opcode',
                                      startLine=block.startLine,
                                      endLine=block.endLine,
                                      text='\n'.join(block.lines),
                                      name=block.name,
                                      attrs={'outargs':block.outargs,
                                              'inargs':block.inargs}))
        elif context and context[-1] in {'instr', 'opcode'}:
            block.lines.append(line)
        elif match := _re.search(r"^\s*(sr|ksmps|kr|A4|0dbfs|nchnls|nchnls_i)\s*=\s*(\d+)", line):
            blocks.append(ParsedBlock(kind='header',
                                      name=match.group(1),
                                      startLine=i,
                                      text=line,
                                      attrs={'value':match.group(2)}))
        elif _re.search(r"^\s*(;|\/\/)", line):
            if keepComments:
                blocks.append(ParsedBlock(kind='comment',
                                          startLine=i,
                                          text=line))
        elif match := _re.search(r"^\s*opcode\s+(\w+)\s*,\s*([0ika\[\]]*),\s*([0ikaoOjJpP\[\]]*)", line):
            context.append('opcode')
            block = _OrcBlock(name=match.group(1),
                              startLine = i,
                              lines = [line],
                              outargs = match.group(2),
                              inargs = match.group(3)
                              )
        elif strippedline.startswith('#include'):
            blocks.append(ParsedBlock(kind='include',
                                      startLine=i,
                                      text=line,
                                      ))
        else:
            blocks.append(ParsedBlock(kind='instr0',
                                      startLine=i,
                                      text=line))
    return blocks


def _hashdict(d: dict) -> int:
    return hash((frozenset(d.keys()), frozenset(d.values())))


@dataclasses.dataclass
class ParsedInstrBody:
    """
    The result of parsing the body of an instrument

    This is used by :func:`instrParseBody`

    """
    pfieldIndexToName: dict[int, str]
    """Maps pfield index to assigned name"""

    pfieldLines: Sequence[str]
    """List of lines where pfields are defined"""

    body: str
    """The body parsed"""

    lines: Sequence[str]
    """The body, split into lines"""

    pfieldIndexToValue: dict[int, float] | None = None
    "Default values of the pfields, by pfield index"

    pfieldsUsed: set[int] | None = None
    "Which pfields are accessed"

    outChannels: set[int] | None = None
    "Which output channels are used"

    @_functools.cached_property
    def pfieldsText(self) -> str:
        """The text containing pfield definitions"""
        return "\n".join(self.pfieldLines)

    @_functools.cached_property
    def pfieldNameToIndex(self):
        """Maps pfield name to its index"""
        return {name: idx for idx, name in self.pfieldIndexToName.items()}

    def numPfields(self) -> int:
        """ Returns the number of pfields in this instrument """
        return 3 if not self.pfieldsUsed else max(self.pfieldsUsed)

    @_functools.cached_property
    def pfieldNameToValue(self) -> dict[str, float]:
        """
        Dict mapping pfield name to default value

        If a pfield has no explicit name assigned, p## is used. If it has no explicit
        value, 0. is used

        Example
        ~~~~~~~

        Given a csound instr:

        >>> parsed = instrParseBody(r'''
        ... pset 0, 0, 0, 0.1, 400, 0.5
        ... iamp = p4
        ... kfreq = p5
        ... ''')
        >>> parsed.pfieldNameToValue
        {'iamp': 0.1, 'kfreq': 400, 'p6': 0.5}

        """
        if not self.pfieldNameToIndex:
            return EMPTYDICT

        if self.pfieldIndexToValue is not None:
            out1 = {(self.pfieldIndexToName.get(idx) or f"p{idx}"): value
                    for idx, value in self.pfieldIndexToValue.items()}
        else:
            out1 = {}
        if self.pfieldIndexToName is not None:
            assert self.pfieldIndexToValue is not None
            out2 = {name: self.pfieldIndexToValue.get(idx, 0.)
                    for idx, name in self.pfieldIndexToName.items()}
        else:
            out2 = {}
        out1.update(out2)
        return out1


def lastAssignmentToVariable(varname: str, lines: list[str]) -> int | None:
    """
    Line of the last assignment to a variable

    Given a piece of code (normally the body of an instrument)
    find the line in which the given variable has its **last**
    assignment

    Args:
        varname: the name of the variable
        lines: the lines which make the instrument body. We need to split
            the body into lines within the function itself and since the
            user might need to split the code anyway afterwards, we
            already ask for the lines instead.

    Returns:
        the line number of the last assignment, or None if there is no
        assignment to the given variable

    Possible matches::

        aout oscili 0.1, 1000
        aout, aout2 pan2 ...
        aout = ...
        aout=...
        aout += ...
        aout2, aout = ...

    Example
    ~~~~~~~

        >>> lastAssignmentToVariable("aout", r'''
        ... aout oscili:a(0.1, 1000)
        ... aout *= linen:a(...)
        ... aout = aout + 10
        ... outch 1, aout
        ... '''.splitlines())
        3
    """
    rgxs = [
        _re.compile(rf'^\s*({varname})\s*(=|\*=|-=|\+=|\/=)'),
        _re.compile(rf'^\s*({varname})\s*,'),
        _re.compile(rf'^\s*({varname})\s+[A-Za-z]\w*'),
        _re.compile(rf'^\s*(?:\w*,\s*)+\b({varname})\b')
    ]
    for i, l in enumerate(reversed(lines)):
        for rgx in rgxs:
            if rgx.search(l):
                return len(lines) - 1 - i
    return None


def locateDocstring(lines: list[str]) -> tuple[int | None, int]:
    """
    Locate the docstring in this instr code

    Args:
        lines: the code to analyze, tipically the code inside an instr
            (between instr/endin), split into lines

    Returns:
        a tuple (firstline, lastline) indicating the location of the docstring
        within the given text. firstline will be None if no docstring was found

    """
    assert isinstance(lines, list)
    docstringStart = None
    docstringEnd = 0
    docstringKind = ''
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if docstringStart is None:
            if _re.search(r'(;|\/\/|\/\*)', line):
                docstringStart = i
                docstringKind = ';' if line[0] == ';' else line[:2]
                continue
            else:
                # Not a docstring, so stop looking
                break
        else:
            # inside docstring
            if docstringKind == '/*':
                # TODO
                pass
            elif line.startswith(docstringKind):
                docstringEnd = i+1
            else:
                break
    if docstringStart is not None and docstringEnd < docstringStart:
        docstringEnd = docstringStart + 1
    return docstringStart, docstringEnd


def splitDocstring(body: str | list[str]) -> tuple[str, str]:
    if isinstance(body, str):
        lines = body.splitlines()
    else:
        lines = body
    docstart, docend = locateDocstring(lines)
    if docstart is not None:
        docstring = '\n'.join(lines[docstart:docend])
        rest = '\n'.join(lines[docend:])
    else:
        docstring = ''
        rest = body
    return docstring, rest


@_functools.cache
def instrParseBody(body: str) -> ParsedInstrBody:
    """
    Parses the body of an instrument, returns pfields used, output channels, etc.

    Args:
        body (str): the body of the instr (between instr/endin)

    Returns:
        a ParsedInstrBody

    Example
    -------

        >>> from csoundengine import csoundlib
        >>> body = r'''
        ... pset 0, 0, 0, 1, 1000
        ... ibus = p4
        ... kfreq = p5
        ... a0 = busin(ibus)
        ... a1 = oscili:a(0.5, kfreq) * a0
        ... outch 1, a1
        ... '''
        >>> csoundlib.instrParseBody(body)
        ParsedInstrBody(pfieldsIndexToName={4: 'ibus', 5: 'kfreq'},
                        pfieldLines=['ibus = p4', ['kfreq = p5'], 
                        body='\\na0 = busin(ibus)\\n
                          a1 = oscili:a(0.5, kfreq) * a0\\noutch 1, a1',
                        pfieldsDefaults={1: 0.0, 2: 0.0, 3: 0.0, 4: 1.0, 5: 1000.0},
                        pfieldsUsed={4, 5},
                        outChannels={1},
                        pfieldsNameToIndex={'ibus': 4, 'kfreq': 5})
    """
    if not body.strip():
        return ParsedInstrBody(pfieldIndexToValue=EMPTYDICT,
                               pfieldLines=(),
                               body='',
                               lines=(),
                               pfieldIndexToName=EMPTYDICT)

    pfieldLines = []
    bodyLines = []
    pfieldIndexToValue = {}
    insideComment = False
    pfieldsUsed = set()
    pfieldIndexToName: dict[int, str] = {}
    outchannels: set[int] = set()
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if insideComment:
            bodyLines.append(line)
            if _re.match(r"\*\/", line):
                insideComment = False
            continue
        elif _re.match(r"^\s*(;|\/\/)", line):
            # A line comment
            bodyLines.append(line)
            continue
        else:
            # Not inside comment
            if pfieldsInLine := _re.findall(r"\bp\d+", line):
                for p in pfieldsInLine:
                    pfieldsUsed.add(int(p[1:]))

            if _re.match(r"^\s*\/\*", line):
                insideComment = True
                bodyLines.append(line)
            elif _re.match(r"\*\/", line) and insideComment:
                insideComment = False
                bodyLines.append(line)
            elif m := _re.search(r"\bpassign\s+(\d+)", line):
                if "[" in line:
                    # array form, iarr[] passign 4, 6
                    bodyLines.append(line)
                else:
                    pfieldLines.append(line)
                    pstart = int(m.group(1))
                    argsstr, rest = line.split("passign")
                    args = argsstr.split(",")
                    for j, name in enumerate(args, start=pstart):
                        pfieldsUsed.add(j)
                        pfieldIndexToName[j] = name.strip()
            elif _re.search(r"^\s*\bpset\b", line):
                s = line.strip()[4:]
                psetValues = {j: float(v) for j, v in enumerate(s.split(","), start=1)
                              if v.strip()[0].isnumeric()}
                pfieldIndexToValue.update(psetValues)
            elif m := _re.search(r"^\s*\b(\w+)\s*(=|init\s)\s*p(\d+)", line):
                # 'ival = p4' / kval = p4 or 'ival init p4'
                pname = m.group(1)
                pfieldIndex = int(m.group(3))
                pfieldLines.append(line)
                pfieldIndexToName[pfieldIndex] = pname.strip()
                pfieldsUsed.add(pfieldIndex)
            else:
                if _re.search(r"\bouts\s+", line):
                    outchannels.update((1, 2))
                elif _re.search(r"\bout\b", line):
                    outchannels.add(1)
                elif _re.search(r"\boutch\b", line):
                    args = line.strip()[5:].split(",")
                    channels = args[::2]
                    for chans in channels:
                        chan = emlib.misc.asnumber(chans)
                        if chan is not None and int(chan) == chan:
                            outchannels.add(chan)
                bodyLines.append(line)

    for pidx in range(1, 4):
        pfieldIndexToValue.pop(pidx, None)
        pfieldIndexToName.pop(pidx, None)

    bodyLines = [line for line in bodyLines if line.strip()]

    return ParsedInstrBody(pfieldIndexToValue=pfieldIndexToValue,
                           pfieldIndexToName=pfieldIndexToName,
                           pfieldsUsed=pfieldsUsed,
                           outChannels=outchannels,
                           pfieldLines=pfieldLines,
                           body="\n".join(bodyLines),
                           lines=lines)


def bestSampleEncodingForExtension(ext: str) -> str | None:
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
        raise ValueError(f"Format {ext} not supported")


def _parsePresetSflistprograms(line:str) -> tuple[str, int, int] | None:
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


class SoundFontIndex:
    """
    Creates an index of presets for a given soundfont

    Attributes:
        instrs: a list of instruments, where each instrument is a tuple (instr. index, name)
        presets: a list of presets, where each preset is a tuple (bank, num, name)
        nameToIndex: a dict mapping instr name to index
        indexToName: a dict mapping instr idx to name
        nameToPreset: a dict mapping preset name to (bank, num)
        presetToName: a dict mapping (bank, num) to preset name
    """
    def __init__(self, soundfont: str):
        assert _os.path.exists(soundfont)
        self.soundfont = soundfont
        instrs, presets = _soundfontInstrumentsAndPresets(soundfont)
        self.instrs: list[tuple[int, str]] = instrs
        self.presets: list[tuple[int, int, str]] = presets
        self.nameToIndex: dict[str, int] = {name:idx for idx, name in self.instrs}
        self.indexToName: dict[int, str] = {idx:name for idx, name in self.instrs}
        self.nameToPreset: dict[str, tuple[int, int]] = {name: (bank, num)
                                                         for bank, num, name in self.presets}
        self.presetToName: dict[tuple[int, int], str] = {(bank, num): name
                                                         for bank, num, name in self.presets}


@_functools.cache
def soundfontIndex(sfpath: str) -> SoundFontIndex:
    """
    Make a SoundFontIndex for the given soundfont

    Args:
        sfpath: the path to a soundfont (.sf2) file

    Returns:
        a SoundFontIndex

    Example
    ~~~~~~~

        >>> from csoundengine import csoundlib
        >>> idx = csoundlib.soundfontIndex("/path/to/piano.sf2")
        >>> idx.nameToPreset
        {'piano': (0, 0)}
        >>> idx.nameToIndex
        {'piano': 0}
    """
    return SoundFontIndex(sfpath)


@_functools.cache
def _soundfontInstrumentsAndPresets(sfpath: str
                                    ) -> tuple[list[tuple[int, str]],
                                               list[tuple[int, int, str]]]:
    """
    Returns a tuple (instruments, presets)

    Where instruments is a list of tuples(instridx, instrname) and presets
    is a list of tuples (bank, presetnum, name)

    Args:
        sfpath: the path to the soundfont

    Returns:
        a tuple (instruments, presets), where instruments is a list
        of tuples (instrindex, instrname) and prests is a list of
        tuples (bank, presetindex, name)
    """
    from sf2utils.sf2parse import Sf2File
    f = open(sfpath, 'rb')
    sf = Sf2File(f)
    instruments: list[tuple[int, str]] = [(num, instr.name.strip())
                                          for num, instr in enumerate(sf.instruments)
                                          if instr.name != 'EOI']
    presets: list[tuple[int, int, str]] = [(p.bank, p.preset, p.name.strip())
                                           for p in sf.presets if p.name != 'EOP']
    presets.sort()
    return instruments, presets


def soundfontInstruments(sfpath: str) -> list[tuple[int, str]]:
    """
    Get instruments for a soundfont

    The instrument index is used by csound opcodes like `sfinstr`. These
    are different from soundfont programs, which are ordered in
    banks/presets

    Args:
        sfpath: the path to the soundfont. "?" to open a file-browser dialog

    Returns:
        list[tuple[int,str]] - a list of tuples, where each tuple has the form
        (index: int, instrname: str)
    """
    if sfpath == "?":
        sfpath = _state.openSoundfont(ensureSelection=True)
    instrs, _ = _soundfontInstrumentsAndPresets(sfpath)
    return instrs


def soundfontPresets(sfpath: str) -> list[tuple[int, int, str]]:
    """
    Get presets from a soundfont

    Args:
        sfpath: the path to the soundfont. "?" to open a file-browser dialog

    Returns:
        a list of tuples ``(bank:int, presetnum:int, name:str)``
    """
    if sfpath == "?":
        sfpath = _state.openSoundfont(ensureSelection=True)
    _, presets = _soundfontInstrumentsAndPresets(sfpath)
    return presets


def soundfontSelectPreset(sfpath: str
                          ) -> tuple[str, int, int] | None:
    """
    Select a preset from a soundfont interactively

    Returns:
        a tuple (preset name, bank, preset number) if a selection was made, None
        otherwise

    .. figure:: ../assets/select-preset.png
    """
    presets = soundfontPresets(sfpath)
    items = [f'{bank:03d}:{pnum:03d}:{name}' for bank, pnum, name in presets]
    item = emlib.dialogs.selectItem(items, ensureSelection=True)
    if item is None:
        return None
    idx = items.index(item)
    preset = presets[idx]
    bank, pnum, name= preset
    return (name, bank, pnum)


def soundfontInstrument(sfpath: str, name:str) -> int | None:
    """
    Get the instrument number from a preset

    The returned instrument number can be used with csound opcodes like `sfinstr`
    or `sfinstr3`

    Args:
        sfpath: the path to a .sf2 file. "?" to open a file-browser dialog
        name: the instrument name

    Returns:
        the instrument index, if exists
    """
    if sfpath == "?":
        sfpath = _state.openSoundfont(ensureSelection=True)
    sfindex = soundfontIndex(sfpath)
    return sfindex.nameToIndex.get(name)


def splitInclude(line: str) -> str:
    """
    Given an include line it splits the include path

    Example
    ~~~~~~~

        >>> splitInclude(r'   #include "foo/bar" ')
        foo/bar

    NB: the quotation marks are not included
    """
    match = _re.search(r'#include\s+"(.+)""', line)
    if not match:
        raise ValueError("Could not parse include")
    return match.group(1)


def makeIncludeLine(include: str) -> str:
    """
    Given a path, creates the #include directive

    In particula, it checks the need for quotation marks

    Args:
        include: path to include

    Returns:

    """
    s = emlib.textlib.quoteIfNeeded(include.strip())
    return f'#include {s}'


@_functools.cache
def _pygmentsOrcLexer():
    import pygments.lexers.csound
    return pygments.lexers.csound.CsoundOrchestraLexer()


def highlightCsoundOrc(code: str, theme='') -> str:
    """
    Converts csound code to html with syntax highlighting

    Args:
        code: the code to highlight
        theme: the theme used, one of 'light', 'dark'. If not given, a default
            is used (see config['html_theme'])

    Returns:
        the corresponding html
    """
    if not theme:
        from .config import config
        theme = config['html_theme']

    import pygments
    if theme == 'light':
        htmlfmt = pygments.formatters.HtmlFormatter(noclasses=True, wrapcode=True)
    else:
        htmlfmt = pygments.formatters.HtmlFormatter(noclasses=True, style='fruity',
                                                    wrapcode=True)
    html = pygments.highlight(code, lexer=_pygmentsOrcLexer(), formatter=htmlfmt)
    return html


def _eventEnd(event) -> float | None:
    if len(event) >= 4:
        # 0 1  2  3
        # i p1 p2 p3
        start = event[2]
        dur = event[3]
        if dur == -1:
            return float('inf')
        return start + dur
    elif len(event) == 2:
        return event[1]
    else:
        return None


def _cropScore(events: list[Sequence], start=0., end=0.) -> list:
    """
    Crop the score so that no event exceeds the given limits

    Args:
        events: a list of events, where each event is a sequence
            representing the pargs [p1, p2, p3, ...]
        start: the min. start time for any event
        end: the max. end time for any event

    Returns:
        the score events which are between start and end
    """
    scoreend = max(_ for ev in events
                   if (_ := _eventEnd(ev)) is not None)
    assert scoreend is not None and scoreend > 0, f"Invalid score duration ({scoreend}): {events}"
    if end == 0:
        end = scoreend
    cropped = []
    for ev in events:
        kind = ev[0]
        if kind == 'e' or kind == 'f':
            evstart = ev[1]
            if start <= evstart < end:
                cropped.append(ev)
        elif kind != 'i':
            cropped.append(ev)
            continue

        evstart = ev[2]
        evdur = ev[3]
        evend = evstart + evdur if evdur >= 0 else float('inf')
        if evend < start or evstart > end:
            continue

        if start <= evstart and evend <= end:
            cropped.append(ev)
        else:
            xstart, xend = emlib.mathlib.intersection(start, end, evstart, evend)
            if xstart is not None:
                if xend == float('inf'):
                    dur = -1
                else:
                    dur = xend - xstart
                ev = list(ev)
                ev[2] = xstart
                ev[3] = dur
                cropped.append(ev)
    return cropped


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


def isPfield(name: str) -> bool:
    """
    Is name a pfield?
    """
    return _re.match(r'\bp[1-9][0-9]*\b', name) is not None

