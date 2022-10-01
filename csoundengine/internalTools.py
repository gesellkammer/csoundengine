from __future__ import annotations
import numpy as np
import sys
from . import jacktools
import signal
import math
import textwrap
from typing import TYPE_CHECKING
import emlib.dialogs
import subprocess

if TYPE_CHECKING:
    from .instr import Instr
    from typing import *
    from csoundlib import AudioDevice, MidiDevice
    

_registry: dict[str, Any] = {}


try:
    import xxhash
    def ndarrayhash(a: np.ndarray) -> str:
        return xxhash.xxh128_hexdigest(a)

except ImportError:
    import hashlib
    def ndarrayhash(a: np.ndarray) -> str:
        return hashlib.sha1(a).hexdigest()


def isrunning(prog: str) -> bool:
    "True if prog is running"
    failed = subprocess.call(['pgrep', '-f', prog],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return not failed


def m2f(midinote: float, a4:float) -> float:
    """
    Convert a midi-note to a frequency
    """
    return 2**((midinote-69)/12.0)*a4


def arrayNumChannels(a: np.ndarray) -> int:
    """
    Return the number of channels in a numpy array holding audio data
    """
    return 1 if len(a.shape) == 1 else a.shape[1]


def unflattenArray(a: np.ndarray, numchannels: int) -> None:
    """
    Unflatten array in place

    Args:
        a: the array to unflatten
        numchannels: the number of audio channels in the data
    """
    if len(a.shape) > 1:
        if a.shape[1] != numchannels:
           raise ValueError("Array is not flat but the number of channels"
                            f"diverge (given numchannels={numchannels}, "
                            f"array number of channels: {a.shape[1]}")
        return a
    numrows = len(a) / numchannels
    if numrows != int(numrows):
        raise ValueError("The array does not have an integral number of frames. "
                         f"(length: {len(a)} / {numchannels} = {numrows}")
    a.shape = (numrows, numchannels)


def getChannel(samples: np.ndarray, channel: int) -> np.ndarray:
    """
    Get a channel of a numpy array holding possibly multichannel audio data

    Args:
        samples: the (multichannel) audio data
        channel: the index of the channel to extract

    Returns:
        a numpy array holding a channel of audio data.
    """
    return samples if len(samples.shape) == 1 else samples[:, channel]


def sigintHandler(sig, frame):
    print(frame)
    raise KeyboardInterrupt("SIGINT (CTRL-C) while waiting")


def setSigintHandler():
    """
    Set own sigint handler to prevent CTRL-C from crashing csound

    It will do nothing if this was already set
    """
    if _registry.get('sigint_handler_set'):
        return
    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, sigintHandler)
    _registry['original_sigint_handler'] = original_handler
    _registry['sigint_handler_set'] = True


def removeSigintHandler():
    """
    Reset the sigint handler to its original state
    This will do nothing if our own handler was not set
    in the first place
    """
    if not _registry.get('sigint_handler_set'):
        return
    signal.signal(signal.SIGINT, _registry['original_sigint_handler'])
    _registry['sigint_handler_set'] = False


def determineNumbuffers(backend:str, buffersize:int) -> int:
    if backend == 'jack':
        info = jacktools.getInfo()
        numbuffers = int(math.ceil(info.blocksize / buffersize))
    else:
        numbuffers = 2
    return numbuffers


def instrResolveArgs(instr: Instr,
                     p4: int,
                     pargs: list[float] | dict[str, float] = None,
                     pkws: dict[str, float] = None,
                     ) -> list[float]:
    """
    Resolves pargs, returns pargs starting from p4

    Args:
        instr: the Instr instance
        p4: the value for p4
        pargs: pargs passed to the instr, starting with p5
        pkws: named pargs

    Returns:
        pargs passed to csound, **starting with p4**
    """
    allargs: list[float] = [float(p4)]
    if not pargs and not instr.pargsIndexToDefaultValue and not pkws:
        return allargs
    if isinstance(pargs, list):
        allargs.extend(instr.pargsTranslate(pargs, pkws))
    else:
        if pkws:
            if pargs:
                pargs.update(pkws)
            else:
                pargs = pkws
        allargs.extend(instr.pargsTranslate(kws=pargs))
    return allargs


def addNotifycationAtStop(body: str, notifyDeallocInstrnum: int) -> str:
    notifystr = f'atstop {notifyDeallocInstrnum}, 0.01, 0.0, p1'
    out = "\n".join([notifystr, body])
    return out


def instrWrapBody(body:str, instrid:Union[int, str, Sequence[str]], comment:str= '',
                  notifyDeallocInstrnum: int = 0
                  ) -> str:
    s = r"""
instr {instrnum}  {commentstr}
    {body}
endin
    """
    if notifyDeallocInstrnum > 0:
        body = addNotifycationAtStop(body, notifyDeallocInstrnum)
    commentstr = "; " + comment if comment else ""
    if isinstance(instrid, (list, tuple)):
        instrid = ", ".join([str(i) for i in instrid])
    s = s.format(instrnum=instrid, body=body, commentstr=commentstr)
    return textwrap.dedent(s)


def addLineNumbers(code: str) -> str:
    lines = [f"{i:03d}  {line}"
             for i, line in enumerate(code.splitlines(), start=1)]
    return "\n".join(lines)


# Maps platform values as given by sys.platform to more readable aliases
_platformAliases = {
    'linux2': 'linux',
    'linux': 'linux',
    'darwin': 'macos',
    'macos': 'macos',
    'win32': 'windows',
    'windows': 'windows'
}

platform = _platformAliases[sys.platform]


# Maps possible platform names to names as returned by sys.platform
_normalizedPlatforms = {
    'linux': 'linux',
    'win32': 'win32',
    'darwin': 'darwin',
    'windows': 'win32',
    'macos': 'darwin'
}

def platformAlias(platform: str) -> str:
    """
    Return the platform alias (macos, windows, linux) for the
    given platform (instead of darwin, win32, etc)

    This is the opposite of `normalizePlatform`
    """
    out = _platformAliases.get(platform)
    if out is None:
        raise KeyError(f"Platform {platform} unknown, possible values are"
                       f" {_platformAliases.keys()}")
    return out


def normalizePlatform(s:str) -> str:
    """Return the platform as given by sys.platform

    This is the opposite of `platformAlias`
    """
    out = _normalizedPlatforms.get(s)
    if out is None:
        raise KeyError(f"Platform {s} not known")
    return out


def resolveOption(prioritizedOptions:list[str], availableOptions:list[str]
                  ) -> Optional[str]:
    for opt in prioritizedOptions:
        if opt in availableOptions:
            return opt
    return None


def selectAudioDevice(devices: list[AudioDevice], title='Select device'
                      ) -> Optional[AudioDevice]:
    if len(devices) == 1:
        return devices[0]
    outnames = [dev.info() for dev in devices]
    selected = emlib.dialogs.selectItem(items=outnames, title=title)
    if not selected:
        return None
    idx = outnames.index(selected)
    outdev = devices[idx]
    return outdev


def selectMidiDevice(devices: list[MidiDevice], title='Select MIDI device'
                     ) -> Optional[MidiDevice]:
    """
    Select a midi device from the given devices

    Args:
        devices: the midi devices to select from, as returned from ...
        title: the title of the dialog

    Returns:
        the deviceid of the selected device, None if no selection was made
        If the devices are input devices, 'all' is added as option. The given
        value can be passed to -M csound option
    """
    if len(devices) == 1:
        return devices[0]
    names = [f"{dev.name} [{dev.deviceid}]" for dev in devices]
    selected = emlib.dialogs.selectItem(items=names, title=title)
    if not selected:
        return None
    else:
        name, devid = selected[:-1].split("[")
        return next(d for d in devices if d.deviceid == devid)



def selectItem(items: list[str], title="Select") -> Optional[str]:
    return emlib.dialogs.selectItem(items=items, title=title)


def instrNameFromP1(p1: Union[float, str]) -> Union[int, str]:
    return int(p1) if isinstance(p1, (int, float)) else p1.split(".")[0]


def resolvePfieldIndex(pfield: Union[int, str], pfieldNameToIndex: dict[str, int] = None
                       ) -> int:
    if isinstance(pfield, int):
        return pfield
    if pfield[0] == 'p':
        return int(pfield[1:])
    if not pfieldNameToIndex:
        return 0
    return pfieldNameToIndex.get(pfield, 0)


def isAscii(s: str) -> bool:
    return all(ord(c)<128 for c in s)



