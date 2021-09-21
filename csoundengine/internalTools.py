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
    from csoundlib import AudioDevice
    

_registry: Dict[str, Any] = {}


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


def getChannel(samples: np.ndarray, channel: int) -> np.ndarray:
    """ Get a channel of a numpy array holding possibly multichannel
    audio data

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
                     pargs: Union[List[float], Dict[str, float]]=None,
                     pkws: Dict[str, float]=None
                     ) -> List[float]:
    allargs: List[float] = [float(p4)]
    if not pargs and not instr.pargsDefaultValues and not pkws:
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


def instrWrapBody(body:str, instrid:Union[int, str, Sequence[str]], comment:str= '',
                  addNotificationCode=False) -> str:
    s = r"""
instr {instrnum}  {commentstr}
    {notifystr}
    {body}
endin
    """
    if addNotificationCode:
        # notifystr = 'defer "outvalue", "__dealloc__", p1'
        # TODO: defer can cause memory corruption in some cases
        notifystr = 'atstop "_notifyDealloc", 0.01, 0.0, p1'
    else:
        notifystr = ''
    commentstr = "; " + comment if comment else ""
    if isinstance(instrid, (list, tuple)):
        instrid = ", ".join([str(i) for i in instrid])
    s = s.format(instrnum=instrid, body=body, notifystr=notifystr,
                 commentstr=commentstr)
    s = textwrap.dedent(s)
    return s


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


def resolveOption(prioritizedOptions:List[str], availableOptions:List[str]
                  ) -> Optional[str]:
    for opt in prioritizedOptions:
        if opt in availableOptions:
            return opt
    return None


def selectAudioDevice(devices: List[AudioDevice], title='Select device'
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


def selectItem(items: List[str], title="Select") -> Optional[str]:
    return emlib.dialogs.selectItem(items=items, title=title)


def instrNameFromP1(p1: Union[float, str]) -> Union[int, str]:
    return int(p1) if isinstance(p1, (int, float)) else p1.split(".")[0]


def resolvePfieldIndex(pfield: Union[int, str], pfieldNameToIndex: Dict[str, int] = None
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