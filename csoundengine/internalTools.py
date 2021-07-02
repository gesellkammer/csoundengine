from __future__ import annotations
import numpy as np
import sys
import os
from typing import Optional as Opt, TYPE_CHECKING, Union as U, List, Dict, Any
from . import jacktools
import signal
import math
import textwrap

if TYPE_CHECKING:
    from .instr import Instr

_registry: Dict[str, Any] = {}


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
                     pargs: U[List[float], Dict[str, float]]=None,
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

def instrWrapBody(body:str, instrid:U[int, str], comment:str= '',
                  addNotificationCode=False) -> str:
    s = """
instr {instrnum}  {commentstr}
    {notifystr}
    {body}
endin
    """
    # notifystr  = 'atstop "_notifyDealloc", 0.01, 0.01, p1' if addNotificationCode else ''
    if addNotificationCode:
        notifystr = 'defer "outvalue", "__dealloc__", p1'
    else:
        notifystr = ''
    commentstr = "; " + comment if comment else ""
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
                  ) -> Opt[str]:
    for opt in prioritizedOptions:
        if opt in availableOptions:
            return opt
    return None


