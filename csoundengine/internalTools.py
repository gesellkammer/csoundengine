from __future__ import annotations
import numpy as np
import sys
import os
from typing import Optional as Opt, TYPE_CHECKING, Union as U, List, Dict, Any
from .config import config
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


def defaultSoundfontPath() -> Opt[str]:
    """
    Returns the path of the fluid sf2 file
    """
    key = 'fluidsf2_path'
    path = _registry.get(key)
    if path:
        return path
    userpath = config['generalmidi_soundfont']
    if userpath and os.path.exists(userpath):
        _registry[key] = userpath
        return userpath
    if sys.platform == 'linux':
        paths = ["/usr/share/sounds/sf2/FluidR3_GM.sf2"]
        path = next((path for path in paths if os.path.exists(path)), None)
        if path:
            _registry[key] = path
            return path
    else:
        raise RuntimeError("only works for linux right now")
    return None


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
    notifystr  = 'atstop "_notifyDealloc", 0, 0.1, p1' if addNotificationCode else ''
    commentstr = "; " + comment if comment else ""
    s = s.format(instrnum=instrid, body=body, notifystr=notifystr,
                 commentstr=commentstr)
    s = textwrap.dedent(s)
    return s


def addLineNumbers(code: str) -> str:
    lines = [f"{i:03d}  {line}"
             for i, line in enumerate(code.splitlines(), start=1)]
    return "\n".join(lines)


_platforms = {
    'linux2': 'linux',
    'linux': 'linux',
    'darwin': 'macos',
    'win32': 'windows',
}

platform = _platforms[sys.platform]

