# depends on JACK-client: https://pypi.python.org/pypi/JACK-Client

import sys
import subprocess
import shutil
import cachetools
import dataclasses
from typing import Optional as Opt
import jack
try:
    import jack
    JACK_INSTALLED = True
except OSError:
    # the jack library was not found so no jack support here
    JACK_INSTALLED = False

class PlatformNotSupportedError(Exception): pass


@cachetools.cached(cache=cachetools.TTLCache(1, 20))
def isJackRunning() -> bool:
    """
    Returns True if jack is running.

    .. note::
        The result is cached for a certain amount of time. Use `jack_running_check`
        for an uncached version
    """
    return isJackRunningUncached()


def isJackRunningUncached() -> bool:
    """
    Returns True if jack is running.
    """
    if sys.platform == "linux":
        jack_control = shutil.which("jack_control")
        if jack_control:
            proc = subprocess.Popen([jack_control, "status"],
                                    stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            if proc.wait() == 0:
                return True
    if not JACK_INSTALLED:
        return False
    try:
        cl = jack.Client("checkjack", no_start_server=True)
    except jack.JackOpenError:
        return False
    return True


@dataclasses.dataclass
class JackInfo:
    running: bool
    samplerate: int = 0
    blocksize: int  = 0


def getInfo() -> JackInfo:
    if not isJackRunning():
        return JackInfo(running=False)
    c = jack.Client("ujacktools", no_start_server=True)
    return JackInfo(running=True,
                    samplerate=c.samplerate,
                    blocksize=c.blocksize)

