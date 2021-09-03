# depends on JACK-client: https://pypi.python.org/pypi/JACK-Client
from __future__ import annotations
import cachetools
import dataclasses
from typing import List, Tuple, Set, Dict, Optional
from . import linuxaudio

try:
    import jack
    JACK_INSTALLED = True
except OSError:
    # the jack library was not found so no jack support here
    JACK_INSTALLED = False


def isJackRunning() -> bool:
    """
    Returns True if jack is running.

    .. note::
        The result is cached for a certain amount of time. Use `jack_running_check`
        for an uncached version
    """
    return getInfo() is not None


@dataclasses.dataclass
class JackInfo:
    running: bool
    samplerate: int
    blocksize: int
    numOutChannelsPhysical: int
    numInChannelsPhysical: int
    systemInput: JackClient
    systemOutput: JackClient
    onPipewire: bool = False


@cachetools.cached(cache=cachetools.TTLCache(1, 20))
def getInfo() -> Optional[JackInfo]:
    """
    Get info about a running jack server

    Returns:
        a JackInfo
    """
    # assumes that jack is running
    if not JACK_INSTALLED:
        return None
    try:
        c = jack.Client("jacktools.getInfo", no_start_server=True)
    except jack.JackOpenError:
        return None
    inports = c.get_ports(is_audio=True, is_physical=True, is_input=True)
    outports = c.get_ports(is_audio=True, is_physical=True, is_output=True)
    systemOutput = _buildClients(inports)[0]
    systemInput = _buildClients(outports)[0]
    onPipewire = linuxaudio.isPipewireRunning()
    return JackInfo(running=True,
                    samplerate=c.samplerate,
                    blocksize=c.blocksize,
                    numOutChannelsPhysical=len(inports),
                    numInChannelsPhysical=len(outports),
                    systemInput=systemOutput,
                    systemOutput=systemInput,
                    onPipewire=onPipewire)


def bufferSizeAndNum() -> Tuple[int, int]:
    # assumes that jack is running
    info = getInfo()
    return info.blocksize, 2


@dataclasses.dataclass
class JackClient:
    name: str
    kind: str
    regex: str
    isPhysical: bool = False
    ports: List[jack.Port] = dataclasses.field(default_factory=list)


def _splitlast(s:str, sep:str) -> Tuple[str, str]:
    if sep not in s:
        return (s, '')
    s2 = s[::-1]
    last, first = s2.split(sep, maxsplit=1)
    return first[::-1], last[::-1]


def _buildClients(ports: List[jack.Port]) -> List[JackClient]:
    assert all(p.is_output for p in ports) or all(p.is_input for p in ports)
    d = {}
    regexes = {}
    for p in ports:
        prefix = _splitlast(p.name, ":")[0]
        # special case system monitor
        if prefix == 'system':
            if p.shortname.startswith('monitor_'):
                prefix = 'system monitor'
            regex = p.name.split("_")[0]
        else:
            regex = prefix + ':'
        if prefix not in d:
            d[prefix] = [p]
            regexes[prefix] = regex
        else:
            d[prefix].append(p)
    return [JackClient(name,
                       regex=regexes[name],
                       kind='output' if ports[0].is_output else 'input',
                       isPhysical=ports[0].is_physical,
                       ports=ports)
            for name, ports in d.items()]


@cachetools.cached(cache=cachetools.TTLCache(1, 20))
def getClients() -> List[JackClient]:
    """
    Get a list of running clients

    A client of kind "output" has only output ports, which means it can be
    an input to another client
    """
    client = jack.Client("jacktools", no_start_server=True)
    inclients = _buildClients(client.get_ports(is_audio=True, is_input=True))
    outclients = _buildClients(client.get_ports(is_audio=True, is_output=True))
    return inclients + outclients


def getSystemClients() -> Tuple[Optional[JackClient], Optional[JackClient]]:
    """
    Returns the hardware (physical) clients (input, output)

    Returns input and output client. An input client has only inputs,
    an output client has outputs. The physical sound-card output (playback)
    is an input client while the microphone/line-in is an output client
    """
    clients = getClients()
    inphysical = next((c for c in clients
                       if c.kind == 'input' and c.isPhysical), None)
    outphysical = next((c for c in clients
                        if c.kind == 'output' and c.isPhysical), None)
    return inphysical, outphysical
