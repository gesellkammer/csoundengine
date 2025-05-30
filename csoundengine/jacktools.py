# depends on JACK-client: https://pypi.python.org/pypi/JACK-Client
from __future__ import annotations

import dataclasses
import time
from typing import Any

from . import linuxaudio

try:
    import jack
    JACK_INSTALLED = True
except OSError:
    # the jack library was not found so no jack support here
    JACK_INSTALLED = False


_cache: dict[str, tuple[Any, float]] = {
}


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
    systemInput: str
    systemOutput: str
    onPipewire: bool = False


def getInfo() -> JackInfo | None:
    """
    Get info about a running jack server

    Returns:
        a JackInfo
    """
    # assumes that jack is running
    if not JACK_INSTALLED:
        return None
    cachedinfo, lasttime = _cache.get('getInfo', (None, 0.))
    if lasttime > 0 and time.time() - lasttime < 20:
        return cachedinfo

    try:
        c = jack.Client("jacktools-getinfo", no_start_server=True)
    except jack.JackOpenError:
        return None
    inports = c.get_ports(is_audio=True, is_physical=True, is_input=True)
    outports = c.get_ports(is_audio=True, is_physical=True, is_output=True)
    systemOutput = _buildClients(inports)[0]
    systemInput = _buildClients(outports)[0] if outports else None
    onPipewire = linuxaudio.isPipewireRunning()
    sr = c.samplerate
    blocksize = c.blocksize
    out = JackInfo(running=True,
                   samplerate=sr,
                   blocksize=blocksize,
                   numOutChannelsPhysical=len(inports),
                   numInChannelsPhysical=len(outports),
                   systemInput=systemOutput.name,
                   systemOutput=systemInput.name if systemInput else '',
                   onPipewire=onPipewire)
    c.close()
    _cache['getInfo'] = (out, time.time())
    return out


def bufferSizeAndNum() -> tuple[int, int]:
    # assumes that jack is running
    info = getInfo()
    if info is None:
        raise RuntimeError("Cannot get buffer size, failed to get info about jack")
    return info.blocksize, 2


@dataclasses.dataclass
class JackClient:
    name: str
    kind: str
    regex: str
    isPhysical: bool = False
    ports: list[jack.Port] = dataclasses.field(default_factory=list)
    firstIndex: int = -1


def _splitlast(s:str, sep:str) -> tuple[str, str]:
    if sep not in s:
        return (s, '')
    s2 = s[::-1]
    last, first = s2.split(sep, maxsplit=1)
    return first[::-1], last[::-1]


def _buildClients(ports: list[jack.Port]) -> list[JackClient]:
    assert all(p.is_output for p in ports) or all(p.is_input for p in ports)
    d = {}
    regexes = {}
    for i, p in enumerate(ports):
        prefix = _splitlast(p.name, ":")[0]
        # special case system monitor
        if prefix == 'system':
            if p.shortname.startswith('monitor_'):
                prefix = 'system monitor'
            regex = p.name.split("_")[0]
        else:
            regex = prefix + ':'
        if prefix not in d:
            d[prefix] = []
            regexes[prefix] = regex
        d[prefix].append(i)

    clients = []
    for name, portindexes in d.items():
        portindexes.sort()
        port0 = ports[portindexes[0]]
        client = JackClient(name,
                            regex=regexes[name],
                            kind='output' if port0.is_output else 'input',
                            isPhysical=port0.is_physical,
                            ports=[ports[i] for i in portindexes],
                            firstIndex=portindexes[0])
        clients.append(client)
    return clients


def getClients() -> list[JackClient]:
    """
    Get a list of running clients

    A client of kind "output" has only output ports, which means it can be
    an input to another client
    """
    cachedclients, lasttime = _cache.get('getClients', ([], 0.))
    if lasttime > 0 and time.time() - lasttime < 20:
        return cachedclients
    client = jack.Client("jacktools", no_start_server=True)
    inclients = _buildClients(client.get_ports(is_audio=True, is_input=True))
    outclients = _buildClients(client.get_ports(is_audio=True, is_output=True))
    allclients = inclients + outclients
    _cache['getClients'] = (allclients, time.time())
    return allclients


def getSystemClients() -> tuple[JackClient | None, JackClient | None]:
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
