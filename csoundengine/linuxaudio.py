from __future__ import annotations
import cachetools
from . import internalTools
import dataclasses
import subprocess
import shutil
import re


@cachetools.cached(cache=cachetools.TTLCache(1, 20))
def isPipewireRunning() -> bool:
    return internalTools.isrunning("pipewire")


@cachetools.cached(cache=cachetools.TTLCache(1, 20))
def isPulseaudioRunning() -> bool:
    """
    Returns True if a pulseaudio is running

    NB: pulseaudio still can be running on pipewire even if
    his returns False
    """
    if not shutil.which("pulseaudio"):
        return False
    status = subprocess.call(["pulseaudio", "--check"])
    return status == 0


@dataclasses.dataclass
class PipewireInfo:
    sr: int
    quantum: int
    isPulseServer: bool

@dataclasses.dataclass
class PulseaudioInfo:
    sr: int
    numchannels: int
    onPipewire: bool = False


@cachetools.cached(cache=cachetools.TTLCache(1, 20))
def pulseaudioInfo() -> PulseaudioInfo:
    sr = 48000
    numchannels = 2
    onPipewire = False
    if not shutil.which("pactl"):
        raise RuntimeError("pactl not found")
    output = subprocess.check_output(['pactl', 'info']).decode('utf-8')
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("Server Name"):
            serverName = line.split(":", maxsplit=1)[1]
            if "PipeWire" in serverName:
                onPipewire = True
        elif line.startswith("Default Sample Specification"):
            spec = line.split(":", maxsplit=1)[1]
            samplefmt, numchnlstr, srstr = spec.split()
            numchannels = int(numchnlstr[:-2])
            sr = int(srstr[:-2])
    return PulseaudioInfo(sr=sr, numchannels=numchannels, onPipewire=onPipewire)


@cachetools.cached(cache=cachetools.TTLCache(1, 20))
def pipewireInfo() -> PipewireInfo:
    assert shutil.which('pw-cli')
    sr = 48000
    quantum = 1024
    output = subprocess.check_output(['pw-cli', 'info', '0']).decode('utf-8')
    for line in output.splitlines():
        if match := re.search(r'default\.clock\.rate = "([0-9]+)"', line):
            sr = int(match.group(1))
        elif match := re.search(r'default\.clock\.quantum = "([0-9]+)"}', line):
            quantum = int(match.group(1))
    pulseinfo = pulseaudioInfo()
    return PipewireInfo(sr=sr, quantum=quantum, isPulseServer=pulseinfo.onPipewire)