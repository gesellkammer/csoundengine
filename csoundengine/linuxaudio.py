from __future__ import annotations
import cachetools
import os
from . import internalTools
from dataclasses import dataclass
import subprocess
import shutil
import re
import logging


logger = logging.getLogger('csoundengine')


def _getprocs() -> list[tuple[int, list[str]]]:
    """
    Get a list of all processes running

    Returns:
        a list where each item is a tuple (pid, args: list[str])
    """
    procs = []
    for dirname in os.listdir('/proc'):
        if dirname == 'curproc' or not dirname.isdigit():
            continue
        try:
            with open('/proc/{}/cmdline'.format(dirname), mode='rb') as fd:
                args = fd.read().decode().split('\x00')[:-1]
        except Exception:
            continue
        if args:
            procs.append((int(dirname), args))
    return procs


# @cachetools.cached(cache=cachetools.TTLCache(1, 20))
def isPipewireRunning() -> bool:
    return internalTools.isrunning("pipewire")


# @cachetools.cached(cache=cachetools.TTLCache(1, 20))
def isPulseaudioRunning() -> bool:
    """
    Returns True if the pulseaudio server is running

    .. note:: pulseaudio can be running on pipewire even if this returns False
    """
    if not shutil.which("pulseaudio"):
        return False
    status = subprocess.call(["pulseaudio", "--check"])
    return status == 0


@dataclass
class PipewireInfo:
    sr: int
    quantum: int
    isPulseServer: bool
    numchannels: int = 2


@dataclass
class PulseaudioInfo:
    sr: int
    numchannels: int
    onPipewire: bool = False


@cachetools.cached(cache=cachetools.TTLCache(1, 5))
def _pactlinfo() -> PulseaudioInfo | None:
    if not shutil.which("pactl"):
        return None
    sr = 48000
    numchannels = 2
    onPipewire = False
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


@cachetools.cached(cache=cachetools.TTLCache(1, 5))
def pulseaudioInfo() -> PulseaudioInfo | None:
    """
    Returns info about the pulseaudio server, or None if not running

    This function will work even if pulseaudio is running as a pipewire
    server.

    Returns:
        a PulseaudioInfo object, or None if pulseaudio is not running
    """
    if internalTools.isrunning('pulseaudio'):
        # pulseaudio server is running
        return _pactlinfo()

    # maybe on pipewire
    if not isPipewireRunning():
        return None

    # ok, pipewire is running, check if the pulseaudio interface is on
    pactlinfo = _pactlinfo()
    if pactlinfo is not None:
        # pactl is installed and working
        return pactlinfo

    # pactl is not installed, this is a pipewire pure installation
    pipeinfo = pipewireInfo()
    if pipeinfo is None or not pipeinfo.isPulseServer:
        # could not get info about pipewire
        return None

    return PulseaudioInfo(sr=pipeinfo.sr, numchannels=pipeinfo.numchannels,
                          onPipewire=True)


@cachetools.cached(cache=cachetools.TTLCache(1, 5))
def pipewireInfo() -> PipewireInfo | None:
    if not internalTools.isrunning('pipewire'):
        return None

    if shutil.which('pw-cli') is None:
        logger.debug("pipewire seems running but could not find pw-cli")
        return None

    sr = 48000
    quantum = 1024
    output = subprocess.check_output(['pw-cli', 'info', '0']).decode('utf-8')
    for line in output.splitlines():
        if match := re.search(r'default\.clock\.rate = "([0-9]+)"', line):
            sr = int(match.group(1))
        elif match := re.search(r'default\.clock\.quantum = "([0-9]+)"}', line):
            quantum = int(match.group(1))
    if shutil.which('pactl'):
        pactlinfo = _pactlinfo()
        isPulseServer = pactlinfo.onPipewire if pactlinfo is not None else False
    elif internalTools.isrunning('pulseaudio'):
        isPulseServer = False
    else:
        isPulseServer = internalTools.isrunning('pipewire-pulse') is not None
    return PipewireInfo(sr=sr, quantum=quantum, isPulseServer=isPulseServer)