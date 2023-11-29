from __future__ import annotations

import os
import cachetools
import numpy as np
import sys
import re
from . import jacktools
import signal
import math
import textwrap
from typing import TYPE_CHECKING
import emlib.dialogs
import emlib.iterlib
import emlib.misc
import subprocess
import bisect
import time
import xxhash
from functools import cache


if TYPE_CHECKING:
    from .instr import Instr
    from typing import *
    from csoundlib import AudioDevice, MidiDevice
    T = TypeVar('T')


_registry: dict[str, Any] = {}


def aslist(seq: Sequence[T] | np.ndarray) -> list[T]:
    if isinstance(seq, list):
        return seq
    elif isinstance(seq, np.ndarray):
        return seq.tolist()
    else:
        return list(seq)


def ndarrayhash(a: np.ndarray) -> str:
    """Calculates a str hash for the data in the array"""
    if a.flags.contiguous:
        return xxhash.xxh128_hexdigest(a)
    else:
        return str(id(a))


@cachetools.cached(cache=cachetools.TTLCache(1, 20))
def isrunning(prog: str) -> bool:
    """True if prog is running"""
    if sys.platform == 'linux':
        failed = subprocess.call(['pgrep', '-f', prog],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return not failed
    else:
        raise RuntimeError(f"This function is not supported for platform '{sys.platform}'")


def m2f(midinote: float, a4: float) -> float:
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
        return
    numrows = len(a) / numchannels
    if numrows != int(numrows):
        raise ValueError("The array does not have an integral number of frames. "
                         f"(length: {len(a)} / {numchannels} = {numrows}")
    a.shape = (int(numrows), numchannels)


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


def determineNumbuffers(backend: str, buffersize: int) -> int:
    """
    Calculates the number of buffers needed by a given backend

    Args:
        backend: the backend
        buffersize: the buffersize used

    Returns:
        the number of buffers
    """
    if backend == 'jack':
        info = jacktools.getInfo()
        if info is None:
            raise RuntimeError("Jack does not seem to be running")
        numbuffers = int(math.ceil(info.blocksize / buffersize))
    else:
        numbuffers = 2
    return numbuffers


def splitDict(d: dict[str, float],
              keys1: Sequence[str] | set[str] | KeysView,
              keys2: Sequence[str] | set[str] | KeysView
              ) -> tuple[dict[str, float], dict[str, float]]:
    """
    Given a dict, distribute its key: value pairs depending on to which group of keys they belong

    Args:
        d: the dict to split
        keys1: first set of keys
        keys2: second set of keys

    Returns:
        two dicts, corresponding to the keys1 and keys2
    """
    out1 = {}
    out2 = {}
    for k, v in d.items():
        if k in keys1:
            out1[k] = v
        else:
            assert k in keys2
            out2[k] = v
    return out1, out2


def resolveInstrArgs(instr: Instr,
                     p4: int,
                     pargs: list[float] | dict[str | int, float] | None = None,
                     pkws: dict[str | int, float] | None = None,
                     ) -> list[float | str]:
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
    allargs: list[float | str] = [float(p4)]
    if not pargs and not instr.pfieldIndexToValue and not pkws:
        return allargs
    if isinstance(pargs, list):
        allargs.extend(instr.pfieldsTranslate(pargs, pkws))
    else:
        if pkws:
            if pargs:
                pargs.update(pkws)
            else:
                pargs = pkws
        allargs.extend(instr.pfieldsTranslate(kws=pargs))
    allargs = [arg if isinstance(arg, str) else float(arg) for arg in allargs]
    return allargs


def instrWrapBody(body: str,
                  instrid: int | str | Sequence[str],
                  comment='',
                  ) -> str:
    s = r"""
instr {instrnum}  {commentstr}
    {body}
endin
    """
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


def normalizePlatform(s: str) -> str:
    """Return the platform as given by sys.platform

    This is the opposite of `platformAlias`
    """
    out = _normalizedPlatforms.get(s)
    if out is None:
        raise KeyError(f"Platform {s} not known")
    return out


def resolveOption(prioritizedOptions: list[str], availableOptions: list[str]
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


def resolvePfieldIndex(pfield: Union[int, str],
                       pfieldNameToIndex: dict[str, int] | None = None
                       ) -> int:
    if isinstance(pfield, int):
        return pfield
    if pfield[0] == 'p':
        return int(pfield[1:])
    if not pfieldNameToIndex:
        return 0
    return pfieldNameToIndex.get(pfield, 0)


def isAscii(s: str) -> bool:
    return all(ord(c) < 128 for c in s)


def consolidateDelay(pairs: Sequence[float], delay: float
                     ) -> tuple[Sequence[float], float]:
    """
    (2, 20, 3, 30, 4, 40), delay=3

    out = (0, 20, 1, 30, 2, 40), delay=5
    """
    t0 = pairs[0]
    assert t0 >= 0
    if t0 == 0:
        return pairs, delay
    out = []
    for t, v in emlib.iterlib.window(pairs, 2, 2):
        out.append(t - t0)
        out.append(v)
    return out, delay + t0


def cropDelayedPairs(pairs: Sequence[float], delay: float, start: float, end: float
                     ) -> tuple[list[float], float]:
    """
    Crop the given pairs between start and end (inclusive)

    Args:
        pairs: a flat list of pairs in the form (t0, value0, t1, value1, ...)
        delay: a time offset to apply to all times
        start: start cropping at this time
        end: end cropping at this time

    Returns:
        a tuple (new pairs, new delay)

    .. code::
        pairs = (2, 20, 3, 30)
        delay = 3
        abspairs = (5, 20, 6, 30)
        t0 = 4, t1 = 5.5   -> t0 = 5, t1 = 5.5

        outpairs = (2, 20, 2.5, 25)
        outdelay = 3

        t0 = 5.5, t1 = 6
        cropPairs(pairs, 5.5-3=2.5, 6-3=3)

        outpairs = (2.5, 25, 3, 30)
        outdelay = 3

        t0 = 1, t1 = 5.5
        cropPairs(pairs, 5-3=2, 5.5-3=2.5)
        outpairs = (2, 20, 2.5, 30)
        outdelay = 3
    """
    pairst0 = pairs[0] + delay
    if start < pairst0:
        start = pairst0
    croppedPairs = cropPairs(pairs, start - delay, end - delay)
    return croppedPairs, delay


def cropPairs(pairs: Sequence[float], t0: float, t1: float) -> list[float]:
    pairsStart, pairsEnd = pairs[0], pairs[-2]

    if t0 < pairsStart and t1 >= pairsEnd:
        return aslist(pairs)

    if t0 >= pairsEnd or t1 <= pairsStart:
        return []

    def interpolate(t: float, times: Sequence[float], values: Sequence[float]
                    ) -> tuple[int, float, float]:
        idx = bisect.bisect(times, t)
        if times[idx - 1] == t:
            return idx, t, values[idx - 1]
        else:
            t0, v0 = times[idx-1], values[idx-1]
            t1, v1 = times[idx], values[idx]
            delta = (t - t0) / (t1 - t0)
            v = v0 + (v1 - v0) * delta
            return idx, t, v

    times = pairs[::2]
    values = pairs[1::2]
    out: list[float] = []
    if t0 <= times[0]:
        chunkstart = 0
    else:
        chunkstart, t, v = interpolate(t0, times, values)
        out.append(t)
        out.append(v)

    if t1 >= times[-1]:
        chunkend = len(times)
        lastbreakpoint = None
    else:
        chunkend, t, v = interpolate(t1, times, values)
        lastbreakpoint = (t, v)
    out.extend(pairs[chunkstart*2:chunkend*2])
    if lastbreakpoint is not None:
        out.extend(lastbreakpoint)
    return out


def _rewindGroup(pairs: Sequence[float], inplace=False) -> Sequence[float]:
    delay = pairs[0]
    if inplace and isinstance(pairs, list):
        for i in range(len(pairs) // 2):
            pairs[i*2] -= delay
        return pairs
    else:
        out = [val if i % 2 == 1 else val - delay
               for i, val in enumerate(pairs)]
        return out


def splitAutomation(flatpairs: Sequence[float], maxpairs: int
                    ) -> list[tuple[float, Sequence[float]]]:
    """
    Split an automation line into chunks

    Args:
        flatpairs: the automation data as a flat list of the form t0, val0, t1, val1, ...
        maxpairs: the max number of pairs per chunk

    Returns:
        a list of tuples (relativedelay, group), where relativedelay is the delay of
        the group from the start of the automation, and group is the automation data
        of this group. Each group starts with t0=0
    """
    groups = splitPairs(flatpairs=flatpairs, maxpairs=maxpairs)
    out: list[tuple[float, Sequence[float]]] = []
    for group in groups:
        groupdelay = group[0]
        group = _rewindGroup(group, inplace=True)
        assert isinstance(group, list)
        assert len(group) <= maxpairs*2, f"group size: {len(group)} = {group}"
        out.append((groupdelay, group))
    return out


def splitPairs(flatpairs: Sequence[float], maxpairs: int) -> list[Sequence[float]]:
    """
    Split automation pairs

    Args:
        flatpairs: automation data of the form time0, value0, time1, value1, ...
        maxpairs: max. number of pairs per group. The length of a group would
            be the number of pairs * 2

    Returns:
        list of pair lists
    """
    chunksize = maxpairs * 2
    lendata = len(flatpairs)
    groups = []
    start = 0
    while start < lendata - 1:
        end = min(start + chunksize, lendata)
        group = flatpairs[start:end]
        groups.append(group)
        start = end
    assert sum(len(group) for group in groups) == len(flatpairs)
    return groups


def soundfileHtml(sndfile: str,
                  withHeader=True,
                  withAudiotag=True,
                  audiotagMaxDuration=10,
                  audiotagWidth='100%',
                  audiotagMaxWidth='1200px',
                  embedThreshold=2.
                  ) -> str:
    """
    Returns an HTML representation of this Sample

    This can be used within a Jupyter notebook to force the
    html display. It is useful inside a block were it would
    not be possible to put this Sample as the last element
    of the cell to force the html representation

    Args:
        sndfile: the path to the soundfile
        withHeader: include a header line with repr text ('Sample(...)')
        withAudiotag: include html for audio playback.
        audiotagMaxDuration: max duration
        audiotagWidth: the width of the audio tag, as a css width value
        audiotagMaxWidth: the max width, as a css width value
        embedThreshold: the max duration of a sound file to be embedded. Longer files
            are saved to disk and loaded

    Returns:
        the HTML repr as str

    """
    import sndfileio
    import IPython.display
    import emlib.img
    from . import plotting
    import tempfile
    pngfile = tempfile.mktemp(suffix=".png", prefix="plot-")
    samples, info = sndfileio.sndget(sndfile)
    if info.duration < 20:
        profile = 'highest'
    elif info.duration < 40:
        profile = 'high'
    elif info.duration < 180:
        profile = 'medium'
    else:
        profile = 'low'
    plotting.plotSamples(samples, samplerate=info.samplerate, profile=profile, saveas=pngfile)
    img = emlib.img.htmlImgBase64(pngfile)
    if info.duration > 60:
        durstr = emlib.misc.sec2str(info.duration)
    else:
        durstr = f"{info.duration:.3g}"
    if withHeader:
        s = (f"<b>Soundfile</b>: '{sndfile}', duration: <code>{durstr}</code>, "
             f"sr: <code>{info.samplerate}</code>, "
             f"numchannels: <code>{info.channels}</code>)<br>")
    else:
        s = ''
    s += img
    if withAudiotag and info.duration/60 < audiotagMaxDuration:
        maxwidth = audiotagMaxWidth
        # embed short audio files, the longer ones are written to disk and read
        # from there
        if info.duration < embedThreshold:
            audioobj = IPython.display.Audio(samples.T, rate=info.samplerate)
            audiotag = audioobj._repr_html_()
        else:
            os.makedirs('tmp', exist_ok=True)
            outfile = tempfile.mktemp(suffix='.mp3')
            sndfileio.sndwrite(outfile, samples=samples, sr=info.samplerate)
            audioobj = IPython.display.Audio(outfile)
            audiotag = audioobj._repr_html_()
        audiotag = audiotag.replace('audio  controls="controls"',
                                    fr'audio controls style="width: {audiotagWidth}; max-width: {maxwidth};"')
        s += "<br>" + audiotag
    return s


safeColors = {
    'blue1': '#9090FF',
    'blue2': '#6666E0',
    'red1': '#FF9090',
    'red2': '#E08080',
    'green1': '#90FF90',
    'green2': '#8080E0',
    'magenta1': '#F090F0',
    'magenta2': '#E080E0',
    'cyan': '#70D0D0',
    'grey1': '#BBBBBB',
    'grey2': '#A0A0A0',
    'grey3': '#909090'
}


def isiterable(obj) -> bool:
    return hasattr(obj, '__iter__')


def interleave(a: Sequence[T], b: Sequence[T]) -> list[T]:
    out: list[T] = []
    for pair in zip(a, b):
        out.extend(pair)
    return out


def flattenAutomationData(pairs: Sequence[float] | tuple[Sequence[float], Sequence[float]]
                          ) -> Sequence[float]:
    if isinstance(pairs, tuple) and len(pairs) == 2 and isinstance(pairs[0], (list, tuple)):
        return interleave(*pairs)
    else:
        return pairs


@cache
def assignInstrNumbers(orc: str, startInstr: int, postInstrNum: int) -> dict[str, int]:
    """
    Given an orc with quoted instrument names, assign numbers to each instr

    Args:
        orc: the orchestra code
        startInstr: the starting instrument number
        postInstrNum: starting instrument number for 'post' instruments. Post
            instruments are those which should be placed after all other
            instruments

    Returns:
        a dict mapping instrument names to their assigned numbers
    """
    names = _extractInstrNames(orc)
    preInstrs = [name for name in names if not name.endswith('_post')]
    postInstrs = [name for name in names if name.endswith('_post')]

    instrs = {name: i for i, name in enumerate(preInstrs, start=startInstr)}
    for i, name in enumerate(postInstrs):
        instrs[name] = postInstrNum + i
    return instrs


@cache
def _extractInstrNames(s: str) -> list[str]:
    return [match.group(1) for line in s.splitlines()
            if (match := re.search(r"\binstr\s+\$\{(\w+)\}", line))]


def waitWhileTrue(func: Callable[[], bool],
                  pollinterval: float = 0.02,
                  sleepfunc=time.sleep
                  ) -> None:
    """
    Wait until this the function returns False

    Args:
        func: the function to call. It should return True if we need to keep waiting,
            False if the waiting is over
        pollinterval: polling interval in seconds
        sleepfunc: the function to call when sleeping, defaults to time.sleep

    Example
    ~~~~~~~

        >>> from csoundengine import *
        >>> from csoundengine import internalTools
        >>> session = Session()
        >>> session.defInstr('test', ...)
        >>> synth = session.sched('test', ...)
        >>> internalTools.waitWhileTrue(synth.playing)


    .. seealso:: :meth:`csoundengine.synth.Synth.wait`
    """
    setSigintHandler()
    while func():
        sleepfunc(pollinterval)
    removeSigintHandler()


def classify(objs: tuple[str, _T]) -> dict[str, _T]:
    """
    Example
    ~~~~~~~

        @dataclass
        def Person:
            name: str
            country: str

        persons = [Person("A", "Germany"),
                   Person("B", "Italy"),
                   Person("C", "Germany")]
        classify([(person.country, person) for person in persons])
        -> {'Germany': [Person(name="A", country="Germany"),
                        Person(name="C", country="Germany)],
            'Italy': [Person(name="B", country="Italy"]}

    """
    groups = {}
    for key, obj in objs:
        groups.setdefault(key, []).append(obj)
    return groups
