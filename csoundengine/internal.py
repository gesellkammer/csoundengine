from __future__ import annotations

import bisect
import math
import os
import re
import sys
import textwrap
import time
import tempfile
from functools import cache
from typing import TYPE_CHECKING

import emlib.iterlib
import emlib.numpytools
import numpy as np

from csoundengine._common import EMPTYDICT


if TYPE_CHECKING:
    from typing import Any, Callable, KeysView, Sequence, TypeVar
    import numpy.typing as npt
    from matplotlib.figure import Figure

    from csoundlib import AudioDevice, MidiDevice

    from .instr import Instr
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
        import xxhash
        return xxhash.xxh128_hexdigest(a)  # type: ignore
    else:
        return str(id(a))


def isrunning(prog: str) -> bool:
    """True if prog is running (only for linux for the moment)"""
    if sys.platform != 'linux':
        raise RuntimeError(f"This function is not supported for platform '{sys.platform}'")
    import subprocess
    failed = subprocess.call(['pgrep', '-f', prog], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return not failed


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


def _dummySigintHandler(sig, frame):
    print(frame)
    raise KeyboardInterrupt("SIGINT (CTRL-C) while waiting")


def setSigintHandler(handler=None):
    """
    Set own sigint handler to prevent CTRL-C from crashing csound

    It will do nothing if this was already set
    """
    if _registry.get('sigint_handler_set'):
        return
    import signal
    if handler is None:
        handler = _dummySigintHandler
    originalHandler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handler)
    _registry['original_sigint_handler'] = originalHandler
    _registry['sigint_handler_set'] = True


def removeSigintHandler():
    """
    Reset the sigint handler to its original state
    This will do nothing if our own handler was not set
    in the first place
    """
    if not _registry.get('sigint_handler_set'):
        return
    import signal
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
        from . import jacktools
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
                     pargs: Sequence[float] | dict[str, float | str] = (),
                     pkws: dict[str, float | str] = EMPTYDICT,
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

    if isinstance(pargs, (list, tuple)):
        allargs.extend(instr.pfieldsTranslate(pargs, pkws))
    else:
        if pkws:
            if pargs:
                pargs.update(pkws)  # type: ignore
            else:
                pargs = pkws        # type: ignore
        assert isinstance(pargs, dict)
        allargs.extend(instr.pfieldsTranslate(kws=pargs))
    return [arg if isinstance(arg, str) else float(arg) for arg in allargs]


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
                  ) -> str | None:
    for opt in prioritizedOptions:
        if opt in availableOptions:
            return opt
    return None


def selectAudioDevice(devices: list[AudioDevice], title='Select device'
                      ) -> AudioDevice | None:
    """
    Select an audio device

    Args:
        devices: the list of AudioDevices to select from. It cannot be empty
        title: the title of the dialog

    Returns:
        the AudioDevice selected, or None if there was no selection
    """
    if not devices:
        raise ValueError("No devices given")
    outnames = [dev.info() for dev in devices]
    import emlib.dialogs
    selected = emlib.dialogs.selectItem(items=outnames, title=title)
    if not selected:
        return None
    idx = outnames.index(selected)
    outdev = devices[idx]
    return outdev


def selectMidiDevice(devices: list[MidiDevice], title='Select MIDI device'
                     ) -> MidiDevice | None:
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
    import emlib.dialogs
    selected = emlib.dialogs.selectItem(items=names, title=title)
    if not selected:
        return None
    else:
        name, devid = selected[:-1].split("[")
        return next(d for d in devices if d.deviceid == devid)


def selectItem(items: list[str], title="Select") -> str | None:
    import emlib.dialogs
    return emlib.dialogs.selectItem(items=items, title=title)


def instrNameFromP1(p1: float | str) -> int | str:
    return int(p1) if isinstance(p1, (int, float)) else p1.split(".")[0]


def resolvePfieldIndex(pfield: int | str,
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


def splitAutomation(flatpairs: Sequence[float] | np.ndarray, maxpairs: int
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
    groups = splitPairs(flatpairs=flatpairs.tolist() if isinstance(flatpairs, np.ndarray) else flatpairs, maxpairs=maxpairs)
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


def hashSoundfile(path: str) -> int:
    """
    Produce a hash for a soundfile

    The hash does not actually depend on the contents, only
    metadata is used (modification time, number of frames, etc.)

    Args:
        path: the path to the soundfile

    Returns:
        a hash in the form of an integer
    """
    import sndfileio
    info = sndfileio.sndinfo(path)
    mtime = os.path.getmtime(path)
    return hash((path, mtime, info.nframes, info.encoding, info.channels, info.samplerate))


_soundfileHtmlCache = {}



def wrapBase64(base64img: str,
               width: int | str = None,
               maxwidth: int | str = None,
               margintop='14px',
               padding='10px',
               ) -> str:
    """
    Read an image and return the data as base64 within an img html tag

    Args:
        imgpath: the path to the image
        width: the width of the displayed image. Either a width
            in pixels or a str as passed to css ('800px', '100%').
        maxwidth: similar to width
        scale: if width is not given, a scale value can be used to display
            the image at a relative width

    Returns:
        the generated html
    """
    attrs = [f'padding:{padding}',
             f'margin-top:{margintop}']
    if maxwidth:
        if isinstance(maxwidth, int):
            maxwidth = f'{maxwidth}px'
        attrs.append(f'max-width: {maxwidth}')
    if width is not None:
        if isinstance(width, int):
            width = f'{width}px'
        attrs.append(f'width:{width}')
    style = ";\n".join(attrs)
    return fr'''
        <img style="display:inline; {style}"
             src="data:image/png;base64,{base64img}"/>'''


def plotSamplesAsHtml(samples: np.ndarray,
                      sr: int,
                      withHeader=True,
                      withAudiotag=True,
                      audiotagWidth='100%',
                      audiotagMaxWidth='1200px',
                      audiotagMaxDuration=0.,
                      profile='',
                      embedAudiotag=True,
                      path='',
                      figure: Figure | None = None,
                      customHeader: str = '',
                      figsize: tuple[int, int] | None = None) -> str:
    """
    Plot the given samples and return a base64 image as html

    Args:
        samples: the samples to plot
        sr: the samplerate
        withHeader: if True, add a header with info about the samples
        withAudiotag: if True, add an <audio> element to play the samples
        audiotagWidth: width of the audio tag element as css width
        audiotagMaxWidth: max. width given to the audio tag element (as css value)
        audiotagMaxDuration: if the samples exceed this duration the audio actually
            passed to the audio tag is shortened to not exceed this duration
        profile: a plotting profile, one of 'low', 'medium', 'high', 'highest'. If
            not given, a profile is chosen based on the length of the sample

    """
    import IPython.display
    from . import plotting
    dur = len(samples) / sr
    if figure is None:
        if not profile:
            if dur < 3:
                profile = 'highest'
            elif dur < 9:
                profile = 'high'
            elif dur < 45:
                profile = 'medium'
            else:
                profile = 'low'
        figure = plotting.plotSamples(samples, samplerate=sr, profile=profile, figsize=figsize)
    imgb64 = plotting.figureToBase64(figure)
    imgtag = wrapBase64(imgb64)
    parts = []
    if customHeader:
        parts.append(customHeader)
    elif withHeader:
        import emlib.misc
        durstr = durstr = emlib.misc.sec2str(dur) if dur > 60 else f"{dur:.3g}"
        nchnls = 1 if len(samples.shape) == 1 else samples.shape[1]
        parts.append(f"<b>Soundfile</b>: '{path}', duration: <code>{durstr}</code>, "
                     f"sr: <code>{sr}</code>, "
                     f"numchannels: <code>{nchnls}</code>)<br>")
    parts.append(imgtag)
    if withAudiotag:
        # embed short audio files, the longer ones are written to disk and read
        # from there
        if embedAudiotag:
            samplestransp = samples.T
            if dur > audiotagMaxDuration:
                samplestransp = samplestransp[:int(audiotagMaxDuration*sr)]
            audioobj = IPython.display.Audio(samplestransp, rate=sr)
            audiotag = audioobj._repr_html_()
        else:
            os.makedirs('tmp', exist_ok=True)
            outfile = tempfile.mktemp(suffix='.mp3')
            if dur > audiotagMaxDuration:
                samples = samples[:int(audiotagMaxDuration*sr)]
            import sndfileio
            sndfileio.sndwrite(outfile, samples=samples, sr=sr)
            audioobj = IPython.display.Audio(outfile)
            audiotag = audioobj._repr_html_()
        audiotag = audiotag.replace('audio  controls="controls"',
                                    fr'audio controls style="width: {audiotagWidth}; max-width: {audiotagMaxWidth};"')
        parts.append("<br>")
        parts.append(audiotag)
    return "\n".join(parts)


def soundfileHtml(sndfile: str,
                  withHeader=True,
                  withAudiotag=True,
                  audiotagMaxDuration=10,
                  audiotagWidth='100%',
                  audiotagMaxWidth='1200px',
                  profile='',
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
    samples, info = sndfileio.sndget(sndfile)
    dur = len(samples) / info.samplerate
    return plotSamplesAsHtml(samples=samples, sr=info.samplerate,
                             withHeader=withHeader, path=sndfile,
                             withAudiotag=withAudiotag,
                             audiotagMaxDuration=audiotagMaxDuration,
                             audiotagMaxWidth=audiotagMaxWidth,
                             audiotagWidth=audiotagWidth,
                             profile=profile,
                             embedAudiotag=dur < embedThreshold)


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


def flattenAutomationData(pairs: npt.ArrayLike | tuple[npt.ArrayLike, npt.ArrayLike]
                          ) -> list[float]:
    if isinstance(pairs, np.ndarray):
        if len(pairs.shape) == 1:
            return pairs.tolist()  # type: ignore
        elif len(pairs.shape) == 2:
            return pairs.flatten().tolist()
        else:
            raise ValueError(f"Expected a 1D or 2D array, got {pairs}")
    if isinstance(pairs, list):
        return pairs
    elif isinstance(pairs, tuple):
        if len(pairs) == 2 and not isinstance(pairs[0], (int, float)):
            xs, ys = pairs
            if isinstance(xs, (list, tuple)):
                assert isinstance(ys, (list, tuple))
                return interleave(xs, ys)  # type: ignore
            elif isinstance(xs, np.ndarray):
                assert isinstance(ys, np.ndarray)
                return emlib.numpytools.interlace(xs, ys).tolist()  # type: ignore
            else:
                raise TypeError(f"Expected a tuple (xs, ys), got {pairs}")
        else:
            return list(pairs)  # type: ignore
    else:
        raise TypeError(f"Expected a list of values or a tuple (list, list), got {pairs}")


@cache
def assignInstrNumbers(orc: str, startInstr: int, postInstr: int = 0) -> dict[str, int]:
    """
    Given an orc with quoted instrument names, assign numbers to each instr

    Args:
        orc: the orchestra code
        startInstr: the starting instrument number
        postInstr: starting instrument number for 'post' instruments. Post
            instruments are those which should be placed after all other
            instruments

    Returns:
        a dict mapping instrument names to their assigned numbers
    """
    names = _extractInstrNames(orc)
    preInstrs = [name for name in names if not name.endswith('_post')]
    postInstrs = [name for name in names if name.endswith('_post')]

    instrs = {name: i for i, name in enumerate(preInstrs, start=startInstr)}
    instrs.update({name: postInstr + i for i, name in enumerate(postInstrs)})
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
        >>> from csoundengine import internal
        >>> session = Session()
        >>> session.defInstr('test', ...)
        >>> synth = session.sched('test', ...)
        >>> internal.waitWhileTrue(synth.playing)


    .. seealso:: :meth:`csoundengine.synth.Synth.wait`
    """
    setSigintHandler()
    while func():
        sleepfunc(pollinterval)
    removeSigintHandler()


def classify(objs: Sequence[tuple[str, T]]) -> dict[str, list[T]]:
    """
    Split the given objects into groups by a given key

    Args:
        objs: the objects to classify, as a list of tuples of the form (key: str, object)

    Returns:
        a dictionary `{key: str, objects: list}` where objects is a list of objects
        with the given key

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
    groups: dict[str, list[T]] = {}
    for key, obj in objs:
        if key in groups:
            groups[key].append(obj)
        else:
            groups[key] = [obj]
    return groups


def stripTrailingEmptyLines(lines: list[str]) -> list[str]:
    """
    Remove empty lines from the top and bottom

    Args:
        lines: lines already split

    Returns:
        a list of lines without any empty lines at the beginning and at the end
    """
    startidx, endidx = 0, 0
    for startidx, line in enumerate(lines):
        if line and not line.isspace():
            break
    for endidx, line in enumerate(reversed(lines)):
        if line and not line.isspace():
            break
    return lines[startidx:len(lines)-endidx]


def splitBytes(s: bytes, maxlen: int) -> list[bytes]:
    """
    Split `s` into strings of max. size `maxlen`

    Args:
        s: the str/bytes to split
        maxlen: the max. length of each substring

    Returns:
        a list of substrings, where each substring has a max. length
        of *maxlen*
    """
    out = []
    idx = 0
    L = len(s)
    while idx < L:
        n = min(L-idx, maxlen)
        subs = s[idx:idx+n]
        out.append(subs)
        idx += n
    return out


def normalizePath(path: str) -> str:
    """
    Convert `path` to an absolute path with user expanded
    (something that can be safely passed to a subprocess)
    """
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def exponcurve(num: int, exp: float, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    """
    Generate an exponential curve between two points

    Args:
        num: number of points to generate
        exp: exponent of the curve
        x0: start x-coordinate
        x1: end x-coordinate
        y0: start y-coordinate
        y1: end y-coordinate

    Returns:
        a numpy array of shape (num,) containing the y-coordinates of the curve
    """
    xs = np.linspace(x0, x1, num)
    dxs = (xs - x0) / (x1 - x0)
    ys = (dxs ** exp) * (y1 - y0) + y0
    return ys
