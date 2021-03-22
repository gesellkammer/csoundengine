from __future__ import annotations

import math as _math
import os 
import sys
import subprocess
import re
import shutil as _shutil
import logging as _logging
import textwrap as _textwrap
import io
import tempfile
import cachetools
import dataclasses
from typing import List, Union as U, Optional as Opt, Generator, \
    Sequence as Seq, Dict, Tuple, IO, NamedTuple, Callable
from functools import lru_cache

import numpy as np

from emlib import misc


logger = _logging.getLogger("csoundengine")


Curve = Callable[[float], float]


@dataclasses.dataclass
class AudioBackend:
    """
    An AudioBackend holds information about a csound audio backend

    Attributes:
        name: the name of this backend
        alwaysAvailable: is this backend always available?
        supportsSystemSr: does this backend have a system samplerate?
        needsRealtime: the backend needs to be run in realtime
        platforms: a list of platform for which this backend is available
        dac: the name of the default output device
        adc: the name of the default input device
        longname: an alternative name for the backend
        alias: an alias for the backend
    """
    name: str
    alwaysAvailable: bool
    supportsSystemSr: bool
    needsRealtime: bool
    platforms: List[str]
    dac: str = "dac"  
    adc: str = "adc"
    longname: str = ""
    alias: str = ""

    def __post_init__(self):
        if not self.longname:
            self.longname = self.name

    def isAvailable(self) -> bool:
        """ Is this backend available? """
        if sys.platform not in self.platforms:
            return False
        return self.alwaysAvailable or isBackendAvailable(self.name)


_backend_jack = AudioBackend('jack',
                             alwaysAvailable=False,
                             supportsSystemSr=True,
                             needsRealtime=False,
                             platforms=['linux', 'darwin', 'win32'],
                             dac="dac:system:playback_",
                             adc="adc:system:capture_")

_backend_pacb = AudioBackend('pa_cb',
                             alwaysAvailable=True,
                             supportsSystemSr=False,
                             needsRealtime=False,
                             longname="portaudio-callback",
                             alias='portaudio',
                             platforms=['linux', 'darwin', 'win32'])

_backend_pabl = AudioBackend('pa_bl',
                             alwaysAvailable=True,
                             supportsSystemSr=False,
                             needsRealtime=False,
                             longname="portaudio-blocking",
                             platforms=['linux', 'darwin', 'win32'])

_backend_auhal = AudioBackend('auhal',
                              alwaysAvailable=True,
                              supportsSystemSr=True,
                              needsRealtime=False,
                              longname="coreaudio",
                              platforms=['darwin'])

_backend_pulse = AudioBackend('pulse',
                              alwaysAvailable=False,
                              supportsSystemSr=False,
                              needsRealtime=False,
                              longname="pulseaudio",
                              platforms=['linux'])

_backend_alsa = AudioBackend('alsa',
                             alwaysAvailable=True,
                             supportsSystemSr=False,
                             needsRealtime=True,
                             platforms=['linux'])

audioBackends: Dict[str, AudioBackend] = {
    'jack' : _backend_jack,
    'auhal': _backend_auhal,
    'pa_cb': _backend_pacb,
    'portaudio': _backend_pacb,
    'pa_bl': _backend_pabl,
    'pulse': _backend_pulse,
    'alsa' : _backend_alsa
}


_platformBackends: Dict[str, List[AudioBackend]] = {
    'linux': [_backend_jack, _backend_pacb, _backend_alsa, _backend_pabl, _backend_pulse],
    'darwin': [_backend_jack, _backend_auhal, _backend_pacb],
    'win32': [_backend_pacb, _backend_pabl]
}

"""
helper functions to work with csound
"""


_csoundbin = None
_OPCODES = None


# --- Exceptions ---

def nextpow2(n:int) -> int:
    """ Returns the power of 2 higher or equal than n"""
    return int(2 ** _math.ceil(_math.log(n, 2)))
    

def findCsound() -> Opt[str]:
    """
    Find the csound binary. If found, returns its path. Otherwise, returns None
    """
    global _csoundbin
    if _csoundbin:
        return _csoundbin
    csound = _shutil.which("csound")
    if csound:
        _csoundbin = csound
        return csound
    logger.error("csound is not in the path!")
    return None


def getVersion() -> Tuple[int, int, int]:
    """
    Returns the csound version as tuple (major, minor, patch) so that
    '6.03.0' is (6, 3, 0)

    Raises IOError if either csound is not present or its version 
    can't be parsed
    """
    csound = findCsound()
    if not csound:
        raise IOError("Csound not found")
    cmd = '{csound} --version'.format(csound=csound).split()
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE)
    proc.wait()
    lines = proc.stderr.readlines()
    if not lines:
        raise IOError("Could not read csounds output")
    for line in lines:
        if line.startswith(b"Csound version"):
            line = line.decode('utf8')
            matches = re.findall(r"(\d+\.\d+(\.\d+)?)", line)
            if matches:
                version = matches[0]
                if isinstance(version, tuple):
                    version = version[0]
                points = version.count(".")
                if points == 1:
                    major, minor = list(map(int, version.split(".")))
                    patch = 0
                else:
                    major, minor, patch = list(map(int, version.split(".")[:3]))
                return (major, minor, patch)
    else:
        raise IOError("Did not found a csound version")


def csoundSubproc(args, piped=True):
    """
    Calls csound with the given args in a subprocess, returns
    such subprocess.
    """
    csound = findCsound()
    if not csound:
        return
    p = subprocess.PIPE if piped else None
    callargs = [csound]
    callargs.extend(args)
    logger.debug(f"csound_subproc> args={callargs}")
    return subprocess.Popen(callargs, stderr=p, stdout=p)
    

def getDefaultBackend() -> AudioBackend:
    """
    Get the default backend for platform, discard any backend which is
    not available at the moment
    """
    return getAudioBackends()[0]


def runCsd(csdfile:str,
           outdev ="",
           indev ="",
           backend = "",
           nodisplay = False,
           comment:str = None,
           piped = False,
           extra:List[str] = None) -> subprocess.Popen:
    """
    Run the given .csd as a csound subprocess

    Args:
        csdfile: the path to a .csd file
        outdev: "dac" to output to the default device, the label of the
            device (dac0, dac1, ...), or a filename to render offline
            (-o option)
        indev: The input to use (for realtime) (-i option)
        backend: The name of the backend to use. If no backend is given,
            the default for the platform is used (this is only meaningful
            if running in realtime)
        nodisplay: if True, eliminates debugging info from output
        piped: if True, the output of the csound process is piped and can be accessed
            through the Popen object (.stdout, .stderr)
        extra: a list of extra arguments to be passed to csound
        comment: if given, will be added to the generated soundfile
            as comment metadata (when running offline)

    Returns:
        the subprocess.Popen object. In order to wait until
        rendering is finished in offline mode, call .wait on the
        returned process
    """
    args = []
    realtime = False
    if outdev:
        args.extend(["-o", outdev])
        if outdev.startswith("dac"):
            realtime = True
    if realtime and not backend:
        backend = getDefaultBackend().name
    if backend:
        args.append(f"-+rtaudio={backend}")
    if indev:
        args.append(f"-i {indev}")
    if nodisplay:
        args.extend(['-d', '-m', '0'])
    if comment and not realtime:
        args.append(f'-+id_comment="{comment}"')
    if extra:
        args.extend(extra)
    args.append(csdfile)
    return csoundSubproc(args, piped=piped)
    

def joinCsd(orc: str, sco="", options:List[str] = None) -> str:
    """
    Join an orc and a score (both as str), returns a consolidated
    csd as string

    """
    optionstr = "" if options is None else "\n".join(options)
    csd = r"""
<CsoundSynthesizer>
<CsOptions>
{optionstr}
</CsOptions>
<CsInstruments>

{orc}

</CsInstruments>
<CsScore>

{sco}

</CsScore>
</CsoundSynthesizer>
    """.format(optionstr=optionstr, orc=orc, sco=sco)
    csd = _textwrap.dedent(csd)
    return csd


@dataclasses.dataclass
class CsoundProc:
    """
    A CsoundProc wraps a running csound subprocess

    Attributes:
        proc: the running csound subprocess
        backend: the backend used
        outdev: the output device
        sr: the sample rate of the running process
        nchnls: the number of channels
        csdstr: the csd being run, as a str
    """
    proc: subprocess.Popen
    backend: str
    outdev: str
    sr: int
    nchnls: int
    csdstr: str = ""


def testCsound(dur=8., nchnls=2, backend=None, device="dac", sr:int=None, verbose=True
               ) -> CsoundProc:
    """
    Test the current csound installation for realtime output with
    the given configuration
    """
    backend = backend or getDefaultBackend().name
    sr = sr or getSamplerateForBackend(backend)
    printchan = "printk2 kchn" if verbose else ""
    orc = f"""
sr = {sr}
ksmps = 128
nchnls = {nchnls}

instr 1
    iperiod = 1
    kchn init -1
    ktrig metro 1/iperiod
    kchn = (kchn + ktrig) % nchnls 
    anoise pinker
    outch kchn+1, anoise
    {printchan}
endin
    """
    sco = f"i1 0 {dur}"
    orc = _textwrap.dedent(orc)
    logger.debug(orc)
    csd = joinCsd(orc, sco=sco)
    tmp = tempfile.mktemp(suffix=".csd")
    open(tmp, "w").write(csd)
    proc = runCsd(tmp, outdev=device, backend=backend)
    return CsoundProc(proc=proc, backend=backend, outdev=device, sr=sr,
                      nchnls=nchnls, csdstr=csd)
    

class ScoreEvent(NamedTuple):
    """
    A ScoreEvent represent an event line in the score

    Attributes:
        kind: 'i' for isntrument event, 'f' for table definition
        p1: the p1 of the event
        start: the start time of the event
        dur: the duration of the event
        args: any other args of the event (starting with p4)
    """
    kind: str
    p1: U[str, int, float]
    start: float
    dur: float
    args: List[U[float, str]]


def parseScore(sco: str) -> Generator[ScoreEvent]:
    """
    Parse a score given as string, returns a seq. of ScoreEvents
    """
    for line in sco.splitlines():
        words = line.split()
        w0 = words[0]
        if w0 in {'i', 'f'}:
            kind = w0
            name = words[1]
            if namenum := misc.asnumber(name) is not None:
                name = namenum
            t0 = float(words[2])
            dur = float(words[3])
            rest = words[4:]
        elif w0[0] in {'i', 'f'}:
            kind = w0[0]
            name = w0[1:]
            t0 = float(words[1])
            dur = float(words[2])
            rest = words[3:]
        else:
            continue
        yield ScoreEvent(kind, name, t0, dur, rest)


def opcodesList(cached=True) -> List[str]:
    """
    Return a list of the opcodes present
    """
    global _OPCODES
    if _OPCODES is not None and cached:
        return _OPCODES
    s = csoundSubproc(['-z'])
    lines = s.stderr.readlines()
    allopcodes = []
    for line in lines:
        if line.startswith(b"end of score"):
            break
        opcodes = line.decode('utf8').split()
        if opcodes:
            allopcodes.extend(opcodes)
    _OPCODES = allopcodes
    return allopcodes

   
def saveAsGen23(data: Seq[float], outfile:str, fmt="%.12f", header="") -> None:
    """
    Saves the data to a gen23 table

    .. note::
        gen23 is a 1D list of numbers in text format, sepparated by a space

    Args:
        data: A 1D sequence (list or array) of floats
        outfile: The path to save the data to. Recommended extension: '.gen23'
        fmt: If saving frequency tables, fmt can be "%.1f" and save space, for
            amplitude the default if "%.12f" is best
        header: If specified it is included as a comment as the first line
            (csound will skip it). It is there just to document what is in the table


    Example
    ~~~~~~~

    .. code-block:: python

        import bpf4
        from csoundengine import csoundlib
        a = bpf.linear(0, 0, 1, 10, 2, 300)
        dt = 0.01
        csoundlib.saveAsGen23(a[::dt].ys, "out.gen23", header=f"dt={dt}")


    In csound:

    .. code-block:: csound

        gi_tab ftgen 0, 0, 0, -23, "out.gen23"

        instr 1
          itotaldur = ftlen(gi_tab) * 0.01
          ay poscil 1, 1/itotaldur, gi_tab
        endin
    """
    if header:
        np.savetxt(outfile, data, fmt=fmt, header="# " + header)
    else:
        np.savetxt(outfile, data, fmt=fmt)


def matrixAsWav(outfile: str, mtx: np.ndarray, dt: float, t0=0.) -> None:
    """
    Save the data in the 2D matrix `mtx` as a wav file. This is not a real
    soundfile but it is used to transfer the data in binary form to be
    read in csound.

    Format::

        header: headerlength, dt, numcols, numrows, t0
        rows: each row has `numcol` number of items

    Args:
        outfile (str): The path where the data is written to (must have a .wav extension)
        mtx (numpy array): a numpy array of shape (numcols, numsamples). A 2D matrix
            representing a series of streams sampled at a regular period (dt)
        dt (float): the sampling period of the matrix
        t0 (float): the sampling offset of the matrix (t = row*dt + t0)
    """
    assert isinstance(outfile, str)
    assert isinstance(dt, float)
    assert isinstance(t0, float)
    import sndfileio
    mtx_flat = mtx.ravel()
    numrows, numcols = mtx.shape
    header = np.array([5, dt, numcols, numrows, t0], dtype=float)
    sndwriter = sndfileio.sndwrite_chunked(sr=44100, outfile=outfile, encoding="flt32")
    sndwriter.write(header)
    sndwriter.write(mtx_flat)
    sndwriter.close()


def matrixAsGen23(outfile: str, mtx: np.ndarray, dt:float, t0=0, include_header=True
                  ) -> None:
    """
    Save a numpy 2D array as gen23

    Args:
        outfile (str): the path to save the data to. Suggestion: use '.gen23' as ext
        mtx (np.ndarray): a 2D array of floats
        dt (float): the sampling period
        t0 (float): start of sampling time
        include_header (bool): if True, the saved data will be preluded by a header
            of the form ``[header size, dt, numcols, numrows, t0]``

    .. note::
        The format used by gen23 is a text format with numbers separated by any space.
        When read inside csound the table is of course 1D but can be interpreted as
        2D with the provided header metadata.
    """
    numrows, numcols = mtx.shape
    mtx = mtx.round(6)
    with open(outfile, "w") as f:
        if include_header:
            header = np.array([5, dt, numcols, numrows, t0], dtype=float)
            f.write(" ".join(header.astype(str)))
            f.write("\n")
        for row in mtx:
            rowstr = " ".join(row.astype(str))
            f.write(rowstr)
            f.write("\n")


@dataclasses.dataclass
class AudioDevice:
    """
    An AudioDevice holds information about a an audio device for a given backend

    Attributes:
        index: the index of this audio device, as passed to adcXX or dacXX
        id: the device identification (dac3, adc2, etc)
        name: the name of the device
        kind: 'output' / 'input'
        ins: the number of inputs
        outs: the number of outputs
    """
    index: int
    id: str
    name: str
    kind: str
    # api: str = ""
    ins: int = -1
    outs: int = -1


@cachetools.cached(cache=cachetools.TTLCache(1, 10))
def getAudioDevices(backend:str=None) -> Tuple[List[AudioDevice], List[AudioDevice]]:
    """
    Returns (indevices, outdevices), where each of these lists is an AudioDevice.

    Args:
        backend: specify a backend supported by your installation of csound.
            None to use a default for you OS

    Returns:
        a tuple of (input devices, output devices)

    .. note::

        For jack an audio device is a client

        Each returned device is an AudioDevice instance with attributes:

        index: The device index
        label: adc{index} for input devices, dac{index} for output devices.
            The label can be passed to csound directly with either the -i or the -o flag
            (``-i{label}`` or ``-o{label}``)
        name: A description of the device
        kind: 'input' for an input device, 'output' for an output device.
        ins: number of input channels
        outs: number of output channels


    Backends::

                OSX  Linux  Win   Multiple-Devices    Description
        jack     x      x    -     -                  Jack
        auhal    x      -    -     x                  CoreAudio
        pa_cb    x      x    x     x                  PortAudio (Callback)
        pa_bl    x      x    x     x                  PortAudio (blocking)

    """
    if not backend:
        backend = getDefaultBackend().name
    if backend == "jack" and not isJackRunning():
        raise RuntimeError("jack is not running")
    if backend == "pulse":
        if not isPulseaudioRunning():
            raise RuntimeError("pulseaudio is not running")
        return ([AudioDevice(0, id="adc", name="adc", kind="input", ins=2, outs=2)],
                [AudioDevice(0, id="dac", name="dac", kind="output", ins=2, outs=2)])

    indevices, outdevices = [], []
    proc = csoundSubproc(['-+rtaudio=%s'%backend, '--devices'])
    proc.wait()
    lines = proc.stderr.readlines()
    # regex_all = r"([0-9]+):\s(adc[0-9]+|dac[0-9]+)\s\((.+)\)"
    # regex_all = r"([0-9]+):\s((?:adc|dac)\d+)\s*\((.*)\)"
    if backend in {'pa_cb', 'pa_bl', 'portaudio'}:
        regex_all = r"([0-9]+):\s((?:adc|dac)\d+)\s*\((.*)\)"
    elif backend in {'alsa'}:
        regex_all = r"([0-9]+):\s((?:adc|dac):.*)\((.*)\)"
    else:
        raise RuntimeError(f"Operation not available for backend {backend}")
    for line in lines:
        line = line.decode("ascii")
        match = re.search(regex_all, line)
        if not match:
            continue
        idxstr, devid, devnameraw = match.groups()
        isinput = devid.startswith("adc")
        if backend in {'portaudio', 'pa_cb', 'pa_bl'}:
            name, api, inch, outch = _parsePortaudioDeviceName(devnameraw)
        else:
            name = devnameraw
            api = ''
            inch = -1
            outch = -1
        dev = AudioDevice(int(idxstr), id=devid, name=name,
                          kind="input" if isinput else "output",
                          ins=inch, outs=outch)
        if isinput:
            indevices.append(dev)
        else:
            outdevices.append(dev)
    return indevices, outdevices


def getSamplerateForBackend(backend: U[str, AudioBackend] = None) -> float:
    """
    Returns the samplerate reported by the given backend, or
    0 if failed
    """
    audiobackend = getAudioBackend(backend)
    FAILED = 0

    if not audiobackend.isAvailable():
        raise AudioBackendNotAvailable

    if not audiobackend.supportsSystemSr:
        return 44100

    if audiobackend.name == 'jack' and _shutil.which('jack_samplerate') is not None:
        sr = int(subprocess.getoutput("jack_samplerate"))
        return sr

    proc = csoundSubproc(f"-odac -+rtaudio={backend} --get-system-sr".split())
    proc.wait()
    srlines = [line for line in proc.stdout.readlines() 
               if line.startswith(b"system sr:")]
    if not srlines:
        logger.error(f"get_sr: Failed to get sr with backend {backend.name}")
        return FAILED
    sr = float(srlines[0].split(b":")[1].strip())
    logger.debug(f"get_sr: sample rate query output: {srlines}")
    return sr if sr > 0 else FAILED


@cachetools.cached(cache=cachetools.TTLCache(1, 30))
def isJackRunning() -> bool:
    """ Return True if jack is available and running"""
    return _csoundTestJackRunning()


@cachetools.cached(cache=cachetools.TTLCache(1, 30))
def isPulseaudioRunning() -> bool:
    """ Return True if Pulseaudio is running """
    retcode = subprocess.call(["pulseaudio", "--check"])
    return retcode == 0


def _csoundTestJackRunning():
    proc = csoundSubproc(['-+rtaudio=jack', '-odac', '--get-system-sr'])
    proc.wait()
    return b'could not connect to JACK server' not in proc.stderr.read()


@cachetools.cached(cache=cachetools.TTLCache(1, 30))
def isBackendAvailable(backend: str) -> bool:
    """ Returns True if the given audio backend is available """
    if backend == 'jack':
        out = isJackRunning()
    elif backend == 'pulse':
        out = isPulseaudioRunning()
    else:
        indevices, outdevices = getAudioDevices(backend=backend)
        out = bool(indevices or outdevices)
    return out


@cachetools.cached(cache=cachetools.TTLCache(1, 30))
def getAudioBackends() -> List[AudioBackend]:
    """
    Return a list of supported audio backends as they would be passed
    to -+rtaudio

    Only those backends currently available are returned
    (for example, jack will not be returned in linux if the
    jack server is not running)

    """
    backends = _platformBackends[sys.platform]
    backends = [backend for backend in backends if backend.isAvailable()]
    return backends


def getAudioBackend(name:str=None) -> Opt[AudioBackend]:
    """ Given the name of the backend, return the AudioBackend structure """
    if name is None:
        return getDefaultBackend()
    return audioBackends.get(name)


def getAudioBackendNames() -> List[str]:
    """
    Similar to get_audiobackends, but returns the names of the
    backends. The AudioBackend class can be retrieved via

    audio_backends[backend_name]

    Returns:
        a list with the names of all available backends for the
        current platform
    """
    backends = getAudioBackends()
    return [backend.name for backend in backends]


def _quoteIfNeeded(arg:U[float, int, str]) -> U[float, int, str]:
    return arg if not isinstance(arg, str) else f'"{arg}"'


def _eventStartTime(event:tuple) -> float:
    kind = event[0]
    if kind == "e":           # end
        return event[1]
    elif kind == "C":         # carry
        return 0.
    else:
        assert len(event) >= 3
        return event[2]

_normalizer = misc.make_replacer({".":"_", ":":"_", " ":"_"})


def normalizeInstrumentName(name:str) -> str:
    """
    Transform name so that it can be accepted as an instrument name
    """
    return _normalizer(name)


_fmtoptions = {
    'pcm16'    : '',
    'pcm24'    : '--format=24bit',
    'float32'  : '--format=float',  # also -f
    'float64'  : '--format=double'
}


_csoundFormatOptions = {'-3', '-f', '--format=24bit', '--format=float',
                          '--format=double', '--format=long', '--format=vorbis',
                          '--format=short'}


def csoundOptionForSampleFormat(fmt:str) -> str:
    """
    Returns the command-line option for the given sample format.

    Given a sample format of the form pcmXX or floatXX, where
    XX is the bit-rate, returns the corresponding command-line option
    for csound

    Args:
        fmt (str): the desired sample format

    Returns:
        the command line option corresponding to the given format

    Example
    =======

        >>> csoundOptionForSampleFormat("pcm24")
        --format=24bit
        >>> csoundOptionForSampleFormat("float64")
        --format=double

    """
    return _fmtoptions.get(fmt)


class Csd:
    """
    A Csd object can be used to build a csound script by adding
    global code, instruments, score events, etc.

    Args:
        sr: the sample rate of the generated audio
        ksmps: the samples per cycle to use
        nchnls: the number of output channels
        a4: the reference frequency
        optiosn (list[str]): any number of extra options passed to csound
        nodisplay: if True, avoid outputting debug information
        carry: should carry be enabled in the score?

    Example
    =======

    .. code::

        >>> from csoundengine import csoundlib
        >>> csd = Csd(ksmps=32, nchnls=4)
        >>> csd.addInstr('sine', r'''
        ...   ifreq = p4
        ...   outch 1, oscili:a(0.1, ifreq)
        ... ''')
        >>> sound1 = csd.addSndfile("sounds/sound1.wav")
        >>> csd.playTable(tabnum)
        >>> csd.addEvent('sine', 0, 2, [1000])
        >>> csd.writeCsd('out.csd')
    """

    _builtinInstrs = {
        '_playgen1': r'''
        pset 0, 0, 0, 0,    1,     1,      1,    0.05, 0,         0
        kgain = p4
        kspeed = p5
        ksampsplayed = 0
        itabnum, ichan, ifade, ioffset passign 6
        inumsamples = nsamp(itabnum)
        itabsr = ftsr(itabnum)
        istartframe = ioffset * itabsr
        ksampsplayed += ksmps * kspeed
        aouts[] loscilx kgain, kspeed, itabnum, 4, 1, istartframe
        aenv = linsegr:a(0, ifade, 1, ifade, 0)
        aouts = aouts * aenv
        inumouts = lenarray(aouts)
        kchan = 0
        while kchan < inumouts do
            outch kchan+ichan, aouts[kchan]
            kchan += 1
        od
        if ksampsplayed >= inumsamples then
            turnoff
        endif
        '''
    }

    def __init__(self, sr=44100, ksmps=64, nchnls=2, a4=442., options:List[str]=None,
                 nodisplay=False, carry=False):
        self._strLastIndex = 20
        self._str2index: Dict[str, int] = {}
        self.score: List[U[list, tuple]] = []
        self.instrs: Dict[U[str, int], str] = {}
        self.globalcodes: List[str] = []
        self.options: List[str] = []
        if options:
            self.setOptions(*options)
        self.sr = sr
        self.ksmps = ksmps
        self.nchnls = nchnls
        self.a4 = a4
        self._sampleFormat: Opt[str] = None
        self._definedTables = set()
        self._minTableIndex = 1
        self.nodisplay = nodisplay
        self.enableCarry = carry
        if not carry:
            self.score.append(("C", 0))

    def addEvent(self,
                 instr: U[int, float, str],
                 start: float,
                 dur: float,
                 args: List[float] = None) -> None:
        """
        Add an instrument ("i") event to the score

        Args:
            instr: the instr number or name, as passed to addInstr
            start: the start time
            dur: the duration of the event
            args: pargs beginning at p4
        """
        start = round(start, 8)
        dur = round(dur, 8)
        event = ["i", _quoteIfNeeded(instr), start, dur]
        if args:
            event.extend(_quoteIfNeeded(arg) for arg in args)
        self.score.append(event)

    def strset(self, s:str) -> int:
        """
        Add a strset to this csd
        """
        idx = self._str2index.get(s)
        if idx is not None:
            return idx
        idx = self._strLastIndex
        self._strLastIndex += 1
        self._str2index[s] = idx
        return idx

    def _assignTableIndex(self, tabnum=0) -> int:
        defined_ftables = self._definedTables
        if tabnum > 0:
            if tabnum in defined_ftables:
                raise ValueError(f"ftable {tabnum} already defined")
        else:
            for tabnum in range(self._minTableIndex, 9999):
                if tabnum not in defined_ftables:
                    break
            else:
                raise IndexError("All possible ftable slots used!")
        defined_ftables.add(tabnum)
        return tabnum


    def _addTable(self, pargs) -> int:
        """
        Adds an ftable to the score

        Args:
            pargs: as passed to csound (without the "f")
                p1 can be 0, in which case a table number
                is assigned

        Returns:
            The index of the new ftable
        """
        tabnum = self._assignTableIndex(pargs[0])
        pargs = ["f", tabnum] + pargs[1:]
        self.score.append(pargs)
        return tabnum

    def addTableFromSeq(self, seq: Seq[float], tabnum:int=0, start=0
                        ) -> int:
        """
        Create a ftable, fill it with seq, return the ftable index

        Args:
            seq: a sequence of floats to fill the table. The size of the
                table is determined by the size of the seq.
            tabnum: 0 to auto-assign an index
            start: the same as f 1 2 3

        Returns:
            the table number

        .. note::

            The length of the data should not excede 2000 items. If the seq is longer,
            it is advised to save the seq. as either a gen23 or a wav file and load
            the table from an external file.

        """
        if len(seq) > 2000:
            raise ValueError("tables longer than 2000 items are currently not supported")
        if start > 0:
            start = round(start, 8)
        pargs = [tabnum, start, -len(seq), -2]
        pargs.extend(seq)

        return self._addTable(pargs)

    def addEmptyTable(self, size:int, tabnum: int=0) -> int:
        """

        Args:
            tabnum: use 0 to autoassign an index
            size: the size of the empty table

        Returns:
            The index of the created table
        """
        pargs = (tabnum, 0, -size, -2, 0)
        return self._addTable(pargs)

    def addSndfile(self, sndfile:str, tabnum=0, start=0.) -> int:
        """ Add a table which will load this sndfile

        Args:
            sndfile: the soundfile to load
            tabnum: fix the table number or use 0 to generate a unique table number
            start: when to load this soundfile (normally this should be left 0)

        Returns:
            the table number
            """
        tabnum = self._assignTableIndex(tabnum)
        pargs = [tabnum, start, 0, -1, sndfile, 0, 0, 0]
        self._addTable(pargs)
        return tabnum

    def destroyTable(self, tabnum:int, time:float) -> None:
        """
        Schedule ftable with index `tabnum` to be destroyed
        at time `time`

        Args:
            tabnum: the index of the table to be destroyed
            time: the time to destroy it
        """
        pargs = ("f", -tabnum, time)
        self.score.append(pargs)

    def setEndMarker(self, dur: float) -> None:
        """
        Add an end marker to the score
        """
        self.score.append(("e", dur))

    def setComment(self, comment:str) -> None:
        """ Add a comment to the renderer soundfile """
        self.setOptions(f'-+id_comment="{comment}"')

    def setSampleFormat(self, fmt: str) -> None:
        """
        Set the sample format for recording

        Args:
            fmt: one of 'pcm16', 'pcm24', 'pcm32', 'float32', 'float64'

        """
        option = csoundOptionForSampleFormat(fmt)
        if option is None:
            fmts = ", ".join(_fmtoptions.keys())
            raise KeyError(f"fmt unknown, should be one of {fmts}")
        if option:
            self.setOptions(option)
            self._sampleFormat = option

    def writeScore(self, stream:IO) -> None:
        """
        Write the score to `stream`

        Args:
            stream (file-like): the open stream to write to

        """
        self.score.sort(key=_eventStartTime)
        for event in self.score:
            line = " ".join(str(arg) for arg in event)
            stream.write(line)
            stream.write("\n")
            
    def addInstr(self, instr: U[int, str], instrstr: str) -> None:
        """ Add an instrument definition to this csd """
        self.instrs[instr] = instrstr

    def addGlobalCode(self, code: str) -> None:
        """ Add code to the instr 0 """
        self.globalcodes.append(code)

    def setOptions(self, *options: str) -> None:
        """ Adds options to this csd """
        for opt in options:
            if opt in _csoundFormatOptions:
                self._sampleFormat = opt
            self.options.append(opt)

    def dump(self) -> str:
        """ Returns a string with the .csd """
        stream = io.StringIO()
        self.writeCsd(stream)
        return stream.getvalue()

    def playTable(self, tabnum:int, start:float, dur:float=-1,
                  gain=1., speed=1., chan=1, fade=0.05,
                  skip=0.) -> None:
        """ Add an event to play the given table

        Example
        =======

        >>> csd = Csd()
        >>> tabnum = csd.addSndfile("stereo.wav")
        >>> csd.playTable(tabnum, tabnum, start=1, fade=0.1, speed=0.5)
        >>> csd.writeCsd("out.csd")
        """
        if self.instrs.get('_playgen1') is None:
            self.addInstr('_playgen1', self._builtinInstrs['_playgen1'])
        args = [gain, speed, tabnum, chan, fade, skip]
        self.addEvent('_playgen1', start=start, dur=dur, args=args)

    def writeCsd(self, stream:U[str, IO]) -> None:
        """
        Args:
            stream: the stream to write to. Either an open file, a io.StringIO
                stream or a path
        """
        if isinstance(stream, str):
            outfile = stream
            stream = open(outfile, "w")
        write = stream.write
        write("<CsoundSynthesizer>\n")

        write("\n<CsOptions>\n")
        options = self.options.copy()
        if self.nodisplay:
            options.append("-m0")

        if self._sampleFormat:
            options.append(csoundOptionForSampleFormat(self._sampleFormat))

        for option in self.options:
            write(option)
            write("\n")
        write("</CsOptions>\n")

        srstr = f"sr     = {self.sr}" if self.sr is not None else ""
        
        txt = f"""
            <CsInstruments>

            {srstr}
            ksmps  = {self.ksmps}
            0dbfs  = 1
            nchnls = {self.nchnls}
            A4     = {self.a4}

            """
        txt = _textwrap.dedent(txt)
        write(txt)
        tab = "  "

        if self._str2index:
            for s, idx in self._str2index.items():
                write(f'strset {idx}, "{s}"\n')
            write("\n")

        if self.globalcodes:
            write("; ----- global code\n")
            for globalcode in self.globalcodes:
                write(globalcode)
                write("\n")
            write("; ----- end global code\n\n")

        for instr, instrcode in self.instrs.items():
            write(f"instr {instr}\n")
            body = _textwrap.dedent(instrcode)
            body = _textwrap.indent(body, tab)
            write(body)
            write("\nendin\n")
        
        write("\n</CsInstruments>\n")
        write("\n<CsScore>\n\n")
        
        self.writeScore(stream)
        
        write("\n</CsScore>\n")
        write("</CsoundSynthesizer")

    def run(self,
            output:str,
            inputdev:str=None,
            backend: str = None,
            supressdisplay=False,
            piped=False,
            extra: List[str] = None) -> subprocess.Popen:
        """
        Run this csd. 
        
        Args:
            output: the file to use as output. This will be passed
                as the -o argument to csound.
            inputdev: the input device to use when running in realtime
            backend: the backend to use
            supressdisplay: if True, debugging information is supressed
            piped: if True, stdout and stderr are piped through
                the Popen object, accessible through .stdout and .stderr
                streams
            extra: any extra args passed to the csound binary

        Returns:
            the subprocess.Popen object

        """
        if self._sampleFormat is None and not output.startswith('dac'):
            self.setSampleFormat(bestSampleFormatForExtension(os.path.splitext(output)[1]))

        tmp = tempfile.mktemp(suffix=".csd")
        with open(tmp, "w") as f:
            self.writeCsd(f)
        logger.debug(f"Csd.run :: tempfile = {tmp}")
        return runCsd(tmp, outdev=output, indev=inputdev,
                      backend=backend, nodisplay=supressdisplay,
                      piped=piped, extra=extra)


def mincer(sndfile:str, outfile:str,
           timecurve:U[Curve, float], pitchcurve:[Curve, float],
           dt=0.002, lock=False, fftsize=2048, ksmps=128, debug=False
           ) -> None:
    """
    Stretch/Pitchshift a soundfile using csound's mincer opcode

    Args:
        sndfile: the path to a soundfile
        timecurve: a func mapping time to playback time or a scalar indicating
            a timeratio (2 means twice as fast, 1 to leave unmodified)
        pitchcurve: a func time to pitchscale, or a scalar indicating a freqratio
        outfile: the path to a resulting outfile. The resulting file is always a
            32-bit float .wav file
        dt: the sampling period to sample the curves
        lock: should mincer be run with phase-locking?
        fftsize: the size of the fft
        ksmps: the ksmps to pass to the csound process
        debug: run csound with debug information

    .. note::

        If the mapped time excedes the bounds of the sndfile, silence is generated.
        For example, a negative time or a time exceding the duration of the sndfile

    .. note::

        The samplerate and number of channels of of the generated file matches
        that of the input file

    Examples
    ========

        # Example 1: stretch a soundfile 2x

        >>> from csoundengine import csoundlib
        >>> import bpf4
        >>> import sndfileio
        >>> snddur = sndfileio.sndinfo("mono.wav").duration
        >>> timecurve = bpf4.linear(0, 0, snddur*2, snddur)
        >>> mincer(sndfile, "mono2.wav", timecurve=timecurve, pitchcurve=1)
    """
    import bpf4 as bpf
    import sndfileio

    info = sndfileio.sndinfo(sndfile)
    sr = info.samplerate
    nchnls = info.channels
    pitchbpf = bpf.asbpf(pitchcurve)
    
    if isinstance(timecurve, (int, float)):
        t0, t1 = 0, info.duration / timecurve
        timebpf = bpf.linear(0, 0, t1, info.duration)
    elif isinstance(timecurve, bpf.core.BpfInterface):
        t0, t1 = timecurve.bounds()
        timebpf = timecurve
    else:
        raise TypeError("timecurve should be either a scalar or a bpf")
    
    assert isinstance(pitchcurve, (int, float, bpf.core.BpfInterface))
    ts = np.arange(t0, t1+dt, dt)
    fmt = "%.12f"
    _, time_gen23 = tempfile.mkstemp(prefix='time-', suffix='.gen23')
    np.savetxt(time_gen23, timebpf.map(ts), fmt=fmt, header=str(dt), comments="")
    _, pitch_gen23 = tempfile.mkstemp(prefix='pitch-', suffix='.gen23')
    np.savetxt(pitch_gen23, pitchbpf.map(ts), fmt=fmt, header=str(dt), comments="")
    csd = f"""
    <CsoundSynthesizer>
    <CsOptions>
    -o {outfile}
    </CsOptions>
    <CsInstruments>

    sr = {sr}
    ksmps = {ksmps}
    nchnls = {nchnls}
    0dbfs = 1.0

    gi_snd   ftgen 0, 0, 0, -1,  "{sndfile}", 0, 0, 0
    gi_time  ftgen 0, 0, 0, -23, "{time_gen23}"
    gi_pitch ftgen 0, 0, 0, -23, "{pitch_gen23}"

    instr vartimepitch
        idt tab_i 0, gi_time
        ilock = {int(lock)}
        ifftsize = {fftsize}
        ikperiod = ksmps/sr
        isndfiledur = ftlen(gi_snd) / ftsr(gi_snd)
        isndchnls = ftchnls(gi_snd)
        ifade = ikperiod*2
        inumsamps = ftlen(gi_time)
        it1 = (inumsamps-2) * idt           ; account for idt and last value
        kt timeinsts
        aidx    linseg 1, it1, inumsamps-1
        at1     tablei aidx, gi_time, 0, 0, 0
        kpitch  tablei k(aidx), gi_pitch, 0, 0, 0
        kat1 = k(at1)
        kgate = (kat1 >= 0 && kat1 <= isndfiledur) ? 1 : 0
        agate = interp(kgate) 
        aenv linseg 0, ifade, 1, it1 - (ifade*2), 1, ifade, 0
        aenv *= agate
        if isndchnls == 1 then
            a0  mincer at1, 1, kpitch, gi_snd, ilock, ifftsize, 8
            outch 1, a0*aenv
        else
            a0, a1   mincer at1, 1, kpitch, gi_snd, ilock, ifftsize, 8
            outs a0*aenv, a1*aenv
        endif
        
      if (kt >= it1 + ikperiod) then
        event "i", "exit", 0.1, 1
            turnoff     
        endif
    endin

    instr exit
        puts "exiting!", 1
        exitnow
    endin

    </CsInstruments>
    <CsScore>
    i "vartimepitch" 0 -1
    f 0 36000

    </CsScore>
    </CsoundSynthesizer>
    """
    _, csdfile = tempfile.mkstemp(suffix=".csd")
    with open(csdfile, "w") as f:
        f.write(csd)
    subprocess.call(["csound", "-f", csdfile])
    if not debug:
        os.remove(time_gen23)
        os.remove(pitch_gen23)
        os.remove(csdfile)
    return {'outfile': outfile, 'csdstr': csd, 'csd': csdfile}


def _instr_as_orc(instrid, body, initstr, sr, ksmps, nchnls):
    orc = """
sr = {sr}
ksmps = {ksmps}
nchnls = {nchnls}
0dbfs = 1

{initstr}

instr {instrid}
    {body}
endin

    """.format(sr=sr, ksmps=ksmps, instrid=instrid, body=body, nchnls=nchnls, initstr=initstr)
    return orc


def recInstr(body:str, events:list, init="", outfile:str=None,
             sr=44100, ksmps=64, nchnls=2, a4=442, samplefmt='float',
             dur=None
             ) -> Tuple[str, subprocess.Popen]:
    """
    Record one instrument for a given duration

    Args:
        dur: the duration of the recording
        body: the body of the instrument
        init: the initialization code (ftgens, global vars, etc)
        outfile: the generated soundfile, or None to generate a temporary file
        events: a list of events, where each event is a list of pargs passed
            to the instrument, beginning with p2: delay, dur, [p4, p5, ...]
        sr: the samplerate
        a4: A4 frequency
        ksmps: block size
        nchnls: number of output channels
        samplefmt: defines the sample format used for outfile, one of (16, 24, 32, 'float')

    Returns:
        a tuple (outfile to be generated, subprocess.Popen running csound)
    """
    if not isinstance(events, list) or not all(isinstance(event, (tuple, list)) for event in events):
        raise ValueError("events is a seq., where each item is a seq. of pargs passed to"
                         "the instrument, beginning with p2: [delay, dur, ...]"
                         f"Got {events} instead")

    csd = Csd(sr=sr, ksmps=ksmps, nchnls=nchnls, a4=a4)
    if not outfile:
        outfile = tempfile.mktemp(suffix='.wav', prefix='csdengine-rec-')

    if init:
        csd.addGlobalCode(init)
    
    instrnum = 100
    
    csd.addInstr(instrnum, body)
    for event in events:
        start, dur = event[0], event[1]
        csd.addEvent(instrnum, start, dur, event[2:])
    
    if dur is not None:
        csd.setEndMarker(dur)

    fmtoption = {16: '', 24: '-3', 32: '-f', 'float': '-f'}.get(samplefmt)
    if fmtoption is None:
        raise ValueError("samplefmt should be one of 16, 24, 32, or 'float'")
    csd.setOptions(fmtoption)

    proc = csd.run(output=outfile)
    return outfile, proc


def _ftsaveReadText(path):
    # a file can have multiple tables saved
    lines = iter(open(path))
    tables = []
    while True:
        tablength = -1
        try:
            headerstart = next(lines)
            if not headerstart.startswith("===="):
                raise IOError(f"Expecting header start, got {headerstart}")
        except StopIteration:
            # no more tables
            break
        # Read header
        for line in lines:
            if line.startswith("flen:"):
                tablength = int(line[5:])
            if 'END OF HEADER' in line:
                break
        if tablength < 0:
            raise IOError("Could not read table length")
        values = np.zeros((tablength+1,), dtype=float)
        # Read data
        for i, line in enumerate(lines):
            if line.startswith("---"):
                break
            values[i] = float(line)
        tables.append(values)
    return tables
 

def ftsaveRead(path, mode="text"):
    """
    Read a file saved by ftsave, returns a list of tables
    """
    if mode == "text":
        return _ftsaveReadText(path)
    else:
        raise ValueError(f"mode {mode} not supported")


def getNchnls(backend: str=None, device:str=None, indevice:str=None,
              defaultin:int=2, defaultout:int=2
              ) -> Tuple[int, int]:
    """
    Get the default number of channels for a given device

    Args:
        backend: the backend, one of 'jack', 'portaudio', etc. None to use default
        device: the output device. Use None for default device. Otherwise either the
            device id ("dac0") or a regex pattern matching the long name of the device
        indevice: the input device. Use None for default device. Otherwise either the
            device id ("dac0") or a regex pattern matching the long name of the device
        defaultin: default value returned if it is not possible to determine
            the number of channels for given backend+device
        defaultout: default value returned if it is not possible to determine
            the number of channels for given backend/device

    Returns:
        a tuple (nchnls_i, nchnls) for that backend+device combination

    """
    assert device != "adc" and indevice != "dac"
    if backend is None:
        backend = getDefaultBackend().name
    if indevice is None:
        indevice = device

    if backend == 'jack':
        if device and device.startswith("dac:"):
            device = device[4:]
        if indevice and indevice.startswith("adc:"):
            indevice = indevice[4:]
        return _getNchnlsJackViaJackclient(indevice=indevice, outdevice=device)
    elif backend == 'portaudio' or backend == 'pa_cb' or backend == 'pa_bl':
        return _getNchnlsPortaudio(indevice=indevice, outdevice=device)
    elif backend == "pulse" or backend == "pulseaudio":
        return 2, 2
    else:
        return defaultin, defaultout

def defaultDevicesForBackend(backendname:str=None) -> Tuple[str, str]:
    """
    Get the default input and output devices for the backend

    Args:
        backendname (str): the name of the backend to use

    Returns:
        A tuple (adc, dac) as passed to -i and -o respectively

    Example::

        >>> defaultDevicesForBackend("jack")
        ('adc:system:capture_.*', 'dac:system:playback_.*')

    """
    if backendname is None:
        backend = getDefaultBackend()
    else:
        backend = getAudioBackend(backendname)
    if not backend:
        backends = getAudioBackendNames()
        raise ValueError(f"Backend {backendname} not known. Possible backends: {backends}")
    return (backend.adc, backend.dac)


def _getNchnlsJack(indevice:str, outdevice:str) -> Tuple[int, int]:
    """
    
    Args:
        indevice (str): regex pattern for input client 
        outdevice (str): regex pattern for output client

    Returns:
        a tuple (number of inputs, number of outputs)
    """
    if indevice is None:
        indevice = outdevice

    if indevice == "adc" or indevice is None:
        indevice = "system:capture_.*"
    if outdevice == "dac" or outdevice is None:
        outdevice = "system:playback_.*"
    if not isJackRunning():
        return 0, 0
    indevs, outdevs = getAudioDevices('jack')
    outports = []
    inports = []
    for outdev in outdevs:
        if re.search(outdevice, outdev.name):
            outports.append(outdev)
    for indev in indevs:
        if re.search(indevice, indev.name):
            inports.append(indev)
    return len(inports), len(outports)


def _getNchnlsJackViaJackclient(indevice:str, outdevice:str
                                ) -> Tuple[int, int]:
    """
    Get the number of ports for the given clients using JACK-Client
    This is faster than csound and should result in the same results

    Args:
        indevice (str): A regex pattern matching the input client
        outdevice (str): A regex pattern matching the output client

    Returns:
        a tuple (number of inputs, number of outputs)
    """
    import jack
    c = jack.Client("query")
    if indevice == "adc" or indevice is None:
        indevice = "system:capture_.*"
    if outdevice == "dac" or outdevice is None:
        outdevice = "system:playback_.*"
    # NB: our output ports are other clients input ports, so the
    # query is reversed
    outports = c.get_ports(outdevice, is_audio=True, is_input=True)
    inports = c.get_ports(indevice, is_audio=True, is_output=True)
    return len(inports), len(outports)


def _getNchnlsPortaudioSounddevice(indevice:str, outdevice:str
                                   ) -> Tuple[int, int]:
    import sounddevice
    devlist = sounddevice.query_devices()
    if indevice is None or indevice == "adc":
        indevice = r"\bdefault\b"
    if outdevice is None or outdevice == "dac":
        outdevice = r"\bdefault\b"
    max_input_channels, max_output_channels = 0, 0
    for dev in devlist:
        if max_input_channels == 0 and re.search(indevice, dev['name']):
            max_input_channels = dev['max_input_channels']
        if max_output_channels == 0 and re.search(outdevice, dev['name']):
            max_output_channels = dev['max_output_channels']
        if max_input_channels and max_output_channels:
            break
    return max_input_channels, max_output_channels


def _parsePortaudioDeviceName(name:str) -> Tuple[str, str, int, int]:
    """
    Parses a string like "HDA Intel PCH: CX20590 Analog (hw:0,0) [ALSA, 2 in, 4 out]"

    Returns a tuple: ("HDA Intel PCH: CX20590 Analog (hw:0,0)", "ALSA", 2, 4)
    """
    devname, rest = name.split("[")
    rest = rest.replace("]", "")
    if "," in rest:
        api, inchstr, outchstr = rest.split(",")
        inch = int(inchstr.split()[0])
        outch = int(outchstr.split()[0])
    else:
        api = rest
        inch = -1
        outch = -1
    return devname.strip(), api, inch, outch


def _getNchnlsPortaudio(indevice:str, outdevice:str
                        ) -> Tuple[int, int]:
    indevs, outdevs = getAudioDevices(backend="portaudio")
    assert indevice != "dac" and outdevice != "adc"
    if indevice is None or indevice == "adc":
        indevice = r"\bdefault\b"
    if outdevice is None or outdevice == "dac":
        outdevice = r"\bdefault\b"

    for indev in indevs:
        if indevice == indev.id or re.search(indevice, indev.name):
            max_input_channels = indev.ins
            break
    else:
        max_input_channels = -1
    for outdev in outdevs:
        if outdevice == outdev.id or re.search(outdevice, outdev.name):
            max_output_channels = outdev.outs
            break
    else:
        max_output_channels = -1
    if max_input_channels == -1 and max_output_channels == -1:
        dumpDevices()
        raise ValueError(f"Device {indevice} / {outdevice} not found")
    return max_input_channels, max_output_channels


def dumpDevices(backend:str=None):
    """
    Print a list of audio devices for the given backend. If backend is not
    given, the default backend (of all available backends for the current
    platform) is chosen
    """
    indevs, outdevs = getAudioDevices(backend=backend)
    print("Input Devices:")
    for dev in indevs:
        print("  ", dev)
    print("Output Devices:")

    for dev in outdevs:
        print("  ", dev)


def instrNames(instrdef: str) -> List[U[int, str]]:
    """
    Returns the list of names/instrument numbers in the instrument definition.

    Most of the time this list will have one single element, either an instrument
    number or a name

    Args:
        code (str): the code defining an instrument

    Returns:
        a list of names/instrument numbers

    Example
    =======

        >>> instr = r'''
        ... instr 10, foo
        ...     outch 1, oscili:a(0.1, 440)
        ... endin
        ... '''
        >>> instrNames(instr)
        [10, "foo"]

    """
    for line in instrdef.splitlines():
        line = line.strip()
        if not line.startswith("instr "):
            continue
        names = line[6:].split(",")
        out = []
        for n in names:
            asnum = misc.asnumber(n)
            out.append(asnum if asnum is not None else n)
        return out
    raise ValueError("No instrument definition found")


@dataclasses.dataclass
class ParsedInstrBody:
    """
    Attributes:
        pfieldsIndexToName: maps p index to name
        pfieldsText: a (multiline) string collecting all lines
            which deal with pfields (ifoo = p4 / ibar, ibaz passign ... / etc)
        body: the body of the instr without any pfields declarations
        pfieldsDefaults: default values used by pfields (via pset)
        pfieldsUsed: a set of all pfields used, both named and unnamed
        outChannels: output channels explicitely used (out, outs, outch with
            a constant)
    """
    pfieldsIndexToName: dict[int, str]
    pfieldsText: str
    body: str
    pfieldsDefaults: Opt[dict[int, float]] = None
    pfieldsUsed: Opt[set[int]] = None
    outChannels: Opt[set[int]] = None

    def numPfields(self) -> int:
        """ Returns the number of pfields in this instrument """
        return max(self.pfieldsUsed)


@lru_cache(maxsize=1000)
def instrParseBody(body: str) -> ParsedInstrBody:
    """
    Parses the body of the instrument and returns information
    about pfields used, output channels, etc.

    Args:
        body (str): the body of the instr (between instr/endin)

    Returns:
        a ParsedInstrBody
    """
    pfield_lines = []
    rest_lines = []
    values = None
    pargs_used = set()
    pfields: dict[int, str] = {}
    outchannels: set[int] = set()

    for line in body.splitlines():
        pargs_in_line = re.findall(r"\bp\d+", line)
        if pargs_in_line:
            for p in pargs_in_line:
                pargs_used.add(int(p[1:]))
        if m := re.search(r"\bpassign\s+(\d+)", line):
            pfield_lines.append(line)
            pstart = int(m.group(1))
            argsstr, rest = line.split("passign")
            args = argsstr.split(",")
            for i, name in enumerate(args, start=pstart):
                pargs_used.add(i)
                pfields[i] = name.strip()
        elif re.search(r"^\s*pset\s+([+-]?([0-9]*[.])?[0-9]+)", line):
            defaults_str = line.strip()[4:]
            values = {i: float(v)
                      for i, v in enumerate(defaults_str.split(","), start=1)}
        elif m := re.search(r"^\s*\b(\w+)\s*(=|init\s)\s*p(\d+)", line):
            pname = m.group(1)
            parg = int(m.group(3))
            pfield_lines.append(line)
            pfields[parg] = pname.strip()
            pargs_used.add(parg)
        else:
            if re.search(r"\bouts\s+", line):
                outchannels.update((1, 2))
            elif re.search(r"\bout\b", line):
                outchannels.add(1)
            elif re.search(r"\boutch\b", line):
                args = line.strip()[5:].split(",")
                channels = args[::2]
                for chans in channels:
                    if chan := misc.asnumber(chans) is not None:
                        outchannels.add(chan)
            rest_lines.append(line)

    return ParsedInstrBody(pfieldsText="\n".join(pfield_lines),
                           pfieldsDefaults=values,
                           pfieldsIndexToName=pfields,
                           pfieldsUsed=pargs_used,
                           outChannels=outchannels,
                           body="\n".join(rest_lines))


def bestSampleFormatForExtension(ext: str) -> str:
    """
    Given an extension, return the best sample format. Float64
    is not considered necessary for holding sound information

    Args:
        ext (str): the extension of the file will determine the format

    Returns:
        a sample format of the form "pcmXX" or "floatXX", where XX determines
        the bit rate ("pcm16", "float32", etc)
    """
    if ext[0] == ".":
        ext = ext[1:]

    if ext in {"wav", "aif", "aiff"}:
        return "float32"
    elif ext == "flac":
        return "pcm24"
    else:
        raise ValueError("Format {ext} not supported")