from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from . import internal



@dataclasses.dataclass
class TableInfo:
    """
    Information about a csound table
    """
    sr: int
    "Sample rate of the audio, if applicable"

    size: int
    "Total number of items (independent of numChannels)"

    nchnls: int = 1
    "Number of channels"

    path: str = ''
    "Path to the source soundfile"

    guard: bool | None = None
    "Has this table a guard point"

    def __post_init__(self):
        if self.guard is None:
            self.guard = self.size == self.numFrames * self.nchnls + 1

    @property
    def numFrames(self):
        """Number of frames (size // numchannels)"""
        return self.size // self.nchnls

    @property
    def duration(self) -> float:
        """
        The duration of this table in seconds, if it has a sr

        This is valid only for samples loaded via gen1 or with metadata
        assigned. If the table does not have a samplerate, 0 is returned
        """
        if self.sr != 0:
            return self.numFrames / self.sr
        else:
            return 0.

    @staticmethod
    def get(path: str) -> TableInfo:
        import sndfileio
        sndinfo = sndfileio.sndinfo(path)
        return TableInfo(sr=sndinfo.samplerate,
                         size=sndinfo.channels * sndinfo.nframes,
                         nchnls=sndinfo.channels,
                         path=path)


def _channelMode(kind: str) -> int:
    import libcsound
    if kind == 'r':
        return libcsound.CSOUND_INPUT_CHANNEL
    elif kind == 'w':
        return libcsound.CSOUND_INPUT_CHANNEL
    elif kind == 'rw':
        return libcsound.CSOUND_INPUT_CHANNEL | libcsound.CSOUND_OUTPUT_CHANNEL
    else:
        raise ValueError(f"Expected r, w or rw, got {kind}")


class _EngineBase(ABC):

    def __init__(self,
                 sr: int,
                 ksmps: int,
                 nchnls: int,
                 numAudioBuses: int,
                 numControlBuses: int,
                 sampleAccurate=False,
                 a4: int = 442
                 ):
        assert sr > 0
        assert ksmps > 0
        assert a4 > 0

        self.version: int = 0

        self.sr = sr
        "Sample rate"

        self.ksmps = ksmps
        "Number of samples per cycle"

        self.nchnls = nchnls
        "Number of output channels"

        self.numAudioBuses = numAudioBuses
        "Number of audio buses"

        self.numControlBuses = numControlBuses
        "Number of control buses"

        self.sampleAccurate = sampleAccurate
        "Use sample accurate scheduling"

        self.a4 = a4
        "Reference frequency for A4"

        self._busTokenCountPtr: np.ndarray = np.empty((1,), dtype=float)
        self._busTokenToKind: dict[int, str] = {}
        self._kbusTable: np.ndarray | None = None
        self._busIndexes: dict[int, int] = {}
        self._builtinInstrs: dict[str, int] = {}
        """Dict of built-in instrs, mapping instr name to number"""

    @abstractmethod
    def elapsedTime(self) -> float: ...

    @abstractmethod
    def strSet(self, s: str) -> int: ...

    @abstractmethod
    def unsched(self, p1: float | str, delay: float = 0) -> None: ...

    @abstractmethod
    def readSoundfile(self, path: str, tabnum=0, chan=0, skiptime=0., delay=0., unique=True
                      ) -> tuple[int, TableInfo]: ...

    @abstractmethod
    def tableInfo(self, size: int, cache=True) -> TableInfo: ...

    @abstractmethod
    def makeEmptyTable(self, size: int, numchannels=1, sr=0, delay=0.) -> int: ...

    @abstractmethod
    def getTableData(self, idx: int) -> np.ndarray: ...

    @abstractmethod
    def getControlChannel(self, channel: str) -> float:
        """
        Get the value of a control channel

        Args:
            channel: the name of the channel

        Returns:
            the value of the channel. Raises KeyError if the channel
            does not exist.
        """
        ...

    @abstractmethod
    def setChannel(self, channel: str, value: float | str | np.ndarray, delay=0.): ...

    @abstractmethod
    def channelPointer(self, channel: str, kind='control', mode='rw') -> np.ndarray:
        """
        Returns a numpy array aliasing the memory of a control or audio channel

        If the channel does not exist, it will be created with the given `kind` and set to
        the given mode.
        The returned numpy arrays are internally cached and are valid as long as this
        Engine is active. Accessing the channel through the pointer is not thread-safe.

        Args:
            channel: the name of the channel
            kind: one of 'control' or 'audio' (string channels are not supported yet)
            mode: the kind of channel, 'r', 'w' or 'rw'

        Returns:
            a numpy array of either 1 or ksmps size

        .. seealso:: :meth:`Engine.setChannel`
        """
        pass

    @abstractmethod
    def makeTable(self,
                  data: np.ndarray | Sequence[float],
                  sr: int = 0,
                  tabnum: int = -1,
                  delay=0.
                  ) -> int: ...

    @abstractmethod
    def initChannel(self,
                    channel: str,
                    value: float | str | np.ndarray = 0,
                    kind='',
                    mode="r") -> None:
        """
        Create a channel and set its initial value

        Args:
            channel: the name of the channel
            value: the initial value of the channel,
                will also determine the type (k, a, S)
            kind: One of 'k', 'S', 'a'. Use None to auto determine the channel type.
            mode: r for read, w for write, rw for both.

        .. note::
                the `mode` is set from the perspective of csound. A read (input)
                channel is a channel which can be written to by the api and read
                from csound. An write channel (output) can be written by csound
                and read from the api

        Example
        ~~~~~~~

        >>> from csoundengine import *
        >>> e = Engine()
        >>> e.initChannel("mastergain", 1.0)
        >>> e.compile(r'''
        ... instr 100
        ...   asig oscili 0.1, 1000
        ...   kmastergain = chnget:k("mastergain")
        ...   asig *= intrp(kmastergain)
        ... endin
        ... ''')
        >>> eventid = e.sched(100)
        >>> e.setChannel("mastergain", 0.5)
        """
        ...

    @abstractmethod
    def includeFile(self, include: str) -> None:
        """
        Add an #include file to this Engine

        Args:
            include: the path to the include file
        """
        pass

    def sched(self,
              instr: int | float | str,
              delay=0.,
              dur=-1.,
              *pfields,
              args: np.ndarray | Sequence[float | str] = (),
              relative=True,
              unique=False,
              **namedpfields
              ) -> float: ...

    @abstractmethod
    def playSoundFromDisk(self, path: str, delay=0., chan=0, speed=1., fade=0.01
                          ) -> float:
        pass

    @abstractmethod
    def setp(self, p1: float | str, *pairs, delay=0.) -> None: ...

    @abstractmethod
    def freeTable(self, tableindex: int, delay=0.) -> None: ...

    def hasBusSupport(self) -> bool:
        """
        Returns True if this Engine was started with bus support

        .. seealso::

            :meth:`~csoundengine.engine.Engine.assignBus`
            :meth:`~csoundengine.engine.Engine.writeBus`
            :meth:`~csoundengine.engine.Engine.readBus`
        """
        return (self.numAudioBuses > 0 or self.numControlBuses > 0)

    def automateBus(self,
                    bus: int,
                    pairs: Sequence[float] | tuple[Sequence[float], Sequence[float]],
                    mode='linear',
                    delay=0.,
                    overtake=False) -> float:
        """
        Automate a control bus

        The automation is performed within csound and is thus assured to stay
        in sync

        Args:
            bus: the bus token as received via :meth:`Engine.assignBus`
            pairs: the automation data as a flat sequence (t0, value0, t1, value1, ...) or
                a tuple (times, values)
                Times are relative to the start of the automation event
            mode: interpolation mode, one of 'linear', 'expon(xx)', 'cos', 'smooth'.
                See the csound opcode 'interp1d' for mode information
                (https://csound-plugins.github.io/csound-plugins/opcodes/interp1d.html)
            delay: when to start the automation
            overtake: if True, the first value of pairs is replaced with the current
                value of the bus. The same effect can be achieved if the first value
                of the automation line is a nan

        .. seealso:: :meth:`Engine.assignBus`, :meth:`Engine.writeBus`, :meth:`Engine.automatep`

        Example
        ~~~~~~~

        >>> e = Engine()
        >>> e.compile(r'''
        ... instr 100
        ...   ifreqbus = p4
        ...   kfreq = busin:k(ifreqbus)
        ...   outch 1, oscili:a(0.1, kfreq)
        ... endin
        ... ''')
        >>> freqbus = e.assignBus(value=440)
        >>> eventid = e.sched(100, args=(freqbus,))
        >>> e.automateBus(freqbus, [0, float('nan'), 3, 200, 5, 200])

        """
        if not self.hasBusSupport():
            raise RuntimeError("This engine does not have bus support")
        pairs = internal.flattenAutomationData(pairs)
        if self.version >= 7000 or len(pairs) <= 1900:
            args = [int(bus), self.strSet(mode), int(overtake), len(pairs), *pairs]
            return self.sched(self._builtinInstrs['automateBusViaPargs'],
                              delay=delay,
                              dur=pairs[-2] + self.ksmps/self.sr,
                              args=args)
        else:
            for subdelay, subgroup in internal.splitAutomation(pairs, 1900 // 2):
                self.automateBus(bus=bus, pairs=subgroup, delay=delay+subdelay,
                                 mode=mode, overtake=overtake)
            return 0

    @abstractmethod
    def writeBus(self, bus: int, value: float, delay=0.) -> None:
        """
        Set the value of a control bus

        Normally a control bus is set via another running instrument,
        but it is possible to set it directly from python. The first
        time a bus is set or queried there is short delay, all
        subsequent operations on the bus are very fast.

        Args:
            bus: the bus token, as returned via :meth:`Engine.assignBus`
            value: the new value
            delay: if given, the modification is scheduled in the future

        .. seealso::

            :meth:`~Engine.readBus`
            :meth:`~Engine.assignBus`
            :meth:`~Engine.automateBus`

        Example
        ~~~~~~~

        >>> e = Engine(...)
        >>> e.compile(r'''
        ... instr 100
        ...   ifreqbus = p4
        ...   kfreq = busin:k(ifreqbus)
        ...   outch 1, vco2:a(0.1, kfreq)
        ... endin
        ... ''')
        >>> freqbus = e.assignBus(value=1000)
        >>> e.sched(100, 0, 4, args=[freqbus])
        >>> e.writeBus(freqbus, 500, delay=0.5)

        """
        raise NotImplementedError
