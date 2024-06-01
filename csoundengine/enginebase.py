from __future__ import annotations
import dataclasses
import subprocess
import functools
from . import internal
import os
import emlib.misc
import sndfileio
import numpy as np
from abc import ABC, abstractmethod


@dataclasses.dataclass
class TableInfo:
    """
    Information about a csound table
    """
    sr: int
    size: int
    numChannels: int = 1
    numFrames: int = -1
    path: str = ''
    hasGuard: bool | None = None

    def __post_init__(self):
        if self.hasGuard is None:
            self.hasGuard = self.size == self.numFrames * self.numChannels + 1
        if self.numFrames == -1:
            self.numFrames = self.size // self.numChannels

    @property
    def duration(self) -> float:
        """
        The duration of this table in seconds, if it has a sr

        This is valid only for samples loaded via gen1 or with metadata
        assigned. If the table does not have a samplerate, 0 is returned
        """
        if self.sr != 0:
            return self.size / self.sr
        else:
            return 0.

    @staticmethod
    def get(path: str) -> TableInfo:
        import sndfileio
        sndinfo = sndfileio.sndinfo(path)
        return TableInfo(sr=sndinfo.samplerate,
                         numChannels=sndinfo.channels,
                         numFrames=sndinfo.nframes,
                         size=sndinfo.channels * sndinfo.nframes,
                         path=path)


def _channelMode(kind: str) -> int:
    if kind == 'r':
        return ctcsound.CSOUND_INPUT_CHANNEL
    elif kind == 'w':
        return ctcsound.CSOUND_INPUT_CHANNEL
    elif kind == 'rw':
        return ctcsound.CSOUND_INPUT_CHANNEL | ctcsound.CSOUND_OUTPUT_CHANNEL
    else:
        raise ValueError(f"Expected r, w or rw, got {kind}")


class _EngineBase(ABC):

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
    def getChannel(self, channel: str) -> float: ...

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
        pass

    @abstractmethod
    def makeTable(self,
                  data: np.ndarray | Sequence[float],
                  tabnum: int = -1,
                  sr: int = 0,
                  delay=0.
                  ) -> int: ...

    @abstractmethod
    def includeFile(self, include: str) -> None:
        """
        Add an #include file to this Engine

        Args:
            include: the path to the include file
        """
        pass

    @abstractmethod
    def playSoundFromDisk(self, path: str, delay=0., chan=0, speed=1., fade=0.01
                          ) -> float:
        pass

    @abstractmethod
    def setp(self, p1: float | str, *pairs, delay=0.) -> None: ...

    @abstractmethod
    def freeTable(self, tableindex: int, delay=0.) -> None: ...











