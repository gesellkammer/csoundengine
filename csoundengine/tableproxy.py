from __future__ import annotations
import os
import numpy as np
from typing import Optional as Opt, TYPE_CHECKING

if TYPE_CHECKING:
    from .synth import Synth
    from .engine import Engine


class TableProxy:
    """
    A TableProxy is a proxy to an existing csound table

    Args:
        tabnum: the csound table number
        sr: the sample rate of the table
        nchnls: the number of channels in the table
        numframes: the number of frames (data = numframes * nchnls)
        engine: the corresponding Engine
        path: the path to the output, if known
        freeself: if True, csound will free the table when this object goes out of scope

    .. warning::

        A TableProxy is **never** created by the user directly. It is returned
        by certain operations within a :class:`~csoundengine.session.Session`,
        like :meth:`~csoundengine.session.Session.readSoundfile` or
        :meth:`~csoundengine.session.Session.makeTable`

    Example
    =======

    .. code::

        >>> from csoundengine import *
        >>> session = Engine().session()
        >>> table = session.readSoundfile("mono.wav")
        >>> table
        TableProxy(tabnum=1, sr=44100, nchnls=1, engine=Engine(…), numframes=102400, path='mono.wav', freeself=False)
        >>> session.playSample(table, loop=True)
        >>> table.plot()

    .. image:: assets/tableproxy-plot.png

    .. code::

        >>> table.plotSpectrogram()

    .. image:: assets/tableproxy-plotspectrogram.png

    """
    def __init__(self,
                 tabnum: int,
                 engine: Engine,
                 numframes: int,
                 sr: int = 0,
                 nchnls: int = 1,
                 path: str = '',
                 freeself=False):
        if path:
            path = os.path.abspath(os.path.expanduser(path))
        self.tabnum = tabnum
        self.sr = sr
        self.nchnls = nchnls
        self.engine = engine
        self.numframes = numframes
        self.path = path
        self.freeself = freeself
        self._array: Opt[np.ndarray] = None

    def __repr__(self):
        return (f"TableProxy(engine={self.engine.name}, source={self.tabnum}, sr={self.sr},"
                f" nchnls={self.nchnls},"
                f" numframes={self.numframes}, path={self.path},"
                f" freeself={self.freeself})")

    def __int__(self):
        return self.tabnum

    def __float__(self):
        return float(self.tabnum)

    def getData(self) -> np.ndarray:
        """
        Get the table data as a numpy array. The returned array is a pointer
        to the csound memory (a view)
        """
        if self._array is None:
            csound = self.engine.csound
            assert csound is not None
            self._array = csound.table(self.tabnum)
        return self._array

    def getDuration(self) -> float:
        """
        Get the duration of the data in this table. This is only possible if
        the table holds sample data and the table has a samplerate

        Returns:
            the duration of the sample data, in seconds
        """
        if not self.sr:
            raise ValueError("This table has no samplerate")
        return self.numframes / self.sr

    def __del__(self):
        if not self.freeself:
            return
        engine = self.engine
        if engine and engine.started:
            engine.freeTable(self.tabnum)

    def plot(self) -> None:
        """
        Plot the table
        """
        from . import plotting
        if self.sr:
            plotting.plotSamples(self.getData(), self.sr, profile='high')
        else:
            # TODO: implement generic plotting
            data = self.getData()
            plotting.plt.plot(data)

    def plotSpectrogram(self, fftsize=2048, mindb=-90, maxfreq:int=None, overlap=4,
                        minfreq:int=0) -> None:
        """
        Plot a spectrogram of the sample data in this table.

        Requires that the samplerate is set

        Args:
            fftsize (int): the size of the fft
            mindb (int): the min. dB to plot
            maxfreq (int): the max. frequency to plot
            overlap (int): the number of overlaps per window
            minfreq (int): the min. frequency to plot

        """
        if not self.sr:
            raise ValueError("This table has no samplerate, cannot plot")
        from . import plotting
        plotting.plotSpectrogram(self.getData(), self.sr, fftsize=fftsize, mindb=mindb,
                                 maxfreq=maxfreq, minfreq=minfreq, overlap=overlap)