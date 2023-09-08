from __future__ import annotations
import os
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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

    __slots__ = ('tabnum', 'sr', 'nchnls', 'engine', 'numframes', 'path', 'freeself', '_array',
                 'skiptime')

    def __init__(self,
                 tabnum: int,
                 numframes: int,
                 engine: Engine | None = None,
                 sr: int = 0,
                 nchnls: int = 1,
                 path: str = '',
                 skiptime=0.,
                 freeself=False):
        if path:
            path = os.path.abspath(os.path.expanduser(path))

        self.tabnum = tabnum
        """The table number assigned to this table"""

        self.sr = sr
        """Samplerate"""

        self.nchnls = nchnls
        """Number of channels, of applicable"""

        self.engine = engine
        """The parent engine, if this is a live table (None if rendering offline)"""

        self.numframes = numframes
        """The number of frames (samples = numframes * nchnls)"""

        self.path = path
        """The path to load the data from, if applicable"""

        self.freeself = freeself
        """If True, bind the lifetime of the proxy to the lifetime of the table itself"""

        self.skiptime = skiptime
        """Skiptime of the table. This applies only to soundfiles"""

        self._array: np.ndarray | None = None
        """The data, if applicable"""

    def __repr__(self):
        enginename = self.engine.name if self.engine else 'none'
        return (f"TableProxy(engine={self.engine.name}, source={self.tabnum}, sr={self.sr},"
                f" nchnls={self.nchnls},"
                f" numframes={self.numframes}, path={self.path},"
                f" freeself={self.freeself})")

    def __int__(self):
        return self.tabnum

    def __float__(self):
        return float(self.tabnum)

    def online(self):
        return self.engine is None

    def data(self) -> np.ndarray:
        """
        Get the table data as a numpy array.

        The returned numpy array is a pointer to the csound memory (a view)
        if the table is live or the samples read from disk otherwise

        Returns:
            the data as a numpy array
        """
        if self._array is None:
            if self.engine is not None:
                assert self.engine.csound is not None
                self._array = self.engine.csound.table(self.tabnum)
            else:
                import sndfileio
                samples, sr = sndfileio.sndread(self.path)
                self._array = samples

        return self._array

    def duration(self) -> float:
        """
        Duration of the sample data in this table.

        This is only possible if the table holds sample data and the
        table has a samplerate

        Raises ValueError if the table has no samplerate

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
            plotting.plotSamples(self.data(), self.sr, profile='high')
        else:
            # TODO: implement generic plotting
            data = self.data()
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
        plotting.plotSpectrogram(self.data(), self.sr, fftsize=fftsize, mindb=mindb,
                                 maxfreq=maxfreq, minfreq=minfreq, overlap=overlap)