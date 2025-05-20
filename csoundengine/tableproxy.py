from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .abstractrenderer import AbstractRenderer


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
        by certain operations within a :class:`~csoundengine.session.Session` or
        a :class:`~csoundengine.offline.OfflineSession`, like
        :meth:`~csoundengine.session.Session.readSoundfile` or
        :meth:`~csoundengine.session.Session.makeTable`

    Example
    ~~~~~~~

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

    __slots__ = ('tabnum', 'sr', 'nchnls', 'parent', 'numframes', 'path', 'freeself', '_array',
                 'skiptime')

    def __init__(self,
                 tabnum: int,
                 numframes: int,
                 parent: AbstractRenderer | None = None,
                 sr: int = 0,
                 nchnls: int = 1,
                 path: str = '',
                 skiptime=0.,
                 freeself=False):

        # if path:
        #    path = os.path.abspath(os.path.expanduser(path))

        self.tabnum = tabnum
        """The table number assigned to this table"""

        self.sr = sr
        """Samplerate"""

        self.nchnls = nchnls
        """Number of channels, of applicable"""

        self.parent = parent
        """The parent Renderer"""

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

        if parent:
            parent._registerTable(self)

    @property
    def size(self) -> int:
        return self.numframes * self.nchnls

    def __repr__(self):
        return (f"TableProxy(source={self.tabnum}, sr={self.sr},"
                f" nchnls={self.nchnls},"
                f" numframes={self.numframes}, path={self.path},"
                f" freeself={self.freeself})")

    def __int__(self):
        return self.tabnum

    def __float__(self):
        return float(self.tabnum)

    def online(self) -> bool:
        return self.parent is not None and self.parent.renderMode() == 'online'

    def data(self) -> np.ndarray:
        """
        Get the table data as a numpy array.

        The returned numpy array is a pointer to the csound memory (a view)
        if the table is live or the samples read from disk otherwise

        Returns:
            the data as a numpy array
        """
        if self._array is not None:
            return self._array

        if self.parent is None:
            import sndfileio
            samples, sr = sndfileio.sndread(self.path)
            out = samples
        else:
            out = self.parent._getTableData(self)
            if out is None:
                raise RuntimeError("Could not access table data")
        self._array = out
        return out

    def free(self, delay=0.):
        if self.parent is None:
            raise RuntimeError("Cannot free this table since it is not associated "
                               "with any online or offline csound process")
        self.parent.freeTable(table=self.tabnum, delay=delay)

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
        if self.parent:
            self.parent.freeTable(table=self.tabnum)

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

    def plotSpectrogram(self, fftsize=2048, mindb=-90, maxfreq=0, overlap=4,
                        minfreq=20) -> None:
        """
        Plot a spectrogram of the sample data in this table.

        Requires that the samplerate is set

        Args:
            fftsize: the size of the fft
            mindb: the min. dB to plot
            maxfreq: the max. frequency to plot (0=default)
            overlap: the number of overlaps per window
            minfreq: the min. frequency to plot

        """
        if not self.sr:
            raise ValueError("This table has no samplerate, cannot plot")
        from . import plotting
        plotting.plotSpectrogram(self.data(), self.sr, fftsize=fftsize, mindb=mindb,
                                 maxfreq=maxfreq, minfreq=minfreq, overlap=overlap)
