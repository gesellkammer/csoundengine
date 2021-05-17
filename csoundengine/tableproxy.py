from __future__ import annotations
import os
import numpy as np
from typing import Optional as Opt, TYPE_CHECKING

if TYPE_CHECKING:
    from .synth import Synth
    from .session import Session


class TableProxy:
    """
    A TableProxy is a proxy to an existing csound table

    .. note::

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
        # table is a TableProxy
        >>> table.plot()

    .. image:: assets/tableproxy-plot.png

    .. code::

        >>> table.plotSpectrogram()

    .. image:: assets/tableproxy-plotspectrogram.png

    """
    def __init__(self,
                 tabnum: int,
                 session: Session,
                 numframes: int,
                 sr: int=0,
                 nchnls: int=1,
                 path:str='',
                 freeself=False):
        """
        Args:
            tabnum (int) - the csound table number
            sr (int) - the sample rate of the table
            nchnls (int) - the number of channels in the table
            numframes (int) - the number of frames (data = numframes * nchnls)
            session (Session) - the corresponding Session
            path (str) - the path to the soundfile, if known
            freeself (bool) - if True, csound will free the table when this object
                goes out of scope

        """
        if path:
            path = os.path.abspath(os.path.expanduser(path))
        self.tabnum = tabnum
        self.sr = sr
        self.nchnls = nchnls
        self.session = session
        self.numframes = numframes
        self.path = path
        self.freeself = freeself
        self._array: Opt[np.ndarray] = None

    def __repr__(self):
        return (f"TableProxy(tabnum={self.tabnum}, sr={self.sr}, nchnls={self.nchnls},"
                f" numframes={self.numframes}, path={self.path}, "
                f"freeself={self.freeself})")

    def getData(self) -> np.ndarray:
        """
        Get the table data as a numpy array. The returned array is a pointer
        to the csound memory (a view)
        """
        if self._array is None:
            csound = self.session.engine.csound
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
        engine = self.session.engine
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
        Plot a spectrogram of the sample data in this table. Requires that
        the samplerate is set

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

    def play(self, **kws) -> Synth:
        """
        A proxy to Session.playSample

        Possible kws
        ------------

        dur:
            the duration of playback (-1 to play the whole sample)
        chan:
            the channel to play the sample to. In the case of multichannel
            samples, this is the first channel
        pan:
            a value between 0-1. -1 means default, which is 0 for mono,
            0.5 for stereo. For multichannel (3+) samples, panning is not
            taken into account
        gain:
            gain factor. See also: gaingroup
        speed:
            speed of playback
        loop:
            True/False or -1 to loop as defined in the file itself (not all
            file formats define loop points)
        delay:
            time to wait before playback starts
        start:
            the starting playback time (0=play from beginning)
        fade:
            fade in/out in secods. -1=default
        gaingroup:
            the idx of a gain group. The gain of all samples routed to the
            same group are scaled by the same value and can be altered as a group
            via Engine.setSubGain(idx, gain)

        Returns:
             A Synth. Modulatable parameters: gain, speed, chan, pan
             (see Synth.pwrite)
        """
        return self.session.playSample(self, **kws)