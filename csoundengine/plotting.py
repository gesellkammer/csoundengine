from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from emlib import numpytools
from emlib.envir import inside_jupyter
from . import internal
from .config import config

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _envelope(x: np.ndarray, hop: int):
    return numpytools.overlapping_frames(x, hop_length=hop,
                                         frame_length=hop).max(axis=0)


def _framesToTime(frames, sr: int, hoplength: int, nfft=0):
    samples = _framesToSamples(frames, hoplength=hoplength, nfft=nfft)
    return samples / sr


def _framesToSamples(frames: np.ndarray, hoplength=512, nfft=0) -> np.ndarray:
    offset = int(nfft // 2) if nfft else 0
    return (np.asanyarray(frames) * hoplength + offset).astype(int)


def _figsizeAsTuple(figsize) -> tuple[int, int]:
    if isinstance(figsize, tuple):
        assert isinstance(figsize[0], int) and isinstance(figsize[1], int)
        return figsize
    elif isinstance(figsize, list):
        assert len(figsize) == 2
        assert isinstance(figsize[0], int) and isinstance(figsize[1], int)
        h, w = figsize
        return (h, w)
    elif isinstance(figsize, str):
        parts = figsize.split(":")
        assert len(parts) == 2
        return int(parts[0]), int(parts[1])
    else:
        raise ValueError(f"Could not interpret {figsize} as a figure size")


def _plot_matplotlib(samples: np.ndarray, samplerate: int, show=False, tight=True,
                     figsize: tuple[int, int] | None = None
                     ) -> Figure:
    numch = internal.arrayNumChannels(samples)
    numsamples = samples.shape[0]
    dur = numsamples / samplerate
    times = np.linspace(0, dur, numsamples)
    figsize = figsize or _figsizeAsTuple(config['samplesplot_figsize'])
    figsize = figsize[0]*2, figsize[1]
    f = plt.figure(figsize=figsize)
    if tight:
        f.set_tight_layout(True)  # type: ignore
    ax1 = f.add_subplot(numch, 1, 1)
    for i in range(numch):
        if i == 0:
            axes = ax1
        else:
            axes = f.add_subplot(numch, 1, i + 1, sharex=ax1, sharey=ax1)
        if i < numch - 1:
            plt.setp(axes.get_xticklabels(), visible=False)
        chan = internal.getChannel(samples, i)
        axes.plot(times, chan, linewidth=1)
        plt.xlim([0, dur])
    if not matplotlibIsInline() and show:
        plt.show()
    return f


def matplotlibIsInline() -> bool:
    """
    Return True if matplotlib is set to display plots inline

    https://stackoverflow.com/questions/15341757/how-to-check-that-pylab-backend-of-matplotlib-runs-inline
    """
    return inside_jupyter() and 'inline' in plt.get_backend()


def _plotSubsample(samples: np.ndarray, samplerate: int, maxpoints: int,
                   maxsr: int, show: bool, figsizeFactor=1., tight=True,
                   figsize: tuple[int, int] | None = None
                   ) -> Figure:
    targetsr = samplerate
    numch = internal.arrayNumChannels(samples)
    numsamples = samples.shape[0]
    if maxpoints < numsamples:
        targetsr = min(maxsr, (samplerate * numsamples) // maxpoints)
    hoplength = samplerate // targetsr
    figsize = figsize or _figsizeAsTuple(config['samplesplot_figsize'])
    figsize = (int(figsize[0] * figsizeFactor), figsize[1])
    f = plt.figure(figsize=figsize)
    if tight:
        f.set_tight_layout(True)
    ax1 = f.add_subplot(numch, 1, 1)
    for i in range(numch):
        ax = ax1 if i == 0 else f.add_subplot(numch, 1, i + 1, sharex=ax1, sharey=ax1)
        if i < numch - 1:
            plt.setp(ax.get_xticklabels(), visible=False)

        chan = internal.getChannel(samples, i)
        env = _envelope(np.ascontiguousarray(chan), hoplength)
        samples_top = env
        samples_bottom = -env
        locs = _framesToTime(np.arange(len(samples_top)),
                               sr=samplerate,
                               hoplength=hoplength)
        ax.fill_between(locs, samples_bottom, samples_top)
        ax.set_xlim([locs.min(), locs.max()])
    # f.subplots_adjust(left=0.1, right=0.1, top=0.1, bottom=0.1)
    f.tight_layout(pad=0.1)
    if not matplotlibIsInline() and show:
        plt.show()
    return f


def plotSamples(samples: np.ndarray,
                samplerate: int,
                profile='auto',
                show=False,
                saveas='',
                figsize: tuple[int, int] | None = None,
                closefig=False
                ) -> Figure:
    """
    Plot the samples

    Args:
        samples: a numpy array holding one or many channels of audio data
        samplerate: the sampling rate of samples
        profile: one of 'low', 'medium', 'high', 'highest', 'auto'
        show: if True, the plot is shown. Otherwise matplotlib.pyplot.show() needs
            to be called explicitely (when not in inline mode inside jupyter)
        saveas: if a path is given, the plot is saved to this path

    Returns:
        the figure used

    """
    if profile == 'auto':
        dur = len(samples)/samplerate
        if dur > 60*2:
            profile = 'low'
        elif dur > 60*1:
            profile = 'medium'
        elif dur > 4:
            profile = 'high'
        else:
            profile = 'highest'

    if profile == 'low':
        fig = _plotSubsample(samples=samples, samplerate=samplerate,
                              maxpoints=2000, maxsr=300, show=show, figsize=figsize)
    elif profile == 'medium':
        fig = _plotSubsample(samples=samples, samplerate=samplerate, maxpoints=4000,
                              maxsr=600, show=show, figsizeFactor=1.4, figsize=figsize)
    elif profile == 'high':
        undersample = min(32, len(samples) // (1024*8))
        fig = _plot_matplotlib(samples[::undersample], samplerate//undersample, show=show)
    elif profile == 'highest':
        fig = _plot_matplotlib(samples, samplerate, show=show)
    else:
        raise ValueError("profile should be one of 'low', 'medium' or 'high'")

    if saveas:
        plt.close(fig)
        fig.savefig(saveas, transparent=False, facecolor="white", bbox_inches='tight')
    return fig


def figureToArray(fig: Figure, removeAlpha=True) -> np.ndarray:
    fig.canvas.draw()
    # buf = fig.canvas.tostring_rgb()
    # X = np.array(fig.canvas.renderer.buffer_rgba())
    buf = fig.canvas.renderer.buffer_rgba()
    data = np.frombuffer(buf, dtype=np.uint8)
    image = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    if removeAlpha:
        image = image[:, :-1]
    return image


def figureToBase64(fig: Figure) -> str:
    imgarray = figureToArray(fig)
    imgb64 = numpyToB64(imgarray)
    plt.close(fig)
    return imgb64


def numpyToB64(array: np.ndarray) -> str:
    from PIL import Image
    from io import BytesIO
    import base64
    impil = Image.fromarray(array)
    if impil.mode != 'RGB':
        impil = impil.convert('RGB')
    buff = BytesIO()
    impil.save(buff, format="png")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def plotSpectrogram(samples: np.ndarray, samplerate: int, fftsize=2048, window='',
                    overlap=4, axes: Axes | None = None, cmap: str = '', interpolation='bilinear',
                    minfreq=40, maxfreq=0,
                    mindb=-90, show=False):
    """
    Plot a spectrogram

    Args:
        samples: a channel of audio data
        samplerate: the samplerate of the audio data
        fftsize: the size of the fft, in samples
        window: a string passed to scipy.signal.get_window
        overlap: the number of overlaps. If fftsize=2048, an overlap of 4 will result
            in a hopsize of 512 samples
        axes: the axes to plot on. If None, new axes will be created
        cmap: colormap, see pyplot.colormaps() (see config['spectrogram_colormap'])
        minfreq: initial min.frequency
        maxfreq: initial max. frequency. If 0, a configurable default will be used
            (see config['spectrogram_maxfreq')
        interpolation: one of 'bilinear'
        mindb: the amplitude threshold
        show: if True, show the plot. Otherwise matplotlib.pyplot.show() needs
            to be called

    Returns:
        the axes object
    """
    if axes is None:
        figsize = _figsizeAsTuple(config['spectrogram_figsize'])
        f = plt.figure(figsize=figsize)
        axes = f.add_subplot(1, 1, 1)
    hopsize = int(fftsize / overlap)
    noverlap = fftsize - hopsize
    if not window:
        window = config['spectrogram_window']
    from scipy import signal
    win = signal.get_window(window, fftsize)
    cmap = cmap if cmap else config['spectrogram_colormap']
    axes.specgram(samples,
                  NFFT=fftsize,
                  Fs=samplerate,
                  noverlap=noverlap,
                  window=win,
                  cmap=cmap,
                  interpolation=interpolation,
                  vmin=mindb)
    if not maxfreq:
        maxfreq = config['spectrogram_maxfreq']
    axes.set_ylim(minfreq, maxfreq)
    if not matplotlibIsInline() and show:
        plt.show()
    return axes
