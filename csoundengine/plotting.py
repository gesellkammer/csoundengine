from emlib import numpytools
import emlib.misc

from scipy import signal
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from . import internalTools
from .config import config
from typing import Tuple


def _envelope(x: np.ndarray, hop:int):
    return numpytools.overlapping_frames(x, hop_length=hop,
                                         frame_length=hop).max(axis=0)


def _frames_to_time(frames, sr:int, hop_length:int, n_fft:int=None):
    samples = _frames_to_samples(frames, hop_length=hop_length, n_fft=n_fft)
    return samples / sr


def _frames_to_samples(frames:np.ndarray, hop_length=512, n_fft:int=None) -> np.ndarray:
    offset = int(n_fft // 2) if n_fft else 0
    return (np.asanyarray(frames) * hop_length + offset).astype(int)


def _figsizeAsTuple(figsize) -> Tuple[int, int]:
    if isinstance(figsize, tuple):
        return figsize
    elif isinstance(figsize, list):
        assert len(figsize) == 2
        return tuple(figsize)
    elif isinstance(figsize, str):
        parts = figsize.split(":")
        assert len(parts) == 2
        return int(parts[0]), int(parts[1])
    else:
        raise ValueError(f"Could not interpret {figsize} as a figure size")


def _plot_matplotlib(samples: np.ndarray, samplerate: int, show=False) -> None:
    numch = internalTools.arrayNumChannels(samples)
    numsamples = samples.shape[0]
    dur = numsamples / samplerate
    times = np.linspace(0, dur, numsamples)
    figsize = _figsizeAsTuple(config['samplesplot_figsize'])
    figsize = figsize[0]*2, figsize[1]
    f = plt.figure(figsize=figsize)
    ax1 = f.add_subplot(numch, 1, 1)
    for i in range(numch):
        if i == 0:
            axes = ax1
        else:
            axes = f.add_subplot(numch, 1, i + 1, sharex=ax1, sharey=ax1)
        if i < numch - 1:
            plt.setp(axes.get_xticklabels(), visible=False)
        chan = internalTools.getChannel(samples, i)
        axes.plot(times, chan, linewidth=1)
        plt.xlim([0, dur])
    if not matplotlibIsInline() and show:
        plt.show()


def matplotlibIsInline():
    """
    Return True if matplotlib is set to display plots inline

    https://stackoverflow.com/questions/15341757/how-to-check-that-pylab-backend-of-matplotlib-runs-inline
    """
    return emlib.misc.inside_jupyter() and 'inline' in matplotlib.get_backend()


def plotSamples(samples: np.ndarray, samplerate: int, profile: str= 'auto',
                show=False
                ) -> None:
    """
    Plot the samples

    Args:
        samples: a numpy array holding one or many channels of audio data
        samplerate: the sampling rate of samples
        profile: one of 'low', 'medium', 'high', 'highest', 'auto'
        show: if True, the plot is shown. Otherwise matplotlib.pyplot.show() needs
            to be called explicitely (when not in inline mode inside jupyter)

    """
    if profile == 'auto':
        dur = len(samples)/samplerate
        if dur > 60*8:
            profile = 'low'
        elif dur > 60*2:
            profile = 'medium'
        elif dur > 60*1:
            profile = 'high'
        else:
            profile = 'highest'
    if profile == 'low':
        maxpoints = 2000
        maxsr = 300
    elif profile == 'medium':
        maxpoints = 4000
        maxsr = 600
    elif profile == 'high':
        undersample = min(32, len(samples) // (1024*8))
        return _plot_matplotlib(samples[::undersample], samplerate//undersample, show=show)
    elif profile == 'highest':
        return _plot_matplotlib(samples, samplerate, show=show)
    else:
        raise ValueError("profile should be one of 'low', 'medium' or 'high'")
    targetsr = samplerate
    numch = internalTools.arrayNumChannels(samples)
    numsamples = samples.shape[0]
    if maxpoints < numsamples:
        targetsr = min(maxsr, (samplerate * numsamples) // maxpoints)
    hop_length = samplerate // targetsr
    figsize = _figsizeAsTuple(config['samplesplot_figsize'])
    if profile == "medium":
        figsize = int(figsize[0]*1.4), figsize[1]
    f = plt.figure(figsize=figsize)
    ax1 = f.add_subplot(numch, 1, 1)
    for i in range(numch):
        ax = ax1 if i==0 else f.add_subplot(numch, 1, i + 1, sharex=ax1, sharey=ax1)
        if i < numch - 1:
            plt.setp(ax.get_xticklabels(), visible=False)

        chan = internalTools.getChannel(samples, i)
        env = _envelope(np.ascontiguousarray(chan), hop_length)
        samples_top = env
        samples_bottom = -env
        locs = _frames_to_time(np.arange(len(samples_top)),
                               sr=samplerate,
                               hop_length=hop_length)
        ax.fill_between(locs, samples_bottom, samples_top)
        ax.set_xlim([locs.min(), locs.max()])
    f.subplots_adjust(left=0.1, right=0.1, top=0.1, bottom=0.1)
    f.tight_layout(pad=0.1)
    if not matplotlibIsInline() and show:
        plt.show()


def plotSpectrogram(samples: np.ndarray, samplerate: int, fftsize=2048, window:str=None,
                    overlap=4, axes:plt.Axes=None, cmap=None, interpolation='bilinear',
                    minfreq=40, maxfreq=None,
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
        maxfreq: initial max. frequency. If None, a configurable default will be used
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
        f: plt.Figure = plt.figure(figsize=figsize)
        axes:plt.Axes = f.add_subplot(1, 1, 1)
    hopsize = int(fftsize / overlap)
    noverlap = fftsize - hopsize
    if window is None:
        window = config['spectrogram_window']
    win = signal.get_window(window, fftsize)
    cmap = cmap if cmap is not None else config['spectrogram_colormap']
    axes.specgram(samples,
                  NFFT=fftsize,
                  Fs=samplerate,
                  noverlap=noverlap,
                  window=win,
                  cmap=cmap,
                  interpolation=interpolation,
                  vmin=mindb)
    if maxfreq is None:
        maxfreq = config['spectrogram_maxfreq']
    axes.set_ylim(minfreq, maxfreq)
    if not matplotlibIsInline() and show:
        plt.show()
    return axes
