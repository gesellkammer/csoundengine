from emlib import numpytools
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from configdict import ConfigDict
from . import internalTools


config = ConfigDict("csoundengine.plotting")
config.addKey('spectrogram.colormap', 'inferno', choices=plt.colormaps())
config.addKey('samplesplot.figsize', (12, 4))
config.addKey('spectrogram.figsize', (24, 8))
config.addKey('spectrogram.maxfreq', 12000,
              doc="Highest frequency in a spectrogram")
config.addKey('spectrogram.window', 'hamming', choices={'hamming', 'hanning'})
config.load()


def _envelope(x: np.ndarray, hop:int):
    return numpytools.overlapping_frames(x, hop_length=hop,
                                         frame_length=hop).max(axis=0)


def _frames_to_time(frames, sr:int, hop_length:int, n_fft:int=None):
    samples = _frames_to_samples(frames, hop_length=hop_length, n_fft=n_fft)
    return samples / sr


def _frames_to_samples(frames:np.ndarray, hop_length=512, n_fft:int=None) -> np.ndarray:
    offset = int(n_fft // 2) if n_fft else 0
    return (np.asanyarray(frames) * hop_length + offset).astype(int)


def _plot_matplotlib(samples: np.ndarray, samplerate: int) -> None:
    numch = internalTools.arrayNumChannels(samples)
    numsamples = samples.shape[0]
    dur = numsamples / samplerate
    times = np.linspace(0, dur, numsamples)
    figsize = config['samplesplot.figsize']
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


def plotSamples(samples: np.ndarray, samplerate: int, profile: str= 'auto') -> None:
    """
    Plot the samples

    Args:
        samples: a numpy array holding one or many channels of audio data
        samplerate: the sampling rate of samples
        profile: one of 'low', 'medium', 'high', 'highest', 'auto'

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
        return _plot_matplotlib(samples[::undersample], samplerate//undersample)
    elif profile == 'highest':
        return _plot_matplotlib(samples, samplerate)
    else:
        raise ValueError("profile should be one of 'low', 'medium' or 'high'")
    targetsr = samplerate
    numch = internalTools.arrayNumChannels(samples)
    numsamples = samples.shape[0]
    if maxpoints < numsamples:
        targetsr = min(maxsr, (samplerate * numsamples) // maxpoints)
    hop_length = samplerate // targetsr
    figsize = config['samplesplot.figsize']
    if profile == "medium":
        figsize = int(figsize[0]*1.4), figsize[1]
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
        env = _envelope(np.ascontiguousarray(chan), hop_length)
        samples_top = env
        samples_bottom = -env
        locs = _frames_to_time(np.arange(len(samples_top)),
                               sr=samplerate,
                               hop_length=hop_length)
        axes.fill_between(locs, samples_bottom, samples_top)
        axes.set_xlim([locs.min(), locs.max()])


def plotSpectrogram(samples: np.ndarray, samplerate: int, fftsize=2048, window:str=None,
                    overlap=4, axes:plt.Axes=None, cmap=None, interpolation='bilinear',
                    minfreq=40, maxfreq=None,
                    mindb=-90):
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
        cmap: colormap, see pyplot.colormaps() (see config['spectrogram.cmap'])
        minfreq: initial min.frequency
        maxfreq: initial max. frequency. If None, a configurable default will be used
            (see config['spectrogram.maxfreq')
        interpolation: one of 'bilinear'
        mindb: the amplitude threshold

    Returns:
        the axes object
    """
    if axes is None:
        f: plt.Figure = plt.figure(figsize=config['spectrogram.figsize'])
        axes:plt.Axes = f.add_subplot(1, 1, 1)
    hopsize = int(fftsize / overlap)
    noverlap = fftsize - hopsize
    if window is None:
        window = config['spectrogram.window']
    win = signal.get_window(window, fftsize)
    cmap = cmap if cmap is not None else config['spectrogram.colormap']
    axes.specgram(samples,
                  NFFT=fftsize,
                  Fs=samplerate,
                  noverlap=noverlap,
                  window=win,
                  cmap=cmap,
                  interpolation=interpolation,
                  vmin=mindb)
    if maxfreq is None:
        maxfreq = config['spectrogram.maxfreq']
    axes.set_ylim(minfreq, maxfreq)
    return axes
