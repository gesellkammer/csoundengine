from __future__ import annotations
import sys
import os as _os


def compressionQualityToBitrate(quality: float, fmt='ogg') -> int:
    """
    Convert compression quality to bitrate

    Args:
        quality: the compression quality (0-1) as passed to --vbr-quality
        fmt: the encoding format (ogg at the moment)

    Returns:
        the resulting bit rate


    =======   =======
    quality   bitrate
    =======   =======
    0.0       64
    0.1       80
    0.2       96
    0.3       112
    0.4       128
    0.5       160
    0.6       192
    0.7       224
    0.8       256
    0.9       320
    1.0       500
    =======   =======
    """
    if fmt == 'ogg':
        idx = int(quality * 10 + 0.5)
        if idx > 10:
            idx = 10
        return (64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 500)[idx]
    else:
        raise ValueError(f"Format {fmt} not supported")


def compressionBitrateToQuality(bitrate: int, fmt='ogg') -> float:
    """
    Convert a bitrate to a compression quality between 0-1, as passed to --vbr-quality

    Args:
        bitrate: the bitrate in kb/s, oneof 64, 80, 96, 128, 160, 192, 224, 256, 320, 500
        fmt: the encoding format (ogg at the moment)
    """
    if fmt == 'ogg':
        bitrates = [64, 80, 96, 128, 128, 160, 192, 224, 256, 320, 500]
        import emlib.misc
        idx = emlib.misc.nearest_index(bitrate, bitrates)
        return idx / 10
    else:
        raise ValueError(f"Format {fmt} not supported")


_pluginsFolders = {
    '6.0': {
        'linux': '$HOME/.local/lib/csound/6.0/plugins64',
        'darwin': '$HOME/Library/csound/6.0/plugins64',
        'win32': '%LOCALAPPDATA%/csound/6.0/plugins64'
    },
    '7.0': {
        'linux': '$HOME/.local/lib/csound/7.0/plugins64',
        'darwin': '$HOME/Library/csound/7.0/plugins64',
        'win32': '%LOCALAPPDATA%/csound/7.0/plugins64'
    },
}


def userPluginsFolder(apiversion='6.0') -> str:
    """
    Returns the user plugins folder for this platform

    This is the folder where csound will search for user-installed
    plugins. The returned folder is always an absolute path. It is not
    checked if the folder actually exists.

    Args:
        apiversion: 6.0 or 7.0

    Returns:
        the user plugins folder for this platform

    **Folders for 64-bit plugins**:

    ======== ===== =================================================
     OS       api  Plugins folder
    ======== ===== =================================================
     Linux    6.0  ``~/.local/lib/csound/6.0/plugins64``
              7.0  ``~/.local/lib/csound/7.0/plugins64``
     macOS    6.0  ``~/Library/csound/6.0/plugins64``
              7.0  ``~/Library/csound/7.0/plugins64``
     windows  6.0  ``C:/Users/<User>/AppData/Local/csound/6.0/plugins64``
              6.0  ``C:/Users/<User>/AppData/Local/csound/7.0/plugins64``
    ======== ===== =================================================

    For 32-bit plugins the folder is the same, without the '64' ending (``.../plugins``)
    """
    key = apiversion
    folders = _pluginsFolders[key]
    if sys.platform not in folders:
        raise RuntimeError(f"Platform {sys.platform} not known")
    folder = folders[sys.platform]
    return _os.path.abspath(_os.path.expandvars(folder))


_formatOptions = {
    'pcm16': '',
    'pcm24': '--format=24bit',
    'float32': '--format=float',  # also -f
    'float64': '--format=double',
    'vorbis': '--format=vorbis'
}


_optionForSampleFormat = {
    'wav': '--format=wav',   # could also be --wave
    'aif': '--format=aiff',
    'aiff': '--format=aiff',
    'flac': '--format=flac',
    'ogg': '--format=ogg'
}


_defaultEncodingForFormat = {
    'wav': 'float32',
    'flac': 'pcm24',
    'aif': 'float32',
    'aiff': 'float32',
    'ogg': 'vorbis'
}

# _csoundFormatOptions = {'-3', '-f', '--format=24bit', '--format=float',
#                         '--format=double', '--format=long', '--format=vorbis',
#                         '--format=short'}


def csoundOptionsForOutputFormat(fmt='wav',
                                 encoding=''
                                 ) -> list[str]:
    """
    Returns the command-line options for the given format+encoding

    Args:
        fmt: the format of the output file ('wav', 'flac', 'aif', etc)
        encoding: the encoding ('pcm16', 'pcm24', 'float32', etc). If not given,
            the best encoding for the given format is chosen

    Returns:
        a tuple of two strings holding the command-line options for the given
        sample format/encoding

    Example
    -------

        >>> csoundOptionsForOutputFormat('flac')
        ('--format=flac', '--format=24bit')
        >>> csoundOptionsForOutputFormat('wav', 'float32')
        ('--format=wav', '--format=float')
        >>> csoundOptionsForOutputFormat('aif', 'pcm16')
        ('--format=aiff', '--format=short')

    .. seealso:: :func:`csoundOptionForSampleEncoding`
    """
    if fmt.startswith("."):
        fmt = fmt[1:]
    assert fmt in _defaultEncodingForFormat, f"Unknown format: {fmt}, possible formats are: " \
                                             f"{_defaultEncodingForFormat.keys()}"
    if not encoding:
        encoding = _defaultEncodingForFormat.get(fmt)
        if not encoding:
            raise ValueError(f"Default encoding unknown for format {fmt}")
    encodingOption = csoundOptionForSampleEncoding(encoding)
    fmtOption = _optionForSampleFormat[fmt]
    options = [fmtOption]
    if encodingOption:
        options.append(encodingOption)
    return options


def csoundOptionForSampleEncoding(encoding: str) -> str:
    """
    Returns the command-line option for the given sample encoding.

    Given a sample encoding of the form pcmXX or floatXX, where
    XX is the bit-rate, returns the corresponding command-line option
    for csound

    Args:
        fmt (str): the desired sample format. Either pcmXX, floatXX, vorbis
          where XX stands for the number of bits per sample (pcm24,
          float32, etc)

    Returns:
        the csound command line option corresponding to the given format

    Example
    -------

        >>> csoundOptionForSampleEncoding("pcm24")
        --format=24bit
        >>> csoundOptionForSampleEncoding("float64")
        --format=double

    .. seealso:: :func:`csoundOptionsForOutputFormat`

    """
    if encoding not in _formatOptions:
        raise ValueError(f'format {encoding} not known. Possible values: '
                         f'{_formatOptions.keys()}')
    return _formatOptions[encoding]


def bestSampleEncodingForExtension(ext: str) -> str:
    """
    Given an extension, return the best sample encoding.

    .. note::

        float64 is not considered necessary for holding sound information

    Args:
        ext (str): the extension of the file will determine the format

    Returns:
        a sample format of the form "pcmXX" or "floatXX", where XX determines
        the bit rate ("pcm16", "float32", etc)

    ========== ================
    Extension  Sample Format
    ========== ================
    wav        float32
    aif        float32
    flac       pcm24
    mp3        pcm16
    ogg        vorbis
    ========== ================

    """
    if ext[0] == ".":
        ext = ext[1:]

    if ext in {"wav", "aif", "aiff"}:
        return "float32"
    elif ext == "flac":
        return "pcm24"
    elif ext == 'ogg':
        return 'vorbis'
    else:
        raise ValueError(f"Format {ext} not supported. Formats supported: wav, aiff, flac and ogg")
