from __future__ import annotations

import os
import sys
import platform
from collections import UserString

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import sndfileio
    import numpy as np


_cache = {}


def defaultSoundfontPath() -> str:
    """
    Returns the path of the fluid sf2 file

    Returns:
        the path of the default soundfont or an empty path if this does not apply
    """
    if (path := _cache.get('defaultSoundfontPath')) is not None:
        return path
    if sys.platform == 'linux':
        paths = ["/usr/share/sounds/sf2/FluidR3_GM.sf2"]
        path = next((path for path in paths if os.path.exists(path)), '')
    else:
        from .config import logger
        logger.info("Default path for soundfonts only defined in linux")
        path = ''
    _cache['defaultSoundfontPath'] = path
    return path


class PlatformId(UserString):
    def __init__(self, osname: str, arch: str):
        assert osname in ('windows', 'linux', 'macos')
        assert arch in ('x86_64', 'arm64', 'arm32')

        self.osname = osname
        self.arch = arch
        super().__init__(osname + "-" + arch)


def platformId() -> PlatformId:
    """
    Query the platform id for the current system

    The paltform id is the duplet <osname>-<architecture>, and is one of
    'linux-x86_64', 'windows-x86_64', 'macos-x86_64', 'linux-arm64', 'windows-arm64',
    'macos-arm64', ...
    """
    if (platformid := _cache.get('platformId')) is not None:
        return platformid

    osname = {
        'linux': 'linux',
        'darwin': 'macos',
        'win32': 'windows'
    }[sys.platform]
    platformid = PlatformId(osname, _platformArch())
    _cache['platformId'] = platformid
    return platformid


def _platformArch() -> str:
    machine = platform.machine().lower()
    bits, linkage = platform.architecture()
    if machine == 'arm':
        if bits == '64bit':
            return 'arm64'
        elif bits == '32bit':
            return 'arm32'
    elif machine.startswith('arm64'):
        return 'arm64'
    elif machine == 'x86_64' or machine.startswith('amd64') or machine.startswith('intel64'):
        return 'x86_64'
    elif machine == 'i386':
        if bits == '64bit':
            return 'x86_64'
        elif bits == '32bit':
            return 'x86'

    raise RuntimeError(f"Architecture not supported ({machine=}, {bits=}, {linkage=})")


def sndfileInfo(path: str) -> sndfileio.SndInfo:
    """
    Get information about a soundfile

    Args:
        path: path to the soundfile

    Returns:
        sndfileio.SndInfo: information about the soundfile
    """
    import sndfileio
    return sndfileio.sndinfo(path)


def sdifToMatrix(path: str, maxpolyphony: int) -> np.ndarray:
    import importlib
    import importlib.util
    if importlib.util.find_spec('loristrck'):
        try:
            lt = importlib.import_module('loristrck')
            partials, labels = lt.read_sdif(path)
            tracks, matrix = lt.util.partials_save_matrix(partials=partials, maxtracks=maxpolyphony)
            return matrix
        except ModuleNotFoundError as e:
            raise RuntimeError(f"Could not import loristrck while trying to read a .sdif file: {e}")
    else:
        raise RuntimeError("loristrck is needed in order to read a .sdif file. "
                        "Install it via `pip install loristrck` (see https://loristrck.readthedocs.io "
                        "for more information)")
