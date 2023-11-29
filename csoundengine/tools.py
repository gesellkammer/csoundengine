import fnmatch
from typing import Optional as Opt
import sys
import os
import tempfile
import emlib.misc
from .config import logger
import platform
from functools import cache
from collections import UserString
from typing import TypeVar


_T = TypeVar("_T")


def makeUniqueFilename(ext: str, prefix='', folder='.') -> str:
    """
    Create a unique filename

    Args:
        ext: the extension of the filename
        prefix: a prefix to the unique part of the filename
        folder: where should this file be created? NB: the file itself is not
            created, but will be unique in the given folder.

    Returns:
        the generated filename
    """
    if not ext.startswith('.'):
        ext = '.' + ext
    return tempfile.mktemp(suffix=ext, dir=folder, prefix=prefix)


@emlib.misc.runonce
def defaultSoundfontPath() -> Opt[str]:
    """
    Returns the path of the fluid sf2 file
    """
    if sys.platform == 'linux':
        paths = ["/usr/share/sounds/sf2/FluidR3_GM.sf2"]
        path = next((path for path in paths if os.path.exists(path)), None)
        return path
    else:
        logger.error("Default path for soundfonts only defined in linux")
    return None


def showSoundfontPrograms(sfpath: str, glob="") -> None:
    """
    Print a list of sounfont presets/programs

    Args:
        sfpath: the path to the soundfont
        glob: if given, it is used to filter the presets to only those
            whose name matches the given glob pattern
    """
    from . import csoundlib
    progs = csoundlib.soundfontPresets(sfpath)
    if glob:
        progs = [p for p in progs
                 if fnmatch.fnmatch(p[2], glob)]
    emlib.misc.print_table(progs, headers=('bank', 'num', 'name'), showindex=False)


class PlatformId(UserString):
    def __init__(self, osname: str, arch: str):
        assert osname in ('windows', 'linux', 'macos')
        assert arch in ('x86_64', 'arm64', 'arm32')

        self.osname = osname
        self.arch = arch
        super().__init__(osname + "-" + arch)


@cache
def platformId() -> PlatformId:
    """
    Query the platform id for the current system

    The paltform id is the duplet <osname>-<architecture>, and is one of
    'linux-x86_64', 'windows-x86_64', 'macos-x86_64', 'linux-arm64', 'windows-arm64',
    'macos-arm64', ...
    """
    osname = {
        'linux': 'linux',
        'darwin': 'macos',
        'win32': 'windows'
    }[sys.platform]
    return PlatformId(osname, _platformArch())


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

