import fnmatch
from typing import Optional as Opt
import sys
import os
import tempfile
import emlib.misc
import numpy as np
from .config import logger

def makeUniqueFilename(ext:str, prefix='', folder:str='.') -> str:
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
    progs = csoundlib.soundfontGetPresets(sfpath)
    if glob:
        progs = [p for p in progs
                 if fnmatch.fnmatch(p[2], glob)]
    emlib.misc.print_table(progs, headers=('bank', 'num', 'name'), showindex=False)
