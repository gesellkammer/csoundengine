from __future__ import annotations
import os
import sys
import urllib.request, urllib.error
import re

from . import csoundlib
from pathlib import Path
import tempfile
import shutil
import subprocess
import logging
from datetime import datetime
import json
from configdict import ConfigDict
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Union as U, List, Dict, Tuple

logger = logging.getLogger("csoundengine")


def _asVersionTriplet(tagname: str) -> Tuple[int, int, int]:
    match = re.search(r"v?(\d+)\.(\d+)(.(\d+))?", tagname)
    if not match:
        raise ValueError("Could not parse tagname")
    major = int(match.group(1))
    minor= int(match.group(2))
    try:
        patch = int(match.group(3))
    except IndexError:
        patch = 0
    return (major, minor, patch)


def getPluginsLatestRelease() -> Dict[str, str]:
    """
    Returns a dict with the urls of the plugins latest release

    The returned dict has the form ``{platform: url}``, where
    *platform* is the str in *sys.platform* for each of the
    supported platforms. Other metadata includes: 'version'

    Returns:
        a dict mapping platform to download url

    Example
    -------

        >>> getPluginsLatestRelease()
        {'linux': 'https://github.com/.../csound-plugins-linux.zip',
         'darwin': 'https://github.com/.../csound-plugins-macos.zip',
         'win32': 'https://github.com/.../csound-plugins-win64.zip'}
    """
    url = f"https://api.github.com/repos/csound-plugins/csound-plugins/releases/latest"
    try:
        tmpfile, _ = urllib.request.urlretrieve(url)
    except urllib.error.URLError as e:
        logger.error(str(e))
        raise RuntimeError(f"Could not download plugins info from {url}")
    info = json.load(open(tmpfile))
    assets = info.get('assets')
    if not assets:
        raise RuntimeError("Could not get release assets")
    asseturls = [asset['browser_download_url'] for asset in assets]
    out = {}
    for asseturl in asseturls:
        asseturl_lower = asseturl.lower()
        if "linux" in asseturl_lower:
            out['linux'] = asseturl
        elif "macos" in asseturl_lower:
            out['darwin'] = asseturl
        elif "win64" in asseturl_lower or "windows" in asseturl_lower:
            out['win32'] = asseturl
    tagname = info.get('tab_name')
    if tagname:
        try:
            versiontriplet = _asVersionTriplet(tagname)
        except ValueError:
            versiontriplet = (0, 0, 0)
        out['version'] = versiontriplet

    return out


def csoundInstalled() -> bool:
    """ Returns True if csound is installed """
    return shutil.which("csound") is not None


def downloadLatestPluginForPlatform(destFolder: Path = None) -> Path:
    """
    Downloads the latest release for a given platform

    Args:
        destFolder: where to save the .zip file. If not given, plugins are
            downloaded to the Downloads folder in your platform

    Returns:
        the full path to the saved .zip file
    """
    if destFolder is None:
        destFolder = _getDownloadsFolder()
    pluginurls = getPluginsLatestRelease()
    pluginurl = pluginurls.get(sys.platform)
    version = pluginurls.get('version', None)
    if version:
        print(f"Downloading latest version ({version}) of csound-plugins...")
    if pluginurl is None:
        raise RuntimeError(f"No plugins released for platform {sys.platform}")
    return _download(pluginurl, destFolder)


def _download(url: str, destFolder: U[str, Path]) -> Path:
    assert os.path.exists(destFolder) and os.path.isdir(destFolder)
    fileName = os.path.split(url)[1]
    dest = Path(destFolder) / fileName
    if dest.exists():
        logger.warning(f"Destination {dest} already exists, overwriting")
        os.remove(dest)
    logger.info(f"Downloading {url}")
    urllib.request.urlretrieve(url, dest)
    logger.info(f"   ... saved to {dest}")
    return dest


def _zipExtract(zippedfile: Path) -> Path:
    import zipfile
    destFolder = tempfile.mktemp(prefix=zippedfile.name)
    os.mkdir(destFolder)
    with zipfile.ZipFile(zippedfile, 'r') as z:
        z.extractall(destFolder)
    return Path(destFolder)


def _copyFiles(files: List[str], dest: str, sudo=False) -> None:
    assert os.path.isdir(dest)
    if sudo:
        args = ["sudo", "cp"] + files
        args.append(dest)
        subprocess.call(args)
    else:
        for f in files:
            shutil.copy(f, dest)


def pluginsInstalled(force=False) -> bool:
    """Returns True if the needed plugins are already installed"""
    opcodes = set(csoundlib.opcodesList(cached=not force,
                                        opcodedir=csoundlib.userPluginsFolder()))
    neededOpcodes = {"atstop", "pwrite", "pread", "initerror",
                     "dict_new", "dict_set", "dict_get",
                     "pool_gen", "pool_pop", "pool_push", "pool_isfull",
                     'interp1d', 'bisect', 'ftsetparams', 'zeroarray'
                     }
    return neededOpcodes.intersection(opcodes) == neededOpcodes


def _getDownloadsFolder() -> Path:
    downloads = Path.home() / "Downloads"
    assert downloads.exists()
    return downloads


def installPlugins(force=False) -> None:
    """ Install all needed plugins
    """
    if pluginsInstalled() and not force:
        logger.info("Plugins are already installed")
        return
    zipped = downloadLatestPluginForPlatform()
    assert zipped.exists() and zipped.suffix == ".zip"
    unzippedFolder = _zipExtract(zipped)
    pluginsFolder = csoundlib.userPluginsFolder(float64=True)
    os.makedirs(pluginsFolder, exist_ok=True)
    if sys.platform == "linux":
        plugins = [plugin.as_posix() for plugin in unzippedFolder.glob("*.so")]
    elif sys.platform == "darwin":
        plugins = [plugin.as_posix() for plugin in unzippedFolder.glob("*.dylib")]
    elif sys.platform == "win32":
        plugins = [plugin.as_posix() for plugin in unzippedFolder.glob("*.dll")]
    else:
        raise OSError(f"Platform {sys.platform} not supported")
    _copyFiles(plugins, pluginsFolder, sudo=False)
    if not pluginsInstalled(force=True):
        raise RuntimeError("There was an error in the installation...")


def _isofmt(t:datetime) -> str:
    """Returns the time in iso format"""
    return t.isoformat(':', 'minutes')


_state = ConfigDict('csoundengine.state',
    default={
        'last_run': _isofmt(datetime(1900, 1, 1))
    }
)


def _getState() -> ConfigDict:
    return _state


def _checkDependencies(tryfix=False, updateState=True):
    if not csoundInstalled():
        raise RuntimeError("csound not installed. See https://csound.com/download.html")

    #version = csoundlib.getVersion()
    #if version  < (6, 16, 0):
    #    print(f"The installed version of csound ({version}) is too old. ")
    #    print("csound should be >= 6.16")
    #    print("Download the latest version from https://csound.com/download.html")
    #    raise RuntimeError("csound version too old")

    if not pluginsInstalled():
        if tryfix:
            print("csound plugins are not installed. I will try to install them now")
            installPlugins()
        else:
            print("csound plugins are not installed. Install them from "
                  "https://github.com/csound-plugins/csound-plugins/releases")
            raise RuntimeError("csound plugins not installed")
    logger.info("Dependencies OK")
    if updateState:
        state = _getState()
        state['last_run'] = _isofmt(datetime.now())


def checkDependencies(force=True, tryfix=True):
    """
    Check that all external dependencies are fullfilled.

    Args:
        force: if True, do not use cached results
        tryfix: if True, try to fix missing dependencies, where possible

    Raises RuntimeError if any dependency is missing
    """
    # Skip checks if only building docs
    if 'sphinx' in sys.modules:
        return

    if force:
        _checkDependencies(tryfix=tryfix)
        return

    state = _getState()
    timeSinceLastrun = datetime.now() - datetime.fromisoformat(state['last_run'])
    if timeSinceLastrun.days > 30:
        _checkDependencies(tryfix=tryfix)
        return
