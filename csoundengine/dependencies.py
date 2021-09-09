from __future__ import annotations
import os
import sys
import urllib.request, urllib.error
import re
from . import csoundlib
from .state import state
from pathlib import Path
import tempfile
import shutil
import subprocess
import logging
from datetime import datetime
import json
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *


logger = logging.getLogger("csoundengine")


def _asVersionTriplet(tagname: str) -> Tuple[int, int, int]:
    assert isinstance(tagname, str)
    match = re.search(r"(\d+)\.(\d+)(.(\d+))?", tagname)
    if not match:
        raise ValueError(f"Could not parse tagname {tagname}")
    major = int(match.group(1))
    minor= int(match.group(2))
    try:
        patch = int(match.group(4))
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
        raise RuntimeError("Could not download plugins info")
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
    tagname = info.get('tag_name')
    if tagname:
        try:
            versiontriplet = _asVersionTriplet(tagname)
        except ValueError:
            logger.error(f"Could not parse tagname: {tagname}")
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

    NB: throws RuntimeError if could not retrieve latest release
    """
    if destFolder is None:
        destFolder = _getDownloadsFolder()
    pluginurls = getPluginsLatestRelease()
    pluginurl = pluginurls.get(sys.platform)
    if pluginurl is None:
        raise RuntimeError(f"No plugins released for platform {sys.platform}")
    version = pluginurls.get('version', None)
    if version:
        print(f"Downloading latest version ({version}) of csound-plugins...")
    return _download(pluginurl, destFolder)


def _download(url: str, destFolder: Union[str, Path]) -> Path:
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


def _copyFiles(files: List[str], dest: str, verbose=False) -> None:
    assert os.path.isdir(dest)
    for f in files:
        if verbose:
            print(f"Copying file {f} to {dest}")
        shutil.copy(f, dest)


def pluginsInstalled(cached=True) -> bool:
    """Returns True if the needed plugins are already installed"""
    opcodes = set(csoundlib.opcodesList(cached=cached,
                                        opcodedir=csoundlib.userPluginsFolder()))
    neededOpcodes = {
        "atstop", "pwrite", "pread", "initerror",
        "dict_new", "dict_set", "dict_get",
        "pool_gen", "pool_pop", "pool_push", "pool_isfull",
        'interp1d', 'bisect', 'ftsetparams', 'zeroarray',
        'panstereo'
    }
    return neededOpcodes.intersection(opcodes) == neededOpcodes


def _getDownloadsFolder() -> Path:
    downloads = Path.home() / "Downloads"
    assert downloads.exists()
    return downloads


def _installPluginsFromZipFile(zipped: Path):
    """
    install plugins from a zipped file downloaded from github release

    NB: throws RuntimeError if there was an error during installation
    """
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
    _copyFiles(plugins, pluginsFolder, verbose=True)
    if not pluginsInstalled(cached=False):
        raise RuntimeError("There was an error in the installation...")


def _installPluginsFromDist():
    rootfolder = Path(os.path.split(__file__)[0]).parent
    assert rootfolder.exists()
    subfolder, globpattern = {
        'darwin': ('macos', '*.dylib'),
        'windows': ('windows', '*.dll'),
        'linux': ('linux', '*.so')
    }.get(sys.platform, (None, None))
    if subfolder is None:
        raise RuntimeError(f"Platform {sys.platform} not supported")
    pluginspath = rootfolder/'data/plugins'/subfolder
    if not pluginspath.exists():
        raise RuntimeError(f"Could not find own csound plugins. Folder: {pluginspath}")
    plugins = list(pluginspath.glob(globpattern))
    if not plugins:
        logger.error(f"Plugins not found. Plugins folder: {pluginspath}, "
                     f"glob patter: {globpattern}")
        raise RuntimeError("Plugins not found")
    pluginsDest = csoundlib.userPluginsFolder()
    os.makedirs(pluginsDest, exist_ok=True)
    _copyFiles([plugin.as_posix() for plugin in plugins], pluginsDest, verbose=True)
    if not pluginsInstalled(cached=False):
        raise RuntimeError("There was an error in the installation")


def installPlugins() -> None:
    """
    Install all needed plugins

    Will raise RuntimeError if failed
    """
    if pluginsInstalled():
        logger.info("Plugins are already installed, installed plugins will be "
                    "(eventually) overwritten")
    try:
        zipped = downloadLatestPluginForPlatform()
        _installPluginsFromZipFile(zipped)
        return
    except RuntimeError as e:
        logger.error(f"Could not install plugins from github: {e}")
    logger.info("Installing plugins from distribution")
    _installPluginsFromDist()
    logger.info("<<< Plugins installed successfully! >>>")


def _checkDependencies(tryfix=False, updateState=True) -> Optional[str]:
    """
    Either returns None or an error message
    """
    if not csoundInstalled():
        return "csound not installed. See https://csound.com/download.html"

    version = csoundlib.getVersion()
    if version  < (6, 16, 0):
        return f"Csound version ({version}) is too old, should be >= 6.16"

    if not pluginsInstalled():
        if tryfix:
            print("csound plugins are not installed or are too old."
                  " I will try to install them now")
            installPlugins()
        else:
            return ("csound plugins are not installed. Install them from "
                    "https://github.com/csound-plugins/csound-plugins/releases")
    logger.info("Dependencies OK")
    if updateState:
        state['last_run'] = datetime.now().isoformat()


def checkDependencies(force=True, tryfix=False) -> bool:
    """
    Check that all external dependencies are fullfilled.

    Args:
        force: if True, do not use cached results
        tryfix: if True, try to fix missing dependencies, where possible

    Returns:
        True if all dependencies are fullfilled

    """
    # Skip checks if only building docs
    if 'sphinx' in sys.modules:
        return True

    timeSincelast_run = datetime.now() - datetime.fromisoformat(state['last_run'])
    if force or timeSincelast_run.days > 30:
        logger.info("Checking dependencies")
        errormsg = _checkDependencies(tryfix=tryfix)
        if errormsg:
            logger.error(errormsg)
            return False
    return True
