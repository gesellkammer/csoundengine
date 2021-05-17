import os
import sys
import urllib.request, urllib.error
from . import csoundlib
from pathlib import Path
import tempfile
import shutil
import subprocess
from typing import Union as U, List, Dict
import json
import logging


logger = logging.getLogger("csoundengine")


def getPluginsLatestRelease() -> Dict[str, str]:
    """
    Returns a dict with the form {platform: url}

    Where *platform* is the str in *sys.platform* for each of the
    supported platforms

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
    pluginurls = {}
    for asseturl in asseturls:
        asseturl_lower = asseturl.lower()
        if "linux" in asseturl_lower:
            pluginurls['linux'] = asseturl
        elif "macos" in asseturl_lower:
            pluginurls['darwinmacos'] = asseturl
        elif "win64" in asseturl_lower or "windows" in asseturl_lower:
            pluginurls['win32'] = asseturl
    return pluginurls


def isCsoundInstalled() -> bool:
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


def arePluginsInstalled(force=False) -> bool:
    """Returns True if the needed plugins are already installed"""
    opcodes = set(csoundlib.opcodesList(cached=not force,
                                        opcodedir=csoundlib.getUserPluginsFolder()))
    neededOpcodes = {"atstop", "pwrite", "pread", "initerror",
                     "dict_new", "dict_set", "dict_get",
                     "pool_gen", "pool_pop", "pool_push", "pool_isfull"
                     }
    return neededOpcodes.intersection(opcodes) == neededOpcodes


def _getDownloadsFolder() -> Path:
    downloads = Path.home() / "Downloads"
    assert downloads.exists()
    return downloads


def installPlugins(force=False) -> None:
    """ Install all needed plugins
    """
    if arePluginsInstalled() and not force:
        logger.info("Plugins are already installed")
        return
    zipped = downloadLatestPluginForPlatform()
    assert zipped.exists() and zipped.suffix == ".zip"
    unzippedFolder = _zipExtract(zipped)
    pluginsFolder = csoundlib.getUserPluginsFolder(float64=True)
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
    if not arePluginsInstalled(force=True):
        raise RuntimeError("There was an error in the installation...")


def checkDependencies():
    """
    Check that all external dependencies are fullfilled. Raises
    RuntimeError if this is not the case.
    """
    if not isCsoundInstalled():
        raise RuntimeError("csound not installed. See https://csound.com/download.html")

    if not arePluginsInstalled():
        print("csound plugins are not installed. I will try to install them now")
        installPlugins()
