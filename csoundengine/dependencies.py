from __future__ import annotations
import os
import sys
import urllib.request, urllib.error
import re
from . import csoundlib
from . import tools
from .state import state
from pathlib import Path
import tempfile
import shutil
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


def downloadLatestPluginForPlatform(destFolder: Path | None = None) -> Path:
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


def pluginsInstalled(cached: bool) -> bool:
    """Returns True if the needed plugins are already installed"""
    #opcodes = set(csoundlib.opcodesList(cached=cached,
    #                                    opcodedir=csoundlib.userPluginsFolder(apiversion=apiversion)))
    opcodes = set(csoundlib.opcodesList(cached=cached))
    neededOpcodes = {
        "atstop", "pwrite", "pread", "initerror",
        "dict_new", "dict_set", "dict_get",
        "pool_gen", "pool_pop", "pool_push", "pool_isfull",
        'interp1d', 'bisect', 'ftsetparams', 'zeroarray',
        'panstereo', 'poly0'
    }
    return neededOpcodes.intersection(opcodes) == neededOpcodes


def _getDownloadsFolder() -> Path:
    downloads = Path.home() / "Downloads"
    if downloads.exists():
        return downloads
    tempdir = Path(tempfile.gettempdir())
    logger.warning(f"Downloads folder {downloads} not found, using temp dir: {tempdir}")
    return tempdir


def _installPluginsFromDist(apiversion=6, codesign=True) -> None:
    platformid = tools.platformId()
    rootfolder = Path(os.path.split(__file__)[0])
    assert rootfolder.exists()
    globpattern = {
        'macos': '*.dylib',
        'windows': '*.dll',
        'linux': '*.so'
    }.get(platformid.osname, None)
    if globpattern is None:
        raise RuntimeError(f"Platform {platformid} not supported")

    subfolder = str(platformid)
    pluginspath = rootfolder/f'data/plugins{apiversion}'/subfolder
    if not pluginspath.exists():
        raise RuntimeError(f"Could not find own csound plugins. Folder: {pluginspath}")
    plugins = list(pluginspath.glob(globpattern))
    if not plugins:
        logger.error(f"Plugins not found. Plugins folder: {pluginspath}, "
                     f"glob patter: {globpattern}")
        raise RuntimeError("Plugins not found")
    pluginsDest = csoundlib.userPluginsFolder(apiversion=f'{apiversion}.0')
    logger.info(f"Installing plugins in folder: {pluginsDest}")
    os.makedirs(pluginsDest, exist_ok=True)
    _copyFiles([plugin.as_posix() for plugin in plugins], pluginsDest, verbose=True)
    if platformid.osname == 'macos' and codesign:
        installedBinaries = [os.path.join(pluginsDest, plugin.name)
                             for plugin in plugins]
        assert all(os.path.exists(binary) for binary in installedBinaries)
        try:
            _codesignBinaries(installedBinaries)
        except RuntimeError as e:
            logger.error(f"Could not code-sign the binaries, error: {e}")
            if platformid.arch == 'arm64':
                logger.error(f"... The needed plugins will probably not work as is. You can still "
                             f"manually authorize them via right-click. The paths are: {installedBinaries}")


def _installPluginsViaRisset(majorversion: int | None = None) -> bool:
    """
    Tries to install plugins via risset

    Does not raise anything itself, but risset might
    """
    logger.info("Trying to install plugins via risset")
    import risset
    idx = risset.MainIndex(update=True, majorversion=majorversion)
    for pluginname in ['else', 'beosc', 'klib', 'poly']:
        p = idx.plugins.get(pluginname)
        if p is None:
            logger.error(f"Plugin '{pluginname}' not found in risset's index")
            return False
        elif idx.is_plugin_installed(p):
            logger.debug(f"Plugin '{pluginname}' already installed, skipping")
        else:
            errmsg = idx.install_plugin(p)
            if errmsg:
                logger.error(f"Error while installing plugin {pluginname}: {errmsg}")
                return False
    return True


def installPlugins(majorversion=6, risset=True) -> bool:
    """
    Install all needed plugins

    Will raise RuntimeError if failed

    Args:
        majorversion: the csound version for which to install plugins. If None,
            will detect the installed version and use that
        risset: if True, install plugins via risset (default). Otherwise, uses
            the bundled plugins

    Returns:
        True if installation succeeded. Any errors are logged
    """
    logger.info("Installing external plugins via risset")

    if risset:
        try:
            ok = _installPluginsViaRisset()
            pluginsok = pluginsInstalled(cached=False)
            if ok and pluginsok:
                logger.info("Plugins installed successfully via risset")
                return True
            else:
                logger.error("Could not install plugins via risset")
                if not pluginsok:
                    logger.error("Tried to load the plugins but the provided opcodes are not"
                                 " listed by csound")
                    opcodes = csoundlib.opcodesList(cached=False)
                    opcodestr = ', '.join(opcodes)
                    logger.error(f"List of opcodes loaded by csound: {opcodestr}")
        except Exception as e:
            logger.error(f"Exception {e} while trying to install plugins via risset")

    logger.info("Installing plugins from distribution")
    try:
        _installPluginsFromDist(apiversion=majorversion)
        ok = pluginsInstalled(cached=False)
        if ok:
            logger.info("Plugins installed successfully from distribution")
        else:
            logger.error("Plugins where installed but do not seem to be detected")
    except Exception as e:
        logger.error(f"Exception {e} while trying to install plugins from distribution")
        return False
    return True


def _checkDependencies(fix=False, updateState=True, quiet=False) -> Optional[str]:
    """
    Either returns None or an error message
    """
    if not csoundInstalled():
        return "csound not installed. See https://csound.com/download.html"

    version = csoundlib.getVersion(useApi=True)
    if version  < (6, 16, 0):
        return f"Csound version ({version}) is too old, should be >= 6.16"

    if version[0] >= 7:
        print(f"WARNING: Csound 7 is not fully supported. Proceed at yout own risk")

    binversion = csoundlib.getVersion(useApi=False)
    if version[:2] != binversion[:2]:
        print(f"WARNING: the csound library found reported a version {version}, different"
              f" from the version reported by the csound binary {binversion}")

    if not pluginsInstalled(cached=False):
        if fix:
            if not quiet:
                print("** csoundengine: Csound external plugins are not installed or are too old."
                      " I will try to install them now")
            ok = installPlugins(version[0])
            if ok:
                if not quiet:
                    print("** csoundengine: csound external plugins installed ok")
            else:
                if not quiet:
                    print("** csoundengine: csound external plugins could not be installed")
                return "csound external plugins could not be installed"
        else:
            return ("Some plugins are not installed. They can be installed via 'import csoundengine; csoundengine.installDependencies()'. "
                    "To install the plugins manually you will need risset installed Install them via risset "
                    "(risset install \"*\"), or manually from "
                    "https://github.com/csound-plugins/csound-plugins/releases")
    logger.info("Dependencies OK")
    if updateState:
        state['last_run'] = datetime.now().isoformat()


def installDependencies() -> bool:
    """
    Install any needed depencendies

    Any problems regarding installation of dependencies will be logged as
    errors

    Returns:
        True if dependencies are installed or were installed successfully, False otherwise
    """
    return checkDependencies(force=True, fix=True)


def checkDependencies(force=False, fix=True) -> bool:
    """
    Check that all external dependencies are fullfilled.

    Args:
        force: if True, do not use cached results
        fix: if True, try to fix missing dependencies, where possible

    Returns:
        True if all dependencies are fullfilled

    """
    # Skip checks if only building docs
    if 'sphinx' in sys.modules:
        logger.info("Called by sphinx? Skipping dependency check")
        return True

    timeSincelast_run = datetime.now() - datetime.fromisoformat(state['last_run'])
    if force or timeSincelast_run.days > 30:
        logger.warning("Checking dependencies")
        errormsg = _checkDependencies(fix=fix)
        if errormsg:
            logger.error(f"*** checkDependencies: {errormsg}")
            if not fix:
                logger.error("*** You can try to fix this by calling installDependencies()")
            return False
    return True


def _codesignBinaries(binaries: list[str]) -> None:
    """
    Calls codesign to sign the binaries with adhoc signature

    Raises RuntimeError on fail
    """
    import risset
    logger.info(f"Codesigning macos binaries: {binaries}")
    risset.macos_codesign(binaries, signature='-')
