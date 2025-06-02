from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from . import tools
from .state import state

logger = logging.getLogger("csoundengine.dependencies")


def _asVersionTriplet(tagname: str) -> tuple[int, int, int]:
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


def getPluginsLatestRelease() -> dict[str, str]:
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
    url = "https://api.github.com/repos/csound-plugins/csound-plugins/releases/latest"
    import urllib.error
    import urllib.request
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


def csoundBinaryInPath() -> bool:
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


def _download(url: str, destFolder: str | Path) -> Path:
    assert os.path.exists(destFolder) and os.path.isdir(destFolder)
    fileName = os.path.split(url)[1]
    dest = Path(destFolder) / fileName
    if dest.exists():
        logger.warning(f"Destination '{dest}' already exists, overwriting")
        os.remove(dest)

    import urllib.request
    urllib.request.urlretrieve(url, dest)
    logger.info(f"Downloaded '{url}', saved to '{dest}'")
    return dest


def _zipExtract(zippedfile: Path) -> Path:
    import zipfile
    destFolder = tempfile.mktemp(prefix=zippedfile.name)
    os.mkdir(destFolder)
    with zipfile.ZipFile(zippedfile, 'r') as z:
        z.extractall(destFolder)
    return Path(destFolder)


def _copyFiles(files: list[str], dest: str, verbose=False) -> None:
    assert os.path.isdir(dest)
    for f in files:
        if verbose:
            print(f"Copying file '{f}' to '{dest}'")
        shutil.copy(f, dest)


def pluginsInstalled(cached=True) -> bool:
    """Returns True if the needed plugins are already installed"""
    from . import csoundlib
    installedOpcodes = csoundlib.installedOpcodes(cached=cached)
    neededOpcodes = {
        "atstop", "pwrite", "pread", "initerror",
        "dict_new", "dict_set", "dict_get",
        "pool_gen", "pool_pop", "pool_push", "pool_isfull",
        'interp1d', 'bisect', 'ftsetparams', 'zeroarray',
        'panstereo', 'poly0'
    }
    return neededOpcodes.issubset(installedOpcodes)


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
        logger.error(f"Plugins not found. Plugins folder: '{pluginspath}', "
                     f"glob patter: '{globpattern}'")
        raise RuntimeError("Plugins not found")
    from . import csoundlib
    pluginsDest = csoundlib.userPluginsFolder(apiversion=f'{apiversion}.0')
    logger.info(f"Installing plugins in folder: '{pluginsDest}'")
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
            logger.error(f"Plugin '{pluginname}' not in risset's index")
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

    if risset:
        logger.info("Installing external plugins via risset")
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
                    from . import csoundlib
                    opcodes = csoundlib.installedOpcodes(cached=False)
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


def _checkDependencies(fix=False, quiet=False) -> str:
    """
    Returns an error message on failure, or an empty string on success
    """
    if not csoundBinaryInPath():
        logger.error("csound not found in the path. See https://csound.com/download.html. Some functionality might not be available")
    from . import csoundlib
    version = csoundlib.getVersion(useApi=True)

    if version < (6, 16, 0):
        return f"Csound version ({version}) is too old, should be >= 6.16"

    if version[0] >= 7:
        logger.debug("Csound 7 support is experimental")

    if not pluginsInstalled():
        if fix:
            print("** csoundengine: Csound external plugins are not installed or are too old."
                  " I will try to install them now")
            ok = installPlugins(version[0])
            if ok:
                print("** csoundengine: csound external plugins installed ok")
            else:
                print("** csoundengine: csound external plugins could not be installed")
                return "csound external plugins could not be installed"
        else:
            return ("Some plugins are not installed. They can be installed via "
                    "'import csoundengine; csoundengine.installDependencies()'. "
                    "To install the plugins manually you will need risset installed. Install them via risset "
                    "(risset install \"*\"), or manually from "
                    "https://github.com/csound-plugins/csound-plugins/releases")
    logger.info("Dependencies OK")
    state['last_check'] = datetime.now().isoformat()
    return ''


def installDependencies() -> bool:
    """
    Install any needed depencendies

    Any problems regarding installation of dependencies will be logged as
    errors

    Returns:
        True if dependencies are installed or were installed successfully, False otherwise
    """
    err = _checkDependencies(fix=True)
    if err:
        logger.error(f"Failed to install dependencies: {err}")
    return not err


def checkDependencies(force=True, fix=False) -> bool:
    """
    Check that all external dependencies are fullfilled.

    Args:
        force: if True, do not use cached results
        fix: if True, try to fix missing dependencies if needed

    Returns:
        True if all dependencies are fullfilled

    """
    # Skip checks if only building docs
    if 'sphinx' in sys.modules:
        logger.debug("Called by sphinx? Skipping dependency check")
        return True

    now = datetime.now()
    timeSinceLastCheck = now - datetime.fromisoformat(state['last_check'])
    if not force and timeSinceLastCheck.days < 30:
        return True
    logger.info("Checking dependencies")
    errormsg = _checkDependencies(fix=fix)
    if errormsg:
        logger.error(f"Failed while checking dependencies: {errormsg}")
        if not fix:
            logger.error("Missing dependencies might be installed by calling installDependencies()")
    return not errormsg


def _codesignBinaries(binaries: list[str]) -> None:
    """
    Calls codesign to sign the binaries with adhoc signature

    Raises RuntimeError on fail
    """
    import risset
    logger.warning(f"Codesigning macos binaries: {binaries}")
    risset.macos_codesign(binaries, signature='-')
