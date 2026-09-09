"""
Check that all dependencies needed by csoundengine (csound and its plugins)
are available even if csound is not preinstalled in the system.

csoundengine does not use the csound binary. The csound library comes
with the pip package ``libcsound``, which installs csound (and the needed
plugins) on demand the first time it is imported if no csound
installation is found.

This test is meant to be run in a clean environment where csound is not
installed (see the ``testdeps`` job in .github/workflows/test2.yml)
"""
import sys
import shutil
import platform
import logging

logging.basicConfig(level=logging.DEBUG,
                    format="%(levelname)s %(name)s: %(message)s")

print("python  :", sys.version)
print("platform:", platform.platform())
print("machine :", platform.machine())

csoundbin = shutil.which("csound")
print(f"csound binary in path: {csoundbin or '<not found>'}")
if csoundbin:
    print("Warning: csound is preinstalled, the auto-installation of csound"
          " by libcsound will not be exercised")

# Importing csoundengine imports libcsound. If no csound installation is
# found, libcsound installs csound (together with its plugins) automatically
import csoundengine  # noqa: F401
from csoundengine import dependencies

ok = dependencies.checkDependencies(force=True, fix=True)
pluginsok = dependencies.pluginsInstalled(cached=False)
print(f"Dependencies ok: {ok}")
print(f"Plugins ok: {pluginsok}")
if not (ok and pluginsok):
    print("******************* Some dependencies are not met")
    sys.exit(1)

from csoundengine import csoundlib
print("csound version:", csoundlib.getVersion())
print(":::::::::::::::: All dependencies are installed")
