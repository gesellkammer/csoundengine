import csoundengine as ce
from csoundengine import dependencies
import sys
import logging
logging.basicConfig(level="DEBUG")

# Test that all opcodes needed are there
ok = dependencies.pluginsInstalled(cached=False)
if not ok:
    print("Some plugins with needed opcodes don't seem to be installed")
    sys.exit(1)