import csoundengine as ce
from csoundengine import dependencies
import sys
import logging
logging.basicConfig(level="DEBUG")

# Test that all dependencies needed are there
ok = dependencies.checkDependencies(force=True, fix=True)
if not ok:
    print("*************** Some dependencies where not met")
    sys.exit(1)
print(":::::::::::::::: Dependencies ok")