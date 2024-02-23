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

ce.csoundlib.dumpAudioBackends()

# Portaudio
print("Audio Devices for portaudio")
ce.csoundlib.dumpAudioDevices(backend='portaudio')

pa = ce.csoundlib.getAudioBackend('portaudio')
print("Default Audio Devices")
indev, outdev = pa.defaultAudioDevices()
print(indev)
print(outdev)
