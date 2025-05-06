import csoundengine as ce
from csoundengine import csoundlib, dependencies
import sys
import logging
logging.basicConfig(level="DEBUG")


print("--------------- test2 -----------------", sys.argv)

# Test that all dependencies needed are there
ok = dependencies.checkDependencies(force=True, fix=True)
if not ok:
    print("*************** Some dependencies where not met")
    sys.exit(1)
print(":::::::::::::::: Dependencies ok")

csoundlib.dumpAudioInfo()

# Portaudio
print("Audio Devices for portaudio")
csoundlib.dumpAudioDevices(backend='portaudio')

pa = csoundlib.getAudioBackend('portaudio')
print("Default Audio Devices")
indev, outdev = pa.defaultAudioDevices()
print(f"Default Input device : '{indev}'")
print(f"Default Output device: '{outdev}'")

if not outdev:
    print("---------- No output device")
else:
    print(f"--------- Testing audio for device {outdev}")
    engine = ce.Engine()
    engine.testAudio(dur=4)
    import time
    t0 = time.time()
    while time.time() - t0 < 4:
        print(".")
        time.sleep(0.25)
    print("------- exiting")    
    
