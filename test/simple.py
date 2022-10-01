import logging
logging.basicConfig(level="DEBUG")
import csoundengine
e = csoundengine.Engine()
s = e.session()
s.defInstr("foo", r"""
imidi = p5
ifreq = mtof:i(imidi)
print imidi, ifreq
a0 oscili 0.1, mtof:i(imidi)
outch 1, a0
""")
import time
s.sched("foo", 0, 4, args=[60])
time.sleep(4)
