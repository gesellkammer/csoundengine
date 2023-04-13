import logging
logging.basicConfig(level="DEBUG")

import csoundengine
e = csoundengine.Engine()
s = e.session()
s.defInstr("sin", r"""
imidi = p5
ifreq = mtof:i(imidi)
print imidi, ifreq
a0 oscili 0.1, mtof:i(imidi)
outch 1, a0
""")

with s.rendering("test1.wav") as r:
    r.sched("sin", 0, 4, args=[60])
