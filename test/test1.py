import csoundengine as ce

import logging
logging.basicConfig(level="DEBUG")

r = ce.Renderer()
r.defInstr("sin", r"""
imidi = p5
ifreq = mtof:i(imidi)
print imidi, ifreq
a0 oscili 0.1, mtof:i(imidi)
a0 *= linsegr:a(0, 0.1, 1, 0.1, 0)
outch 1, a0
""")


r.sched("sin", 0, 4, args=[60])
r.sched("sin", 1, 3.5, args=[60.5])
r.writeCsd("test1.csd")
r.render('test1.wav')
