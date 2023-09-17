import csoundengine as ce
import argparse
import logging
logging.basicConfig(level="DEBUG")


# Test offline rendering

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--outfile', default='test1.wav')
args = parser.parse_args()

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
r.render(args.outfile, verbose=True)
