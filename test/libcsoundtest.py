import libcsound 

cs = libcsound.Csound()
cs.setOption('-+rtaudio=jack')
cs.setOption('-odac0')
cs.setOption('-B1280')
cs.setOption('-b256')
cs.compileOrc(r'''
sr = 48000
0dbfs = 1
nchnls = 1
ksmps = 64
''')
cs.start()
pt = cs.performanceThread()
pt.play()
input("\nkey...")
