import csoundengine as ce

s = ce.Session()

s.defInstr('foo', r'''
|iarg1, iarg2, iarg3, iarg4=10, kval=20|
prints "iarg1: %f. iarg2: %f, iarg3: %f, iarg4: %f\n", iarg1, iarg2, iarg3, iarg4
if eventcycles() == 0 then
    println "kval: %f", kval
endif
turnoff
''')

s.sched('foo', 0, 1, args=dict(iarg1=100, iarg2=101.5, iarg3=102.))
s.sched('foo', 0, 1, iarg1=100, iarg2=101.5, iarg3=102.)


import time
time.sleep(1)
