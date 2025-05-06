import time
t0 = time.time()
from csoundengine import *
t1 = time.time()
print(f"{(t1 - t0) * 1000} ms")
