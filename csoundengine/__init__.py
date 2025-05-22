"""
csoundengine
============

csoundengine implements a simple interface to run and control a csound
process.


.. code::

    from csoundengine import Engine
    # create an engine with default options for the platform
    engine = Engine()
    engine.compile('''
      instr synth
        kmidinote = p4
        kamp = p5
        kcutoff = p6
        kfreq = mtof:k(kmidinote)
        asig = vco2:a(kamp, kfreq)
        asig = moogladder2(asig, kcutoff, 0.9)
        aenv = linsegr:a(0, 0.1, 1, 0.1, 0)
        asig *= aenv
        outs asig, asig
      endin
    ''')
    # start a synth with indefinite duration
    event = engine.sched("synth", args=[67, 0.1, 3000])

    # any parameter can be modified afterwords:
    # change midinote
    engine.setp(event, 4, 67)

    # modify cutoff
    engine.setp(event, 6, 1000, delay=4)

    # stop the synth:
    engine.unsched(event)

A higher level layer (:class:`Session`) allows a more ergonomic interface, with features like
named arguments for instruments, default values for arguments, the possibility
to schedule an instrument after or before another instrument to build process chains,
built-in instruments to play a sample from disk/memory, offline rendering, etc.

.. code::

    from csoundengine import *

    # create an Engine and a corresponding Session with default options
    session = Engine().session()

    # create a master audio channel
    session.engine.initChannel("master", kind='a')

    # define instruments
    session.defInstr("synth", r'''
      |ibus, kmidi=60, kamp=0.1, ktransp=0, ifade=0.5|
      asig vco2 kamp, mtof:k(kmidi+ktransp)
      asig *= linsegr:a(0, ifade, 1, ifade, 0)
      busout(ibus, asig)
    ''')

    session.defInstr("filter", r'''
      |ibus, kcutoff=1000, kresonance=0.9|
      asig = busin(ibus)
      asig = moogladder2(asig, kcutoff, kresonance)
      chnmix(asig, "master")
    ''')

    session.defInstr("master", r'''
      asig = chnget:a("master")
      asig compress2 asig, asig, -120, -40, -12, 3, 0.1, 0.01, 0.05
      outch 1, asig
      chnclear("master")
    ''')

    # start a master instance at the end of the evaluation chain
    master = session.sched("master", priority=10)

    dur = 5
    for i, midinote in enumerate(range(60, 72, 2)):
        # for each synth, we create a bus to plug it to an effect, in this case a filter
        bus = session.assignBus()

        # start time for synth and effect
        start = i * 1

        # Schedule a synth
        synth = session.sched("synth", delay=start, dur=5, kmidi=midinote, ibus=bus)

        # Automate the transposition of the pitch so that it goes 2 semitones
        synth.automate('ktransp', [0, 0, dur, -2], delay=start)

        # Schedule the filter for this synth, with a priority higher than the
        # synth, so that it is evaluated later in the chain
        filt = session.sched("filter", delay=start, dur=5, priority=synth.priority+1,
                             kcutoff=2000, kresonance=0.92, ibus=bus)

        # Automate the cutoff freq. of the filter
        filt.automate('kcutoff', [0, 2000, dur*0.8, 500, dur, 6000], delay=start)
"""
# This disables warning about denormals
# import numpy as np

# np.finfo(np.dtype("float32"))
# np.finfo(np.dtype("float64"))

import sys

import emlib.envir

from .config import config, logger
from .engine import Engine
# from .event import Event
from .session import Session
# from .synth import SynthGroup


__all__ = [
    'config',
    'logger',
    'Engine',
    'Session',
    # 'SynthGroup',
    # 'Event',
    'dumpAudioInfo',
    'installDependencies',
]


if 'sphinx' not in sys.modules:
    # Not building docs
    from .dependencies import checkDependencies, installDependencies
    _ok = checkDependencies(force=False, fix=True)
    if not _ok:
        raise RuntimeError("csoundengine: Depencencies not fullfilled")
else:
    print("Building docs")
    from sphinx.ext.autodoc.mock import _MockObject
    libcsound = _MockObject()
    sys.modules['libcsound'] = libcsound


if emlib.envir.inside_ipython() and config['ipython_load_magics_at_startup']:
    from IPython.core.getipython import get_ipython
    ipython = get_ipython()
    if ipython is not None and ipython.extension_manager is not None:
        ipython.extension_manager.load_extension('csoundengine.magic')


def dumpAudioInfo(backend=''):
    """
    Dump information about the audio backend.
    """
    from . import csoundlib
    return csoundlib.dumpAudioInfo(backend=backend)
