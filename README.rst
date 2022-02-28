csoundengine
============

This package implements an intuitive interface to run and control a csound process


Documentation
-------------

https://csoundengine.readthedocs.io/en/latest/index.html


-----


Introduction 
------------

The core of this package is the ``Engine`` class, which wraps a csound
process and allows transparent control over all parameters, while providing 
sane defaults. It uses the csound API to communicate with a running csound
instance. All audio processing is run in a separate performance thread.


.. code-block:: python

    from csoundengine import *
    # create an Engine with default options for the platform.
    engine = Engine()
    
    # Define an instrument
    engine.compile(r'''
      instr synth
        ; pfields of the instrument
        kmidinote = p4
        kamp = p5
        kcutoff = p6
        kdetune = p7

        kfreq = mtof:k(kmidinote)
        ; A filtered sawtooth
        asig  = vco2:a(kamp*0.7, kfreq)
        asig += vco2:a(kamp*0.7, kfreq + kdetune)
        asig = moogladder2(asig, kcutoff, 0.9)
        ; Attack / Release
        aenv = linsegr:a(0, 0.01, 1, 0.2, 0)
        asig *= aenv
        outs asig, asig
      endin
    ''')

    # Start a synth with indefinite duration. This returns the eventid (p1)
    # of the running instrument, which can be used to further control it
    event = engine.sched("synth", args=[48, 0.2, 3000, 4])

    # Change midinote. setp means: set p-field. This sets p4 (kmidinote) to 50
    engine.setp(event, 4, 50)

    # Modify cutoff
    engine.setp(event, 6, 1000, delay=4)

    # Create a ui for this event:
    engine.eventui(event, p4=(0, 127), p5=(0, 1), kcutoff=(100, 5000))


.. figure:: https://raw.githubusercontent.com/gesellkammer/csoundengine/master/docs/assets/eventui2.png



Session - high level interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each engine can have an associated ``Session``. A Session provides a
higher level interface, allowing to:

* Define instrument templates (an ``Instr``), which can be instantiated at any order of evaluation, allowing to implement processing chains of any complexity

* An ``Instr`` can have named parameters which can be used to control the event.

* A ``Session`` provides a series of built-in ``Instr``'s to perform some common tasks, like playing samples from memory or from disk, perform audio analysis, etc.


.. figure:: https://raw.githubusercontent.com/gesellkammer/csoundengine/master/docs/assets/synthui.png


.. code-block:: python
    
    from csoundengine import *

    # Create an Engine and a corresponding Session using default options
    session = Engine().session()

    # create a master audio bus
    masterbus = session.assignBus()

    # define instruments
    session.defInstr("synth", r'''
      |ibus, kmidi=60, kamp=0.1, ktransp=0, ifade=0.5|
      ; a simple sawtooth
      asig vco2 kamp, mtof:k(kmidi+ktransp)
      asig *= linsegr:a(0, ifade, 1, ifade, 0)
      ; output is routed to a bus
      busout(ibus, asig)
    ''')

    session.defInstr("filter", r'''
      |ibus, imasterbus, kcutoff=1000, kresonance=0.9|
      asig = busin(ibus)
      asig = moogladder2(asig, kcutoff, kresonance)
      busmix(imasterbus, asig)
    ''')

    session.defInstr("master", r'''
      imasterbus = p4
      asig = busin(imasterbus)
      asig compress2 asig, asig, -120, -40, -12, 3, 0.1, 0.01, 0.05
      outch 1, asig
    ''')

    # Start a master instance at the end of the evaluation chain
    master = session.sched("master", imasterbus=masterbus, priority=10)

    # Launch some notes
    for i, midinote in enumerate(range(60, 72, 2)):
        # for each synth, we create a bus to plug it to an effect, in this case a filter
        # The bus will be collected once all clients are finished
        bus = session.assignBus()
        
        # start time for synth and effect
        start = i * 1
        
        # Schedule a synth
        synth = session.sched("synth", delay=start, dur=5, kmidi=midinote, ibus=bus)
        
        # Automate pitch transposition so that it descends 2 semitones over the
        # duration of the event
        synth.automatep('ktransp', [0, 0, dur, -2], delay=start)
        
        # Schedule the filter for this synth, with a priority higher than the
        # synth, so that it is evaluated later in the chain
        filt = session.sched("filter", 
                             delay=start, 
                             dur=synth.dur, 
                             priority=synth.priority+1,
                             kcutoff=2000, 
                             kresonance=0.92, 
                             ibus=bus, 
                             imasterbus=masterbus)
        
        # Automate the cutoff freq. of the filter
        filt.automatep('kcutoff', [0, 2000, dur*0.8, 500, dur, 6000], delay=start) 


-----

Installation
------------

https://csoundengine.readthedocs.io/en/latest/Installation.html

Dependencies
~~~~~~~~~~~~

* python >= 3.8
* csound 6 >= 6.16 (https://github.com/csound/csound/releases). 


.. note:: 

	csound 7 is not supported at the moment


.. code-block:: bash

    pip install csoundengine

**csoundengine** also needs many csound plugins (https://github.com/csound-plugins/csound-plugins/releases),
but these are installed automatically if needed.


Documentation
-------------

https://csoundengine.readthedocs.io
