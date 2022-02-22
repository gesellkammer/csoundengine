Introduction 
============

**csoundengine** is a library to run and control a csound process using
its API (via `ctcsound <https://csound.com/docs/ctcsound/>`_).

Engine
------

The core of **csoundengine** is the :class:`~csoundengine.engine.Engine` class,
which wraps a csound process and allows transparent control over all parameters,
while providing sane defaults adapted to the running system. It uses the csound
API to communicate with csound. All audio processing is run in a thread with
realtime priority to avoid dropouts.

A csound process is launched by creating a new Engine. If not given any options,
the current system is queried regarding number of channels, samplerate or buffer size,
most appropriate audio backend, etc.

.. code-block:: python

    from csoundengine import *
    # create an Engine with default/detected options for the platform.
    engine = Engine()
    
    # Define an instrument
    engine.compile('''
      instr synth
        ; pfields of the instrument
        kmidinote = p4
        kamp = p5
        kcutoff = p6
        kdetune = p7

        kfreq = mtof:k(kmidinote)
        ; A filtered sawtooth
        asig  = vco2:a(kamp*0.7, kfreq)
        asig += vco2.a(kamp*0.7, kfreq + kdetune)
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

    # Modify cutoff (p6)
    engine.setp(event, 6, 1000, delay=4)

    # Stop the synth
    engine.unsched(event)


----------------------------------


Session (high level interface)
------------------------------

Each Engine has an associated :class:`~csoundengine.session.Session`. A Session provides a
higher level interface, allowing to:

* Define instrument templates (an :class:`~csoundengine.instr.Instr`), which can be
  instantiated at any order of evaluation, allowing to implement processing chains
  of any complexity
* An :class:`~csoundengine.instr.Instr` can have named parameters which can be
  used to control the scheduled event.
* A :class:`~csoundengine.session.Session` provides a series of built-in
  :class:`~csoundengine.instr.Instr`'s to perform some common tasks, like playing
  samples from memory or from disk, perform audio analysis, etc.


.. code-block:: python
    
    from csoundengine import *

    # Create an Engine and a corresponding Session. It is possible to be specific about
    # Engine parameters.
    session = Engine(nchnls=4, ksmps=32).session()

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

    # create a master audio channel
    masterbus = session.assignBus()

    # Start a master instance at the end of the evaluation chain
    master = session.sched("master", imasterbus=masterbus, priority=10)

    # Launch some notes
    for i, midinote in enumerate(range(60, 72, 2)):
        # for each synth, we create a bus to plug it to an effect, in this case a filter
        bus = session.assignBus()

        delay = i
        
        # Schedule a synth
        synth = session.sched("synth", delay=delay, dur=5, kmidi=midinote, ibus=bus)
        
        # Automate pitch transposition so that it descends 2 semitones over the
        # duration of the event
        synth.automatep('ktransp', [0, 0, dur, -2], delay=delay)
        
        # Schedule the filter for this synth, with a priority higher than the
        # synth, so that it is evaluated later in the chain
        filt = session.sched("filter", 
                             delay=delay,
                             dur=synth.dur, 
                             priority=synth.priority+1,
                             kcutoff=2000, 
                             kresonance=0.92, 
                             ibus=bus, 
                             imasterbus=masterbus)
        
        # Automate the cutoff freq. of the filter, so that it starts at 2000 Hz,
        # it drops to 500 Hz by 80% of the note and goes up to 6000 Hz at the end
        filt.automatep('kcutoff', [0, 2000, dur*0.8, 500, dur, 6000], delay=start) 


csoundengine vs ctcsound
------------------------

**csoundengine** uses `ctcsound <https://github.com/csound/csound/blob/master/interfaces/ctcsound.py>`_
to interact with csound. **ctcsound** follows the csound API very closely and requires good knowledge
of it in order to avoid crashes and provide good performance. **csoundengine** bundles
this knowledge into a wrapper which is flexible for advanced use cases but enables a casual
user to start and control a csound process very easily. See below for a detailed description of
*csoundengine* ´s features

Features
--------

* **Detection of current environment** - *csoundengine* queries the os/hardware to determine the
  system samplerate, hardware number of channels and most appropriate buffer size
* **Named parameters and defaults** - An instrument in **csoundengine** can have named
  parameters and default values. This makes it very easy to create instruments with
  many parameters. When an instance of such an instrument is scheduled **csoundengine**
  fills the values of any parameter which is not explicitely given with the default
  value. Any parg can also be modulated in real-time. See :meth:`Engine.setp() <csoundengine.engine.Engine.setp>`
  and :meth:`Engine.setp() <csoundengine.engine.Engine.getp>`
* **Event ids / Modulation** - in *csoundengine* every event is assigned a unique id, allowing the user
  to control it during performance, from python or from csound directly.
* **Informed use of the Csound API** - *csoundengine* uses the most convenient part of the
  API for each task (create a table, communicate with a running event, load a soundfile),
  in order to minimize latency and/or increase performance.
* **Automation** - *csoundengine* provides a built-in method to automate the parameters of a
  running event, either via break-point curves or in realtime via any python process.
  See :meth:`Engine.automatep() <csoundengine.engine.Engine.automatep>` or
  :meth:`Engine.setp() <csoundengine.engine.Engine.setp>`
* **Bus system** - an :class:`~csoundengine.engine.Engine` provides a bus system (both for
  audio and control values) to make communication between running events much easier. See
  :meth:`~csoundengine.engine.Engine.assignBus` and :ref:`Bus opcodes<busopcodes>`
* **Jupyter notebook** - When used inside a jupyter notebook *csoundengine* generates customized
  html output and interactive widgets. For any scheduled event *csoundengine*
  can generate an interactive UI to control its parameters in realtime. It also provides
  %magic routines to compile csound code and interact with a running *Engine*.
  See :ref:`Inside Jupyter<jupyternotebook>`
* **Processing chains** - An instrument defined in a Session can be scheduled at any
  point within a processing chain, making instrument definitions more modular and reusable
* **Built-in functions** - Any Engine / Session has built-in functionality for soundfile/sample
  playback, loading sf2/sf3 soundfonts, jsfx effects, audio analysis, etc.



