Introduction
============

**csoundengine** is a library to run and control a csound process using
its API (via `libcsound <https://github.com/csound-plugins/libcsound>`__).

Engine
------

The core of **csoundengine** is the :class:`~csoundengine.engine.Engine` class.
An :class:`~csoundengine.engine.Engine` wraps a live csound process transparently:
it lets the user compile csound code and schedule real-time events.

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

A csound process is launched by creating a new :class:`~csoundengine.engine.Engine`.
**csoundengine** will query the system regarding audio backend, audio device, number
of channels, samplerate, etc., for any option that is not explicitly given.
For example, in linux **csoundengine** will first check if jack is running (either as
jack itself or within *pipewire*) and, if so, use that as backend, or fallback to using
portaudio otherwise. By default **csoundengine** will use the default audio devices for
the backend and query the number of channels and samplerate to match them, choosing
an appropriate buffer size to match the selected device.

An :class:`~csoundengine.engine.Engine` uses the csound API to communicate with
csound. **All audio processing is run in a thread with realtime priority to avoid
dropouts**.

An :class:`~csoundengine.engine.Engine` has an offline pendant, :class:`~csoundengine.offlineengine.OfflineEngine`,
which has the same interface but renders offline to a soundfile.


Common Tasks
~~~~~~~~~~~~

An :class:`~csoundengine.engine.Engine` provides built-in functionality to
perform common tasks. For example:

* :meth:`~csoundengine.engine.Engine.readSoundfile`: loads a soundfile into a table
* :meth:`~csoundengine.engine.Engine.playSample`: plays a sample from a previously loaded table
* :meth:`~csoundengine.engine.Engine.playSoundFromDisk`: plays an audio file directly from
  disk, without loading the sample data first
* :meth:`~csoundengine.engine.Engine.testAudio`: tests the Engine's output


Modulation / Automation
~~~~~~~~~~~~~~~~~~~~~~~

Within **csoundengine** instruments can declare *pfields* as dynamic values (*k-variables*),
which can be modified, modulated and automated after the event has started. Notice
that in the definition of the 'synth' instrument, ``kmidinote = p4`` or ``kcutoff = p6``
assign a parameter (``p4``, ``p6``) to a control variable.

.. code-block:: python

    # Schedule an event with a unique id
    event = engine.sched("synth", dur=20, args=[48, 0.2, 3000, 4])

    # Change midinote. setp means: set p-field. This sets p4 (kmidinote) to 50
    engine.setp(event, 4, 50)

    # Automate cutoff (p6), from 500 to 2000 hz in 3 seconds, starting in 4 seconds
    # Notice that csoundengine is aware of the assigned variable and the parameter
    # can be adressed by name
    engine.automatep(event, "kcutoff", (0, 500, 3, 2000), delay=4)



----------------------------------


Session (high level interface)
------------------------------

Each Engine can have an associated :class:`~csoundengine.session.Session`. A Session provides a
higher level interface to an existing :class:`Engine`, allowing to:

* Define instrument templates (an :class:`~csoundengine.instr.Instr`), which can be
  instantiated at **any order of evaluation**, allowing to implement **processing chains**
  of any complexity
* Define **named parameters** and **default values**. An :class:`~csoundengine.instr.Instr`
  can use named parameters and assign default values; when an instrument is scheduled,
  only parameters which diverge from the default need to be passed.
* A :class:`~csoundengine.session.Session` provides a series of built-in
  :class:`~csoundengine.instr.Instr`'s to perform some common tasks, like playing
  samples from memory or from disk, perform audio analysis, etc.


.. code-block:: python

    from csoundengine import *

    # When a session is created, the underlying Engine is created as well. The engine
    # is thus created with default values
    session = Session()

    # If the Engine needs to be customized in some way, then the Engine needs to be
    # created first
    session = Engine(nchnls=4, ksmps=32).session()

    # An Engine has only one Session assigned to it. Calling .session() on the engine
    # again will return the same session
    assert session.engine.session() is session

    # Within a Session, instruments can have named parameters and default values
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

    # NB: p4 is reserved, attempting to use it will result in an error
    session.defInstr("master", r'''
      imasterbus = p5
      asig = busin(imasterbus)
      asig compress2 asig, asig, -120, -40, -12, 3, 0.1, 0.01, 0.05
      outch 1, asig
    ''')

    # create a master audio bus
    masterbus = session.assignBus()

    # Start a master instance at the end of the evaluation chain
    master = session.sched("master", imasterbus=masterbus, priority=3)

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



-----------------------------------------------------------

Offline Rendering
-----------------

Offline rendering follows real-time processing closely. Direct access to an offline engine
is provided by the :class:`~csoundengine.offlineengine.OfflineEngine` class. High-level
rendering is implemented via the :class:`~csoundengine.offline.OfflineSession` class,
which has the same interface as a :class:`~csoundengine.session.Session` and
can be used as a drop-in replacement.

.. code-block:: python

    from csoundengine import *
    from pitchtools import *

    renderer = OfflineSession(sr=44100, nchnls=2)

    renderer.defInstr('saw', r'''
      kmidi = p5
      outch 1, oscili:a(0.1, mtof:k(kfreq))
    ''')

    events = [
        renderer.sched('saw', 0, 2, kmidi=ntom('C4')),
        renderer.sched('saw', 1.5, 4, kmidi=ntom('4G')),
        renderer.sched('saw', 1.5, 4, kmidi=ntom('4G+10'))
    ]

    # offline events can be modified just like real-time events
    events[0].automate('kmidi', (0, 0, 2, ntom('B3')), overtake=True)

    events[1].set(delay=3, kmidi=67.2)
    events[2].set(kmidi=80, delay=4)
    renderer.render("out.wav")

A :class:`~csoundengine.offline.OfflineSession` can also be created from an existing :class:`~csoundengine.session.Session`, either via
:meth:`~csoundengine.session.Session.makeRenderer` or via the context manager
:meth:`~csoundengine.session.Session.rendering`. In both cases an
:class:`~csoundengine.offline.OfflineSession` is created in which all instruments and
data defined in the Session are also available.

Taking the first example, the same can be rendered offline by placing this:

.. code-block:: python

    ...

    masterbus = session.assignBus()
    master = session.sched("master", imasterbus=masterbus, priority=3)
    for i, midinote in enumerate(range(60, 72, 2)):
        bus = session.assignBus()
        delay = i
        synth = session.sched("synth", delay=delay, dur=5, kmidi=midinote, ibus=bus)
        synth.automatep('ktransp', [0, 0, dur, -2], delay=delay)
        filt = session.sched("filter", delay=delay, dur=synth.dur,
                             priority=synth.priority+1, kcutoff=2000,
                             ibus=bus,
                             imasterbus=masterbus)
        filt.automatep('kcutoff', [0, 2000, dur*0.8, 500, dur, 6000], delay=start)


inside the ``rendering`` context manager:

.. code-block:: python


    with session.rendering("out.wav") as session:
        masterbus = session.assignBus()
        master = session.sched("master", imasterbus=masterbus, priority=3)
        for i, midinote in enumerate(range(60, 72, 2)):
            bus = session.assignBus()
            delay = i
            synth = session.sched("synth", delay=delay, dur=5, kmidi=midinote, ibus=bus)
            synth.automatep('ktransp', [0, 0, dur, -2], delay=delay)
            filt = session.sched("filter", delay=delay, dur=synth.dur,
                             priority=synth.priority+1, kcutoff=2000,
                             ibus=bus,
                             imasterbus=masterbus)
            filt.automatep('kcutoff', [0, 2000, dur*0.8, 500, dur, 6000], delay=start)


----------------------------


csoundengine vs libcsound / ctcsound
------------------------------------

**csoundengine** uses `libcsound <https://github.com/csound-plugins/libcsound>`__
to interact with csound. **libcsound** (the same applies to ctcsound) follows the csound API
very closely and requires good knowledge of it in order to avoid crashes and provide good
performance. **csoundengine** bundles this knowledge into a wrapper which is flexible for
advanced use cases but enables a casual user to start and control a csound process very easily.


Features
--------

* **Detection of current environment** - *csoundengine* queries the os/hardware to determine the
  system samplerate, hardware number of channels and most appropriate buffer size
* **Named parameters and defaults** - An instrument in **csoundengine** can have named
  parameters and default values. This makes it very easy to create instruments with
  many parameters. When an instance of such an instrument is scheduled **csoundengine**
  fills the values of any parameter which is not explicitely given with the default
  value. Any parg can also be modulated in real-time. See :meth:`Engine.setp() <csoundengine.engine.Engine.setp>`
  and :meth:`Engine.getp() <csoundengine.engine.Engine.getp>`
* **Event ids / Modulation** - in *csoundengine* every event can have a unique id assigned,
  allowing the user to control it during performance, from python or from csound directly.
* **Informed use of the Csound API** - *csoundengine* uses the most convenient part of the
  API for each task (create a table, communicate with a running event, load a soundfile),
  in order to minimize latency and/or increase performance.
* **Automation** - *csoundengine* provides a built-in method to automate the parameters of a
  running event, either via break-point curves or in realtime via any python process.
  See :meth:`Engine.automatep() <csoundengine.engine.Engine.automatep>`,
  :meth:`Engine.setp() <csoundengine.engine.Engine.setp>` or the corresponding
  :class:`~csoundengine.synth.Synth` methods: :meth:`~csoundengine.synth.Synth.set` and
  :meth:`~csoundengine.synth.Synth.automate`
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
