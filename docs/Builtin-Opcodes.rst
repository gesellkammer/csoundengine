================
Built-in Opcodes
================

Whenever an :class:`~csoundengine.engine.Engine` is created, some user-defined opcodes (``UDO``) 
are defined, which can be used from any code within this :class:`~csoundengine.engine.Engine`.

.. contents:: Table of Contents
   :depth: 3
   :local:
   :backlinks: none

.. _busopcodes:

Bus Opcodes
===========


These opcodes are present whenever a csound :class:`~csoundengine.engine.Engine` is created with
``numAudioBuses > 0`` or ``numControlBuses > 0`` (enabled by default). They implement a pool
of audio and control buses. The number of buses in the pool is determined by these variables and
the default can be customized in the configuration (key :ref:`num_audio_buses <config_num_audio_buses>`
and :ref:`num_control_buses <config_num_control_buses>`).

Buses are reference counted: they stay alive as long as there are events using them. As soon
as the last event usign a bus ends the bus is freed and returned to the pool. In order to keep
a bus alive it can be created at the orchestra's header (instr 0) or

An audio bus contains audio from the same performance cycle; all active buses are cleared
at the end of the cycle so they cannot be used to implement feedback or pass audio to
instrument instances with a lower priority. Control buses behave like global k-variables

A bus can be created in python via :meth:`csoundengine.engine.Engine.assignBus` or directly
in csound via :ref:`busassign`.

These user-defined-opcodes are also present for offline rendering

-----


.. _busin:

busin
-----

Receives audio/control data from a bus

**Syntax**

.. code::

    asig busin ibus
    kvalue busin ibus

* ``ibus``: the bus token as returned by :ref:`busassign` or from python by
  :meth:`Engine.assignBus <csoundengine.engine.Engine.assignBus>`


**Example**

.. code-block:: python

    from csoundengine import *
    engine = Engine(numAudioBuses=64)
    engine.compile(r'''
        instr mysynth
          ibus = p4
          kfreq = p5
          asig oscili 0.1, kfreq
          busout ibus, asig
        endin
    
        instr filter
          ibus = p4
          kcutoff = p5
          asig busin ibus
          asig moogladder2 asig, kcutoff, 0.9
          outch 1, asig
        endin
    ''')
    bus = engine.assignBus('audio')
    engine.sched('mysynth', dur=4, args=[bus, 220])
    engine.sched('filter', dur=4, args=[bus, 1000])


-----

.. _busout:

busout
------

Sends audio to a bus or sets a control bus to the given value.
Audio already in the bus is replaced. In order to allow
multiple sends to a bus use :ref:`busmix`.


**Syntax**

.. code::

    busout ibus, asig
    busout ibus, kvalue

* ``asig``: the signal to output
* ``ibus``: the bus token as returned by busassign or from python by :meth:`Engine.assignBus <csoundengine.engine.Engine.assignBus>`

**Example**

.. code-block:: csound

    instr mysynth
      ibus = busassign("a")   ; "a" = audio; "k" = control
      kfreq = p4
      asig oscili 0.1, kfreq
      busout ibus, asig
      schedule "filt", 0, p3, ibus, 2000
    endin

    instr filt
      ibus = p4
      kcutoff = p5
      asig busin ibus
      asig moogladder2 asig, kcutoff, 0.9
      outch 1, asig
    endin

Buses can also be used globally. **NB**: buses are cleared automatically at
the end of a cycle, they do not need to be zeroed by the user. 

.. code-block:: csound

    gimasterL = busassign("a")
    gimasterR = busassign("a")

    instr mysynth
      kfreq = p4
      asig oscili 0.1, kfreq
      busout gimasterL, asig
    endin

    instr 999
      aL busin gimasterL
      aR busin gimasterR
      outch 1, aL, 2, aR
    endin 

    schedule(999, 0, -1)

    
-----

.. _busassign:

busassign
----------

Assigns an unused bus

**Syntax**

.. code::

   ibus busassign Skind

* ``Skind``: the kind of bus, "a" or "k"
* ``ibus``: the bus id of the newly assigned bus. This can be passed to :ref:`busin` or
  :ref:`busout`

.. code-block:: csound

    gimasterL = busassign("a")
    gimasterR = busassign("a")

    instr mysynth
      kfreq = p4
      asig oscili 0.1, kfreq
      busout gimasterL, asig
    endin

    instr 999
      aL busin gimasterL
      aR busin gimasterR
      outch 1, aL, 2, aR
    endin

    schedule(999, 0, -1)


-----


.. _busmix:

busmix
------

Send audio to a bus, mixing it with other sends

The difference with :ref:`busout` is that here the audio of the bus is mixed with
the new audio. **NB**: **busmix** is only available for audio buses

**Syntax**

.. code::

   busmix ibus, asig

* ``ibus``: the bus token as returned by :ref:`busassign` or from python by
  :meth:`Engine.assignBus <csoundengine.engine.Engine.assignBus>`
* ``asig``: the audio signal to mix with the contents of the bus

**Example**

.. code-block:: python

    from csoundengine import *
    e = Engine(numAudioBuses=64)
    e.compile(r'''
      instr vco
        ibus = p4
        ifreq = p5
        asig vco2 0.1, ifreq
        busmix ibus, asig
      endin

      instr group
        ibus = p4
        asig busin ibus
        iatt, irel, ilook = 0.1, 0.2, 0.02       
        asig compress2 asig, asig, -120, -40, -20, /*knee*/3, iatt, irel, ilook
        outch 1, asig
      endin  
    ''')
    bus = e.assignBus("audio")
    freqs = [200, 210, 214]
    for freq in freqs:
        e.sched('vco', dur=4, args=[bus, freq])
    e.sched('master', dur=4, args=[bus])
    
    

Other opcodes
==============

sfloadonce
----------

Load a soundfont if needed

Like `sfload <http://www.csounds.com/manual/html/sfload.html>`_, but
can be used repeatedly. If a soundfont with the given path was already
loaded, it will return the handle number of the loaded instance.


**Syntax**

.. code::

    ihandle sfloadonce "/path/to/soundfont.sf2"


------------------


sfpresetindex
-------------

Assigns an index to a soundfont program

This opcode loads the soundfont if not already loaded (like `sfload`) and assigns an index
(like `sfpassign`) without the user needing to explicitely assign a number.

**Syntax**

.. code::

    ipresetIndex sfpresetindex "/path/to/soundfont.sf2", ibank, ipresetnumber

**Example**

.. code-block:: python

    from csoundengine import *
    e = Engine()
    e.compile(r'''
      instr piano
        ivel, ipitch passign 4
        iamp = ivel/127
        inote = int(ipitch)
        ; assign an index to the program (bank=0, preset=1)
        ipresetidx sfpresetindex "/path/to/piano.sf2", 0, 1
        aL, aR sfplay3 ivel, inote, iamp/16384, mtof:i(ipitch), ipresetidx, 1
        outch 1, aL, 2, aR
      endin
    ''')

.. note::
    There will be a delay when playing a note using this opcode if the soundfont
    is read inside a note for the first time. To avoid this delay, load the 
    soundfont beforehand, via `sfloadonce`. `sfpresetindex` will detect this
    and use the loaded instance (this will not happen with `sfload`).    

**See Also**: :meth:`~csoundengine.engine.Engine.soundfontPreparePreset`, which
does the same operation. 
