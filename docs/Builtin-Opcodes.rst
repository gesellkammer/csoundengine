================
Built-in Opcodes
================

Whenever an :class:`~csoundengine.engine.Engine` is created, some user-defined opcodes (``UDO``) 
are defined, which can be used from any code within this :class:`~csoundengine.engine.Engine`.
 

1. Bus Opcodes
==============

These opcodes are present whenever a csound :class:`~csoundengine.engine.Engine` is created with
``numAudioBuses > 0`` (enabled by default). They implement a pool of audio buses. The number of buses in 
the pool is determined by the `numAudioBuses` argument. An audio bus contains audio from the same 
performance cycle; all active buses are cleared at the end of the cycle so they cannot be used to 
implement feedback or pass audio to instrument instances with a lower priority.

A bus can be created in python via :meth:`csoundengine.engine.Engine.assignBus` or directly in csound via ``busassign``

These user-defined-opcodes are also present for offline rendering

-----

busin
-----

Receives audio from a bus

**Syntax**

.. code::

    asig busin ibus

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
    bus = engine.assignBus()
    engine.sched('mysynth', dur=4, args=[bus, 220])
    engine.sched('filter', dur=4, args=[bus, 1000])


-----

busout
------

Sends audio to a bus. Audio already in the bus is replaced. In order to allow
multiple sends to a bus use ``busmix``

**Syntax**

.. code::

    busout ibus, asig

**Example**

.. code-block:: csound

    instr mysynth
      ibus = busassign()
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
      
-----

busassign
----------

Assigns an unused bus

**Syntax**

.. code::

   ibus busassign

-----


busmix
------

Send audio to a bus, mixing it with other sends

**Syntax**

.. code::

   busmix ibus, asig

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
    bus = e.assignBus()
    freqs = [200, 210, 214]
    for freq in freqs:
        e.sched('vco', dur=4, args=[bus, freq])
    e.sched('master', dur=4, args=[bus])
    
    
