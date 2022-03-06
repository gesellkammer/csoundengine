Instr – Instrument Templates
============================

An Instr is a template used to schedule a concrete instrument at a :class:`~csoundengine.session.Session` 
or a :class:`~csoundengine.offline.Renderer`.
It must be registered to be used.

Example
-------

.. code-block:: python

    s = Engine().session()
    Instr('sine', r'''
        kfreq = p5
        kamp = p6
        a0 = oscili:a(kamp, kfreq)
        outch 1, a0
    ''').register(s)
    synth = s.sched('sine', kfreq=440, kamp=0.1)
    ...
    synth.stop()


.. autoclass:: csoundengine.instr.Instr
    :members:

