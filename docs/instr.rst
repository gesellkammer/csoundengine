Instr – Instrument Templates
============================

An :class:`~csoundengine.instr.Instr` is a template used to schedule a concrete instrument at
a :class:`~csoundengine.session.Session` or a :class:`~csoundengine.offlineengine.OfflineEngine`. It must
be registered to be used (see :meth:`~csoundengine.session.Session.registerInstr`) or created
via :meth:`~csoundengine.session.Session.defInstr`.

**Example**

.. code-block:: python

    from csoundengine import *

    s = Session()

    s.defInstr('sine', r'''
        kfreq = p5
        kamp = p6
        a0 = oscili:a(kamp, kfreq)
        outch 1, a0
    ''')

    synth = s.sched('sine', kfreq=440, kamp=0.1)
    ...
    synth.stop()


**Named arguments / Inline arguments**

An :class:`Instr` can define named arguments and assign default values to any argument.
Named arguments can be created either by using the supported csound syntax, using
``ivar = p5`` or ``kvar = p5``. Default values can be assigned via ``pset`` (https://csound.com/manual/pset.html)
or through the ``args`` argument to :class:`Instr` or :meth:`~csoundengine.session.Session.defInstr`.
Alternatively inline arguments can be used:

An inline args declaration can set both parameter name and default value:

.. code-block:: python

    s = Engine().session()
    Instr('sine', r'''
        |iamp=0.1, kfreq=1000|
        a0 = oscili:a(iamp, kfreq)
        outch 1, a0
    ''').register(s)
    synth = s.sched('sine', kfreq=440)
    synth.stop()



.. autoclass:: csoundengine.instr.Instr
    :members:
    :autosummary:

