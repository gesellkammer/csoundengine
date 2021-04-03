Synth
=====

A Synth is a wrapper for a unique event scheduled via a Session.

.. note::
 
    A user does NOT normally create a Synth: a Synth is created when an 
    :class:`~csoundengine.instr.Instr` is scheduled in a Session via
    :meth:`~csoundengine.session.Session.sched`


The lifetime of the underlying csound event is not bound to the Synth
object. In order to stop a synth :meth:`~csoundengine.Synth.stop` must 
be called explicitely


Example
-------

.. code::

    from csoundengine import *
    session = Engine().session()
    session.defInstr('vco', r'''
        |kamp=0.1, kmidi=60, ktransp=0|
        asig vco2 kamp, mtof:k(kmidi+ktransp)
        asig *= linsegr:a(0, 0.1, 1, 0.1, 0)
        outch 1, asig
    ''')
    midis = [60, 62, 64]
    synths = [session.sched('vco', kamp=0.2, kmidi=midi) for midi in midis]
    # each sched returns a Synth
    synths[1].automatep('ktransp', [0, 0, 10, -1])


-----

.. automodapi:: csoundengine.synth
