.. _offlinemod:

Offline Rendering
=================

Offline rendering follows real-time processing very closely. The :class:`~csoundengine.offlineengine.OfflineEngine`
class represents a csound process running in offline mode and is an almost drop-in replacement to the
:class:`~csoundengine.engine.Engine` class.

High-level rendering is implemented via the :class:`~csoundengine.offline.OfflineSession` class,
which has the same interface as a :class:`~csoundengine.session.Session`


Example low-level API
---------------------

.. include:: offlineengine_example.inc


Example (high-level API)
------------------------

.. code-block:: python

    from csoundengine import *
    from pitchtools import *

    session = OfflineSession(sr=44100, nchnls=2)

    session.defInstr('saw', r'''
      kmidi = p5
      outch 1, oscili:a(0.1, mtof:k(kfreq))
    ''')

    events = [
        session.sched('saw', 0, 2, kmidi=ntom('C4')),
        session.sched('saw', 1.5, 4, kmidi=ntom('4G')),
        session.sched('saw', 1.5, 4, kmidi=ntom('4G+10'))
    ]

    # offline events can be automated just like real-time events
    events[0].automate('kmidi', (0, 0, 2, ntom('B3')), overtake=True)

    events[1].set(delay=3, kmidi=67.2)
    events[2].set(kmidi=80, delay=4)
    session.render("out.wav")


It is possible to create an :class:`~csoundengine.offline.OfflineSession` out of an existing
:class:`~csoundengine.session.Session`, by calling :meth:`session.makeRenderer <csoundengine.session.Session.makeRenderer>`.
This creates an :class:`~csoundengine.offline.OfflineSession` with all :class:`~csoundengine.instr.Instr` and resources
(tables, include files, global code, etc.) in the :class:`~csoundengine.session.Session` already defined.

.. code-block:: python

    from csoundengine import *
    session = Session()
    session.defInstr('test', ...)
    table = session.readSoundfile('path/to/soundfile')
    session.sched('test', ...)
    session.playSample(table)

    # Render offline
    renderer = session.makeRenderer()
    renderer.sched('test', ...)
    renderer.playSample('path/to/soundfile')
    renderer.render('out.wav')

An alternative way to render offline given a live :class:`~csoundengine.session.Session` is to use the
:meth:`~csoundengine.session.Session.rendering` method:

.. code-block:: python

    from csoundengine import *
    session = Session()
    session.defInstr('test', ...)
    table = session.readSoundfile('path/to/soundfile')

    with session.rendering('out.wav') as r:
        r.sched('test', ...)
        r.playSample(table)

----------------------

.. toctree::
    :maxdepth: 1

    offlineengine
    offlinerenderer


.. toctree::
    :hidden:

    renderjob
    schedevent


