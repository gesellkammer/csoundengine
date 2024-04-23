
Reference
=========

Overview
--------

**Engine**
    An :class:`~csoundengine.engine.Engine` wraps a live csound process and provides a simple way to define instruments,
    schedule events, load soundfiles, etc., for realtime audio processing / synthesis.
    Any ``.csd`` script can be adapted to be run using an :class:`~csoundengine.engine.Engine`.

**Session**
    The :class:`~csoundengine.session.Session` class implements a high-level interface on top of a running
    :class:`~csoundengine.engine.Engine`. Within a :class:`Session` it is possible to define instrument
    templates (:class:`~csoundengine.instr.Instr`), which can be scheduled at any order to construct
    complex processing chains (see the :ref:`bus opcodes<busopcodes>` or using the ``chnget``/``chnset`` opcodes to
    communicate between instruments). Such instrument template provide other extra features, like named arguments,
    automatic gui generation, groups, etc.

    The same instrument templates can be reused for
    :ref:`offline (non-real-time) rendering<offlineintro>`

**Synth**
    Within a :class:`~csoundengine.session.Session`, each scheduled event is wrapped in a
    :class:`~csoundengine.synth.Synth`. A :class:`Synth` has methods for setting and automating/modulating
    parameters, querying audio inputs and outputs and can auto-generate a user-interface to interact with
    its parameters in real-time.

.. _offlineintro:

**Offline Rendering**
    Both an :class:`~csoundengine.engine.Engine` and its associated
    :class:`~csoundengine.session.Session` are concieved to run in real-time. For offline rendering
    **csoundengine** provides the :class:`~csoundengine.offline.Renderer` class, which has the same
    interface as a :class:`~csoundengine.session.Session` but collects all scheduled events,
    soundfiles, automation, etc. and renders everything in non-real-time (and probably much faster).
    A :class:`~csoundengine.offline.Renderer` is a drop-in replacement for a
    :class:`~csoundengine.session.Session`. It can also be used to generate a csound project to be
    further edited and/or rendered by the csound executable. See :ref:`offlinemod`

---------------

Table of Contents
-----------------

.. toctree::
    :maxdepth: 1

    csoundengine
    session
    instr
    synth
    offline
    tables
    Builtin-Opcodes
    config
    csoundlib
    magics
    jupyternotebook
