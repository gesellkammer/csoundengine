
Reference
=========

Overview
--------

Engine
    At the core of **csoundengine** is the :class:`~csoundengine.engine.Engine` class, which
    wraps a csound process and provides a simple way to define instruments, schedule events,
    load soundfiles, etc., for realtime audio processing / synthesis.
    Any ``.csd`` script can be adapted to be run using an :class:`~csoundengine.engine.Engine`.

Session
    The :class:`~csoundengine.session.Session` class implements a high-level interface on top of a running
    :class:`~csoundengine.engine.Engine`. Within a Session it is possible to define instrument
    templates (:class:`~csoundengine.instr.Instr`), which can be scheduled at any order to construct
    complex processing chains (see the :ref:`bus opcodes<busopcodes>`). Such instrument templates
    provide other extra features, like named arguments, automatic gui generation, groups, etc.

    The same instrument templates can be reused for
    :ref:`offline (non-real-time) rendering<offlineintro>`

Synth
    Within a :class:`~csoundengine.session.Session`, each scheduled event is wrapped in a
    :class:`~csoundengine.synth.Synth`, which provides shortcuts for setting and automating/modulating
    parameters, query audio inputs and outputs and auto-generate a user-interface to interact with
    its parameters in real-time.

.. _offlineintro:

Offline rendering
    Both an :class:`~csoundengine.engine.Engine` and its associated
    :class:`~csoundengine.session.Session` are conceived to run in real-time. For offline rendering
    **csoundengine** provides the :class:`~csoundengine.offline.Renderer` class, which has the same
    interface as a :class:`~csoundengine.session.Session` but collects all scheduled events,
    soundfiles, automation, etc. and renders everything in non-real-time (and probably much faster).
    A :class:`~csoundengine.offline.Renderer` is a drop-in replacement for a
    :class:`~csoundengine.session.Session`. See :ref:`offlinemod

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
