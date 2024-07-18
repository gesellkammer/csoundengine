
Reference
=========

**Engine**
    An :class:`~csoundengine.engine.Engine` wraps a live csound process and provides a simple way to define instruments,
    schedule events, load soundfiles, etc., for realtime audio processing / synthesis.
    Any ``.csd`` script can be adapted to be run using an :class:`~csoundengine.engine.Engine`.

**Session**
    The :class:`~csoundengine.session.Session` class implements a high-level interface on top of a running
    :class:`~csoundengine.engine.Engine`. Within a :class:`Session` it is possible to define instrument
    templates (:class:`~csoundengine.instr.Instr`), which can be scheduled at any order to construct
    complex processing chains using :ref:`buses<busopcodes>` or using the ``chnget``/``chnset`` opcodes to
    communicate between instruments


**Instr**
    An :class:`~csoundengine.instr.Instr` is an instrument template. It defines the
    csound code and its parameters and default values. A concrete csound instrument is
    created only when an :class:`~csoundengine.instr.Instr` is scheduled.
    The main difference with a csound ``instr`` is that an
    :class:`~csoundengine.instr.Instr` can be scheduled at any level within the
    evaluation chain. Similar to plugins in a DAW, ``Instr`` can be organized to build
    processing chains of any depth. They can also be used for :ref:`offline (non-real-time) rendering<offlineintro>`

**Synth**
    A :class:`~csoundengine.synth.Synth` wraps an event scheduled within a
    :class:`~csoundengine.session.Session`. It has methods for setting and
    automating/modulating parameters, querying audio inputs and outputs and
    can auto-generate a user-interface to interact with its parameters in real-time.

.. _offlineintro:

**Offline Rendering**
    Both an :class:`~csoundengine.engine.Engine` and its associated
    :class:`~csoundengine.session.Session` are concieved to run in real-time. For offline rendering
    *csoundengine* provides for each an offline version, :class:`~csoundengine.offlineengine.OfflineEngine`
    and :class:`~csoundengine.offline.OfflineSession`. These have the same interface as their real-time counterparts,
    but render everything offline to a soundfile (and probably much faster). They can also be used to generate a
    csound project (a .csd file plus needed assets) to be further edited and/or rendered by the csound executable.
    See :ref:`offlinemod`

---------------

.. toctree::
    :maxdepth: 1
    :hidden:

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
