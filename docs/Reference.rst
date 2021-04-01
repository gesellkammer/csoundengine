
Reference
=========

Introduction
------------

At the core of the `csoundengine` library is the :class:`~csoundengine.engine.Engine` class, which wraps a csound process.
This class provides a simple way to start and communicate with csound, define instruments,
schedule events, etc. The interaction is very transparent and any .csd script can be adapted to be run using an Engine.

The :class:`~csoundengine.session.Session` class implements a high-level interface. It builds on top of :class:`~csoundengine.Engine`.
Both an Engine and its associated Session are conceived to run in real-time. See :class:`~csoundengine.offline.Renderer` for offline rendering

.. automodapi:: csoundengine
    :no-inheritance-diagram:

----------

.. automodapi:: csoundengine.session
    :no-inheritance-diagram:

----------

.. automodapi:: csoundengine.offline
    :no-inheritance-diagram:

----------

.. automodapi:: csoundengine.csoundlib
    :no-inheritance-diagram:





