
OfflineEngine Class
===================

.. currentmodule:: csoundengine.offlineengine

The :class:`OfflineEngine` class represents an offline csound process rendering to a soundfile
instead of to the ``dac``. An :class:`OfflineEngine` is, for the most part, a drop-in replacement
of the :class:`~csoundengine.engine.Engine` class. The main difference is that for an :class:`OfflineEngine`
**time needs to be advanced explicitely** via its :meth:`~OfflineEngine.perform` method.

At any moment it is possible to interact with the offline csound process. While an :class:`OfflineEngine` is
running it is possible to query variables, load resources, read and write to channels, etc.


Example 1
---------

.. include:: offlineengine_example.inc



Example 2: sound analysis
-------------------------

TODO

----------------------------


.. autoclass:: csoundengine.offlineengine.OfflineEngine
    :members:
    :autosummary:

