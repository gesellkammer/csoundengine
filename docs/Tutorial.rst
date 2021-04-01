Tutorial
========

XXX
---

When using csound in real-time the first step is always to start a csound process
by creating an Engine.

.. code-block:: python

    from csoundengine import *
    engine = Engine()

This creates an Engine with default / inferred settings. For example the audio
backend selected will depend on the platform and availability of a given backend.
For example, in linux jack will be the default backend if it's running otherwise
we will fall back to portaudio.