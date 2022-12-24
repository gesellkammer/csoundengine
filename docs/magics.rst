.. _magics-label:

Magics
======

**csoundengine** defines a set of ipython/jupyter ``%magic`` / ``%%magic`` commands

.. contents:: Table of Contents
   :depth: 3
   :local:
   :backlinks: none

%csound
-------

Syntax::

    %csound setengine <enginename>         : Sets the default Engine


%%csound
--------

Compile the code in this cell within the current Engine (notice the difference between `%csound`
which is a line-magic and `%%csound` which is a cell-magic)

.. admonition:: Setting the current Engine / Session

    The current engine can be explicitely selected via ``%csound setengine <enginename>``.
    Otherwise the last started Engine will be used. Whenever an Engine/Session is explicitly
    set as active, any new Engine will not override this setting.


.. code-block:: python

    from csoundengine import *
    e = Engine()

.. code-block:: csound

    %%csound
    instr 100
        kamp = p4
        kmidi = p5
        a0 oscili kamp, mtof:k(kmidi)
        aenv linsegr 0, 0.01, 1, 0.01, 0
        a0 *= aenv
        outch 1, a0
    endin

.. code-block:: python

    event = e.sched(100, args=[0.1, 67])

%%definstr
----------

Defines a new Instr inside the current Session (the current Session is the Session
associated with the current Engine)

.. code-block:: python

    from csoundengine import *
    e = Engine()

.. code-block:: csound

    %%definstr foo
    |kamp=0.5, kfreq=1000|
    a0 oscili kamp, kfreq
    aenv linsegr 0, 0.01, 1, 0.01, 0
    a0 *= aenv
    outch 1, a0

The last block is equivalent to

.. code-block:: python

    s = e.session()
    s.defInstr("foo", r'''
        |kamp=0.5, kfreq=1000|
        a0 oscili kamp, kfreq
        aenv linsegr 0, 0.01, 1, 0.01, 0
        a0 *= aenv
        outch 1, a0
    ''')
