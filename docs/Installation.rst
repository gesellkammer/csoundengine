Installation
============

`csoundengine` needs a python version >= 3.8. For all platforms, the installation is 
the same::

    pip install csoundengine

Dependencies
------------

`csoundengine` has following dependencies:

* csound >= 6.15 (https://github.com/csound/csound/releases)

--------------

Linux
-----

Install csound
^^^^^^^^^^^^^^

In most distributions (with the exception of arch) the packaged version of csound might
be too old. The best way to install csound is to install it from source. For ubuntu
based distributions this is done by:

.. code-block:: shell

    sudo apt build-dep csound
    git clone https://github.com/csound/csound
    cd csound 
    mkdir build && cd build
    cmake ..
    make
    sudo make install

--------------

macOS
-----

Install csound
^^^^^^^^^^^^^^

Download and install the ``.dmg`` package from https://github.com/csound/csound/releases

--------------

Windows
-------

Install csound
^^^^^^^^^^^^^^

Download and install the `.exe` installer from https://github.com/csound/csound/releases
