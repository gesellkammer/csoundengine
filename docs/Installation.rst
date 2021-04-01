Installation
============

`csoundengine` needs a python version >= 3.8. For all platforms, the installation is 
the same::

    pip install csoundengine --user


Dependencies
------------

`csoundengine` has following dependencies:

* csound >= 6.15 (https://github.com/csound/csound/releases)
* csound-plugins >= 1.3.1 (https://github.com/csound-plugins/csound-plugins/releases)


--------------

Linux
-----

Install csound
^^^^^^^^^^^^^^

In most distributions (with the exception of arch) the packaged version of csound is 
too old. The best way to install csound is by insalling from source. For debian/ubuntu 
based distributions this is done by:

.. code-block:: shell

    sudo apt build-dep csound
    git clone https://github.com/csound/csound
    cd csound 
    mkdir build && cd build
    cmake ..
    make
    sudo make install

Install csound-plugins
^^^^^^^^^^^^^^^^^^^^^^

Download the linux package from https://github.com/csound-plugins/csound-plugins/releases, unzip 
and copy the contents to ``/usr/local/lib/csound/plugins64-6.0``

--------------

macOS
-----

Install csound
^^^^^^^^^^^^^^

Download and install the ``.dmg`` package from https://github.com/csound/csound/releases

Install csound-plugins
^^^^^^^^^^^^^^^^^^^^^^

Download the macOS package from https://github.com/csound-plugins/csound-plugins/releases, unzip 
and copy the contents to ``/Library/Frameworks/CsoundLib64.framework/Versions/6.0/Resources/Opcodes64``
(which should exist and be already populated with the plugins shipped with csound itself)


--------------

Windows
-------

Install csound
^^^^^^^^^^^^^^

Download and install the `.exe` installer from https://github.com/csound/csound/releases

Install csound-plugins
^^^^^^^^^^^^^^^^^^^^^^

Download the windows package from https://github.com/csound-plugins/csound-plugins/releases, unzip 
and copy the contents to ``C:\Program Files\Csound6_x64\plugins64`` (which should exist and be already
populated with the plugins shipped with csound itself)

