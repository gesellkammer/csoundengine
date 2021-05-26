=============
Configuration
=============

Many defaults can be configured via :py:obj:`csoundengine.config.config` (an instance 
of :class:`configdict.ConfigDict`, see https://configdict.readthedocs.io). 
This is a persistent dictionary: any modification will
be automatically saved and loaded the next time this module is imported

Example
=======

.. code::

    import csoundengine as ce
    # Modify the default number of channels, force it to 2
    ce.config['nchnls'] = 2

    # Set default A4 value 
    ce.config['A4'] = 443

    # Larger number of audio buses
    ce.config['num_audio_buses'] = 200

    
Any :class:`~csoundengine.engine.Engine` created after this will pick up these
new defaults::


    >>> from csoundengine import *
    >>> engine = Engine()
    >>> engine.numAudioBuses
    200
    >>> engine.A4
    443


Edit the config interactively
-----------------------------


.. code::

    from csoundengine import *
    config.edit()


This will open the config in a text editor and any changes there will be reflected back in
the config

-----
    
Keys
====

sr:
    | Default: **0**  -- `int`
    | Choices: ``0, 22050, 44100, 48000, 88200, 96000``
    | *samplerate - 0=default sr for the backend*

rec.sr:
    | Default: **44100**  -- `int`
    | Choices: ``44100, 48000, 88200, 96000, 192000``
    | *default samplerate when rendering*

nchnls:
    | Default: **0**  -- `int`
    | Between 0 - 128
    | *Number of output channels. 0=default for used device*

nchnls_i:
    | Default: **0**  -- `int`
    | Between 0 - 128
    | *Number of input channels to use. 0 = default for used device*

ksmps:
    | Default: **64**  -- `int`
    | Choices: ``16, 32, 64, 128, 256``
    | *corresponds to csound's ksmps*

rec.ksmps:
    | Default: **64**  -- `int`
    | Choices: ``16, 32, 64, 128, 256``
    | *samples per cycle when rendering*

rec.sample_format:
    | Default: **float**  -- `(str, int)`
    | Choices: ``16, 24, 32, float``
    | *Sample format used when rendering*

buffersize:
    | Default: **0**  -- `int`
    | *-b value. 0=determine buffersize depending on ksmps & backend*

numbuffers:
    | Default: **0**  -- `int`
    | *determines the -B value as a multiple of the buffersize. 0=auto*

linux.backend:
    | Default: **jack, pulse, pa_cb**  -- `str`
    | A comma-separated list of backends, in descending priority

macos.backend:
    | Default: **auhal, pa_cb**  -- `str`
    | A comma-separated list of backends, in descending priority
    | Possible backends: auhal (coreaudio), pa_cb (portaudio callbacks)

windows.backend:
    | Default: **pa_cb**  -- `str`
    | A comma-separated list of backends, in descending priority
    | Possible backends: pa_cb (portaudio callbacks), pa_bl (portaudio blocking)

A4:
    | Default: **442**  -- `int`
    | Between 410 - 460
    | *Frequency for A4*

check_pargs:
    | Default: **False**  -- `bool`
    | *Check number of pargs passed to instr*

fail_if_unmatched_pargs:
    | Default: **False**  -- `bool`
    | *Fail if the # of passed pargs doesnt match the # of pargs*

set_sigint_handler:
    | Default: **True**  -- `bool`
    | *Set a sigint handler to prevent csound crash with CTRL-C*

generalmidi_soundfont:
    | Default: **None**  -- `str`

suppress_output:
    | Default: **True**  -- `bool`
    | *Supress csound´s debugging information*

unknown_parameter_fail_silently:
    | Default: **True**  -- `bool`
    | *Do not raise if a synth tries to set an unknown parameter*

define_builtin_instrs:
    | Default: **True**  -- `bool`
    | *If True, a Session with have all builtin instruments defined*

sample_fade_time:
    | Default: **0.05**  -- `float`
    | *Fade time when playing samples via a Session*

prefer_udp:
    | Default: **True**  -- `bool`
    | *If true and a server was defined prefer UDP over the API for communication*

start_udp_server:
    | Default: **False**  -- `bool`
    | *Start an engine with udp communication support*

associated_table_min_size:
    | Default: **16**  -- `int`
    | *Min. size of the param table associated with a synth*

num_audio_buses:
    | Default: **64**  -- `int`
    | *Num. of audio buses in an Engine/Session*
