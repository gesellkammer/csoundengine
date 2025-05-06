.. _config_sr:

sr:
    | Default: **0**  -- ``int``
    | Choices: ``0, 22050, 24000, 44100, 48000, 88200, 96000, 144000, 192000``
    | *samplerate - 0=default sr for the backend*

.. _config_rec_sr:

rec_sr:
    | Default: **44100**  -- ``int``
    | Choices: ``0, 22050, 24000, 44100, 48000, 88200, 96000, 144000, 192000``
    | *default samplerate when rendering*

.. _config_nchnls:

nchnls:
    | Default: **0**  -- ``int``
    | Between 0 - 128
    | *Number of output channels. 0=default for used device*

.. _config_nchnls_i:

nchnls_i:
    | Default: **0**  -- ``int``
    | Between 0 - 128
    | *Number of input channels to use. 0 = default for used device*

.. _config_ksmps:

ksmps:
    | Default: **64**  -- ``int``
    | Choices: ``16, 32, 64, 128, 256, 512, 1024``
    | *corresponds to csound's ksmps*

.. _config_rec_ksmps:

rec_ksmps:
    | Default: **64**  -- ``int``
    | Choices: ``1, 2, 4, 8, 10, 16, 20, 32, 64, 128, 256, 512, 1024``
    | *samples per cycle when rendering*

.. _config_rec_sample_format:

rec_sample_format:
    | Default: **float**  -- ``(int, str)``
    | Choices: ``16, 24, 32, float``
    | *Sample format used when rendering*

.. _config_rec_suppress_output:

rec_suppress_output:
    | Default: **True**  -- ``bool``
    | *Supress debugging output when rendering offline*

.. _config_buffersize:

buffersize:
    | Default: **0**  -- ``int``
    | *-b value. 0=determine buffersize depending on ksmps & backend*

.. _config_numbuffers:

numbuffers:
    | Default: **0**  -- ``int``
    | *determines the -B value as a multiple of the buffersize. 0=auto*

.. _config_a4:

A4:
    | Default: **442**  -- ``int``
    | Between 410 - 460
    | *Frequency for A4*

.. _config_numthreads:

numthreads:
    | Default: **1**  -- ``int``
    | *Number of threads to use for realtime performance. This is an experimental feature and might not necessarily result in better performance*

.. _config_rec_numthreads:

rec_numthreads:
    | Default: **1**  -- ``int``
    | *Number of threads to use when rendering online. If not given, the value set in `numthreads` is used*

.. _config_dynamic_pfields:

dynamic_pfields:
    | Default: **True**  -- ``bool``
    | *If True, use pfields for dynamic parameters (named args starting with k). Otherwise, dynamic controls are implemented via a global table*

.. _config_set_sigint_handler:

set_sigint_handler:
    | Default: **True**  -- ``bool``
    | *Set a sigint handler to prevent csound crash with CTRL-C*

.. _config_disable_signals:

disable_signals:
    | Default: **True**  -- ``bool``
    | *Disable atexit and sigint signal handler*

.. _config_suppress_output:

suppress_output:
    | Default: **True**  -- ``bool``
    | *Suppress csound´s debugging information*

.. _config_unknown_parameter_fail_silently:

unknown_parameter_fail_silently:
    | Default: **True**  -- ``bool``
    | *Do not raise if a synth tries to set an unknown parameter*

.. _config_define_builtin_instrs:

define_builtin_instrs:
    | Default: **True**  -- ``bool``
    | *If True, a Session with have all builtin instruments defined*

.. _config_sample_fade_time:

sample_fade_time:
    | Default: **0.02**  -- ``float``
    | *Fade time (in seconds) when playing samples via a Session*

.. _config_prefer_udp:

prefer_udp:
    | Default: **False**  -- ``bool``
    | *If true and a udp server was defined,  prefer UDP over the API for communication*

.. _config_num_audio_buses:

num_audio_buses:
    | Default: **64**  -- ``int``
    | *Num. of audio buses in an Engine/Session*

.. _config_num_control_buses:

num_control_buses:
    | Default: **512**  -- ``int``
    | *Num. of control buses in an Engine/Session. This sets the upper limit to the number of simultaneous control buses in use*

.. _config_bus_support:

bus_support:
    | Default: **False**  -- ``bool``
    | *If True, an Engine/OfflineEngine will have bus support*

.. _config_html_theme:

html_theme:
    | Default: **light**  -- ``str``
    | Choices: ``dark, light``
    | *Style to use when displaying syntax highlighting in jupyter*

.. _config_html_args_fontsize:

html_args_fontsize:
    | Default: **12px**  -- ``str``
    | *Font size used for args when outputing html (in jupyter)*

.. _config_synth_repr_max_args:

synth_repr_max_args:
    | Default: **12**  -- ``int``
    | *Max. number of pfields shown when in a synth's repr*

.. _config_synth_repr_show_pfield_index:

synth_repr_show_pfield_index:
    | Default: **False**  -- ``bool``
    | *Show the pfield index for named pfields in a Synths repr*

.. _config_synthgroup_repr_max_rows:

synthgroup_repr_max_rows:
    | Default: **4**  -- ``int``
    | *Max. number of rows for a SynthGroup repr. Use 0 to disable*

.. _config_synthgroup_html_table_style:

synthgroup_html_table_style:
    | Default: **font-size: smaller**  -- ``str``
    | *Inline css style applied to the table displayed as html for synthgroups*

.. _config_jupyter_synth_repr_stopbutton:

jupyter_synth_repr_stopbutton:
    | Default: **True**  -- ``bool``
    | *When running inside a jupyter notebook, display a stop buttonfor Synths and SynthGroups*

.. _config_jupyter_synth_repr_interact:

jupyter_synth_repr_interact:
    | Default: **True**  -- ``bool``
    | *When inside jupyter, add interactive widgets if a synth hasnamed parameters*

.. _config_jupyter_instr_repr_show_code:

jupyter_instr_repr_show_code:
    | Default: **True**  -- ``bool``
    | *Show code when displaying an Instr inside jupyter*

.. _config_ipython_load_magics_at_startup:

ipython_load_magics_at_startup:
    | Default: **False**  -- ``bool``
    | *Load csoundengine.magic at startup when inside ipython. If False, magics can still be loaded via `%load_ext csoundengine.magic`*

.. _config_magics_print_info:

magics_print_info:
    | Default: **True**  -- ``bool``
    | *Print some informative information when the csoundengine.magic extension is loaded*

.. _config_jupyter_slider_width:

jupyter_slider_width:
    | Default: **80%**  -- ``str``
    | *CSS Width used by an interactive slider in jupyter*

.. _config_timeout:

timeout:
    | Default: **2**  -- ``int``
    | *Timeout for any action waiting a response from csound*

.. _config_sched_latency:

sched_latency:
    | Default: **0.05**  -- ``float``
    | *Time delay added to any event scheduled to ensure that simultameous events arenot offset by scheduling overhead*

.. _config_datafile_format:

datafile_format:
    | Default: **gen23**  -- ``str``
    | Choices: ``gen23, wav``
    | *Format used when saving a table as a datafile*

.. _config_max_dynamic_args_per_instr:

max_dynamic_args_per_instr:
    | Default: **10**  -- ``int``
    | Between 2 - 512
    | *Max. number of dynamic parameters per instr. This applies only if dynamic args are implemented via a global table*

.. _config_session_priorities:

session_priorities:
    | Default: **10**  -- ``int``
    | Between 1 - 99
    | *Number of priorities within a session*

.. _config_dynamic_args_num_slots:

dynamic_args_num_slots:
    | Default: **10000**  -- ``int``
    | Between 10 - 999999
    | *Number of slots for dynamic parameters. args slices. Dynamic args are implemented as a big array divided in slices. This parameter sets the max. number of such slices, and thus the max number of simultaneous events with named args which can coexist. The size of the allocated table will be size = num_dynamic_args_slices * max_instr_dynamic_args. For 10000 slots, theamount of memory is ~0.8Mb*

.. _config_instr_repr_show_pfield_pnumber:

instr_repr_show_pfield_pnumber:
    | Default: **False**  -- ``bool``
    | *Add pfield number when printing pfields in instruments*

.. _config_spectrogram_colormap:

spectrogram_colormap:
    | Default: **inferno**  -- ``str``
    | Choices: ``cividis, inferno, magma, plasma, viridis``
    | *Colormap used for spectrograms*

.. _config_samplesplot_figsize:

samplesplot_figsize:
    | Default: **12:4**  -- ``str``
    | *Figure size of the plot in the form "<width>:<height>"*

.. _config_spectrogram_figsize:

spectrogram_figsize:
    | Default: **24:8**  -- ``str``
    | *Figure size of the plot in the form "<width>:<height>"*

.. _config_spectrogram_maxfreq:

spectrogram_maxfreq:
    | Default: **12000**  -- ``int``
    | *Highest freq. in a spectrogram*

.. _config_spectrogram_window:

spectrogram_window:
    | Default: **hamming**  -- ``str``
    | Choices: ``hamming, hanning``
    | *Window function used for spectrograms*
