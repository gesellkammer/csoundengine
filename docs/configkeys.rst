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
    | Choices: ``16, 32, 64, 128, 256, 512, 1024``
    | *samples per cycle when rendering*

.. _config_rec_sample_format:

rec_sample_format:
    | Default: **float**  -- ``(int, str)``
    | Choices: ``16, 24, 32, float``
    | *Sample format used when rendering*

.. _config_rec_suppress_output:

rec_suppress_output:
    | Default: **False**  -- ``bool``
    | *Supress debugging output when rendering offline*

.. _config_buffersize:

buffersize:
    | Default: **0**  -- ``int``
    | *-b value. 0=determine buffersize depending on ksmps & backend*

.. _config_numbuffers:

numbuffers:
    | Default: **0**  -- ``int``
    | *determines the -B value as a multiple of the buffersize. 0=auto*

.. _config_linux_backend:

linux_backend:
    | Default: **jack, pulse, pa_cb**  -- ``str``
    | *a comma separated list of backends (possible backends: jack, pulse, pa_cb, alsa)*

.. _config_macos_backend:

macos_backend:
    | Default: **pa_cb**  -- ``str``
    | *a comma separated list of backends (possible backends: pa_cb, auhal)*

.. _config_windows_backend:

windows_backend:
    | Default: **pa_cb**  -- ``str``
    | *a comma separated list of backends (possible backends: pa_cb, pa_bl)*

.. _config_a4:

A4:
    | Default: **442**  -- ``int``
    | Between 410 - 460
    | *Frequency for A4*

.. _config_check_pargs:

check_pargs:
    | Default: **False**  -- ``bool``
    | *Check number of pargs passed to instr*

.. _config_offline_score_table_size_limit:

offline_score_table_size_limit:
    | Default: **1900**  -- ``int``
    | *size limit when writing tables as f score statements via gen2. If a table is bigger than this size, it is saved as a datafile as gen23 or wav*

.. _config_fail_if_unmatched_pargs:

fail_if_unmatched_pargs:
    | Default: **False**  -- ``bool``
    | *Fail if the # of passed pargs doesnt match the # of pargs*

.. _config_set_sigint_handler:

set_sigint_handler:
    | Default: **True**  -- ``bool``
    | *Set a sigint handler to prevent csound crash with CTRL-C*

.. _config_generalmidi_soundfont:

generalmidi_soundfont:
    | Default: **''**  -- ``str``

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
    | Default: **0.05**  -- ``float``
    | *Fade time when playing samples via a Session*

.. _config_prefer_udp:

prefer_udp:
    | Default: **True**  -- ``bool``
    | *If true and a server was defined prefer UDP over the API for communication*

.. _config_start_udp_server:

start_udp_server:
    | Default: **False**  -- ``bool``
    | *Start an engine with udp communication support*

.. _config_associated_table_min_size:

associated_table_min_size:
    | Default: **16**  -- ``int``
    | *Min. size of the param table associated with a synth*

.. _config_num_audio_buses:

num_audio_buses:
    | Default: **64**  -- ``int``
    | *Num. of audio buses in an Engine/Session*

.. _config_num_control_buses:

num_control_buses:
    | Default: **512**  -- ``int``
    | *Num. of control buses in an Engine/Session*

.. _config_html_theme:

html_theme:
    | Default: **light**  -- ``str``
    | Choices: ``dark, light``
    | *Style to use when displaying syntax highlighting*

.. _config_html_args_fontsize:

html_args_fontsize:
    | Default: **12px**  -- ``str``
    | *Font size used for args when outputing html (in jupyter)*

.. _config_synth_repr_max_args:

synth_repr_max_args:
    | Default: **12**  -- ``int``
    | *Max. number of pfields shown when in a synth's repr*

.. _config_synthgroup_repr_max_rows:

synthgroup_repr_max_rows:
    | Default: **4**  -- ``int``
    | *Max. number of rows for a SynthGroup repr. Use 0 to disable*

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
    | Default: **True**  -- ``bool``
    | *Load csoundengine.magic at startup when inside ipython. If False, magics can still be loaded via `%load_ext csoundengine.magic`*

.. _config_magics_print_info:

magics_print_info:
    | Default: **False**  -- ``bool``
    | *Print some informative information when the csounengine.magic extension is loaded*

.. _config_jupyter_slider_width:

jupyter_slider_width:
    | Default: **80%**  -- ``str``
    | *CSS Width used by an interactive slider in jupyter*

.. _config_timeout:

timeout:
    | Default: **2.0**  -- ``float``
    | *Timeout for any action waiting a response from csound*

.. _config_sched_latency:

sched_latency:
    | Default: **0.0**  -- ``float``
    | *Time delay added to any event scheduled to ensure that simultameous events arenot offset by scheduling overhead*

.. _config_datafile_format:

datafile_format:
    | Default: **gen23**  -- ``str``
    | Choices: ``gen23, wav``
    | *Format used when saving a table as a datafile*

.. _config_spectrogram_colormap:

spectrogram_colormap:
    | Default: **inferno**  -- ``str``
    | Choices: ``cividis, inferno, magma, plasma, viridis``

.. _config_samplesplot_figsize:

samplesplot_figsize:
    | Default: **12:4**  -- ``str``
    | *The figure size of the plot in the form '<width>:<height>'*

.. _config_spectrogram_figsize:

spectrogram_figsize:
    | Default: **24:8**  -- ``str``
    | *The figure size of the plot in the form '<width>:<height>'*

.. _config_spectrogram_maxfreq:

spectrogram_maxfreq:
    | Default: **12000**  -- ``int``
    | *Highest freq. in a spectrogram*

.. _config_spectrogram_window:

spectrogram_window:
    | Default: **hamming**  -- ``str``
    | Choices: ``hamming, hanning``

.. _config_dependencies_check_timeout_days:

dependencies_check_timeout_days:
    | Default: **7**  -- ``int``
    | Between 1 - 365
    | *Elapsed time (in days) after which dependencies will be checked*
