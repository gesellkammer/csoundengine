from __future__ import annotations

import logging
from configdict import ConfigDict

logger = logging.getLogger('csoundengine')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                  CONFIG                   #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def _validateFigsize(cfg: dict, key: str, val) -> bool:
    if not isinstance(val, str):
        return False
    parts = val.split(":")
    return len(parts) == 2 and all(p.isnumeric() for p in parts)


_defaultconf = {
    'A4': 442,
    'buffersize': 0,
    'datafile_format': 'gen23',
    'disable_signals': True,
    'define_builtin_instrs': True,
    'dynamic_pfields': True,
    'html_theme': 'light',
    'html_repr_fontsize': '12px',
    'html_args_fontsize': '12px',
    'jupyter_synth_repr_stopbutton': True,
    'jupyter_synth_repr_interact': True,
    'jupyter_instr_repr_show_code': True,
    'ipython_load_magics_at_startup': False,
    'ksmps': 64,
    'magics_print_info': True,
    'nchnls': 0,
    'nchnls_i' : 0,
    'num_audio_buses': 64,
    'num_control_buses': 512,
    'numbuffers': 0,
    'numthreads': 1,
    'offline_score_table_size_limit': 1000,
    'prefer_udp': False,
    'rec_sr': 44100,
    'rec_ksmps': 64,
    'rec_numthreads': 1,
    'rec_sample_format': 'float',
    'rec_suppress_output': True,
    'sample_fade_time': 0.02,
    'sched_latency': 0.05,
    'set_sigint_handler': True,
    'sr': 0,
    'synth_repr_max_args': 12,
    'synth_repr_show_pfield_index': False,
    'synthgroup_repr_max_rows': 4,
    'synthgroup_html_table_style': 'font-size: smaller',
    'timeout': 2,
    'unknown_parameter_fail_silently': True,
    'jupyter_slider_width': '80%',
    'max_dynamic_args_per_instr': 10,
    'session_priorities': 10,
    'dynamic_args_num_slots': 10000,
    'instr_repr_show_pfield_pnumber': False,
    'spectrogram_colormap': 'inferno',
    'samplesplot_figsize': '12:4',
    'spectrogram_figsize': '24:8',
    'spectrogram_maxfreq': 12000,
    'spectrogram_window': 'hamming'
}

_validator = {
    'sr::choices':  {0, 22050, 24000, 44100, 48000, 88200, 96000, 144000, 192000},
    'rec_sr::choices': {0, 22050, 24000, 44100, 48000, 88200, 96000, 144000, 192000},
    'nchnls::range': (0, 128),
    'nchnls_i::range': (0, 128),
    'ksmps::choices': {16, 32, 64, 128, 256, 512, 1024},
    'rec_ksmps::choices': {1, 2, 4, 8, 10, 16, 20, 32, 64, 128, 256, 512, 1024},
    'rec_sample_format::choices': (16, 24, 32, 'float'),
    'A4::range': (410, 460),
    'html_theme::choices': {'dark', 'light'},
    'datafile_format::choices': ('gen23', 'wav'),
    'max_dynamic_args_per_instr::range': (2, 512),
    'session_priorities::range': (1, 99),
    'dynamic_args_num_slots::range': (10, 999999),
    'spectrogram_colormap::choices': {'viridis', 'plasma', 'inferno', 'magma', 'cividis'},
    'samplesplot_figsize': _validateFigsize,
    'spectrogram_figsize': _validateFigsize,
    'spectrogram_window::choices': {'hamming', 'hanning'},
    'offline_score_table_size_limit::range': (8, 10000),
}

_docs = {
    'sr':
        'Samplerate. 0=system sr',
    'rec_sr':
        'Default samplerate for rendering',
    'nchnls':
        'Number of output channels. 0=device default',
    'nchnls_i':
        'Number of input channels. 0=device default',
    'ksmps':
        "Samples per cycle",
    'rec_ksmps':
        "Samples per cycle for rendering",
    'rec_sample_format':
        "Sample format used when rendering",
    'rec_suppress_output':
        'Suppress debug output when rendering offline',
    'buffersize':
        "-b value. 0=derive from ksmps & backend",
    'numbuffers':
        "-B as a multiple of the buffersize. 0=auto",
    'A4':
        "Frequency for A4",
    'numthreads':
        "Threads for realtime performance. Experimental, may not help",
    'rec_numthreads':
        'Number of threads to use when rendering offline. Defaults to `numthreads`',
    'dynamic_pfields':
        'If True, use pfields for dynamic params (named args starting with k). '
        'Otherwise, use a global table',
    'set_sigint_handler':
        'Install a SIGINT handler to avoid CTRL-C crashes',
    'disable_signals':
        'Disable atexit and SIGINT signal handler',
    'unknown_parameter_fail_silently':
        'Don`t raise if a synth tries to set an unknown param',
    'define_builtin_instrs':
        'If True, a Session has all builtin instruments defined',
    'sample_fade_time':
        'Fade time (secs) when playing samples via a Session',
    'prefer_udp':
        'Prefer UDP over the API if a UDP server is defined',
    'num_audio_buses':
        'Num. of audio buses in an Engine/Session',
    'num_control_buses':
        'Num. of control buses in an Engine/Session',
    'html_theme':
        'Syntax highlighting style in Jupyter',
    'html_args_fontsize':
        'HTMLs font size for args in Jupyter',
    'synth_repr_max_args':
        "Max. number of pfields shown in a synth's repr",
    'synth_repr_show_pfield_index':
        'Show the pfield index in a Synths repr',
    'synthgroup_repr_max_rows':
        'Max. number of rows for a SynthGroup repr. Use 0 to disable',
    'synthgroup_html_table_style':
        'Inline CSS style applied to the HTMLs tables for synthgroups',
    'jupyter_synth_repr_stopbutton':
        'Display a stop button for synths/groups inside Jupyter',
    'jupyter_synth_repr_interact':
        'Add interactive widgets for named parameters inside Jupyter',
    'jupyter_instr_repr_show_code':
        'Show code when displaying an Instr inside Jupyter',
    'ipython_load_magics_at_startup':
        'Load csoundengine.magic at ipython/Jupyter startup (also via `%load_ext csoundengine.magic`)',
    'magics_print_info':
        'Print info when csoundengine.magic is loaded',
    'jupyter_slider_width':
        'CSS Width for interactive sliders in Jupyter',
    'timeout':
        'Timeout for any action waiting a response from csound',
    'sched_latency':
        'Delay added to events to absorb scheduling overhead',
    'datafile_format':
        'Format for saving a table as a datafile',
    'max_dynamic_args_per_instr':
        'Max. number of dynamic parameters per instr when using a global table',
    'session_priorities':
        'Number of priorities within a session',
    'dynamic_args_num_slots':
        'Slices for dynamic params (max coexisting named-arg events). Table size = slots * max_dynamic_args_per_instr',
    'instr_repr_show_pfield_pnumber':
        'Add pfield number when printing pfields in instruments',
    'spectrogram_colormap':
        'Colormap used for spectrograms',
    'samplesplot_figsize':
        'Figure size of the plot as "<width>:<height>"',
    'spectrogram_figsize':
        'Figure size of the plot as "<width>:<height>"',
    'spectrogram_maxfreq':
        'Highest freq. in a spectrogram',
    'spectrogram_window':
        'Window function used for spectrograms',
    'offline_score_table_size_limit':
        'Max. table size embedded as an f statement; larger tables are saved next to the .csd'
}


config = ConfigDict('csoundengine', persistent=False, default=_defaultconf, validator=_validator, docs=_docs, load=False)
