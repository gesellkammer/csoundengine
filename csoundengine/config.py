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
    'suppress_output': True,
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
        'samplerate - 0=default sr for the backend',
    'rec_sr':
        'default samplerate when rendering',
    'nchnls':
        'Number of output channels. 0=default for used device',
    'nchnls_i':
        'Number of input channels to use. 0 = default for used device',
    'ksmps':
        "corresponds to csound's ksmps",
    'rec_ksmps':
        "samples per cycle when rendering",
    'rec_sample_format':
        "Sample format used when rendering",
    'rec_suppress_output':
        'Supress debugging output when rendering offline',
    'buffersize':
        "-b value. 0=determine buffersize depending on ksmps & backend",
    'numbuffers':
        "determines the -B value as a multiple of the buffersize. 0=auto",
    'A4':
        "Frequency for A4",
    'numthreads':
        "Number of threads to use for realtime performance. This is an experimental feature "
        "and might not necessarily result in better performance",
    'rec_numthreads':
       'Number of threads to use when rendering online. If not given, the value set '
       'in `numthreads` is used',
    'dynamic_pfields':
        'If True, use pfields for dynamic parameters (named args starting with k). '
        'Otherwise, dynamic controls are implemented via a global table',
    'set_sigint_handler':
        'Set a sigint handler to prevent csound crash with CTRL-C',
    'disable_signals':
        'Disable atexit and sigint signal handler',
    'suppress_output':
        'Suppress csound´s debugging information',
    'unknown_parameter_fail_silently':
        'Do not raise if a synth tries to set an unknown parameter',
    'define_builtin_instrs':
        'If True, a Session with have all builtin instruments defined',
    'sample_fade_time':
        'Fade time (in seconds) when playing samples via a Session',
    'prefer_udp':
        'If true and a udp server was defined,  prefer UDP over the API for communication',
    'num_audio_buses':
        'Num. of audio buses in an Engine/Session',
    'num_control_buses':
        'Num. of control buses in an Engine/Session. This sets the upper limit to the '
        'number of simultaneous control buses in use',
    'html_theme':
        'Style to use when displaying syntax highlighting in jupyter',
    'html_args_fontsize':
        'Font size used for args when outputing html (in jupyter)',
    'synth_repr_max_args':
        "Max. number of pfields shown when in a synth's repr",
    'synth_repr_show_pfield_index':
        'Show the pfield index for named pfields in a Synths repr',
    'synthgroup_repr_max_rows':
        'Max. number of rows for a SynthGroup repr. Use 0 to disable',
    'synthgroup_html_table_style':
        'Inline css style applied to the table displayed as html for synthgroups',
    'jupyter_synth_repr_stopbutton':
        'When running inside a jupyter notebook, display a stop button'
        'for Synths and SynthGroups',
    'jupyter_synth_repr_interact':
        'When inside jupyter, add interactive widgets if a synth has'
        'named parameters',
    'jupyter_instr_repr_show_code':
        'Show code when displaying an Instr inside jupyter',
    'ipython_load_magics_at_startup':
        'Load csoundengine.magic at startup when inside ipython. If False, magics can '
        'still be loaded via `%load_ext csoundengine.magic`',
    'magics_print_info':
        'Print some informative information when the csoundengine.magic extension is loaded',
    'jupyter_slider_width':
        'CSS Width used by an interactive slider in jupyter',
    'timeout':
        'Timeout for any action waiting a response from csound',
    'sched_latency':
        'Time delay added to any event scheduled to ensure that simultameous events are'
        'not offset by scheduling overhead',
    'datafile_format':
        'Format used when saving a table as a datafile',
    'max_dynamic_args_per_instr':
        'Max. number of dynamic parameters per instr. This applies only if dynamic args '
        'are implemented via a global table',
    'session_priorities':
        'Number of priorities within a session',
    'dynamic_args_num_slots':
        'Number of slots for dynamic parameters. args slices. Dynamic args are implemented as a big '
        'array divided in slices. This parameter sets the max. number of '
        'such slices, and thus the max number of simultaneous events with named '
        'args which can coexist. The size of the allocated table will be '
        'size = num_dynamic_args_slices * max_instr_dynamic_args. For 10000 slots, the'
        'amount of memory is ~0.8Mb',
    'instr_repr_show_pfield_pnumber':
        'Add pfield number when printing pfields in instruments',
    'spectrogram_colormap':
        'Colormap used for spectrograms',
    'samplesplot_figsize':
        'Figure size of the plot in the form "<width>:<height>"',
    'spectrogram_figsize':
        'Figure size of the plot in the form "<width>:<height>"',
    'spectrogram_maxfreq':
        'Highest freq. in a spectrogram',
    'spectrogram_window':
        'Window function used for spectrograms',
    'offline_score_table_size_limit':
        'Max. size of a table to be embedded within the score. Larger tables are saved as data files along the .csd'
}


config = ConfigDict('csoundengine', persistent=False, default=_defaultconf, validator=_validator, docs=_docs, load=False)
