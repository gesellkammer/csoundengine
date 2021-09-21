import logging
from configdict import ConfigDict
from . import csoundlib

modulename = 'csoundengine.engine'

logger = logging.getLogger('csoundengine')


def setLoggingLevel(level: str) -> None:
    """
    Utility to set the logging level of csoundengine
    """
    level = level.upper()
    logging.basicConfig(level=level)
    logger.setLevel(level)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                  CONFIG                   #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

config = ConfigDict(modulename)
_ = config.addKey


def _validateBackend(cfg: dict, key:str, s: str) -> bool:
    platform = key.split("_")[0]
    possibleBackends = csoundlib.getAudioBackendNames(available=False, platform=platform)
    for backend in (b.strip() for b in s.split(',')):
        if backend not in possibleBackends:
            logger.error(f"Backend '{backend}' unknown. "
                         f"Possible backends: {possibleBackends}")
            return False
    return True


_('sr', 0,
  choices=(0, 22050, 44100, 48000, 88200, 96000),
  doc='samplerate - 0=default sr for the backend')
_('rec_sr', 44100,
  choices=(44100, 48000, 88200, 96000, 192000),
  doc='default samplerate when rendering')
_('nchnls', 0,
  range=(0, 128),
  doc='Number of output channels. 0=default for used device')
_('nchnls_i', 0,
  range=(0, 128),
  doc='Number of input channels to use. 0 = default for used device')
_('ksmps', 64,
  choices=(16, 32, 64, 128, 256),
  doc="corresponds to csound's ksmps")
_('rec_ksmps', 64,
  choices=(16, 32, 64, 128, 256),
  doc="samples per cycle when rendering")
_('rec_sample_format', 'float',
  choices=(16, 24, 32, 'float'),
  doc="Sample format used when rendering")
_('buffersize', 0,
  doc="-b value. 0=determine buffersize depending on ksmps & backend")
_('numbuffers', 0,
  doc="determines the -B value as a multiple of the buffersize. 0=auto")
_('linux_backend', 'jack, pulse, pa_cb',
  doc="a comma separated list of backends (possible backends: jack, pulse, pa_cb, alsa)",
  validatefunc=_validateBackend),
_('macos_backend', 'pa_cb',
  doc="a comma separated list of backends (possible backends: pa_cb, auhal)",
  validatefunc=_validateBackend),
_('windows_backend', 'pa_cb',
  doc="a comma separated list of backends (possible backends: pa_cb, pa_bl)",
  validatefunc=_validateBackend)
_('A4', 442,
  range=(410, 460),
  doc="Frequency for A4")
_('check_pargs', False,
  doc='Check number of pargs passed to instr')
_('fail_if_unmatched_pargs', False,
  doc='Fail if the # of passed pargs doesnt match the # of pargs'),
_('set_sigint_handler', True,
  doc='Set a sigint handler to prevent csound crash with CTRL-C')
_('generalmidi_soundfont', '')
_('suppress_output', True,
  doc='Suppress csound´s debugging information')
_('unknown_parameter_fail_silently', True,
  doc='Do not raise if a synth tries to set an unknown parameter')
_('define_builtin_instrs', True,
  doc="If True, a Session with have all builtin instruments defined")
_('sample_fade_time', 0.05,
  doc="Fade time when playing samples via a Session")
_("prefer_udp", True,
  doc="If true and a server was defined prefer UDP over the API for communication")
_("start_udp_server", False,
  doc="Start an engine with udp communication support")
_('associated_table_min_size', 16,
  doc="Min. size of the param table associated with a synth")
_('num_audio_buses', 64,
  doc="Num. of audio buses in an Engine/Session")
_('num_control_buses', 512,
  doc="Num. of control buses in an Engine/Session")
_('html_theme', 'light',
  choices={'dark', 'light'},
  doc="Style to use when displaying syntax highlighting")
_('html_args_fontsize', '12px',
  doc="Font size used for args when outputing html (in jupyter)")
_('synth_repr_max_args', 12,
  doc="Max. number of pfields shown when in a synth's repr")
_('synthgroup_repr_max_rows', 16,
  doc='Max. number of rows for a SynthGroup repr')
_('jupyter_synth_repr_stopbutton', True,
  doc='When running inside a jupyter notebook, display a stop button'
      'for Synths and SynthGroups')
_('jupyter_synth_repr_interact', True,
  doc='When inside jupyter, add interactive widgets if a synth has'
      'named parameters')
_('jupyter_instr_repr_show_code', True,
  doc='Show code when displaying an Instr inside jupyter')
_('ipython_load_magics_at_startup', True,
  doc='Load csoundengine.magic at startup when inside ipython. If False, magics can '
      'still be loaded via `%load_ext csoundengine.magic`')
_('magics_print_info', True,
  doc='Print some informative information when the csounengine.magic extension is loaded')
_('jupyter_slider_width', '80%',
  doc='CSS Width used by an interactive slider in jupyter')
_('timeout', 2.,
  doc='Timeout for any action waiting a response from csound')

assert 'num_control_buses' in config.default
config.load()
