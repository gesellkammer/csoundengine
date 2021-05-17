import logging
from configdict import ConfigDict

modulename = 'csoundengine.engine'

logger = logging.getLogger('csoundengine')
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("*** csoundengine: %(message)s\n"))
logger.addHandler(_handler)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                  CONFIG                   #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

config = ConfigDict(modulename.replace(".", ":"))
_ = config.addKey

_('sr', 0,
  choices=(0, 22050, 44100, 48000, 88200, 96000),
  doc='samplerate - 0=default sr for the backend')
_('rec.sr', 44100,
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
_('rec.ksmps', 64,
  choices=(16, 32, 64, 128, 256),
  doc="samples per cycle when rendering")
_('rec.sample_format', 'float',
  choices=(16, 24, 32, 'float'),
  doc="Sample format used when rendering")
_('buffersize', 0,
  doc="-b value. 0=determine buffersize depending on ksmps & backend")
_('numbuffers', 0,
  doc="determines the -B value as a multiple of the buffersize. 0=auto")
_('linux.backend', 'jack',
  choices=('jack', 'pa_cb', 'pa_bl', 'pulse', 'alsa')),
_('macos.backend', 'pa_cb',
  choices=('auhal', 'pa_cb', 'pa_bl'))
_('windows.backend', 'pa_cb',
  choices=('pa_cb', 'pa_bl'))
_('fallback_backend', 'pa_cb',
  choices=('pa_cb', 'pa_bl', ''),
  doc="Fallback backend if the preferred backend is not available.")
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
  doc='Supress csound´s debugging information')
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

config.load()
