from __future__ import annotations
from configdict import ConfigDict
from datetime import datetime
import emlib.dialogs
import os


__all__ = ('state')

_home = os.path.expanduser("~")

def _isofmt(t:datetime) -> str:
    """Returns the time in iso format"""
    return t.isoformat(':', 'minutes')


_defaultState = {
    'last_run': datetime(1900, 1, 1).isoformat(),
    'soundfont_last_dir': _home,
    'soundfile_last_dir': _home,
    'soundfile_save_last_dir': _home,
}


state = ConfigDict("csoundengine.state", _defaultState, persistent=True)


def openFile(key, filter="All (*.*)", title="Open File"):
    folder = state[key]
    f = emlib.dialogs.selectFile(directory=folder, filter=filter, title=title)
    if f:
        folder = os.path.split(f)[0]
        state[key] = folder
    return f


def saveFile(key, filter="All (*.*)", title="Save File"):
    folder = state[key]
    f = emlib.dialogs.saveDialog(directory=folder, filter=filter, title=title)
    if f:
        folder = os.path.split(f)[0]
        state[key] = folder
    return f


def saveSoundfile(filter="Soundfiles (*.wav, *.flac, *.aif, *.aiff)",
                  title="Save Soundfile",
                  ensureSelection=False):
    out = saveFile(key="soundfile_save_last_dir", filter=filter, title=title)
    if not out and ensureSelection:
        raise ValueError("No file was selected for saving")
    return out


def openSoundfile(filter="Soundfiles (*.wav, *.flac, *.aif, *.aiff)",
                  title="Open Soundfile",
                  ensureSelection=False):
    out = openFile("soundfile_last_dir", filter=filter, title=title)
    if not out and ensureSelection:
        raise ValueError("No output selected")
    return out


def openSoundfont(filter="Soundfont (*.sf2)", title="Open Soundfont", ensureSelection=False):
    out = openFile("soundfont_last_dir", filter=filter, title=title)
    if not out and ensureSelection:
        raise ValueError("No soundfont selected")
    return out




