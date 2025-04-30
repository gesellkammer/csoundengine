from __future__ import annotations

import functools
import os

import typing
if typing.TYPE_CHECKING:
    from sf2utils.sf2parse import Sf2File


class SoundFontIndex:
    """
    Creates an index of presets for a given soundfont

    Attributes:
        instrs: a list of instruments, where each instrument is a tuple (instr. index, name)
        presets: a list of presets, where each preset is a tuple (bank, num, name)
        nameToIndex: a dict mapping instr name to index
        indexToName: a dict mapping instr idx to name
        nameToPreset: a dict mapping preset name to (bank, num)
        presetToName: a dict mapping (bank, num) to preset name
    """
    def __init__(self, soundfont: str):
        assert os.path.exists(soundfont)
        self.soundfont = soundfont
        instrs, presets = _soundfontInstrumentsAndPresets(soundfont)
        self.instrs: list[tuple[int, str]] = instrs
        self.presets: list[tuple[int, int, str]] = presets
        self.nameToIndex: dict[str, int] = {name:idx for idx, name in self.instrs}
        self.indexToName: dict[int, str] = {idx:name for idx, name in self.instrs}
        self.nameToPreset: dict[str, tuple[int, int]] = {name: (bank, num)
                                                         for bank, num, name in self.presets}
        self.presetToName: dict[tuple[int, int], str] = {(bank, num): name
                                                         for bank, num, name in self.presets}



@functools.cache
def soundfontIndex(sfpath: str) -> SoundFontIndex:
    """
    Make a SoundFontIndex for the given soundfont

    Args:
        sfpath: the path to a soundfont (.sf2) file

    Returns:
        a SoundFontIndex


    Example
    -------

        >>> from csoundengine import csoundlib
        >>> idx = csoundlib.soundfontIndex("/path/to/piano.sf2")
        >>> idx.nameToPreset
        {'piano': (0, 0)}
        >>> idx.nameToIndex
        {'piano': 0}
    """
    return SoundFontIndex(sfpath)


@functools.cache
def _sf2file(path: str) -> Sf2File:
    from sf2utils.sf2parse import Sf2File
    f = open(path, 'rb')
    return Sf2File(f)


@functools.cache
def _soundfontInstrumentsAndPresets(sfpath: str
                                    ) -> tuple[list[tuple[int, str]],
                                               list[tuple[int, int, str]]]:
    """
    Returns a tuple (instruments, presets)

    Where instruments is a list of tuples(instridx, instrname) and presets
    is a list of tuples (bank, presetnum, name)

    Args:
        sfpath: the path to the soundfont

    Returns:
        a tuple (instruments, presets), where instruments is a list
        of tuples (instrindex, instrname) and prests is a list of
        tuples (bank, presetindex, name)
    """
    sf = _sf2file(sfpath)
    instruments: list[tuple[int, str]] = [(num, instr.name.strip())
                                          for num, instr in enumerate(sf.instruments)
                                          if instr.name and instr.name != 'EOI']
    presets: list[tuple[int, int, str]] = [(p.bank, p.preset, p.name.strip())
                                           for p in sf.presets
                                           if p.name and p.name != 'EOP']
    presets.sort()
    return instruments, presets


def soundfontInstruments(sfpath: str) -> list[tuple[int, str]]:
    """
    Get instruments for a soundfont

    The instrument index is used by csound opcodes like `sfinstr`. These
    are different from soundfont programs, which are ordered in
    banks/presets

    Args:
        sfpath: the path to the soundfont. "?" to open a file-browser dialog

    Returns:
        list[tuple[int,str]] - a list of tuples, where each tuple has the form
        (index: int, instrname: str)
    """
    if sfpath == "?":
        from . import state
        sfpath = state.openSoundfont(ensureSelection=True)
    instrs, _ = _soundfontInstrumentsAndPresets(sfpath)
    return instrs


def soundfontPresets(sfpath: str) -> list[tuple[int, int, str]]:
    """
    Get presets from a soundfont

    Args:
        sfpath: the path to the soundfont. "?" to open a file-browser dialog

    Returns:
        a list of tuples ``(bank:int, presetnum:int, name: str)``
    """
    if sfpath == "?":
        from . import state
        sfpath = state.openSoundfont(ensureSelection=True)
    _, presets = _soundfontInstrumentsAndPresets(sfpath)
    return presets


def soundfontSelectPreset(sfpath: str
                          ) -> tuple[str, int, int] | None:
    """
    Select a preset from a soundfont interactively

    Returns:
        a tuple (preset name, bank, preset number) if a selection was made, None
        otherwise

    .. figure:: ../assets/select-preset.png
    """
    presets = soundfontPresets(sfpath)
    items = [f'{bank:03d}:{pnum:03d}:{name}' for bank, pnum, name in presets]
    import emlib.dialogs
    item = emlib.dialogs.selectItem(items, ensureSelection=True)
    if item is None:
        return None
    idx = items.index(item)
    preset = presets[idx]
    bank, pnum, name = preset
    return (name, bank, pnum)


def soundfontInstrument(sfpath: str, name: str) -> int | None:
    """
    Get the instrument number from a preset

    The returned instrument number can be used with csound opcodes like `sfinstr`
    or `sfinstr3`

    Args:
        sfpath: the path to a .sf2 file. "?" to open a file-browser dialog
        name: the instrument name

    Returns:
        the instrument index, if exists
    """
    if sfpath == "?":
        from . import state
        sfpath = state.openSoundfont(ensureSelection=True)
    sfindex = soundfontIndex(sfpath)
    return sfindex.nameToIndex.get(name)


@functools.cache
def soundfontKeyrange(sfpath: str, preset: tuple[int, int]) -> tuple[int, int] | None:
    """
    Determines the key range of a preset in a soundfont file.

    Args:
        sfpath: the path to a .sf2 file.
        preset: the preset number (bank, program)

    Returns:
        the key range of the preset, if exists
    """
    sf = _sf2file(sfpath)
    for p in sf.presets:
        if p.bank == preset[0] and p.preset == preset[1]:
            return p.key_range.start, p.key_range.stop
    return None


def soundfontPeak(sfpath: str, preset: tuple[int, int],
                  pitches: tuple[int, int] | None = None, dur=0.05
                  ) -> float:
    """
    Finds the peak amplitude of a soundfont preset.

    Args:
        sfpath: the path to a .sf2 file.
        preset: the preset number (bank, program)
        pitches: the pitches to play (min, max)
        dur: the duration of each note

    Returns:
        the peak amplitude of the soundfont preset
    """
    from csoundengine.offline import OfflineEngine
    e = OfflineEngine(nchnls=0, ksmps=128, numAudioBuses=0, numControlBuses=0)
    bank, prog = preset
    presetnum = 1
    if pitches is None:
        keyrange = soundfontKeyrange(sfpath, preset)
        if not keyrange:
            raise ValueError(f"No defined key range for preset {preset} in soundfont {sfpath}")
        minpitch, maxpitch = keyrange
        pitch1 = int((maxpitch - minpitch) * 0.2 + minpitch)
        pitch2 = int((maxpitch - minpitch) * 0.8 + minpitch)
        pitches = (pitch1, pitch2)
    e.compile(fr'''
    gi_sfhandle sfload "{sfpath}"
    gi_presetindex sfpreset {prog}, {bank}, gi_sfhandle, {presetnum}
    chnset 0, "sfpeak"

    instr sfpeak
        ipreset = p4
        ipitch1 = p5
        ipitch2 = p6
        kmax0 init 0
        a1 sfplaym 127, ipitch1, 1, 1, ipreset, 0
        a2 sfplaym 127, ipitch2, 1, 1, ipreset, 0
        kmax1 peak a1
        kmax2 peak a2
        kmax = max(kmax1, kmax2)
        if kmax > kmax0 then
            chnset kmax, "sfpeak"
        endif
        kmax0 = kmax
    endin
    ''')

    e.sched('sfpeak', 0, dur, (presetnum, pitches[0], pitches[1]))
    e.perform(extratime=0.1)
    assert e.csound is not None
    value = e.getControlChannel("sfpeak")
    e.stop()
    return float(value)


def showSoundfontPrograms(sfpath: str, glob="") -> None:
    """
    Print a list of sounfont presets/programs

    Args:
        sfpath: the path to the soundfont
        glob: if given, it is used to filter the presets to only those
            whose name matches the given glob pattern
    """
    import emlib.misc
    import fnmatch
    progs = soundfontPresets(sfpath)
    if glob:
        progs = [p for p in progs
                 if fnmatch.fnmatch(p[2], glob)]
    emlib.misc.print_table(progs, headers=('bank', 'num', 'name'), showindex=False)
