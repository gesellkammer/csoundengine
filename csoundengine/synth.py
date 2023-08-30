from __future__ import annotations
import time
import numpy as np
from functools import cache
from .config import logger, config
from . import internalTools
from . import baseevent
from emlib import iterlib
import emlib.misc
from . import jupytertools
from abc import abstractmethod

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import Engine
    from .instr import Instr
    from .paramtable import ParamTable
    from .session import Session


__all__ = (
    'AbstrSynth',
    'Synth',
    'SynthGroup'
)

#
_EMPTYSET = set()


class AbstrSynth(baseevent.BaseEvent):
    """
    A base class for Synth and SynthGroup

    Args:
        p1: the event id
        start: the start time relative to the start of the engine
        dur: the duration of the synth
        engine: the parent engine
        autostop: if True, link the lifetime of the csound synth to the lifetime
            of this object
        priority: the priority (order of evaluation) of the synth, where events with
            a low priority are evaluated before events with a higher priority
    """
    __slots__ = ('engine', 'autostop', 'priority')

    def __init__(self,
                 p1: float,
                 start: float,
                 dur: float,
                 engine: Engine,
                 autostop: bool = False,
                 priority: int = 1):
        super().__init__(p1, start, dur, ())

        self.engine: Engine = engine
        "The engine used to schedule this synth"

        self.autostop: bool = autostop
        "If True, stop the underlying csound synth when this object is deleted"

        self.priority: int = priority
        "The priority of this synth. Lower priorities are evaluated first in the chain"

    def __del__(self):
        if self.autostop:
            self.stop(stopParent=False)

    def _repr_html_(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def stop(self, delay=0., stopParent=False) -> None:
        """ Stop this synth """
        raise NotImplementedError()

    def playing(self) -> bool:
        """ Is this synth playing? """
        raise NotImplementedError()

    def finished(self) -> bool:
        """ Has this synth ceased to play? """
        raise NotImplementedError()

    def wait(self, pollinterval: float = 0.02, sleepfunc=time.sleep) -> None:
        """
        Wait until this synth has stopped

        Args:
            pollinterval: polling interval in seconds
            sleepfunc: the function to call when sleeping, defaults to time.sleep
        """
        internalTools.setSigintHandler()
        try:
            while self.playing():
                sleepfunc(pollinterval)
        except:
            raise
        internalTools.removeSigintHandler()

    def set(self, *args, delay=0., **kws) -> None:
        """
        Set a value of a named parameter

        Example
        ~~~~~~~

            >>> from csoundengine import *
            >>> s = Engine().session()
            >>> s.defInstr('osc', r'''
            ... kfreq = p5
            ... kamp = p6
            ... outch 1, oscili:a(kamp, kfreq)
            ... ''')
            >>> synth = s.sched('osc', kfreq=1000, kamp=0.5)
            >>> synth.set('kfreq', 440)
            >>> # Parameters can be given as keyword args:
            >>> synth.set(kfreq=440, delay=2.5)
            >>> # Multiple params can be set at a time
            >>> synth.set(kfreq=442, kamp=0.1)
        """
        mode = self.paramMode()
        if self.paramMode() == 'parg':
            return self.setp(*args, delay=delay, **kws)
        elif mode == 'table':
            return self._setTable(*args, **kws)
        else:
            logger.error(f"This {type(self)} does not have any dynamic parameters")

    def _setTable(self, *args, **kws) -> None:
        """
        Set a value of a param table

        Multiple syntaxes are possible::

            synth.set('key1', value1, ['key2', value2, ...])
            synth.set(key1=value1, [key2=value2, ...])

        .. seealso::

            * :meth:`~AbstrSynth.set`

        Example
        =======

        .. code::

            from csoundengine import *
            session = Engine().session()
            session.defInstr('sine', r'''
                {kamp=0.1, kfreq=1000}
                outch 1, oscili:a(kamp, kfreq)
            ''')
            synth = session.sched('sine', tabargs={'kfreq': 440})
            synth.set('kfreq', 2000, delay=3)
        """
        raise NotImplementedError()

    def get(self, slot: int | str, default: float | None = None) -> float | None:
        """
        Get the value of a named parameter

        Args:
            slot (int|str): the slot name/index
            default (float): if given, this value will be returned if the slot
                does not exist

        Returns:
            the current value of the given table slot / named pfield, or default
            if the key does not match any named parameter

        .. seealso:: :meth:`~Synth.getp`, :meth:`~Synth.set`

        """
        raise NotImplementedError()

    def _tableParams(self) -> set[str]:
        """
        Return a set of all named table parameters

        Returns None if this synth does not have a parameters table

        .. seealso::

            * :meth:`~AbstrSynth._automateTable`
            * :meth:`~AbstrSynth.hasParamTable`
            * :meth:`~AbstrSynth._tableState`
            * :meth:`~AbstrSynth._namedPfields`

        """
        raise NotImplementedError()

    def _tableState(self) -> dict[str, float]:
        """
        Get the state of all named parameters defined through a param table

        Returns:
            a dict mapping parameter name to its current value
        """
        raise NotImplementedError()

    def hasParamTable(self) -> bool:
        """ Does this synth/group have an associated parameter table?"""
        raise NotImplementedError()

    def paramMode(self) -> str:
        """
        Returns the dynamic parameter mode, or None

        Returns one of 'parg' or 'table'
        """
        return 'parg'

    def _automateTable(self, param: str, pairs: list[float] | np.ndarray,
                       mode="linear", delay=0., overtake=False) -> float:
        """
        Automate a table parameter. Time stamps are relative to the start
        of the automation

        Args:
            param: the named parameter as defined in the Instr
            pairs: a flat list of pairs of the form [time0, val0, time1, val1, ...]
            mode: one of 'linear', 'cos', 'expon(xx)', 'smooth'. See the csound opcode
                `interp1d` for more information
                (https://csound-plugins.github.io/csound-plugins/opcodes/interp1d.html)
            delay: time offset to start automation
            overtake: if True, the first value of pairs is replaced with
                the current value in the running instance

        .. seealso::

            * :meth:`~AbstrSynth.set`
            * :meth:`~AbstrSynth.get`

        """
        raise NotImplementedError

    def _namedPfields(self) -> set[str]:
        """
        Returns a set of all named pfields
        """
        raise NotImplementedError

    def dynamicParams(self) -> set[str]:
        raise NotImplementedError

    def namedParams(self) -> set[str]:
        """
        Returns a set of named parameters, or None if this Synth has no named parameters

        These parameters can be modified via :meth:`~AbstrSynth.set` or
        :meth:`~AbstrSynth.automate`
        """
        mode = self.paramMode()
        if mode == 'parg':
            return self._namedPfields()
        else:
            return self._tableParams()
        
    def _automatep(self,
                   param: int | str,
                   pairs: list[float] | np.ndarray,
                   mode="linear",
                   delay=0.,
                   overtake=False) -> float:
        """
        Automate the value of a pfield.

        Args:
            param: either the pfield index (5=p5) or the name of the pfield
                as used in the body of the instrument (for example, if the
                body contains the line "kfreq = p5", "kfreq" could be used as param)
            pairs: 1D sequence of floats with the form
                [x0, y0, x1, y1, x2, y2, ...]
            mode: one of 'linear', 'cos', 'expon(xx)', 'smooth'. See the csound opcode
                `interp1d` for more information
                (https://csound-plugins.github.io/csound-plugins/opcodes/interp1d.html)
            delay: 0 to start as soon as possible, otherwise the delay
                before the automateion starts
            overtake: if True, the first value of pairs is replaced with
                the current value in the running instance

        Returns:
            a float with the synthid (p1) of the automation event. A value of 0 indicates
            that no automation event was scheduled, possibly because there is no
            intersection between the time of the synth and that of the automation

        .. seealso:: :meth:`~AbstrSynth.setp`, :meth:`~AbstrSynth.automate`
        """
        raise NotImplementedError()

    def automate(self,
                 param: str | int,
                 pairs: list[float] | np.ndarray,
                 mode='linear',
                 delay=0.,
                 overtake=False
                 ) -> float:
        """
        Automate any named parameter of this Synth

        This method will automate this synth's pfields / param table, depending of
        how the instrument was defined.

        Args:
            param: the name of the parameter to automate, or the param index
            pairs: automation data as a flat array with the form [time0, value0, time1, value1, ...]
            mode: one of 'linear', 'cos'. Determines the curve between values
            delay: when to start the automation
            overtake: if True, do not use the first value in pairs but overtake the current value

        Returns:
            the eventid of the automation event
        """
        paramMode = self.paramMode()
        if paramMode == 'table':
            return self._automateTable(param=param, pairs=pairs, mode=mode, delay=delay, overtake=overtake)
        elif paramMode == 'parg':
            return self._automatep(param=param, pairs=pairs, mode=mode, delay=delay, overtake=overtake)
        else:
            raise RuntimeError("This Synth does not define any dynamic parameters")


    def setp(self, *args, delay=0., **kws) -> None:
        """
        Modify the value of a pfield.

        .. seealso:: :meth:`~AbstrSynth.set`, :meth:`~AbstrSynth.automate`
        """
        raise NotImplementedError()

    @property
    def session(self) -> Session:
        """
        Returns the Session which scheduled self
        """
        return self.engine.session()

    def show(self) -> None:
        """
        If inside jupyter, display the html representation of self
        """
        if jupytertools.inside_jupyter():
            from IPython.display import display
            display(self)


_synthStatusIcon = {
    'playing': '▶',
    'stopped': '■',
    'future': '𝍪'
}


class Synth(AbstrSynth):
    """
    A Synth represents one running csound event

    Args:
        engine: the engine instance where this synth belongs to
        synthid: the synth id inside csound (p1, a fractional instrument number)
        instr: the Instr which originated this Synth
        start: start time of the synth, relative to the engine's elapsed time
        dur: duration of the event (can be -1 for infinite)
        args: the pfields used to create this synth (starting at p4)
        autostop: should this synth autostop? If True, the lifetime of the csound note
            is associated with this Synth object, so if this Synth goes out of
            scope or is deleted, the underlying note is unscheduled
        table: an associated Table (if needed)

    Example
    =======

    .. code::

        from csoundengine import *
        from pitchtools import n2m
        session = Engine().session()
        session.defInstr('vco', r'''
            |kamp=0.1, kmidi=60, ktransp=0|
            asig vco2 kamp, mtof:k(kmidi+ktransp)
            asig *= linsegr:a(0, 0.1, 1, 0.1, 0)
            outch 1, asig
        ''')
        notes = ['4C', '4E', '4G']
        synths = [session.sched('vco', kamp=0.2, kmidi=n2m(n)) for n in notes]
        # synths is a list of Synth
        # automate ktransp in synth 1 to produce 10 second gliss of 1 semitone downwards
        synths[1].automate('ktransp', [0, 0, 10, -1])

    """
    __slots__ = ('instr', 'table', 'group', '_playing')

    def __init__(self,
                 engine: Engine,
                 p1: float,
                 instr: Instr,
                 start: float,
                 dur: float = -1,
                 args: list[float] | None = None,
                 autostop=False,
                 table: ParamTable = None,
                 priority: int = 1,
                 ) -> None:

        AbstrSynth.__init__(self, p1=p1, start=start, dur=dur, engine=engine, autostop=autostop, priority=priority)

        self.instr: Instr = instr
        """The Instr used to play this synth"""

        self.table: ParamTable | None = table
        """A ParamTable used to define parameters if using a table"""

        self.args = args
        """The args used to schedule this synth"""

        self._playing: bool = True

    def _html(self) -> str:
        argsfontsize = config['html_args_fontsize']
        maxi = config['synth_repr_max_args']
        style = jupytertools.defaultPalette
        playstr = _synthStatusIcon[self.playStatus()]
        parts = [
            f'{playstr} <strong style="color:{style["name.color"]}">'
            f'{self.instr.name}</strong>:{self.p1:.4f}',
            ]
        if self.table is not None:
            parts.append(self.table._mappingRepr())
        if self.args:
            i2n = self.instr.pargsIndexToName
            argsstrs = []
            pargs = self.args[0:]
            if any(arg.startswith('k') for arg in self.instr.pargsNameToIndex):
                maxi = max(i+4 for i, n in i2n.items()
                           if n.startswith('k'))
            for i, parg in enumerate(pargs, start=4):
                if i > maxi:
                    argsstrs.append("…")
                    break
                name = i2n.get(i)
                if not isinstance(parg, str):
                    parg = f'{parg:.6g}'
                if name:
                    idxstr = str(i)
                    if self.instr.aliases and (alias := self.instr.aliases.get(name)):
                        s = f"{idxstr}:<b>{alias}({name})</b>=<code>{parg}</code>"
                    else:
                        s = f"{idxstr}:<b>{name}</b>=<code>{parg}</code>"
                else:
                    # s = f"<b>p{i + 4}</b>=<code>{parg:.6g}</code>"
                    s = f"<b>{i}</b>=<code>{parg}</code>"
                argsstrs.append(s)
            argsstr = " ".join(argsstrs)
            argsstr = fr'<span style="font-size:{argsfontsize};">{argsstr}</span>'
            parts.append(argsstr)
        # return '<span style="font-size:12px;">∿(' + ', '.join(parts) + ')</span>'
        return '<span style="font-size:12px;">Synth('+', '.join(parts)+')</span>'

    def _repr_html_(self) -> str:
        if jupytertools.inside_jupyter():
            if self.playing():
                if config['jupyter_synth_repr_stopbutton']:
                    jupytertools.displayButton("Stop", self.stop)
                if config['jupyter_synth_repr_interact'] and self.args:
                    pass
        return f"<p>{self._html()}</p>"

    def __repr__(self):
        playstr = _synthStatusIcon[self.playStatus()]
        parts = [f'{playstr} {self.instr.name}:{self.p1} start:{self.start:.3f} dur:{self.dur:.3f}']
        if self.table is not None:
            parts.append(self.table._mappingRepr())
        if self.args:
            maxi = config['synth_repr_max_args']
            i2n = self.instr.pargsIndexToName
            maxi = max((i for i, name in i2n.items() if name.startswith("k")),
                       default=maxi)
            argsstrs = []
            pargs = self.args[0:]
            for i, parg in enumerate(pargs, start=0):
                if i > maxi:
                    argsstrs.append("...")
                    break
                name = i2n.get(i+4)
                if not isinstance(parg, str):
                    parg = f'{parg:.6g}'
                if name:
                    s = f"{name}:{i+4}={parg}"
                    #s = f"p{i+4}:{name}={parg:.8g}"
                else:
                    s = f"p{i+4}={parg}"
                argsstrs.append(s)
            argsstr = " ".join(argsstrs)
            parts.append(argsstr)
        lines = ["Synth(" + " ".join(parts) + ")"]
        # add a line for k- pargs
        return "\n".join(lines)

    def playStatus(self) -> str:
        """
        Returns the playing status of this synth (playing, stopped or future)

        Returns:
            'playing' if currently playing, 'stopped' if this synth has already stopped
            or 'future' if it has not started

        """
        if self._playing:
            return "playing"
        else:
            if self.start > self.engine.elapsedTime():
                return "future"
            else:
                return "stopped"

    def playing(self) -> bool:
        """ Is this Synth playing """
        return self.playStatus() == "playing"

    def finished(self) -> bool:
        return self.playStatus() == 'stopped'

    def _tableState(self) -> dict[str, float] | None:
        if self.table is None:
            return None
        return self.table.asDict()

    def _tableParams(self) -> set[str] | None:
        if self.table is None:
            return None
        return set(self.table.mapping.keys())

    def _setTable(self, *args, delay=0., **kws) -> None:
        if not self._playing:
            logger.info("synth not playing")

        if not self.table:
            logger.error("This synth has no associated table, skipping")
            return

        if delay > 0:
            if args:
                for key, value in iterlib.pairwise(args):
                    slotidx = self.table.paramIndex(key)
                    if slotidx is None:
                        logger.debug(f"Param {key} unknown")
                    else:
                        self.engine.tableWrite(self.table.tableIndex, slotidx, value, delay=delay)
            if kws:
                for key, value in kws.items():
                    slotidx = self.table.paramIndex(key)
                    if slotidx is None:
                        logger.debug(f"Param {key} unknown")
                    else:
                        self.engine.tableWrite(self.table.tableIndex, slotidx, value,
                                               delay=delay)
        else:
            if args:
                for key, value in iterlib.pairwise(args):
                    self.table[key] = value
            if kws:
                for key, value in kws.items():
                    self.table[key] = value

    def get(self, slot: int | str, default: float = None
            ) -> float | None:
        if not self._playing:
            logger.error("Synth not playing")
            return

        if self.paramMode() is None:
            logger.info("This synth has no dynamic parameters")
            return default
        if self.table:
            return self.table.get(slot, default)
        elif slot in self._namedPfields():
            return self.getp(slot)
        else:
            return default

    def dynamicParams(self) -> set[str]:
        """
        The set of all dynamic parameters accepted by this Synth

        If the instrument used in this synth uses aliases, these are
        also included

        Returns:
            a set of the dynamic (modifiable) parameters accepted by this synth
        """
        return self.instr.dynamicParamKeys(includeRealNames=True)

    def _namedPfields(self) -> set[str]:
        name2idx = self.instr.pargsNameToIndex
        return set(name2idx.keys()) if name2idx else _EMPTYSET

    def setp(self, *args, delay=0., **kws) -> None:
        """
        Modify a pfield of this synth.

        Multiple pfields can be modified simultaneously. It only makes sense
        to modify a pfield if a k-rate variable was assigned to it.
        (see example). A pfield can be referred to via an integer, corresponding
        to the p index (5 would refer to p5), or to the name of the assigned k-rate
        variable as a string (for example, if there is a line "kfreq = p6",
        both 6 and "kfreq" refer to the same pfield).

        Example
        =======

            >>> session = Engine(...).session()
            >>> session.defInstr("sine",
            '''
            kamp = p5
            kfreq = p6
            outch 1, oscili:ar(kamp, kfreq)
            '''
            )
            >>> synth = session.sched('sine', args=[0.1, 440])
            >>> synth.setp(5, 0.5)
            >>> synth.setp(kfreq=880)
            >>> synth.setp(5, 0.1, 6, 1000)
            >>> synth.setp("kamp", 0.2, 6, 440)

        .. seealso::

            - :meth:`AbstrSynth.set`
            - :meth:`Synth.automate`
            - :meth:`Synth.getp`
            - :meth:`Synth.automate`

        """
        if self.playStatus() == 'future':
            # Can we just modify the scheduled value?
            return

        # most common use: just one pair
        if not kws and len(args) == 2:
            k = args[0]
            idx = k if isinstance(k, int) else self.instr.pargIndex(k)
            self.engine.setp(self.p1, idx, args[1], delay=delay)
            return
        pairsd = {}
        instr = self.instr
        if args:
            assert len(args) % 2 == 0, f"Arguments should be even, got {args}"
            for i in range(len(args) // 2):
                k = args[i*2]
                v = args[i*2+1]
                idx = instr.pargIndex(k)
                pairsd[idx] = v
        if kws:
            for k, v in kws.items():
                idx = instr.pargIndex(k)
                pairsd[idx] = v
        pairs = iterlib.flatdict(pairsd)
        self.engine.setp(self.p1, *pairs, delay=delay)

    def getp(self, pfield: int | str) -> float | None:
        """
        Get the current value of a pfield

        Args:
            pfield: the name/index of the pfield

        Returns:
            the current value of the given pfield

        .. seealso::

            - :meth:`~Synth.setp`
            - :meth:`~Synth.automate`
            - :meth:`~Synth.ui`
        """
        if self.playStatus() == 'future':
            # Can we just modify the scheduled value?
            return
        idx = pfield if isinstance(pfield, int) else self.instr.pargIndex(pfield)
        return self.engine.getp(self.p1, idx)

    def ui(self, **specs: dict[str, tuple[float, float]]) -> None:
        """
        Modify dynamic (named) arguments through an interactive user-interface

        If run inside a jupyter notebook, this method will create embedded widgets
        to control the values of the dynamic pfields of an event. Dynamic pfields
        are those assigned to a k-variable or declared as ``|karg|`` (see below)

        Args:
            specs: a dict mapping named arg to a tuple (minvalue, maxvalue)

        Example
        =======

        .. code::

            # Inside jupyter
            from csoundengine import *
            s = Engine().session()
            s.defInstr('vco', r'''
              |kmidinote, kampdb=-12, kcutoff=3000, kres=0.9|
              kfreq = mtof:k(kmidinote)
              asig = vco2:a(ampdb(kampdb), kfreq)
              asig = moogladder2(asig, kcutoff, kres)
              asig *= linsegr:a(0, 0.1, 1, 0.1, 0)
              outs asig, asig
            ''')
            synth = s.sched('vco', kmidinote=67)
            # Specify the ranges for some sliders. All named parameters
            # are assigned a widget
            synth.ui(kampdb=(-48, 0), kres=(0, 1))

        .. figure:: assets/synthui.png

        .. seealso::

            - :meth:`Engine.eventui`

        """
        from . import interact
        if not self.instr.pargsIndexToName:
            logger.info(f"This synth has no named arguments (instr='{self.instr.name}')")
            return
        pairs = list((idx, name) for idx, name in self.instr.pargsIndexToName.items()
                     if name.startswith('k'))
        if not pairs:
            logger.error("The instrument has no dynamic (k) arguments")
            return

        pairs.sort()
        pargindexes, pargnames = zip(*pairs)

        if self.playStatus() == 'future':
            pvalues = [self.instr.pargsIndexToDefaultValue[idx]
                       for idx in pargindexes]
        else:
            pvalues = [self.engine.getp(self.p1, idx) for idx in pargindexes]
        paramspecs = {}
        for idx, pargname, value in zip(pargindexes, pargnames, pvalues):
            if pargname in specs:
                minval, maxval = specs[pargname]
            else:
                minval, maxval = interact._guessRange(value, pargname)
            paramspecs[idx] = interact.ParamSpec(pargname,
                                                 minvalue=minval,
                                                 maxvalue=maxval,
                                                 startvalue=value)
        return interact.interactPargs(self.engine, self.p1, specs=paramspecs)


    def hasParamTable(self) -> bool:
        """ Returns True if this synth has an associated parameter table """
        return self.table is not None

    def _automateTable(self,
                       param: str, pairs: list[float] | np.ndarray,
                       mode="linear", delay=0., overtake=False) -> float:
        """
        Automate a table parameter. Time stamps are relative to the start
        of the automation

        Args:
            param: the named parameter as defined in the Instr
            pairs: a flat list of pairs of the form [time0, val0, time1, val1, ...]
            mode: one of 'linear', 'cos', 'expon(xx)', 'smooth'. See the csound opcode
                `interp1d` for more information
                (https://csound-plugins.github.io/csound-plugins/opcodes/interp1d.html)
            delay: time offset to start automation
            overtake: if True, the first value of pairs is replaced with
                the current value in the running instance

        .. seealso::

            * :meth:`~Synth.set`
            * :meth:`~Synth.get`

        """
        if not self.table:
            raise RuntimeError(f"{self.instr.name} (id={self.p1}) has no parameter table")
        paramidx = self.table.paramIndex(param)
        if paramidx is None:
            raise KeyError(f"Unknown param {param} for synth {self.p1}")
        if len(pairs)>1900:
            raise ValueError(f"pairs is too long (max. pairs = 900, got {len(pairs)/2})")
        return self.engine.automateTable(self.table.tableIndex, paramidx, pairs,
                                         mode=mode, delay=delay, overtake=overtake)

    def paramMode(self) -> str | None:
        return self.instr.paramMode()

    def automate(self,
                 param: str,
                 pairs: list[float] | np.ndarray,
                 mode='linear',
                 delay=0.,
                 overtake=False
                 ) -> float:
        """
        Automate any named parameter of this Synth

        Raises KeyError if the parameter is unknown

        Args:
            param: the name of the parameter to automate
            pairs: automation data as a flat array with the form [time0, value0, time1, value1, ...]
            mode: one of 'linear', 'cos'. Determines the curve between values
            delay: when to start the automation
            overtake: if True, do not use the first value in pairs but overtake the current value

        Returns:
            the eventid of the automation event.
        """
        now = self.engine.elapsedTime()
        automStart = now + delay + pairs[0]
        automEnd = now + delay + pairs[-2]
        if automEnd <= self.start or automStart >= self.end:
            # automation line ends before the actual event!!
            logger.debug(f"Automation times outside of this synth: {param=}, "
                         f"automation start-end: {automStart} - {automEnd}, "
                         f"synth: {self}")
            return 0

        if automStart > self.start or automEnd < self.end:
            pairs, delay = internalTools.cropDelayedPairs(pairs=pairs, delay=delay+now, start=automStart, end=automEnd)
            if not pairs:
                return 0
            delay = delay - now

        if pairs[0] > 0:
            pairs, delay = internalTools.consolidateDelay(pairs, delay)

        paramMode = self.instr.paramMode()
        if paramMode == 'table':
            return self._automateTable(param=param, pairs=pairs, mode=mode, delay=delay, overtake=overtake)
        elif paramMode == 'parg':
            return self._automatep(param=param, pairs=pairs, mode=mode, delay=delay, overtake=overtake)
        else:
            raise RuntimeError("This Synth does not define any dynamic parameters")


    def _automatep(self, param: int | str, pairs: list[float] | np.ndarray,
                   mode="linear", delay=0., overtake=False) -> float:
        if self.playStatus() == 'stopped':
            raise RuntimeError("This synth has already stopped, cannot automate")

        if isinstance(param, str):
            pidx = self.instr.pargIndex(param)
            if not pidx:
                raise KeyError(f"parg {param} not known. "
                               f"Known pargs: {self.instr.pargsIndexToName}")
        else:
            pidx = param
        synthid = self.engine.automatep(self.p1, pidx=pidx, pairs=pairs,
                                        mode=mode, delay=delay, overtake=overtake)
        return synthid

    def stop(self, delay=0., stopParent=False) -> None:
        if self.finished():
            return
        self.session.unsched(self.p1, delay=delay)


def _synthsCreateHtmlTable(synths: list[Synth]) -> str:
    synth0 = synths[0]
    instr0 = synth0.instr
    if any(synth.instr != instr0 for synth in synths):
        # multiple instrs per group, not allowed here
        raise ValueError("Only synths of the same instr allowed here")
    colnames = ["p1", "start", "dur"]
    maxrows = config['synthgroup_repr_max_rows']
    if maxrows and len(synths) > maxrows:
        limitSynths = True
        synths = synths[:maxrows]
    else:
        limitSynths = False
    rows: list[list[str]] = [[] for _ in synths]
    now = synth0.engine.elapsedTime()
    for row, synth in zip(rows, synths):
        row.append(f'{synth.p1} <b>{_synthStatusIcon[synth.playStatus()]}</b>')
        row.append("%.3f" % (synth.start - now))
        row.append("%.3f"%synth.dur)
    if synth0.table is not None:
        keys = list(synth0.table.mapping.keys())
        colnames.extend(synth0.table.mapping.keys())
        for row, synth in zip(rows, synths):
            if synth.playStatus() != 'stopped':
                values = synth.table.array[:len(keys)]
                for value in values:
                    row.append(f'<code>{value}</code>')
            else:
                row.extend(["-"] * len(keys))

    if synth0.args:
        maxi = config['synth_repr_max_args']
        i2n = instr0.pargsIndexToName
        maxi = max((i for i, name in i2n.items() if name.startswith("k")),
                   default=maxi)
        for i, parg in enumerate(synth0.args):
            if i > maxi:
                colnames.append("...")
                break
            name = i2n.get(i+4)
            if name:
                colnames.append(f"{i+4}:{name}")
            else:
                colnames.append(str(i+4))
        for row, synth in zip(rows, synths):
            row.extend("%.5g"%parg if not isinstance(parg, str) else parg
                       for parg in synth.args[:maxi])
            if len(synth.args) > maxi:
                row.append("...")

    if limitSynths:
        rows.append(["..."])
    return emlib.misc.html_table(rows, headers=colnames)


class SynthGroup(AbstrSynth):
    """
    A SynthGroup is used to control multiple synths

    Such multiple synths can be groups of similar synths, as in additive
    synthesis, or processing chains which work as an unity.

    Attributes:
        synths (list[AbstrSynth]): the list of synths in this group

    Example
    ~~~~~~~

        >>> import csoundengine as ce
        >>> session = ce.Engine().session()
        >>> session.defInstr('oscil', r'''
        ... |kfreq, kamp=0.1, kcutoffratio=5, kresonance=0.9|
        ... a0 = vco2(kamp, kfreq)
        ... a0 = moogladder2(a0, kfreq * kcutoffratio, kresonance)
        ... outch 1, a0
        ... ''')
        >>> synths = [session.sched('oscil', kfreq=freq)
        ...           for freq in range(200, 1000, 75)]
        >>> group = ce.synth.SynthGroup(synths)
        >>> group.set(kcutoffratio=3, delay=3)
        >>> group.automate('kresonance', (1, 0.3, 10, 0.99))
        >>> group.stop(delay=11)

    """
    __slots__ = ('synths', '__weakref__')

    def __init__(self, synths: list[Synth], autostop=False) -> None:
        # assert isinstance(synths, list) and len(synths) > 0
        priority = max(synth.priority for synth in synths)
        start = min(synth.start for synth in synths)
        end = max(synth.end for synth in synths)
        dur = end - start
        AbstrSynth.__init__(self, p1=0, start=start, dur=dur,
                            engine=synths[0].engine, autostop=autostop,
                            priority=priority)
        flatsynths: list[Synth] = []
        for synth in synths:
            if isinstance(synth, SynthGroup):
                flatsynths.extend(synth)
            else:
                flatsynths.append(synth)
        self.synths: list[Synth] = flatsynths

    def extend(self, synths: list[Synth]) -> None:
        """
        Add the given synths to the synths in this group
        """
        self.synths.extend(synths)

    def stop(self, delay=0, stopParent=False) -> None:
        for s in self.synths:
            s.stop(stopParent=False, delay=delay)

    def playing(self) -> bool:
        return any(s.playing() for s in self.synths)

    def finished(self) -> bool:
        return all(s.finished() for s in self.synths)

    def _automateTable(self, param: str, pairs, mode="linear", delay=0.,
                       overtake=False) -> None:
        for synth in self.synths:
            if isinstance(synth, Synth):
                if synth.table and param in synth._tableParams():
                    synth._automateTable(param, pairs, mode=mode, delay=delay,
                                         overtake=overtake)
            elif isinstance(synth, SynthGroup):
                synth._automateTable(param=param, pairs=pairs, mode=mode, delay=delay,
                                     overtake=overtake)

    @cache
    def dynamicParams(self) -> set[str]:
        out: set[str] = set()
        for synth in self.synths:
            dynamicParams = synth.dynamicParams()
            out.update(dynamicParams)
        return out

    @cache
    def _namedPfields(self) -> set[str]:
        out: set[str] = set()
        for synth in self.synths:
            namedPargs = synth._namedPfields()
            if namedPargs:
                out.update(namedPargs)
        return out

    def automate(self,
                 param: int | str,
                 pairs: list[float] | np.ndarray,
                 mode="linear",
                 delay=0.,
                 overtake=False) -> list[float]:
        """
        Automate the given parameter for all the synths in this group

        If the parameter is not found in a given synth, the automation is skipped
        for the given synth. This is useful when a group includes synths using
        different instruments so an automation would only adress those synths
        which support a given parameter. Synths which have no time overlap with
        the automation are also skipped.

        Raises KeyError if the param used is not supported by any synth in this group
        Supported parameters can be checked via :meth:`SynthGroup.dynamicParams`

        Args:
            param: the parameter to automate
            pairs: a flat list of pairs (time0, value0, time1, value1, ...)
            mode: the iterpolation mode
            delay: the delay to start the automation
            overtake: if True, the first value is dropped and instead the current
                value of the given parameter is used. The same effect is achieved
                if the first value is given as 'nan', in this case also the current
                value of the synth is overtaken.

        Returns:
            a list of synthids. There will be one synthid per synth in this group.
            A synthid of 0 indicates that for that given synth no automation was
            scheduled, either because that synth does not support the given
            param or because the automation times have no intersection with the
            synth
        """
        synthids = []
        allparams = self.dynamicParams()
        if param not in allparams:
            lines = [f"Parameter '{param}' unknown. Known parameters: {allparams}",
                     f"Synths in this group:"]
            for synth in self.synths:
                lines.append("    " + repr(synth))
            logger.error("\n".join(lines))
            raise KeyError(f"Parameter {param} not supported by any synth in this group. "
                           f"Possible parameters: {allparams}")
        for synth in self.synths:
            if param in synth.instr.dynamicParams():
                synthid = synth.automate(param=param, pairs=pairs, mode=mode, delay=delay, overtake=overtake)
            else:
                synthid = 0
            synthids.append(synthid)
        return synthids

    def _automatep(self,
                   param: int | str,
                   pairs: list[float] | np.ndarray,
                   mode="linear",
                   delay=0.,
                   overtake=False) -> list[float]:
        return [synth._automatep(param, pairs, mode=mode, delay=delay, overtake=overtake)
                for synth in self.synths if synth.paramMode() == 'parg']

    def hasParamTable(self) -> bool:
        return any(s.hasParamTable() is not None for s in self.synths)

    def paramMode(self) -> str | None:
        modes = set(mode for synth in self.synths
                    if (mode:=synth.paramMode()) is not None)
        if len(modes) == 0:
            return None
        elif len(modes) == 1:
            return modes.pop()
        else:
            raise ValueError("This group has multiple param modes")

    def _tableState(self) -> dict[str, float] | None:
        dicts = [d for s in self.synths if (d:=s._tableState())]
        if not dicts:
            return None
        out = dicts[0]
        for d in dicts[1:]:
            out.update(d)
        return out

    def _uniqueInstr(self) -> bool:
        instr0 = self.synths[0].instr
        return all(synth.instr == instr0 for synth in self.synths if synth.playing())

    def _htmlTable(self) -> str:
        subgroups = iterlib.classify(self.synths, lambda synth: synth.instr.name)
        lines = []
        instrcol = jupytertools.defaultPalette["name.color"]
        for instrname, synths in subgroups.items():
            lines.append(f'<p>instr: <strong style="color:{instrcol}">'
                         f'{instrname}'
                         f'</strong> - <b>{len(synths)}</b> synths</p>')
            htmltable = _synthsCreateHtmlTable(synths)
            lines.append(htmltable)
        return '\n'.join(lines)

    def _repr_html_(self) -> str:
        if config['jupyter_synth_repr_stopbutton'] and jupytertools.inside_jupyter():
            jupytertools.displayButton("Stop", self.stop)
        now = self.synths[0].engine.elapsedTime()
        start = min(max(0., s.start - now) for s in self.synths)
        end = max(s.dur + s.start - now for s in self.synths)
        if any(s.dur < 0 for s in self.synths):
            end = float('inf')
        dur = end - start
        lines = [f'<small>SynthGroup - start: {start:.3f}, dur: {dur:.3f}, synths: {len(self.synths)}</small>']
        numrows = config['synthgroup_repr_max_rows']
        if numrows > 0:
            lines.append(self._htmlTable())
        else:
            subgroups = iterlib.classify(self.synths, lambda synth: synth.instr.name)
            instrline = []
            instrcol = jupytertools.defaultPalette["name.color"]
            for instrname, synths in subgroups.items():
                s = f'<strong style="color:{instrcol}">{instrname}</strong> - {len(synths)} synths'
                namedparams = synths[0]._namedPfields()
                kparams = [p for p in namedparams if p[0] == 'k']
                if kparams:
                    s += ' (' + ', '.join(kparams) + ')'
                instrline.append(s)
            line = f'<p><small>Instrs: {", ".join(instrline)}</small></p>'
            lines.append(line)
        return "\n".join(lines)

    def __repr__(self) -> str:
        lines = [f"SynthGroup(n={len(self.synths)})"]
        for synth in self.synths:
            lines.append("    "+repr(synth))
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.synths)

    def __getitem__(self, idx) -> Synth:
        return self.synths[idx]

    def __iter__(self):
        return iter(self.synths)

    def _setTable(self, *args, delay=0, **kws) -> None:
        for synth in self.synths:
            synth._setTable(*args, delay=delay, **kws)

    def get(self, idx: int | str, default=None) -> list[float | None]:
        """
        Get the value of a named parameter

        If a synth in this group is not playing or hasn't a tabarg
        with the given name/idx, `default` is returned for that
        slot. The returned list has the same size as the number of
        synths in this group
        """
        return [synth.get(idx, default=default) for synth in self.synths]

    def setp(self, *args, delay=0., **kws) -> None:
        for synth in self.synths:
            synth.setp(*args, delay=delay, **kws)

    def _tableParams(self) -> set[str]:
        """
        Returns a set of available table named parameters for this group
        """
        allparams = set()
        for synth in self.synths:
            params = synth._tableParams()
            if params:
                allparams.update(params)
        return allparams



