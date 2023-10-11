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

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .engine import Engine
    from .instr import Instr
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
        start: the start time relative to the start of the engine
        dur: the duration of the synth
        engine: the parent engine
        autostop: if True, link the lifetime of the csound synth to the lifetime
            of this object
    """

    __slots__ = ('engine', 'autostop')

    def __init__(self,
                 start: float,
                 dur: float,
                 engine: Engine,
                 autostop: bool = False):
        super().__init__(start=start, dur=dur)

        self.engine: Engine = engine
        "The engine used to schedule this synth"

        self.autostop: bool = autostop
        "If True, stop the underlying csound synth when this object is deleted"

    def stop(self, delay=0.) -> None:
        raise NotImplementedError

    def __del__(self):
        if self.autostop:
            self.stop()

    def playing(self) -> bool:
        """ Is this synth playing? """
        raise NotImplementedError

    def finished(self) -> bool:
        """ Has this synth ceased to play? """
        raise NotImplementedError

    def wait(self, pollinterval: float = 0.02, sleepfunc=time.sleep) -> None:
        """
        Wait until this synth has stopped

        Args:
            pollinterval: polling interval in seconds
            sleepfunc: the function to call when sleeping, defaults to time.sleep
        """
        internalTools.setSigintHandler()
        while self.playing():
            sleepfunc(pollinterval)
        internalTools.removeSigintHandler()

    @property
    def session(self) -> Session:
        """
        Returns the Session which scheduled self
        """
        return self.engine.session()


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
        getInstr: the Instr which originated this Synth
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
    __slots__ = ('instr', 'session', 'group', '_playing', 'priority', 'p1', 'args', 'controlsSlot')

    def __init__(self,
                 session: Session,
                 p1: float,
                 instr: Instr,
                 start: float,
                 dur: float = -1,
                 args: list[float] | None = None,
                 autostop=False,
                 priority: int = 1,
                 controlsSlot: int = -1,
                 ) -> None:

        AbstrSynth.__init__(self, start=start, dur=dur, engine=session.engine, autostop=autostop)

        if controlsSlot < 0 and instr.dynamicParams():
            raise ValueError("Synth has dynamic args but was not assigned a control slot")
        elif controlsSlot >= 1 and not instr.dynamicParams():
            logger.warning("A control slot was assigned but this synth does not have any controls")

        self.session: Session = session
        """The parent Session of this event"""

        self.p1: float = p1
        """Event id for this synth"""

        self.priority: int = priority
        """Priority of this synth (lower priority is evaluated first)"""

        self.instr: Instr = instr
        """The Instr used to create this synth"""

        self.args = args
        """The args used to schedule this synth"""

        self.controlsSlot: int = controlsSlot
        """Holds the slot for dynamic controls, 0 if this synth has no assigned slot"""

        self._playing: bool = True

    @property
    def body(self) -> str:
        return self.session.instrGenerateBody(self.instr)

    def _controlNames(self) -> frozenset[str] | None:
        return self.instr.controlNames()

    def getInstr(self) -> Instr:
        """
        The Instr used to generate this synth
        """
        return self.instr

    def _html(self) -> str:
        argsfontsize = config['html_args_fontsize']
        maxi = config['synth_repr_max_args']
        style = jupytertools.defaultPalette
        playstr = _synthStatusIcon[self.playStatus()]
        parts = [
            f'{playstr} <strong style="color:{style["name.color"]}">'
            f'{self.instr.name}</strong>:{self.p1:.4f}',
            ]

        if self.args and len(self.args) > 1:
            i2n = self.instr.pfieldIndexToName
            argsstrs = []
            # exclude p4
            firstpargs = 5
            pargs = self.args[1:]
            if any(arg.startswith('k') for arg in self.instr.pfieldNameToIndex):
                maxi = max(i+4 for i, n in i2n.items()
                           if n.startswith('k'))
            for i, parg in enumerate(pargs, start=5):
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

        if self.args:
            maxi = config['synth_repr_max_args']
            i2n = self.instr.pfieldIndexToName
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

    def dynamicParams(self) -> set[str]:
        """
        The set of all dynamic parameters accepted by this Synth

        If the instrument used in this synth uses aliases, these are
        also included

        Returns:
            a set of the dynamic (modifiable) parameters accepted by this synth
        """
        return self.instr.dynamicParamNames(includeRealNames=True)

    def _pfieldNames(self) -> set[str]:
        return self.instr.pfieldNames()

    def _sliceStart(self) -> int:
        return self.controlsSlot * self.session.dynamicArgsPerInstr

    def _setp(self, param: str, value: float, delay=0.) -> None:
        """
        Modify a pfield of this synth.

        This makes only sense if the pfield is assigned to a krate variable.
        A pfield can be referred as 'p4', 'p5', etc., or to the
        name of the assigned k-rate variable as a string (for example, if there
        is a line "kfreq = p6", both 'p6' and 'kfreq' refer to the same pfield).

        If the parameter name does not fit any known parameter a KeyError exception
        is raised

        Example
        ~~~~~~~

            >>> session = Engine(...).session()
            >>> session.defInstr("sine",
            '''
            kamp = p5
            kfreq = p6
            outch 1, oscili:ar(kamp, kfreq)
            '''
            )
            >>> synth = session.sched('sine', args=dict(p5=0.1, p6=440))
            >>> synth.set(kfreq=880)
            >>> synth.set(p5=0.1, p6=1000)
            >>> synth.set(kamp=0.2, p6=440)

        .. seealso::

            - :meth:`AbstrSynth.set`
            - :meth:`Synth.automate`
            - :meth:`Synth.getp`
            - :meth:`Synth.automate`

        """
        if self.playStatus() == 'future':
            # TODO: schedule the set operatior
            # (Can we just modify the scheduled value?)
            return

        idx = self.instr.pfieldIndex(param, default=0)
        if idx == 0:
            raise KeyError(f"Unknown parameter {param} for synth {self}. "
                           f"Possible parameters: {self.dynamicParams()}")
        self.engine.setp(self.p1, idx, value, delay=delay)

    def ui(self, **specs: dict[str, tuple[float, float]]) -> None:
        """
        Modify dynamic (named) arguments through an interactive user-interface

        If run inside a jupyter notebook, this method will create embedded widgets
        to control the values of the dynamic pfields of an event. Dynamic pfields
        are those assigned to a k-variable or declared as ``|karg|`` (see below)

        Args:
            specs: a dict mapping named arg to a tuple (minvalue, maxvalue)

        Example
        ~~~~~~~

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
        if not self.instr.pfieldIndexToName:
            logger.info(f"This synth has no named arguments (getInstr='{self.instr.name}')")
            return
        pairs = list((idx, name) for idx, name in self.instr.pfieldIndexToName.items()
                     if name.startswith('k'))
        if not pairs:
            logger.error("The instrument has no dynamic (k) arguments")
            return

        pairs.sort()
        pargindexes, pargnames = zip(*pairs)

        if self.playStatus() == 'future':
            pvalues = [self.instr.pfieldIndexToValue[idx]
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

    def _setTable(self, param: str, value: float, delay=0.) -> None:
        if self.playStatus() == 'stopped':
            logger.error(f"Synth {self} has already stopped, cannot "
                         f"set param '{param}'")
            return

        if not self.controlsSlot:
            raise RuntimeError("This synth has no associated controls slot")

        slot = self.instr.controlIndex(param)
        if delay > 0:
            session = self.session
            session.engine.tableWrite(tabnum=session._dynargsTabnum,
                                      idx=self._sliceStart() + slot,
                                      value=value,
                                      delay=delay)
        else:
            self.session._setNamedControl(slicenum=self.controlsSlot, slot=slot, value=value)

    def _automateTable(self,
                       param: str,
                       pairs: list[float] | np.ndarray,
                       mode="linear",
                       delay=0.,
                       overtake=False) -> float:
        """
        Automate a named parameter, time is relative to the automation start

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
        if not self.controlsSlot:
            raise RuntimeError(f"{self.instr.name} (id={self.p1}) was not assigned "
                               f"a control slice")

        if self.playStatus() == 'stopped':
            logger.error(f"Synth {self} has already stopped, cannot "
                         f"mset param '{param}'")
            return 0.

        return self.session.automateDynamicParam(synth=self,
                                                 param=param,
                                                 pairs=pairs,
                                                 mode=mode,
                                                 overtake=overtake,
                                                 delay=delay)

    def automate(self,
                 param: str,
                 pairs: list[float] | np.ndarray,
                 mode='linear',
                 delay=0.,
                 overtake=False,
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
        params = self.dynamicParams()
        if param not in params:
            raise KeyError(f"Unknown parameter {param} for {self}. Possible parameters: {params}")
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

        if (controlnames := self._controlNames()) and param in controlnames:
            return self._automateTable(param=param, pairs=pairs, mode=mode, delay=delay, overtake=overtake)
        elif param.startswith('p') or ((pargs := self._pfieldNames()) and param in pargs):
            return self._automatePfield(param=param, pairs=pairs, mode=mode, delay=delay, overtake=overtake)
        else:
            raise KeyError(f"Unknown parameter '{param}', supported parameters: {self.dynamicParams()}")

    def _automatePfield(self, param: int | str, pairs: list[float] | np.ndarray,
                        mode="linear", delay=0., overtake=False) -> float:
        if self.playStatus() == 'stopped':
            raise RuntimeError("This synth has already stopped, cannot automate")

        if isinstance(param, str):
            pidx = self.instr.pfieldIndex(param)
            if not pidx:
                raise KeyError(f"pfield '{param}' not known. "
                               f"Known pfields: {self.instr.pfieldIndexToName}")
        else:
            pidx = param
        synthid = self.engine.automatep(self.p1, pidx=pidx, pairs=pairs,
                                        mode=mode, delay=delay, overtake=overtake)
        return synthid

    def stop(self, delay=0.) -> None:
        if self.finished():
            return
        self.session.unsched(self.p1, delay=delay)


def _synthsCreateHtmlTable(synths: list[Synth]) -> str:
    synth0 = synths[0]
    instr0 = synth0.instr
    if any(synth.instr != instr0 for synth in synths):
        # multiple instrs per group, not allowed here
        raise ValueError("Only synths of the same getInstr allowed here")
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
    if synth0._table is not None:
        keys = list(synth0._table.mapping.keys())
        colnames.extend(synth0._table.mapping.keys())
        for row, synth in zip(rows, synths):
            if synth.playStatus() != 'stopped':
                values = synth._table.array[:len(keys)]
                for value in values:
                    row.append(f'<code>{value}</code>')
            else:
                row.extend(["-"] * len(keys))

    if synth0.args:
        maxi = config['synth_repr_max_args']
        i2n = instr0.pfieldIndexToName
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
        start = min(synth.start for synth in synths)
        end = max(synth.end for synth in synths)
        dur = end - start
        AbstrSynth.__init__(self, start=start, dur=dur,
                            engine=synths[0].engine, autostop=autostop)
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

    def stop(self, delay=0.) -> None:
        for s in self.synths:
            s.stop(delay=delay)

    def playing(self) -> bool:
        return any(s.playing() for s in self.synths)

    def finished(self) -> bool:
        return all(s.finished() for s in self.synths)

    def _automateTable(self, param: str, pairs, mode="linear", delay=0.,
                       overtake=False) -> None:
        for synth in self.synths:
            if isinstance(synth, Synth):
                if synth._table and param in synth._controlNames():
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
    def _pfieldNames(self) -> set[str]:
        out: set[str] = set()
        for synth in self.synths:
            namedPargs = synth._pfieldNames()
            if namedPargs:
                out.update(namedPargs)
        return out

    def automate(self,
                 param: int | str,
                 pairs: Sequence[float] | np.ndarray,
                 mode="linear",
                 delay=0.,
                 overtake=False,
                 strict=True
                 ) -> list[float]:
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
            strict: if True, raise an Exception if the parameter does not match any
                synth in this group

        Returns:
            a list of synthids. There will be one synthid per synth in this group.
            A synthid of 0 indicates that for that given synth no automation was
            scheduled, either because that synth does not support the given
            param or because the automation times have no intersection with the
            synth
        """
        synthids = []
        if strict:
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

    def _automatePfield(self,
                        param: int | str,
                        pairs: list[float] | np.ndarray,
                        mode="linear",
                        delay=0.,
                        overtake=False) -> list[float]:
        eventids = [synth._automatePfield(param, pairs, mode=mode, delay=delay, overtake=overtake)
                    for synth in self.synths
                    if param in synth.instr.dynamicPfieldNames()]
        if not eventids:
            raise ValueError(f"Parameter '{param}' unknown for group, possible "
                             f"parameters: {self.dynamicParams()}")
        return eventids

    def _htmlTable(self) -> str:
        subgroups = iterlib.classify(self.synths, lambda synth: synth.getInstr.name)
        lines = []
        instrcol = jupytertools.defaultPalette["name.color"]
        for instrname, synths in subgroups.items():
            lines.append(f'<p>getInstr: <strong style="color:{instrcol}">'
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
            subgroups = iterlib.classify(self.synths, lambda synth: synth.getInstr.name)
            instrline = []
            instrcol = jupytertools.defaultPalette["name.color"]
            for instrname, synths in subgroups.items():
                s = f'<strong style="color:{instrcol}">{instrname}</strong> - {len(synths)} synths'
                namedparams = synths[0]._pfieldNames()
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

    def _setTable(self, param: str, value: float, delay=0) -> None:
        count = 0
        for synth in self.synths:
            if param in synth.dynamicParams():
                synth._setTable(param=param, value=value, delay=delay)
                count += 1
        if count == 0:
            raise KeyError(f"Parameter '{param}' unknown. "
                           f"Possible parameters: {self.dynamicParams()}")
            
    def _setp(self, param: str, value: float, delay=0.) -> None:
        allparams = self.dynamicParams()
        if param not in allparams:
            raise KeyError(f"Parameter {param} unknown. Possible parameters: {allparams}")
        for synth in self.synths:
            synth._setp(param=param, value=value, delay=delay)

    def _controlNames(self) -> set[str]:
        """
        Returns a set of available table named parameters for this group
        """
        allparams = set()
        for synth in self.synths:
            params = synth._controlNames()
            if params:
                allparams.update(params)
        return allparams



