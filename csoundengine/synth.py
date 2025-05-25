from __future__ import annotations

import time
from abc import ABC, abstractmethod
from functools import cache
from emlib.envir import inside_jupyter

from . import internal
from .baseschedevent import BaseSchedEvent
from .config import config, logger
from .schedevent import SchedEvent

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence, Mapping
    from .instr import Instr
    from .session import Session
    import numpy as np


__all__ = (
    'Synth',
    'SynthGroup',
    'ui'
)


class ISynth(ABC):

    """
    Minimal interface defining a Synth

    This is not used inside csoundengine at the moment
    but is used in downstream projects like maelzel
    """

    @abstractmethod
    def playing(self) -> bool:
        """ Is this synth playing? """
        raise NotImplementedError

    @abstractmethod
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
        internal.setSigintHandler()
        while self.playing():
            sleepfunc(pollinterval)
        internal.removeSigintHandler()

    @abstractmethod
    def stop(self, delay=0.) -> None:
        raise NotImplementedError


    @abstractmethod
    def ui(self, **specs: tuple[float, float]) -> None:
        """
        Modify dynamic (named) arguments through an interactive user-interface

        If run inside a jupyter notebook, this method will create embedded widgets
        to control the values of the dynamic pfields of an event. Dynamic pfields
        are those assigned to a k-variable or declared as ``|karg|`` (see below)

        Args:
            specs: map named arg to a tuple (minvalue, maxvalue), the keyword
                is the name of the parameter, the value is a tuple with the
                range

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
        return ui(self, specs=specs)


def ui(event, specs: dict[str, tuple[float, float]]):
    from . import interact
    dynparams = event.dynamicParamNames(aliases=True, aliased=False)
    if not dynparams:
        logger.error(f"No named parameters for {event}")
        return
    params = {param: event.paramValue(param) for param in sorted(dynparams)}
    paramspecs = interact.guessParamSpecs(params, ranges=specs)
    return interact.interactSynth(event, specs=paramspecs)


_synthStatusIcon = {
    'playing': '▶',
    'stopped': '◼',
    'future': '‖',
}

class Synth(SchedEvent, ISynth):
    """
    A Synth represents a realtime csound event

    A user never creates a Synth directly, it is created by a Session when
    :meth:`Session.sched <csoundengine.session.Session.sched>` is called

    Args:
        session: the Session this synth belongs to
        p1: the p1 assigned
        instr: the Instr of this synth
        start: start time (absolute)
        dur: duration of the synth (-1 if no end)
        args: the pfields used to create the actual event, starting with p5 (p4 is
            reserved for the controlsSlot
        autostop: if True, the underlying csound event is stopped when this object
            is deallocated
        priority: the priority at which this event was scheduled
        controls: the dynamic controls used to schedule this synth
        controlsSlot: the control slot assigned to this synth, if the instrument
            defines named controls
        uniqueId: an integer identifying this synth, if applicable

    Example
    ~~~~~~~

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
    __slots__ = ('session', 'autostop', '_scheduled')

    def __init__(self,
                 session: Session,
                 p1: float,
                 start: float,
                 dur: float = -1,
                 instr: Instr | None = None,
                 args: Sequence[float|str] = (),
                 autostop=False,
                 priority: int = 1,
                 controls: Mapping[str, float] | None = {},
                 controlsSlot: int = -1,
                 uniqueId=0,
                 name=''
                 ) -> None:
        assert controls is None or isinstance(controls, dict)
        SchedEvent.__init__(self, instrname=instr.name if instr else '', start=start, dur=dur, args=args,
                            p1=p1, uniqueId=uniqueId, parent=session, priority=priority,
                            controlsSlot=controlsSlot, controls=controls, username=name)
        # AbstrSynth.__init__(self, start=start, dur=dur, session=session, autostop=autostop)

        if instr:
            if controlsSlot < 0 and instr.dynamicParams():
                raise ValueError("Synth has dynamic args but was not assigned a control slot")
            elif controlsSlot >= 1 and not instr.dynamicParams():
                logger.warning("A control slot was assigned but this synth does not have any controls")

        self.p1: float = p1
        """Event id for this synth"""

        self.session = session
        """The Session to which this Synth belongs"""

        self.autostop = autostop
        """If True, stop the underlying csound event when this object is freed"""

        if name and autostop:
            logger.warning("Autostop is disabled for named synths")

    def __del__(self):
        if self.autostop:
            self.stop()

    @staticmethod
    def makeGroup(synths: list[Synth]) -> SynthGroup:
        return SynthGroup(synths)

    def wait(self, pollinterval: float = 0.02, sleepfunc=time.sleep) -> None:
        """
        Wait until this synth has stopped

        Args:
            pollinterval: polling interval in seconds
            sleepfunc: the function to call when sleeping, defaults to time.sleep
        """
        internal.waitWhileTrue(self.playing, pollinterval=pollinterval, sleepfunc=sleepfunc)

    def aliases(self) -> dict[str, str]:
        """The parameter aliases of this synth, or an empty dict if no aliases defined"""
        return self.instr.aliases

    @property
    def body(self) -> str:
        return self.session.generateInstrBody(self.instr)

    def ui(self, **specs: tuple[float, float]) -> None:
        """
        Modify dynamic (named) arguments through an interactive user-interface

        If run inside a jupyter notebook, this method will create embedded widgets
        to control the values of the dynamic pfields of an event. Dynamic pfields
        are those assigned to a k-variable or declared as ``|karg|`` (see below)

        Args:
            specs: map named arg to a tuple (minvalue, maxvalue), the keyword
                is the name of the parameter, the value is a tuple with the
                range

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
        return ui(event=self, specs=specs)

    def _html(self, playstatus: str = '') -> str:
        from . import _palette
        argsfontsize = config['html_args_fontsize']
        maxi = config['synth_repr_max_args']
        style = _palette.defaultPalette
        if not playstatus:
            playstatus = self.playStatus()
        playstr = _synthStatusIcon[playstatus]
        parts = [
            f'{playstr} <strong style="color:{style["name.color"]}">'
            f'{self.instr.name}</strong>:{self.p1:.4f}',
            ]

        if self.args and len(self.args) > 1:
            i2n = self.instr.pfieldIndexToName
            argsstrs = []
            pargs = self.args
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
        status = self.playStatus()
        if status != 'stopped' and inside_jupyter():
            if config['jupyter_synth_repr_stopbutton']:
                from . import jupytertools
                jupytertools.displayButton("Stop", self.stop)
        return f"<p>{self._html(playstatus=status)}</p>"

    def __repr__(self):
        playstr = _synthStatusIcon[self.playStatus()]
        def f3(x) -> str:
            return f"{x:.3f}".strip('0').rstrip('.')
        parts = [f'{playstr} {self.instr.name}={self.p1} start={f3(self.start)} dur={f3(self.dur)}']

        if self.instr.hasControls():
            parts.append(f'slot={self.controlsSlot}')
            ctrlparts = []
            for k, v in self.instr.controls.items():
                if self.controls is not None and k in self.controls:
                    v = self.controls[k]
                ctrlparts.append(f'{k}={v}')
            parts.append(f"|{' '.join(ctrlparts)}|")
        if self.args:
            showpidx = config['synth_repr_show_pfield_index']
            maxi = config['synth_repr_max_args']
            i2n = self.instr.pfieldIndexToName
            maxi = max((i for i, name in i2n.items() if name.startswith("k")),
                       default=maxi)
            argsstrs = []
            pargs = self.args
            for i, parg in enumerate(pargs):
                if i > maxi:
                    argsstrs.append("…")
                    break
                pindex = i+5
                name = i2n.get(pindex)
                if not isinstance(parg, str):
                    parg = f'{parg:.6g}'
                if name:
                    if showpidx:
                        s = f"{name}:{pindex}={parg}"
                    else:
                        s = f"{name}={parg}"
                else:
                    # pargs start at 5
                    s = f"p{i+5}={parg}"
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
        if self.p1 not in self.session._synths:
            return "stopped"
        now = self.session.engine.realElapsedTime()
        return "playing" if now >= self.start else "future"

    def playing(self) -> bool:
        """ Is this Synth playing """
        return self.playStatus() == "playing"

    def finished(self) -> bool:
        return self.playStatus() == 'stopped'

    #def pfieldNames(self, aliases=True, aliased=False) -> frozenset[str]:
    #    return self.instr.pfieldNames(aliases=aliases, aliased=aliased)

    def _sliceStart(self) -> int:
        return self.controlsSlot * self.session.maxDynamicArgs

    def _setPfield(self, param: str, value: float, delay=0.) -> None:
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
        if self.playStatus() == 'stopped':
            logger.error(f"Synth {self} has already stopped, cannot "
                         f"set param '{param}'")
        return self.session._setPfield(event=self, delay=delay, param=param, value=value)

    def _setTable(self, param: str, value: float, delay=0.) -> None:
        if self.playStatus() == 'stopped':
            logger.error(f"Synth {self} has already stopped, cannot "
                         f"set param '{param}'")
        else:
            self.session._setNamedControl(event=self, param=param, value=value, delay=delay)

    def paramValue(self, param: str | int) -> float | str | None:
        """
        Get the value of a parameter

        Args:
            param: the parameter name or a pfield index

        Returns:
            the value, or None if the parameter has no value
        """
        if isinstance(param, int):
            paramidx = param - 4
            return self.args[paramidx] if self.args and 0 <= paramidx < len(self.args) else None
        elif isinstance(param, str):
            param = self.unaliasParam(param, param)
            if (paramidx := self.instr.pfieldIndex(param, -1)) >= 0:
                paramidx0 = paramidx - 5
                if self.args and paramidx0 < len(self.args):
                    return self.args[paramidx0]
                else:
                    return None
            elif param in self.instr.controls:
                if self.playing():
                    return self.session._getNamedControl(slicenum=self.controlsSlot,
                                                         paramslot=self.instr.controlIndex(param))
                else:
                    assert self.controls is not None
                    value = self.controls.get(param)
                    if value is not None:
                        return value
                    return self.instr.controls.get(param)
            return None
        else:
            raise TypeError(f"Expected an integer index or a parameter name, got {param}")

    def relativeStart(self) -> float:
        """
        The relative start time of this Synth

        The .start attribute of the synth carries the absolute timestamp
        (since the start of the engine) at which this Synth was scheduled.
        The relative start time is an offset from the current elapsed
        time. **If this synth has already started then the returned value
        will be negative**

        Returns:
            the relative start time of the synth
        """
        return self.start - self.session.engine.elapsedTime()

    def automate(self,
                 param: str,
                 pairs: Sequence[float] | np.ndarray | tuple[np.ndarray, np.ndarray],
                 mode='linear',
                 delay: float | None = 0.,
                 overtake=False,
                 ) -> float:
        """
        Automate any named parameter of this Synth

        Raises KeyError if the parameter is unknown

        Args:
            param: the name of the parameter to automate
            pairs: automation data as a flat array with the form [time0, value0, time1, value1, ...] or a
                tuple of the form (times, values)
            mode: one of 'linear', 'cos'. Determines the curve between values
            delay: when to start the automation, relative to the current time. If None is given,
                the delay is set to the start of this synth. To set an absolute start time, use
                ``abstime - engine.elapsedTime()`` as delay
            overtake: if True, do not use the first value in pairs but overtake the current value

        Returns:
            the eventid of the automation event.
        """
        return self.session.automate(event=self, param=param, pairs=pairs, mode=mode,
                                     delay=delay, overtake=overtake)

    def stop(self, delay=0.) -> None:
        self.session.unsched(self.p1, delay=delay)


def _synthsCreateHtmlTable(synths: list[Synth], maxrows: int | None = None, tablestyle='',
                           ) -> str:
    synth0 = synths[0]
    instr0 = synth0.instr
    if any(synth.instr.name != instr0.name for synth in synths):
        # multiple instrs per group, not allowed here
        raise ValueError("Only synths of the same instr allowed here")

    colnames = ["p1", "start", "dur", "p4"]
    if maxrows is None:
        maxrows = config['synthgroup_repr_max_rows']
    if maxrows and len(synths) > maxrows:
        limitSynths = True
        synths = synths[:maxrows]
    else:
        limitSynths = False

    rows: list[list[str]] = [[] for _ in synths]
    now = synth0.session.engine.elapsedTime()
    for row, synth in zip(rows, synths):
        row.append(f'{synth.p1} <b>{_synthStatusIcon[synth.playStatus()]}</b>')
        row.append("%.3f" % (synth.start - now))
        row.append("%.3f" % synth.dur)
        row.append(str(synth.controlsSlot))

    if keys := synth0.controlNames():
        colnames.extend(keys)
        for row, synth in zip(rows, synths):
            if synth.playStatus() != 'stopped':
                values = [synth.paramValue(param) for param in keys]
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
            pidx = i + 5
            name = i2n.get(pidx)
            if config['synth_repr_show_pfield_index']:
                colname = f"{pidx}:{name}" if name else str(pidx)
            else:
                colname = name if name else str(pidx)
            colnames.append(colname)
        for row, synth in zip(rows, synths):
            if synth.args:
                row.extend(f"{parg:.5g}" if not isinstance(parg, str) else parg
                        for parg in synth.args[:maxi])
                if len(synth.args) > maxi:
                    row.append("...")

    if limitSynths:
        rows.append(["..."])

    import emlib.misc
    return emlib.misc.html_table(rows, headers=colnames, tablestyle=tablestyle)


class SynthGroup(BaseSchedEvent):
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
    __slots__ = ('synths', 'session', 'autostop', '__weakref__')

    def __init__(self, synths: list[Synth], autostop=False) -> None:
        if not synths:
            start = 0.
            end = 0.
            dur = 0.
        else:
            start = min(synth.start for synth in synths)
            end = max(synth.end for synth in synths)
            dur = end - start
        BaseSchedEvent.__init__(self, start=start, dur=dur)
        flatsynths: list[Synth] = []
        for synth in synths:
            if isinstance(synth, SynthGroup):
                flatsynths.extend(synth)
            else:
                flatsynths.append(synth)
        self.synths: list[Synth] = flatsynths
        self.autostop = autostop
        self.session = self.synths[0].session if synths else None

    def __del__(self):
        if self.autostop:
            for synth in self:
                if synth.playStatus() != 'stopped':
                    synth.stop()

    def extend(self, synths: list[Synth]) -> None:
        """
        Add the given synths to the synths in this group
        """
        self.synths.extend(synths)

    def stop(self, delay=0.) -> None:
        for s in self.synths:
            s.stop(delay=delay)

    def playing(self) -> bool:
        return any(s.playing() for s in self)

    def finished(self) -> bool:
        return all(s.finished() for s in self)

    def _automateTable(self, param: str, pairs, mode="linear", delay=0.,
                       overtake=False) -> list[float]:
        synthids = []
        for synth in self.synths:
            if isinstance(synth, Synth):
                controls = synth.dynamicParamNames()
                if controls and param in controls:
                    synthid = synth._automateTable(param, pairs, mode=mode, delay=delay,
                                                   overtake=overtake)
                    synthids.append(synthid)
            elif isinstance(synth, SynthGroup):
                controls = synth.dynamicParamNames()
                if controls and param in controls:
                    synthid = synth._automateTable(param=param, pairs=pairs, mode=mode, delay=delay,
                                                   overtake=overtake)
                    synthids.append(synthid)
        if not synthids:
            raise KeyError(f"Parameter '{param}' not known. "
                           f"Possible parameters: {self.dynamicParamNames()}")
        return synthids

    @cache
    def dynamicParamNames(self, aliases=True, aliased=False) -> set[str]:
        out: set[str] = set()
        for synth in self:
            dynamicParams = synth.dynamicParamNames(aliases=aliases, aliased=aliased)
            out.update(dynamicParams)
        return out

    @cache
    def aliases(self) -> dict[str, str]:
        out = {}
        for synth in self:
            if synth.instr.aliases:
                out.update(synth.instr.aliases)
        return out

    @cache
    def pfieldNames(self, aliases=True, aliased=False) -> frozenset[str]:
        out: set[str] = set()
        for synth in self:
            namedPargs = synth.pfieldNames(aliases=aliases, aliased=aliased)
            if namedPargs:
                out.update(namedPargs)
        return frozenset(out)

    def automate(self,
                 param: int | str,
                 pairs: Sequence[float] | np.ndarray | tuple[np.ndarray, np.ndarray],
                 mode="linear",
                 delay=0.,
                 overtake=False,
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

        Returns:
            a list of synthids. There will be one synthid per synth in this group.
            A synthid of 0 indicates that for that given synth no automation was
            scheduled, either because that synth does not support the given
            param or because the automation times have no intersection with the
            synth
        """
        synthids = []
        for synth in self:
            if param in synth.instr.dynamicParams():
                synthid = synth.automate(param=param, pairs=pairs, mode=mode, delay=delay, overtake=overtake)
            else:
                synthid = 0
            synthids.append(synthid)
        if all(synthid == 0 for synthid in synthids):
            raise KeyError(f"Parameter '{param}' not known. Possible parameters: "
                           f"{self.dynamicParamNames(aliases=True, aliased=True)}")
        return synthids

    def _automatePfield(self,
                        param: int | str,
                        pairs: list[float] | np.ndarray,
                        mode="linear",
                        delay=0.,
                        overtake=False) -> list[float]:
        eventids = [synth._automatePfield(param, pairs, mode=mode, delay=delay, overtake=overtake)
                    for synth in self
                    if param in synth.instr.dynamicPfieldNames()]
        if not eventids:
            raise ValueError(f"Parameter '{param}' unknown for group, possible "
                             f"parameters: {self.dynamicParamNames()}")
        return eventids

    def _htmlTable(self, style='', maxrows: int | None = None) -> str:
        import emlib.iterlib
        from . import _palette
        subgroups = emlib.iterlib.classify(self.synths, lambda synth: synth.instr.name)
        lines = []
        instrcol = _palette.defaultPalette["name.color"]
        for instrname, synths in subgroups.items():
            lines.append(f'<p><small>Instr: <strong style="color:{instrcol}">'
                         f'{instrname}'
                         f'</strong> - <b>{len(synths)}</b> synths</small></p>')
            htmltable = _synthsCreateHtmlTable(synths, maxrows=maxrows, tablestyle=style)
            lines.append(htmltable)
        out = '\n'.join(lines)
        return out

    def ui(self, **specs: tuple[float, float]) -> None:
        """
        Modify dynamic (named) arguments through an interactive user-interface

        If run inside a jupyter notebook, this method will create embedded widgets
        to control the values of the dynamic pfields of an event. Dynamic pfields
        are those assigned to a k-variable or declared as ``|karg|`` (see below)

        Args:
            specs: map named arg to a tuple (minvalue, maxvalue), the keyword
                is the name of the parameter, the value is a tuple with the
                range

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
            synths = [
                s.sched('vco', kmidinote=67)
                s.sched('vco', kmidinote=69)
            ]
            group = SynthGroup(synths)

            # Specify the ranges for some sliders. All named parameters
            # are assigned a widget
            group.ui(kampdb=(-48, 0), kres=(0, 1))

        .. seealso::

            - :meth:`Engine.eventui`
        """
        ui(event=self, specs=specs)

    def _repr_html_(self) -> str:
        from . import jupytertools

        def bold(txt):
            return span(txt, bold=True)

        span = jupytertools.htmlSpan

        if not self.synths:
            return f'{bold("SynthGroup")}(synths=[])'

        if config['jupyter_synth_repr_stopbutton']:
            jupytertools.displayButton("Stop", self.stop)
        header = f'{bold("SynthGroup")}(synths={span(len(self), tag="code")})'
        lines = [f'<small>{header}</small>']
        numrows = config['synthgroup_repr_max_rows']
        style = config['synthgroup_html_table_style']
        lines.append(self._htmlTable(style=style, maxrows=numrows))
        return "\n".join(lines)

    def __repr__(self) -> str:
        lines = [f"SynthGroup(n={len(self.synths)})"]
        for synth in self:
            lines.append("    "+repr(synth))
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.synths)

    def __getitem__(self, idx) -> Synth:
        return self.synths.__getitem__(idx)

    def __iter__(self):
        return iter(self.synths)

    def _setTable(self, param: str, value: float, delay=0) -> None:
        count = 0
        for synth in self:
            if param in synth.dynamicParamNames(aliases=True, aliased=True):
                synth._setTable(param=param, value=value, delay=delay)
                count += 1
        if count == 0:
            params = list(self.dynamicParamNames(aliases=False))
            if aliases := self.aliases():
                params.extend(f'{alias}>{orig}' for alias, orig in aliases.items())
            raise KeyError(f"Parameter '{param}' unknown. "
                           f"Possible parameters: {params}")

    def paramValue(self, param: str) -> float | str | None:
        """
        Returns the parameter value for the given parameter

        Within a group the first synth which has the given parameter
        will be used to determine the parameter value
        """
        if param not in self.paramNames():
            raise KeyError(f"Parameter '{param}' not known. Possible parameters: "
                           f"{self.paramNames()}")
        for synth in self:
            value = synth.paramValue(param)
            if value is not None:
                return value
        return None

    def _setPfield(self, param: str, value: float, delay=0.) -> None:
        count = 0
        for synth in self:
            if param in synth.instr.pfieldNames(aliases=False):
                synth._setPfield(param=param, value=value, delay=delay)
                count += 1
        if count == 0:
            raise KeyError(f"Parameter {param} unknown. "
                           f"Possible parameters: {self.dynamicParamNames(aliased=True)}")

    def controlNames(self, aliases=True, aliased=False) -> set[str]:
        """
        Returns a set of available table named parameters for this group
        """
        allparams = set()
        for synth in self:
            params = synth.controlNames(aliases=aliases, aliased=aliased)
            if params:
                allparams.update(params)
        return allparams

    def wait(self, pollinterval: float = 0.02, sleepfunc=time.sleep) -> None:
        """
        Wait until this synth has stopped

        Args:
            pollinterval: polling interval in seconds
            sleepfunc: the function to call when sleeping, defaults to time.sleep
        """
        internal.waitWhileTrue(self.playing, pollinterval=pollinterval, sleepfunc=sleepfunc)
