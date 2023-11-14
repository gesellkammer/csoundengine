from __future__ import annotations
import copy
from dataclasses import dataclass
from functools import cache
from .baseschedevent import BaseSchedEvent
from .config import logger
from . import instr
from ._common import EMPTYSET


from typing import TYPE_CHECKING, Sequence
if TYPE_CHECKING:
    import numpy as np
    from .abstractrenderer import AbstractRenderer


@dataclass
class SchedAutomation:

    param: str
    """The parameter to automate (pfield or control)"""

    pairs: Sequence[float] | np.ndarray
    """A flat list of automation data (t0, v1, t1, v1, ...)"""

    interpolation: str = 'linear'
    """Interpolation kind, one of linear, cos, ..."""

    delay: float | None = None
    """A delay of None indicates that the delay is the same as the event
    to which this automation belongs"""

    overtake: bool = False


class SchedEvent(BaseSchedEvent):
    """
    Represents a scheduled event.

    It is used to control / automate / keep track of scheduled events.

    Args:
        p1: the p1 of the scheduled event
        start: start time
        dur: duration
        args: pfields starting with p4
        controls: the dynamic controls used to schedule this event
        uniqueId: an integer unique to this event
        parent: the renderer which scheduled this event
        instrname: the instr name of this event (if applies)
        priority: the priority at which this event was scheduled (if applies)
        controlsSlot: the slot/token assigned for dynamic controls
    """

    __slots__ = ('uniqueId', 'parent', 'instrname', 'priority',
                 'args', 'p1', 'controlsSlot', 'automations', 'controls')

    def __init__(self,
                 instrname: str = '',
                 start: float = 0.,
                 dur: float = -1,
                 args: Sequence[float|str] | None = None,
                 p1: float | str = 0,
                 uniqueId: int = 0,
                 parent: AbstractRenderer = None,
                 priority: int = 0,
                 controls: dict[str, float] | None = None,
                 controlsSlot: int = -1):

        if parent and instrname:
            assert instrname in parent.registeredInstrs()

        super().__init__(start=start, dur=dur)

        self.p1 = p1
        """p1 of this event"""

        self.instrname: str = instrname
        """The instrument template this event was created from, if applicable"""

        self.args = args
        """Args used for this event (p4, p5, ...)"""

        self.priority: int = priority
        """The priority of this event, if applicable"""

        self.uniqueId: int = uniqueId
        """A unique id of this event, as integer"""

        self.parent: AbstractRenderer | None = parent
        """The Renderer to which this event belongs (can be None)"""

        self.controls: dict[str, float] | None = controls
        """The dynamic controls used to schedule this event"""

        self.controlsSlot: int = controlsSlot
        """The slot/token assigned for dynamic controls"""

        self.automations: list[SchedAutomation] | None = None

    def __hash__(self) -> int:
        return hash(('SchedEvent', self.uniqueId))
        # return hash((self.p1, self.uniqueId, self.instrname, self.priority, hash(tuple(self.args))))

    def __repr__(self):
        parts = [f"p1={self.p1}, start={self.start}, dur={self.dur}, uniqueId={self.uniqueId}"]
        if self.args:
            parts.append(f'args={self.args}')
        if self.instrname:
            parts.append(f'instrname={self.instrname}')
        if self.priority:
            parts.append(f'priority={self.priority}')
        partsstr = ', '.join(parts)
        return f"{type(self).__name__}({partsstr})"

    def playStatus(self) -> str:
        """
        Returns the playing status of this event (offline, playing, stopped or future)

        For offline events this will always return 'offline'

        Returns:
            'playing' if currently playing, 'stopped' if this event has already stopped
            or 'future' if it has not started. For offline events always returns 'offline'

        """
        return 'offline'

    def clone(self, **kws) -> SchedEvent:
        event = copy.copy(self)
        for k, v in kws.items():
            setattr(event, k, v)
        return event

    def _setTable(self, param: str, value: float, delay=0.) -> None:
        if not self.parent:
            raise RuntimeError("This event has no parent")

        if not self.start <= delay <= self.end:
            logger.error(f"This operation's time offset ({delay}) is not within "
                         f"the time range of the event ({self.start}-{self.end}")

        if not self.controlsSlot:
            raise RuntimeError("This event has no associated controls slot")

        self.parent._setNamedControl(event=self, param=param, value=value, delay=delay)

    def _setPfield(self, param: str, value: float, delay=0.) -> None:
        """
        Modify a parg of this synth (offline).

        Multiple pfields can be modified simultaneously. It only makes sense
        to modify a parg if a k-rate variable was assigned to this parg
        (see Renderer.setp for an example). A parg can be referred to via an integer,
        corresponding to the p index (5 would refer to p5), or to the name
        of the assigned k-rate variable as a string (for example, if there
        is a line "kfreq = p6", both 6 and "kfreq" refer to the same parg).

        Example
        ~~~~~~~

            >>> from csoundengine import *
            >>> r = Renderer()
            >>> Instr('sine', r'''
            ... |kamp=0.1, kfreq=1000|
            ... outch 1, oscili:ar(kamp, freq)
            ... ''')
            >>> event = r.sched('sine', 0, dur=4, args=[0.1, 440])
            >>> event._setPfield(2, kfreq=880)
            >>> event._setPfield(3, kfreq=660, kamp=0.5)

        """
        if self.parent is None:
            raise RuntimeError("This event is not assigned to a Renderer")
        self.parent._setPfield(self, delay=delay, param=param, value=value)

    @property
    def instr(self) -> instr.Instr:
        """
        The Instr corresponding to this Event, if applicable

        Raises ValueError if this event cannot access to the Instr
        instance (if it has no parent or its instrument name is invalid)
        """
        if not self.parent:
            raise ValueError(f"This event {self} has no parent")
        instr = self.parent.getInstr(self.instrname)
        if instr is None:
            raise ValueError(f"Instrument {self.instrname} not known")
        return instr

    def paramNames(self, aliases=True, aliased=False) -> frozenset[str]:
        return self.instr.paramNames(aliases=aliases, aliased=aliased)

    def dynamicParamNames(self, aliases=True, aliased=False) -> frozenset[str]:
        instr = self.instr
        return instr.dynamicParamNames(aliases=aliases, ) if instr else EMPTYSET

    def automate(self,
                 param: str,
                 pairs: Sequence[float] | np.ndarray,
                 mode="linear",
                 delay: float = None,
                 overtake=False,
                 ) -> float:
        param = self.unaliasParam(param, param)
        if self.parent is None:
            if param not in (params := self.instr.dynamicParams(aliased=True)):
                raise KeyError(f"Unknown parameter '{param}' for {self}. Possible parameters: {params}")

            automation = SchedAutomation(param=param, pairs=pairs, interpolation=mode, delay=delay)
            if self.automations is None:
                self.automations = [automation]
            else:
                self.automations.append(automation)
        else:
            self.parent.automate(self, param=param, pairs=pairs, mode=mode, delay=delay)
        return 0

    def stop(self, delay=0.) -> None:
        """
        Stop this event

        When using this in offline mode, delay is an absolute time

        Args:
            delay: when to stop

        """
        if self.parent is None:
            if self.start < delay < self.end:
                self.dur = delay - self.start
            else:
                logger.error(f"Stop time {delay} outside the lifetime of this event "
                             f"({self.start} - {self.end})")
        else:
            self.parent.unsched(self, delay=delay)

    def controlNames(self, aliases=True, aliased=False) -> frozenset[str]:
        return self.instr.controlNames(aliases=aliases, aliased=aliased)

    def pfieldNames(self, aliases=True, aliased=False) -> frozenset[str]:
        return self.instr.pfieldNames(aliases=aliases, aliased=aliased)

    def paramValue(self, param: str) -> float | str | None:
        param = self.unaliasParam(param, param)
        instr = self.instr
        if param in self.pfieldNames(aliases=False):
            pindex = instr.pfieldIndex(param)
            argindex = pindex - 4
            if self.args and 0 >= argindex < len(self.args):
                return self.args[argindex]
            return instr.pfieldDefaultValue(pindex)
        elif param in self.controlNames(aliases=False):
            if self.controls and param in self.controls:
                return self.controls[param]
            return instr.controls[param]
        else:
            raise KeyError(f"Parameter '{param}' unknown. Possible parameters: {instr.paramNames(aliases=False)},"
                           f" (aliases={instr.aliases})")


class SchedEventGroup(BaseSchedEvent):
    """
    Represents a group of scheduled events

    These events can be controlled together, similar to a SynthGroup
    """

    def __init__(self, events: list[SchedEvent]):
        if not events:
            raise ValueError("No events given")

        start = min(ev.start for ev in events)
        end = max(ev.end for ev in events)
        dur = end - start
        super().__init__(start=start, dur=dur)
        self.events = events

    def __iter__(self):
        return iter(self.events)

    def __getitem__(self, item):
        return self.events.__getitem__(item)

    def __len__(self):
        return len(self.events)

    def stop(self, delay=0.) -> None:
        for ev in self:
            ev.stop(delay=delay)

    def _setPfield(self, param: str, value: float, delay=0.) -> None:
        count = 0
        for ev in self:
            if param in ev.pfieldNames(aliased=True):
                ev._setPfield(delay=delay, param=param, value=value)
                count += 1
        if count == 0:
            raise KeyError(f"Parameter '{param}' unknown. Possible paramters: {self.dynamicParamNames(aliased=True)}")

    def _setTable(self, param: str, value: float, delay=0.) -> None:
        count = 0
        for ev in self:
            if param in ev.controlNames(aliases=True, aliased=True):
                ev._setTable(param=param, value=value, delay=delay)
                count += 1
        if count == 0:
            raise KeyError(f"Parameter '{param}' unknown. "
                           f"Possible parameters: {self.dynamicParamNames(aliased=True)}")

    @cache
    def paramNames(self, aliases=True, aliased=False) -> frozenset[str]:
        allparams = set()
        for ev in self:
            allparams.update(ev.paramNames(aliases=aliases, aliased=aliased))
        return frozenset(allparams)

    def paramValue(self, param: str) -> float | str | None:
        """
        Returns the parameter value for the given parameter

        Within a group the first synth which has the given parameter
        will be used to determine the parameter value
        """
        if param not in self.paramNames():
            raise KeyError(f"Parameter '{param}' not known. Possible parameters: "
                           f"{self.paramNames()}")
        for ev in self:
            value = ev.paramValue(param)
            if value is not None:
                return value
        return None

    @cache
    def dynamicParamNames(self, aliases=True, aliased=False) -> frozenset[str]:
        params = set()
        for ev in self:
            params.update(ev.dynamicParamNames(aliases=aliases, aliased=aliased))
        return frozenset(params)

    def __hash__(self):
        return hash(tuple(hash(ev) for ev in self))

    @cache
    def controlNames(self, aliases=True, aliased=False) -> frozenset[str]:
        """
        Returns a set of available table named parameters for this group
        """
        allparams = set()
        for event in self:
            params = event.controlNames(aliases=aliases, aliased=aliased)
            if params:
                allparams.update(params)
        return frozenset(allparams)

    def set(self, param='', value: float = 0., delay=0., **kws) -> None:
        if kws:
            for k, v in kws.items():
                self.set(param=k, value=v, delay=delay)

        if param:
            count = 0
            for ev in self:
                if param in ev.dynamicParamNames(aliases=True, aliased=True):
                    count += 1
                    ev.set(param=param, value=value, delay=delay)
            if count == 0:
                raise KeyError(f"Param '{param}' not known by any events in this group. "
                               f"Possible parameters: {self.dynamicParamNames(aliased=True)}")

    def automate(self,
                 param: str,
                 pairs: Sequence[float] | np.ndarray,
                 mode="linear",
                 delay: float = None,
                 overtake=False,
                 ) -> float:
        count = 0
        for ev in self:
            if param in ev.dynamicParamNames(aliases=True, aliased=True):
                count += 1
                ev.automate(param=param, pairs=pairs, mode=mode,
                            delay=delay, overtake=overtake)
        if count == 0:
            raise KeyError(f"Param '{param}' not known by any events in this group. "
                           f"Possible parameters: {self.dynamicParamNames(aliased=True)}")
        return 0.


