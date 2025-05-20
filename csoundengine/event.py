from __future__ import annotations

from dataclasses import dataclass
from .schedevent import SchedAutomation

import typing as _t


@dataclass
class Event:
    """
    Represents a future event, groups all its parameters and automations

    Args:
        instrname: the name of the instrument

    Example
    ~~~~~~~

        >>> from csoundengine import *
        >>> session = Session()
        >>> session.defInstr('test', '''
        ... |iamp=0.1, kfreq=1000|
        ... outch 1, oscili:a(iamp, kfreq)
        ... ''')
        >>> event = Event(instrname='test', delay=0, dur=2, args=dict(kfreq=800))
        >>> event.automate('kfreq', (0, 800, 2, 400))
        >>> session.schedEvent(event)
        >>> # This is the same as
        >>> synth = session.sched('test', delay=0, dur=2, kfreq=800)
        >>> synth.automate('kfreq', (0, 800, 2, 400))

    """
    instrname: str = ''
    """The name of the instrument template (not to be confused with p1)"""

    delay: float = 0.
    """The time offset to start the event"""

    dur: float = -1
    """The duration of the event"""

    priority: int = 1
    "The events priority (>1)"

    args: _t.Sequence[float | str] | dict[str, float] = ()
    "Numbered pfields (starting on p5) or a dict of parameters `{name: value}`"

    whenfinished: _t.Callable | None = None
    "A callback to be fired when this event is finished"

    p1: float | str = 0.
    """The event p1, if applicable"""

    relative: bool = True
    "Is the delay expressed in relative time?"

    kws: dict[str, float | str] | None = None
    "Named parameters passed to the instrument if args is a list of pfields"

    uniqueId: int = 0
    """If applicable, a unique id identifying this event. 0 indicates no id"""

    automations: list[SchedAutomation] | None = None
    """Automations attached to this event"""

    def automate(self, param: str, pairs: _t.Sequence[float], delay=0.,
                 interpolation='linear', overtake=False) -> None:
        if self.automations is None:
            self.automations = []
        autom = SchedAutomation(param=param, pairs=pairs, delay=delay,
                                interpolation=interpolation, overtake=overtake)
        self.automations.append(autom)

    def set(self, param='', value=0., delay=0., **kws):
        if param:
            self.automate(param=param, pairs=(0, value), delay=delay)
        if kws:
            for param, value in kws.items():
                self.set(param=param, value=value, delay=delay)

    def stop(self, delay=0.):
        dur = delay - self.delay
        if dur < 0:
            raise ValueError("Event has negative duration")
        self.dur = dur
