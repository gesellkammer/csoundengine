from __future__ import annotations

from typing import Union, Sequence
import numpy as np
from abc import abstractmethod


class BaseEvent:
    """
    Base class for all scheduled events (both offline and realtime)

    Args:
        p1: the event id
        start: the start time relative to the start of the engine
        dur: the duration of the synth
        args: the pfields of this event, beginning with p4
    """

    __slots__ = ('p1', 'start', 'dur', 'args')

    def __init__(self, p1: float | str, start: float, dur: float, args: Sequence[float]):
        self.p1 = p1
        """The instrument number/name"""

        self.start = start
        """Start time of the event (p2)"""

        self.dur = dur
        """Duration of the event (p3). Can be negative to indicate an endless/tied event"""

        self.args = args
        """Numerical arguments beginning with p4"""

    @property
    def end(self) -> float:
        """End of this event (can be inf if the duration is given as negative)"""
        return float('inf') if self.dur < 0 else self.start + self.dur

    @abstractmethod
    def setp(self, delay: float, *args, **kws) -> None:
        """
        Set the value of a pfield for this event

        """
        raise NotImplementedError

    @abstractmethod
    def set(self, delay: float, *args, **kws) -> None:
        """
        Set the value of a parameter for this event

        Args:
            delay: the time offset at which to schedule the operation

        """
        return self.setp(delay=delay, *args, **kws)

    @abstractmethod
    def automate(self,
                 param: str | int,
                 pairs: list[float] | np.ndarray,
                 mode='linear',
                 delay=0.
                 ) -> None:
        """
        Automate a named parameter of this event
        """
        raise NotImplementedError

