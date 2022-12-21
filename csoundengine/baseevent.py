from typing import Union, Sequence
import numpy as np
from abc import abstractmethod


class BaseEvent:
    __slots__ = ('p1', 'start', 'dur', 'args')

    def __init__(self, p1: Union[float, str], start: float, dur: float, args: Sequence[float]):
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
        """Set the value of a parameter for this event"""
        raise NotImplementedError


    @abstractmethod
    def automatep(self, param: str, pairs: Union[list[float], np.ndarray],
                  mode='linear', delay=0.
                  ) -> None:
        """Automate a named parameter of this event"""
        raise NotImplementedError

    @abstractmethod
    def automateTable(self, param: str, pairs: Union[list[float], np.ndarray],
                      mode='linear', delay=0.
                      ) -> None:
        """Automate a table based named parameter of this event"""
        raise NotImplementedError
