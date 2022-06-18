from typing import Union, Sequence
import numpy as np
from abc import abstractmethod


class BaseEvent:
    __slots__ = ('p1', 'start', 'dur', 'pargs')

    def __init__(self, p1: Union[float, str], start: float, dur: float, pargs: Sequence[float]):
        self.p1 = p1
        self.start = start
        self.dur = dur
        self.pargs = pargs

    @property
    def end(self) -> float:
        return float('inf') if self.dur < 0 else self.start + self.dur

    @abstractmethod
    def setp(self, delay: float, *args, **kws) -> None: ...

    @abstractmethod
    def automatep(self, param: str, pairs: Union[list[float], np.ndarray],
                  mode='linear', delay=0.
                  ) -> None: ...

    @abstractmethod
    def automateTable(self, param: str, pairs: Union[list[float], np.ndarray],
                      mode='linear', delay=0.
                      ) -> None: ...
