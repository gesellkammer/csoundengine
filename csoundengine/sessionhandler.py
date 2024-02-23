from __future__ import annotations
from csoundengine.event import Event
from abc import ABC, abstractmethod


class SessionHandler(ABC):

    @abstractmethod
    def sched(self, event: Event):
        raise NotImplementedError

    def makeTable(self,
                  data: np.ndarray | list[float] | None = None,
                  size: int | tuple[int, int] = 0,
                  sr: int = 0,
                  ) -> TableProxy:
        raise NotImplementedError

    def readSoundfile(self,
                      path: str,
                      chan=0,
                      skiptime=0.,
                      delay=0.,
                      force=False,
                      ) -> TableProxy:
        raise NotImplementedError
