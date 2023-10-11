from __future__ import annotations

from .baseevent import BaseEvent
from abc import abstractmethod
from . import instr as _instr


class AbstractRenderer:

    @abstractmethod
    def renderMode(self) -> str:
        """
        The render mode of this Renderer, one of 'online', 'offline'
        """
        raise NotImplementedError

    @abstractmethod
    def sched(self,
              instrname: str,
              delay=0.,
              dur=-1.,
              priority=1,
              args: list[float] | dict[str, float] = None,
              **kws) -> BaseEvent:
        raise NotImplementedError

    @abstractmethod
    def defInstr(self,
                 name: str,
                 body: str,
                 args: dict[str, float] = None,
                 init: str = None,
                 priority: int = None,
                 **kws) -> instr.Instr:
        raise NotImplementedError

    @abstractmethod
    def registerInstr(self, instr: _instr.Instr) -> bool:
        raise NotImplementedError

    @abstractmethod
    def assignBus(self, kind='audio', persist=False) -> int:
        raise NotImplementedError

    @abstractmethod
    def hasBusSupport(self) -> bool:
        raise NotImplementedError

