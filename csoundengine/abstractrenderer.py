from __future__ import annotations

from .baseevent import BaseEvent
from abc import abstractmethod
from . import instr as _instr

import typing
if typing.TYPE_CHECKING:
    from csoundengine import busproxy


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
              **kwargs) -> BaseEvent:
        raise NotImplementedError

    @abstractmethod
    def defInstr(self,
                 name: str,
                 body: str,
                 args: dict[str, float] = None,
                 init: str = None,
                 priority: int = None,
                 **kws) -> _instr.Instr:
        raise NotImplementedError

    @abstractmethod
    def generateInstrBody(self,
                          instr: _instr.Instr
                          ) -> str:
        raise NotImplementedError

    @abstractmethod
    def registerInstr(self, instr: _instr.Instr) -> bool:
        raise NotImplementedError

    @abstractmethod
    def assignBus(self, kind='', value=None, persist=False) -> busproxy.Bus:
        raise NotImplementedError

    @abstractmethod
    def _releaseBus(self, bus: busproxy.Bus) -> None:
        raise NotImplementedError

    @abstractmethod
    def _writeBus(self, bus: busproxy.Bus, value: float, delay=0.) -> None:
        raise NotImplementedError

    @abstractmethod
    def _readBus(self, bus: busproxy.Bus) -> float | None:
        raise NotImplementedError

    @abstractmethod
    def _automateBus(self, bus: busproxy.Bus, pairs: typing.Sequence[float],
                     mode='linear', delay=0., overtake=False):
        raise NotImplementedError

    @abstractmethod
    def hasBusSupport(self) -> bool:
        raise NotImplementedError

