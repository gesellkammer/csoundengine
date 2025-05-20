from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Sequence

if TYPE_CHECKING:
    import numpy as np

    from . import busproxy, tableproxy
    from . import instr as _instr
    from .event import Event
    from .schedevent import SchedEvent, SchedEventGroup


class AbstractRenderer(ABC):
    """
    Base class for rendering (both live and offline)
    """
    def __init__(self):
        self._instrInitCallbackRegistry: set[str] = set()
        self.namedEvents: dict[str, SchedEvent] = {}

    def _initInstr(self, instr: _instr.Instr):
        if instr._initCallback is not None and instr.name not in self._instrInitCallbackRegistry:
            instr._initCallback(self)
            self._instrInitCallbackRegistry.add(instr.name)

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
              *pfields,
              args: Sequence[float|str] | dict[str, float] | None = None,
              priority=1,
              whenfinished: Callable | None = None,
              relative=True,
              **kwargs) -> SchedEvent:
        raise NotImplementedError

    @abstractmethod
    def unsched(self, event: int | float | SchedEvent, delay: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def defInstr(self,
                 name: str,
                 body: str,
                 args: dict[str, float|str] | None = None,
                 init: str = '',
                 priority: int | None = None,
                 doc: str = '',
                 includes: list[str] | None = None,
                 aliases: dict[str, str] | None = None,
                 useDynamicPfields: bool | None = None,
                 initCallback: Callable[[AbstractRenderer], None] = None,
                 **kws) -> _instr.Instr:
        raise NotImplementedError

    @abstractmethod
    def makeTable(self,
                  data: np.ndarray | list[float] | None = None,
                  size: int | tuple[int, int] = 0,
                  tabnum: int = 0,
                  sr: int = 0,
                  delay: float = 0.,
                  unique=True
                  ) -> tableproxy.TableProxy:
        raise NotImplementedError

    @abstractmethod
    def _getTableData(self, table: int | tableproxy.TableProxy) -> np.ndarray | None:
        raise NotImplementedError

    @abstractmethod
    def freeTable(self,
                  table: int | tableproxy.TableProxy,
                  delay: float = 0.) -> None:
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
    def registeredInstrs(self) -> dict[str, _instr.Instr]:
        """
        Returns a dict (instrname: Instr) with all registered Instrs
        """
        raise NotImplementedError

    @abstractmethod
    def assignBus(self, kind='', value: float | None = None, persist=False
                  ) -> busproxy.Bus:
        raise NotImplementedError

    @abstractmethod
    def _releaseBus(self, bus: busproxy.Bus) -> None:
        raise NotImplementedError

    @abstractmethod
    def _writeBus(self, bus: busproxy.Bus, value: float, delay=0.) -> None:
        raise NotImplementedError

    def _readBus(self, bus: busproxy.Bus) -> float | None:
        return None

    @abstractmethod
    def _automateBus(self, bus: busproxy.Bus, pairs: Sequence[float],
                     mode='linear', delay=0., overtake=False):
        raise NotImplementedError

    @abstractmethod
    def getInstr(self, instrname: str) -> _instr.Instr | None:
        """
        Returns the Instr instance corresponding to instrname, or None if no such instr
        """
        raise NotImplementedError

    @abstractmethod
    def automate(self,
                 event: SchedEvent,
                 param: str,
                 pairs: Sequence[float] | np.ndarray,
                 mode="linear",
                 delay: float | None = None,
                 overtake=False
                 ) -> float:
        raise NotImplementedError

    @abstractmethod
    def hasBusSupport(self) -> bool:
        """Does this renderer have bus support?"""
        raise NotImplementedError

    @abstractmethod
    def _setNamedControl(self,
                         event: SchedEvent,
                         param: str,
                         value: float,
                         delay: float = 0.
                         ) -> None:
        raise NotImplementedError

    @abstractmethod
    def _setPfield(self, event: SchedEvent, delay: float,
                   param: str, value: float
                   ) -> None:
        raise NotImplementedError

    @abstractmethod
    def includeFile(self, path: str) -> None:
        """
        Include a file

        Args:
            path: the path to the include file
        """
        raise NotImplementedError

    @abstractmethod
    def schedEvent(self, event: Event) -> SchedEvent:
        """
        Schedule the given event
        """
        raise NotImplementedError

    def schedEvents(self, events: Sequence[Event]) -> SchedEventGroup:
        # naive implementation
        schedevents = [self.schedEvent(event)
                       for event in events]
        return SchedEventGroup(schedevents)

    @abstractmethod
    def playSample(self,
                   source: int | tableproxy.TableProxy | str | tuple[np.ndarray, int],
                   delay=0.,
                   dur=0.,
                   chan=1,
                   gain=1.,
                   speed=1.,
                   loop=False,
                   pan=0.5,
                   skip=0.,
                   fade: float | tuple[float, float] | None = None,
                   crossfade=0.02,
                   ) -> SchedEvent:
        raise NotImplementedError

    @abstractmethod
    def readSoundfile(self,
                      path="?",
                      chan=0,
                      skiptime=0.,
                      delay=0.,
                      force=False,
                      ) -> tableproxy.TableProxy:
        raise NotImplementedError

    @abstractmethod
    def _registerTable(self, tabproxy: tableproxy.TableProxy):
        raise NotImplementedError
