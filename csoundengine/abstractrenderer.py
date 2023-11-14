from __future__ import annotations

from abc import abstractmethod, ABC

from typing import TYPE_CHECKING, Sequence, Callable
if TYPE_CHECKING:
    from csoundengine import busproxy
    from csoundengine.schedevent import SchedEvent, SchedEventGroup
    import numpy as np
    from csoundengine.event import Event
    from csoundeninge import tableproxy
    from csoundengine import instr


__all__ = (
    'AbstractRenderer'
)


class AbstractRenderer(ABC):

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
              args: Sequence[float|str] | dict[str, float] | None = None,
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
                 args: dict[str, float|str] = None,
                 init: str = '',
                 priority: int = None,
                 doc: str = '',
                 includes: list[str] | None = None,
                 aliases: dict[str, str] = None,
                 useDynamicPfields: bool = None,
                 **kws) -> instr.Instr:
        raise NotImplementedError

    @abstractmethod
    def makeTable(self,
                  data: np.ndarray | list[float] | None = None,
                  size: int = 0,
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
                          instr: instr.Instr
                          ) -> str:
        raise NotImplementedError

    @abstractmethod
    def registerInstr(self, instr: instr.Instr) -> bool:
        raise NotImplementedError

    @abstractmethod
    def registeredInstrs(self) -> dict[str, instr.Instr]:
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
    def getInstr(self, instrname: str) -> instr.Instr | None:
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
                 delay: float = None,
                 overtake=False
                 ) -> float:
        raise NotImplementedError

    @abstractmethod
    def hasBusSupport(self) -> bool:
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
                   dur=-1.,
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

