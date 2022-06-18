from __future__ import annotations
from csoundengine.instr import Instr
from abc import abstractmethod
from typing import Optional, Union
import numpy as np



class BaseSession:

    @abstractmethod
    def registerInstr(self, instr: Instr) -> bool: ...

    @abstractmethod
    def defInstr(self, name: str, body: str,
                 args: dict[str, float] = None,
                 init: str = None,
                 tabledef: dict[str, float] = None,
                 **kws) -> Instr: ...

    def registeredInstrs(self) -> dict[str, Instr]: ...

    def isInstrRegistered(self, instr: Instr) -> bool: ...

    def getInstr(self, name: str) -> Optional[Instr]:
        return self.registeredInstrs().get(name)

    def assignBus(self) -> int: ...

    def sched(self,
              instrname: str,
              delay=0.,
              dur=-1.,
              priority: int = 1,
              pargs: Union[List[float], Dict[str, float]] = [],
              tabargs: Dict[str, float] = None,
              whenfinished=None,
              relative=True,
              **pkws
              ): ...




