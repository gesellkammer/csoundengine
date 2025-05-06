from __future__ import annotations

from abc import ABC, abstractmethod

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import tableproxy
    from . import schedevent
    import numpy as np
    from . import event as _event


class SessionHandler(ABC):

    @abstractmethod
    def schedEvent(self, event: _event.Event) -> schedevent.SchedEvent:
        raise NotImplementedError

    def makeTable(self,
                  data: np.ndarray | list[float] | None = None,
                  size: int | tuple[int, int] = 0,
                  sr: int = 0,
                  ) -> tableproxy.TableProxy:
        # Raising NotImplementedError bypasses the handler and lets the
        # session take ove
        raise NotImplementedError

    def readSoundfile(self,
                      path: str,
                      chan=0,
                      skiptime=0.,
                      delay=0.,
                      force=False,
                      ) -> tableproxy.TableProxy:
        # Raising NotImplementedError bypasses the handler and lets the
        # session take ove
        raise NotImplementedError
