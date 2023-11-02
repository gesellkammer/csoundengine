from __future__ import annotations
from dataclasses import dataclass
from typing import Callable


@dataclass
class SessionEvent:
    """
    A class to store a Session event to be scheduled

    """
    instrname: str
    """The name of the instrument"""

    delay: float = 0
    """The time offset to start the event"""

    dur: float = -1
    """The duration of the event"""

    priority: int = 1
    "The events priority (>1)"

    args: list[float] | dict[str, float] | None = None
    "Numbered pfields or a dict of parameters `{name: value}`"

    whenfinished: Callable = None
    "A callback to be fired when this event is finished"

    relative: bool = True
    "Is the delay expressed in relative time?"

    kws: dict[str, float] | None = None
    "Keywords passed to the instrument"

    uniqueId: int = 0
    """If applicable, a unique id identifying this event. 0 indicates no id"""

