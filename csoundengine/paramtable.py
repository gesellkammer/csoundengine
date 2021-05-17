from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Optional as Opt, Union as U
import numpy as np
from .config import config

if TYPE_CHECKING:
    from .engine import Engine

class ParamTable:
    """
    A ParamTable is a csound table used as a multivalue communication channel
        between a running instrument and the outside world. ParamTables are used
        to communicate with a specific instance of an instrument.
        Channels or globals should be used to address global state.

        Each instrument can define the need of a param table attached at creation time,
        together with initial/default values for all slots. It is also possible
        to assign names to each slot

        The underlying memory can be accessed either directly via the .array
        attribute (a numpy array pointing to the table memory), or via
        table[idx] or table[key]

        A ParamTable does not currently check if the underlying csound table
        exists.

    Attributes:
        tableIndex (int): the table number of the csound table
        mapping (Dict[str, int]): a dict mapping slot name to index
        instrName (str): the name of the instrument which defines this table
        engine (Engine): the engine where this table exists
        deallocated (bool): has this table been deallocated?
    """
    def __init__(self,
                 idx: int,
                 engine: Engine,
                 mapping: Dict[str, int] = None,
                 instrName: str = None):
        """

        Args:
            engine: the engine where this table is define
            idx: the index of the table
            mapping: an optional mapping from keys to table indices
                (see setNamed)
            instrName: the name of the instrument to which this table belongs
                (optional, used for the case where a Table is used as
                communication channel)
        """
        self.tableIndex: int = int(idx)
        self.mapping: Dict[str, int] = mapping or {}
        self.instrName = instrName
        self.engine: Engine = engine
        self._array: Opt[np.ndarray] = None
        self.deallocated = False
        self._failSilently = config['unknown_parameter_fail_silently']

    def __repr__(self):
        return f"ParamTable(tableIndex={self.tableIndex}, " \
               f"groupName={self.engine.name})"

    def __len__(self) -> int:
        return len(self.array)

    def getSize(self) -> int:
        return len(self.array)

    def paramIndex(self, param: str) -> Opt[int]:
        """
        Returns the index corresponding to the named parameter
        Returns None if the parameter does not exist
        """
        if not self.mapping:
            return None
        return self.mapping.get(param, None)

    @property
    def array(self):
        if self._array is not None:
            return self._array
        self._array = a = self.engine.csound.table(self.tableIndex)
        return a

    def __setitem__(self, idx, value):
        if isinstance(idx, int):
            self.array[idx] = value
        else:
            self._setNamed(idx, value)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.array[idx]
        return self._getNamed(idx)

    def _setNamed(self, key: str, value: float) -> None:
        """
        Set a value via a named index

        Args:

            key: a key as defined in mapping
            value: the value to set the corresponding index to
        """
        idx = self.mapping.get(key, -1)
        if idx<0:
            if not self._failSilently:
                raise KeyError(f"key {key} not known (keys={list(self.mapping.keys())}")
        else:
            self.array[idx] = value

    def get(self, key: str, default=None) -> Opt[float]:
        """
        Get the value of a named slot. If key is not found, return default
        (similar to dict.get)
        """
        idx = self.mapping.get(key, -1)
        if idx == -1:
            return default
        return self.array[idx]

    def _mappingRepr(self) -> str:
        values = self.array
        if not self.mapping:
            if values is None:
                return ""
            return str(values)
        if values is None:
            # table is not "live"
            return ", ".join(f"{key}=?" for key in self.mapping.keys())
        keys = list(self.mapping.keys())
        return ", ".join(f"{key}={value}" for key, value in zip(keys, values))

    def asDict(self) -> dict:
        """
        Return a dictionary mapping keys to their current value. This is
        only valid if this Table has a mapping, associating keys to indices
        in the table
        """
        if not self.mapping:
            raise ValueError("This Table has no mapping")
        values = self.array
        return {key:values[idx] for key, idx in self.mapping.items()}

    def _getNamed(self, key: str) -> float:
        idx: int = self.mapping.get(key, -1)
        if idx<0:
            raise KeyError(f"key {key} not known (keys={list(self.mapping.keys())}")
        return self.array[idx]

    def free(self) -> None:
        """ Free this table """
        if self.deallocated:
            return
        self.engine.freeTable(self.tableIndex)
        self.deallocated = True
