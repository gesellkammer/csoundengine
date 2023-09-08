from __future__ import annotations

from abc import abstractmethod
import numpy as np
from typing import Sequence
from .config import logger

from . import jupytertools


class BaseEvent:
    """
    Base class for all scheduled events (both offline and realtime) / groups

    Args:
        p1: the event id
        start: the start time relative to the start of the engine
        dur: the duration of the synth
        args: the pfields of this event, beginning with p4
    """

    __slots__ = ('start', 'dur')

    def __init__(self, start: float, dur: float):
        self.start = start
        """Start time of the event (p2)"""

        self.dur = dur
        """Duration of the event (p3). Can be negative to indicate an endless/tied event"""

    def _repr_html_(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def stop(self, delay=0.) -> None:
        """ Stop this event """
        raise NotImplementedError

    @property
    def end(self) -> float:
        """End of this event (can be inf if the duration is given as negative)"""
        return float('inf') if self.dur < 0 else self.start + self.dur

    @abstractmethod
    def setp(self, delay=0., strict=True, **kws) -> None:
        """
        Set the value of a pfield for this event
        """
        raise NotImplementedError

    def _setTable(self, delay=0., **kws) -> None:
        """
        Set a value of a param table
        """
        raise NotImplementedError

    def set(self, delay=0., strict=True, **kws) -> None:
        """
        Set a value of a named parameter

        Args:
            delay: when to set this parameter
            strict: if True, any mismatched parameter will raise an Exception
            kws: the key should be a named parameter, or p5, p6, etc., if
                setting a parameter by index. Bear in mind that only parameters
                assigned to a control variable will see any modification

        Example
        ~~~~~~~

            >>> from csoundengine import *
            >>> s = Engine().session()
            >>> s.defInstr('osc', r'''
            ... kfreq = p5
            ... kamp = p6
            ... outch 1, oscili:a(kamp, kfreq)
            ... ''')
            >>> synth = s.sched('osc', kfreq=1000, kamp=0.5)
            >>> synth.set(kfreq=440)
            >>> # Parameters can be given as index also:
            >>> synth.set(p5=440, delay=2.5)
            >>> # Multiple params can be set at a time
            >>> synth.set(kfreq=442, kamp=0.1)
        """
        if strict:
            params = self.dynamicParams()
            for param in kws.keys():
                if param not in params:
                    raise KeyError(f"Parameter {param} not known. Dynamic parameters: {params}")

        mode = self.paramMode()
        if mode == 'parg':
            # We set strict to false since we already checked
            return self.setp(delay=delay, strict=False, **kws)
        elif mode == 'table':
            return self._setTable(delay=delay, **kws)
        else:
            logger.error(f"Parameter mode {mode} not supported for {self}")

    def hasParamTable(self) -> bool:
        """ Does this event have an associated parameter table?"""
        raise False

    def _automatep(self,
                   param: int | str,
                   pairs: list[float] | np.ndarray,
                   mode="linear",
                   delay=0.,
                   overtake=False) -> float:
        """
        Automate the value of a pfield.

        Args:
            param: either the pfield index (5=p5) or the name of the pfield
                as used in the body of the instrument (for example, if the
                body contains the line "kfreq = p5", "kfreq" could be used as param)
            pairs: 1D sequence of floats with the form
                [x0, y0, x1, y1, x2, y2, ...]
            mode: one of 'linear', 'cos', 'expon(xx)', 'smooth'. See the csound opcode
                `interp1d` for more information
                (https://csound-plugins.github.io/csound-plugins/opcodes/interp1d.html)
            delay: 0 to start as soon as possible, otherwise the delay
                before the automateion starts
            overtake: if True, the first value of pairs is replaced with
                the current value in the running instance

        Returns:
            a float with the synthid (p1) of the automation event. A value of 0 indicates
            that no automation event was scheduled, possibly because there is no
            intersection between the time of the synth and that of the automation

        .. seealso:: :meth:`~AbstrSynth.setp`, :meth:`~AbstrSynth.automate`
        """
        raise NotImplementedError

    def _automateTable(self,
                       param: str,
                       pairs: Sequence[float] | np.ndarray,
                       mode="linear",
                       delay=0.,
                       overtake=False
                       ) -> float:
        """
        Automate a table parameter. Time stamps are relative to the start
        of the automation

        Args:
            param: the named parameter as defined in the Instr
            pairs: a flat list of pairs of the form [time0, val0, time1, val1, ...]
            mode: one of 'linear', 'cos', 'expon(xx)', 'smooth'. See the csound opcode
                `interp1d` for more information
                (https://csound-plugins.github.io/csound-plugins/opcodes/interp1d.html)
            delay: time offset to start automation
            overtake: if True, the first value of pairs is replaced with
                the current value in the running instance

        .. seealso::

            * :meth:`~AbstrSynth.set`
            * :meth:`~AbstrSynth.get`

        """
        raise NotImplementedError

    def automate(self,
                 param: str | int,
                 pairs: Sequence[float] | np.ndarray,
                 mode='linear',
                 delay=0.,
                 overtake=False,
                 strict=True
                 ) -> float:
        """
        Automate any named parameter of this Synth

        This method will automate this synth's pfields / param table, depending of
        how the instrument was defined.

        Args:
            param: the name of the parameter to automate, or the param index
            pairs: automation data as a flat array with the form [time0, value0, time1, value1, ...]
            mode: one of 'linear', 'cos'. Determines the curve between values
            delay: when to start the automation
            overtake: if True, do not use the first value in pairs but overtake the current value

        Returns:
            the eventid of the automation event
        """
        if strict:
            params = self.dynamicParams()
            if param not in params:
                raise KeyError(f"Parameter {param} not known. Dynamic parameters: {params}")

        paramMode = self.paramMode()
        if paramMode == 'table':
            return self._automateTable(param=param, pairs=pairs, mode=mode, delay=delay, overtake=overtake)
        elif paramMode == 'parg':
            return self._automatep(param=param, pairs=pairs, mode=mode, delay=delay, overtake=overtake)
        else:
            raise RuntimeError("This Synth does not define any dynamic parameters")

    def paramMode(self) -> str:
        """
        Returns the dynamic parameter mode, or None

        Returns one of 'parg' or 'table'
        """
        return 'parg'

    def dynamicParams(self) -> set[str]:
        """
        The set of all dynamic parameters accepted by this Synth

        Returns:
            a set of the dynamic (modifiable) parameters accepted by this event
        """
        params = self.namedParams()
        return set(p for p in params if p.startswith('k'))

    def _namedPfields(self) -> set[str]:
        """
        Returns a set of all named pfields
        """
        raise NotImplementedError

    def _tableParams(self) -> set[str]:
        """
        Return a set of all named table parameters

        Returns None if this synth does not have a parameters table
        """
        return None

    def namedParams(self) -> set[str]:
        """
        Returns a set of named parameters, or None if this Synth has no named parameters

        These parameters can be modified via :meth:`~AbstrSynth.set` or
        :meth:`~AbstrSynth.automate`
        """
        mode = self.paramMode()
        if mode == 'parg':
            return self._namedPfields()
        else:
            return self._tableParams() or set()

    def show(self) -> None:
        """
        If inside jupyter, display the html representation of self
        """
        if jupytertools.inside_jupyter():
            from IPython.display import display
            display(self)

