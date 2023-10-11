from __future__ import annotations

from abc import abstractmethod
import numpy as np
from typing import Sequence
from . import jupytertools


_EMPTYSET = frozenset()


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
    def _setp(self, param: str, value: float, delay=0.) -> None:
        """
        Set the value of a k-rate pfield for this event
        """
        raise NotImplementedError

    @abstractmethod
    def _setTable(self, param: str, value: float, delay=0.) -> None:
        """
        Set a value of a param table

        Args:
            param: the parameter to set
            value: the value
            delay: when to set it. This is a relative value if the event
                is in a live renderer, or an absolute value if the
                event is offline
        """
        raise NotImplementedError

    def set(self, param='', value: float = 0., delay=0., **kws) -> None:
        """
        Set a value of a named parameter

        Args:
            param: the parameter name. Can also be an unnamed param, like 'p5'
            value: the value to set
            delay: when to set this parameter
            kws: the key should be a named parameter, or p5, p6, etc., if
                setting a parameter by index. Bear in mind that only parameters
                assigned to a control variable will see any modification

        Example
        ~~~~~~~

            >>> from csoundengine import *
            >>> s = Engine().session()
            >>> s.defInstr('osc', r'''
            ... |kfreq, kamp|
            ... outch 1, oscili:a(kamp, kfreq)
            ... ''')
            >>> synth = s.sched('osc', kfreq=1000, kamp=0.5)
            >>> synth.set(kfreq=440)
            >>> # Parameters can be given as index also:
            >>> synth.set(p5=440, delay=2.5)
            >>> # Multiple parameters can be set at a time
            >>> synth.set(kfreq=442, kamp=0.1)
        """
        if kws:
            for k, v in kws.items():
                self.set(param=k, value=v, delay=delay)

        if param:
            tabkeys = self._controlNames()
            if param in tabkeys:
                self._setTable(param=param, value=value, delay=delay)
            elif param.startswith('p') or param in self._pfieldNames():
                self._setp(param=param, value=value, delay=delay)
            else:
                raise KeyError(f"Unknown parameter: '{param}'. "
                               f"Possible parameters for this event: {self.dynamicParams()}")

    def _automatePfield(self,
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
        params = self.dynamicParams()
        if param not in params:
            raise KeyError(f"Parameter {param} not known. Dynamic parameters: {params}")

        if (tabargs := self._controlNames()) and param in tabargs:
            return self._automateTable(param=param, pairs=pairs, mode=mode, delay=delay, overtake=overtake)
        elif param.startswith('p') or ((pargs := self._pfieldNames()) and param in pargs):
            return self._automatePfield(param=param, pairs=pairs, mode=mode, delay=delay, overtake=overtake)
        else:
            raise KeyError(f"Unknown parameter '{param}', supported parameters: {self.dynamicParams()}")

    def dynamicParams(self) -> set[str]:
        """
        The set of all dynamic parameters accepted by this event

        Returns:
            a set of the dynamic (modifiable) parameters accepted by this event
        """
        params = self.paramNames()
        return set(p for p in params if p.startswith('k'))

    def _pfieldNames(self) -> frozenset[str]:
        """
        Returns a set of all named pfields
        """
        raise NotImplementedError

    def _controlNames(self) -> frozenset[str]:
        """
        The names of all controls for this event

        Returns an empty set if this event does not have controls
        """
        raise NotImplementedError

    def paramNames(self) -> frozenset[str]:
        """
        Set of all named parameters

        .. seealso::

            :meth:`~BaseEvent.dynamicParams`, :meth:`~BaseEvent.set`,
            :meth:`~BaseEvent.automate`
        """
        pargs = self._pfieldNames()
        tableargs = self._controlNames()
        if not pargs and not tableargs:
            return _EMPTYSET
        elif pargs:
            return pargs
        elif tableargs:
            return tableargs
        return pargs | tableargs

    def show(self) -> None:
        """
        If inside jupyter, display the html representation of self
        """
        if jupytertools.inside_jupyter():
            from IPython.display import display
            display(self)
