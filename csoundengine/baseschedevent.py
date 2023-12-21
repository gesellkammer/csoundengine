from __future__ import annotations

from abc import abstractmethod, ABC
import numpy as np
from . import jupytertools
from . import csoundlib
from ._common import EMPTYSET, EMPTYDICT

from typing import Sequence


class BaseSchedEvent(ABC):
    """
    Interface for all scheduled events / groups

    Args:
        start: the start time relative to the start of the engine
        dur: the duration of the synth
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
    def paramValue(self, param: str) -> float | str | None:
        """
        Query the value of a parameter, if possible

        If the event is live the actual value is queried; otherwise,
        the value used at event initialization

        Args:
            the parameter to query

        Returns:
            the value of the parameter.

        TODO: check if it actual returns None and why
        """
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
    def _setPfield(self, param: str, value: float, delay=0.) -> None:
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
            param = self.unaliasParam(param, param)
            if csoundlib.isPfield(param) or param in self.pfieldNames(aliases=False):
                self._setPfield(param=param, value=value, delay=delay)
            elif param in self.controlNames(aliases=False):
                self._setTable(param=param, value=value, delay=delay)
            else:
                raise KeyError(f"Unknown parameter: '{param}'. "
                               f"Possible parameters for this event: {self.dynamicParamNames(aliased=True)}")

    def _automatePfield(self,
                        param: int | str,
                        pairs: Sequence[float] | np.ndarray,
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
        lenpairs = len(pairs)
        assert lenpairs % 2 == 0 and lenpairs >= 2

        if lenpairs == 2:
            # A single value, set it
            delay, value = pairs
            self.set(param=param, value=value, delay=delay)
            return 0

        if isinstance(param, str):
            param = self.unaliasParam(param, param)

        if isinstance(param, int) or csoundlib.isPfield(param) or (
                (pargs := self.pfieldNames(aliases=False)) and param in pargs):
            return self._automatePfield(param=param, pairs=pairs, mode=mode, delay=delay, overtake=overtake)
        elif (tabargs := self.controlNames(aliases=False)) and param in tabargs:
            return self._automateTable(param=param, pairs=pairs, mode=mode, delay=delay, overtake=overtake)
        else:
            raise KeyError(f"Unknown parameter '{param}', supported parameters: {self.dynamicParamNames()}")

    def dynamicParamNames(self, aliases=True, aliased=False) -> frozenset[str]:
        """
        The set of all dynamic parameters accepted by this event

        Returns:
            a set of the dynamic (modifiable) parameters accepted by this event
        """
        params = self.paramNames()
        dynparams = set(p for p in params if p.startswith('k'))
        if aliases and (_aliases := self.aliases()):
            dynparams |= _aliases.keys()
            if not aliased:
                dynparams.difference_update(_aliases.values())
        return frozenset(dynparams)

    def aliases(self) -> dict[str, str]:
        return EMPTYDICT

    def unaliasParam(self, param: str, default='') -> str:
        """
        Returns the alias of param or default if no alias was found

        Args:
            param: the parameter to unalias
            default: the value to return if `param` has no alias

        Returns:
            the original name or `default` if no alias was found
        """
        orig = self.aliases().get(param)
        return orig if orig is not None else default

    def pfieldNames(self, aliases=True, aliased=False) -> frozenset[str]:
        """
        Returns a set of all named pfields

        Args:
            aliases: if True, included aliases for pfields (if applicable)
            aliased: if True, include the aliased names. Otherwise if aliases
                are defined and included (aliases=True), the original names
                will be excluded

        Returns:
            a set with the all pfield names.

        """
        raise NotImplementedError

    def controlNames(self, aliases=True, aliased=False) -> frozenset[str]:
        """
        The names of all controls

        Args:
            aliases: if True, included control names aliases (if applicable)
            aliased: if True, include the aliased names

        Returns:
            the names of all controls. Returns an empty set if this event
            does not have any controls
        """
        raise NotImplementedError

    def paramNames(self) -> frozenset[str]:
        """
        Set of all named parameters

        .. seealso::

            :meth:`~BaseEvent.dynamicParams`, :meth:`~BaseEvent.set`,
            :meth:`~BaseEvent.automate`
        """
        pargs = self.pfieldNames()
        tableargs = self.controlNames()
        if pargs and tableargs:
            return pargs | tableargs
        elif pargs:
            return pargs
        elif tableargs:
            return tableargs
        return EMPTYSET

    def show(self) -> None:
        """
        If inside jupyter, display the html representation of self
        """
        if jupytertools.inside_jupyter():
            from IPython.display import display
            display(self)
        else:
            print(repr(self))

