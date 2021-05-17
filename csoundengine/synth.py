from __future__ import annotations
import time
import numpy as np
from .config import config, logger
from . import internalTools
from emlib import iterlib
import weakref as _weakref

from typing import TYPE_CHECKING, Union as U, Optional as Opt, KeysView, Dict, List, Set


if TYPE_CHECKING:
    from .engine import Engine
    from .instr import Instr
    from .paramtable import ParamTable
    from .session import Session


__all__ = ['AbstrSynth', 'Synth', 'SynthGroup']


class AbstrSynth:
    """
    A base class for Synth and SynthGroup

    Attributes:
        engine (Engine): the Engine which scheduled this synth
        autostop (bool): stop the underlying csound synth when this object
            is deleted
    """

    def __init__(self, engine: Engine, autostop: bool = False, priority: int = 1):
        self.engine = engine
        self.autostop = autostop
        self.priority = priority

    def __del__(self):
        try:
            if self.autostop:
                self.stop(stopParent=False)
        except:
            pass

    def stop(self, delay=0., stopParent=False) -> None:
        """ Stop this synth """
        raise NotImplementedError()

    def isPlaying(self) -> bool:
        """ Is this synth playing? """
        raise NotImplementedError()

    def wait(self, pollinterval: float = 0.02, sleepfunc=time.sleep) -> None:
        """
        Wait until this synth has stopped

        Args:
            pollinterval: polling interval in seconds
            sleepfunc: the function to call when sleeping, defaults to time.sleep
        """
        internalTools.setSigintHandler()
        try:
            while self.isPlaying():
                sleepfunc(pollinterval)
        except:
            raise
        internalTools.removeSigintHandler()

    def set(self, *args, **kws) -> None:
        """
        Set a value of a param table. Multiple syntaxes are possible::

            synth.set('key1', value1, ['key2', value2, ...])
            synth.set(0, value1, 1, value2)
            synth.set(key1=value1, [key2=value2, ...])

        See Also:

            * :meth:`~AbstractSynth.setp`

        Example
        =======

        .. code::

            from csoundengine import *
            session = Engine().session()
            session.defInstr('sine', r'''
                {amp=0.1, freq=1000}
                outch 1, oscili:a(kamp, kfreq)
            ''')
            synth = session.sched('sine', kfreq=440)
            synth.set('kfreq', 2000, delay=3)
        """
        raise NotImplementedError()

    def get(self, slot: U[int, str], default: float = None) -> Opt[float]:
        """
        Get the value of a named parameter

        Args:
            slot (int|str): the slot name/index
            default (float): if given, this value will be returned if the slot
                does not exist

        Returns:
            the current value of the given slot, or default if a slot with
            the given key does not exist
        """
        raise NotImplementedError()

    def tableParams(self) -> Opt[KeysView[str]]:
        """
        Return a seq. of a all named parameters if this synth has a
        parameters table
        """
        return self.tableState().keys()

    def tableState(self) -> Dict[str, float]:
        """
        Get the state of all named parameters

        Returns:
            a dict mapping parameter name to its current value
        """
        raise NotImplementedError()

    def hasParamTable(self) -> bool:
        """ Does this synth have an associated parameter table?"""
        raise NotImplementedError()

    def automateTable(self, param: str, pairs: U[List[float], np.ndarray],
                      mode="linear", delay=0.) -> None:
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
        """
        raise NotImplementedError()

    def namedPfields(self) -> Opt[Set[str]]:
        """
        Returns a set of all named pfields
        """
        raise NotImplementedError()

    def automatep(self, param: U[int, str], pairs: U[List[float], np.ndarray],
                  mode="linear", delay=0.) -> AbstrSynth:
        """
        Automate the value of a pfield.

        Args:
            param (int|str): either the parg index (5=p5) or the name of the parg
                as used in the body of the instrument (for example, if the
                body contains the line "kfreq = p5", "kfreq" could be used as param)
            pairs (List[float] | np.ndarray): 1D sequence of floats with the form
                [x0, y0, x1, y1, x2, y2, ...]
            mode: one of 'linear', 'cos', 'expon(xx)', 'smooth'. See the csound opcode
                `interp1d` for more information
                (https://csound-plugins.github.io/csound-plugins/opcodes/interp1d.html)
            delay (float): 0 to start as soon as possible, otherwise the delay
                before the automateion starts

        Returns:
            a Synth representing the automation routine

        See also: pwrite
        """
        raise NotImplementedError()

    def setp(self, *args, delay=0., **kws) -> None:
        """
        Modify the value of a pfield.

        See also automateParg
        """
        raise NotImplementedError()

    def session(self) -> Session:
        """
        Returns the Session which schedules this Synth
        """
        return self.engine.session()


class Synth(AbstrSynth):
    """

    Args:
        engine: the engine instance where this synth belongs to
        synthid: the synth id inside csound (p1, a fractional instrument number)
        instr: the Instr which originated this Synth
        starttime: when was this synth started
        dur: duration of the note (can be -1 for infinite)
        pargs: the pargs used to create this synth
        synthgroup: the group this synth belongs to (if any)
        autostop: should this synth autostop? If True, the lifetime of the csound note
            is associated with this Synth object, so if this Synth goes out of
            scope or is deleted, the underlying note is unscheduled
        table: an associated Table (if needed)

    Attributes:
        synthid: the synth id inside csound (p1, a fractional instrument number)
        engine: the engine instance where this synth belongs to
        synthid: the synth id inside csound (p1, a fractional instrument number)
        instr: the Instr which originated this Synth
        startTime: when was this synth started
        dur: duration of the note (can be -1 for infinite)
        pargs: the pargs used to create this synth
        synthGroup: the group this synth belongs to (if any)
        table (ParamTable): an associated Table (if defined)

    Example
    =======

    .. code::

        from csoundengine import *
        session = Engine().session()
        session.defInstr('vco', r'''
            |kamp=0.1, kmidi=60, ktransp=0|
            asig vco2 kamp, mtof:k(kmidi+ktransp)
            asig *= linsegr:a(0, 0.1, 1, 0.1, 0)
            outch 1, asig
        ''')
        midis = [60, 62, 64]
        synths = [session.sched('vco', kamp=0.2, kmidi=midi) for midi in midis]
        # synths is a list of Synth
        synths[1].automatep('ktransp', [0, 0, 10, -1])
    """

    def __init__(self,
                 engine: Engine,
                 synthid: float,
                 instr: Instr,
                 starttime: float = None,
                 dur: float = -1,
                 pargs=None,
                 synthgroup: SynthGroup = None,
                 autostop=False,
                 table: ParamTable = None,
                 priority: int = 1,
                 ) -> None:
        """

        """
        AbstrSynth.__init__(self, engine=engine, autostop=autostop, priority=priority)
        self.synthid: float = synthid
        self.instr: Instr = instr
        self.startTime: float = starttime or time.time()
        self.dur: float = dur
        self.pargs: List[float] = pargs
        self.table: Opt[ParamTable] = table
        self.synthGroup = synthgroup
        self._playing: bool = True

    def __repr__(self):
        parts = [self.instr.name, f"id={self.synthid}"]
        if self.table:
            parts.append(self.table._mappingRepr())
        if self.pargs:
            i2n = self.instr.pargsIndexToName
            argsstrs = []
            pargs = self.pargs[0:]
            for i, parg in enumerate(pargs, start=0):

                name = i2n.get(i+4)
                if name:
                    s = f"p{i+4}:{name}={parg}"
                else:
                    s = f"p{i+4}={parg}"
                argsstrs.append(s)
            argsstr = ", ".join(argsstrs)
            parts.append(argsstr)
        return "Synth(" + ", ".join(parts) + ")"

    @property
    def p1(self) -> float:
        """ The synth id (corresponds to the p1 value) """
        return self.synthid

    def isPlaying(self) -> bool:
        """ Is this Synth playing """
        now = time.time()
        if self.dur>0:
            return self._playing and self.startTime<now<self.startTime+self.dur
        return self._playing and self.startTime<now

    def tableState(self) -> Opt[Dict[str, float]]:
        if self.table is None:
            return None
        return self.table.asDict()

    def tableParams(self) -> Opt[KeysView]:
        if self.table is None:
            return None
        return self.table.mapping.keys()

    def set(self, *args, delay=0., **kws) -> None:
        if not self._playing:
            logger.error("synth not playing")
            return

        if not self.table:
            logger.error("This synth has no associated table, skipping")
            return

        if delay > 0:
            if args:
                for key, value in iterlib.pairwise(args):
                    slotidx = self.table.paramIndex(key)
                    self.engine.tableWrite(self.table.tableIndex, slotidx, value, delay=delay)
            if kws:
                for key, value in kws.items():
                    slotidx = self.table.paramIndex(key)
                    self.engine.tableWrite(self.table.tableIndex, slotidx, value,
                                           delay=delay)
        else:
            if args:
                for key, value in iterlib.pairwise(args):
                    self.table[key] = value
            if kws:
                for key, value in kws.items():
                    self.table[key] = value

    def get(self, slot: U[int, str], default: float = None) -> Opt[float]:
        if not self._playing:
            logger.error("Synth not playing")
            return

        if not self.table:
            logger.error("This synth has no associated table, skipping")
            return

        return self.table.get(slot, default)

    def namedPfields(self) -> Opt[Set[str]]:
        name2idx = self.instr.pargsNameToIndex
        return set(name2idx.keys()) if name2idx else None

    def setp(self, *args, delay=0., **kws) -> None:
        """
        Modify a parg of this synth.

        Multiple pargs can be modified simultaneously. It only makes sense
        to modify a parg if a k-rate variable was assigned to this parg
        (see example). A parg can be referred to via an integer, corresponding
        to the p index (5 would refer to p5), or to the name of the assigned k-rate
        variable as a string (for example, if there is a line "kfreq = p6",
        both 6 and "kfreq" refer to the same parg).

        Example
        =======

            >>> session = Engine(...).session()
            >>> session.defInstr("sine",
            '''
            kamp = p5
            kfreq = p6
            outch 1, oscili:ar(kamp, kfreq)
            '''
            )
            >>> synth = session.sched('sine', pargs=[0.1, 440])
            >>> synth.setp(5, 0.5)
            >>> synth.setp(kfreq=880)
            >>> synth.setp(5, 0.1, 6, 1000)
            >>> synth.setp("kamp", 0.2, 6, 440)
        """
        # most common use: just one pair
        if not kws and len(args) == 2:
            k = args[0]
            idx = k if isinstance(k, int) else self.instr.pargIndex(k)
            self.engine.setp(self.synthid, idx, args[1], delay=delay)
        pairsd = {}
        instr = self.instr
        if args:
            assert len(args)%2 == 0
            for i in range(len(args)//2):
                k = args[i*2]
                v = args[i*2+1]
                idx = instr.pargIndex(k)
                pairsd[idx] = v
        if kws:
            for k, v in kws.items():
                idx = instr.pargIndex(k)
                pairsd[idx] = v
        pairs = iterlib.flatdict(pairsd)
        self.engine.setp(self.synthid, *pairs, delay=delay)

    def hasParamTable(self) -> bool:
        """ Returns True if this synth has an associated parameter table """
        return self.table is not None

    def automateTable(self, param: str, pairs: U[List[float], np.ndarray],
                      mode="linear", delay=0.) -> float:
        if not self.table:
            raise RuntimeError(
                f"{self.instr.name} (id={self.synthid}) has no parameter table")
        paramidx = self.table.paramIndex(param)
        if paramidx is None:
            raise KeyError(f"Unknown param {param} for synth {self.synthid}")
        if len(pairs)>1900:
            raise ValueError(f"pairs is too long (max. pairs = 900, got {len(pairs)/2})")
        return self.engine.automateTable(self.table.tableIndex, paramidx, pairs,
                                         mode=mode, delay=delay)

    def automatep(self, param: U[int, str], pairs: U[List[float], np.ndarray],
                  mode="linear", delay=0.) -> float:
        if isinstance(param, str):
            pidx = self.instr.pargIndex(param)
            if not pidx:
                raise KeyError(f"parg {param} not known. "
                               f"Known pargs: {self.instr.pargsIndexToName}")
        else:
            pidx = param
        synthid = self.engine.automatep(self.p1, pidx=pidx, pairs=pairs,
                                        mode=mode, delay=delay)
        return synthid

    def stop(self, delay=0., stopParent=False) -> None:
        if not self.isPlaying():
            return
        if self.synthGroup is not None and stopParent:
            self.synthGroup.stop(delay=delay)
        else:
            self.session().unsched(self.synthid, delay=delay)


class SynthGroup(AbstrSynth):
    """
    A SynthGroup is used to control multiple synths created
    to work together. Such multiple synths can be groups of
     similar synths, as in additive synthesis, or processing
     chains which work as an unity.

    Attributes:
        synths (List[AbstrSynth]): the list of synths in this group
    """

    def __init__(self, synths: List[Synth], autostop=False) -> None:
        assert isinstance(synths, list) and len(synths)>0
        priority = max(synth.priority for synth in synths)
        AbstrSynth.__init__(self, engine=synths[0].engine, autostop=autostop,
                            priority=priority)
        groupref = _weakref.ref(self)
        for synth in synths:
            synth.synthgroup = groupref
        self.synths: List[Synth] = synths

    def stop(self, delay=0, stopParent=False) -> None:
        for s in self.synths:
            s.stop(stopParent=False, delay=delay)

    def isPlaying(self) -> bool:
        return any(s.isPlaying() for s in self.synths)

    def automateTable(self, param: str, pairs, mode="linear", delay=0.) -> None:
        for synth in self.synths:
            if isinstance(synth, Synth):
                if synth.table and param in synth.tableParams():
                    synth.automateTable(param, pairs, mode=mode, delay=delay)
            elif isinstance(synth, SynthGroup):
                synth.automateTable(param=param, pairs=pairs, mode=mode, delay=delay)

    def namedPfields(self) -> Set[str]:
        out: Set[str] = set()
        for synth in self.synths:
            namedPargs = synth.namedPfields()
            if namedPargs:
                out.update(namedPargs)
        return out

    def automatep(self, param: U[int, str], pairs: U[List[float], np.ndarray],
                  mode="linear", delay=0.) -> List[float]:
        eventids = []
        for synth in self.synths:
            if synth.table and param in synth.tableParams():
                eventids.append(synth.automatep(param, pairs, mode=mode, delay=delay))
        return eventids

    def hasParamTable(self) -> bool:
        return any(s.hasParamTable() is not None for s in self.synths)

    def tableState(self) -> Opt[Dict[str, float]]:
        dicts = []
        for s in self.synths:
            d = s.tableState()
            if d:
                dicts.append(d)
        if not dicts:
            return None
        out = dicts[0]
        for d in dicts[1:]:
            out.update(d)
        return out

    def __repr__(self) -> str:
        lines = [f"SynthGroup(n={len(self.synths)})"]
        for synth in self.synths:
            lines.append("    "+repr(synth))
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.synths)

    def __getitem__(self, idx) -> Synth:
        return self.synths[idx]

    def __iter__(self):
        return iter(self.synths)

    def set(self, *args, **kws) -> None:
        for synth in self.synths:
            synth.set(*args, **kws)

    def get(self, idx: U[int, str], default=None) -> List[float]:
        return [synth.get(idx, default=default) for synth in self.synths]

    def setp(self, *args, delay=0., **kws) -> None:
        for synth in self.synths:
            synth.setp(*args, delay=delay, **kws)