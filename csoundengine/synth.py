from __future__ import annotations
import time
import numpy as np
from .config import logger, config
from . import internalTools
from emlib import iterlib
import emlib.misc
import weakref as _weakref
from . import jupytertools

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

    def playing(self) -> bool:
        """ Is this synth playing? """
        raise NotImplementedError()


    def finished(self) -> bool:
        """ Has this synth ceased to play? """
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
            while self.playing():
                sleepfunc(pollinterval)
        except:
            raise
        internalTools.removeSigintHandler()

    def set(self, *args, **kws) -> None:
        """
        Set a value of a param table.

        Multiple syntaxes are possible::

            synth.set('key1', value1, ['key2', value2, ...])
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

    def tableParams(self) -> Opt[Set[str]]:
        """
        Return a seq. of all named parameters

        Returns None if this synth does not have a parameters table
        """
        raise NotImplementedError()

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
                      mode="linear", delay=0., overtake=False) -> None:
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

        """
        raise NotImplementedError()

    def namedPfields(self) -> Opt[Set[str]]:
        """
        Returns a set of all named pfields
        """
        raise NotImplementedError()

    def automatep(self, param: U[int, str], pairs: U[List[float], np.ndarray],
                  mode="linear", delay=0., overtake=False) -> AbstrSynth:
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
            overtake: if True, the first value of pairs is replaced with
                the current value in the running instance

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

    @property
    def session(self) -> Session:
        """
        Returns the Session which scheduled this Synth
        """
        return self.engine.session()


_synthStatusIcon = {
    'playing': '▶',
    'stopped': '■',
    'future': '𝍪'
}

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

    def _html(self) -> str:
        argsfontsize = config['html_args_fontsize']
        maxi = config['synth_repr_max_args']
        style = jupytertools.defaultStyle
        playstr = _synthStatusIcon[self.playStatus()]
        parts = [
            f'{playstr} <strong style="color:{style["name.color"]}">'
            f'{self.instr.name}</strong>:{self.synthid:.4f}',
            ]
        if self.table is not None:
            parts.append(self.table._mappingRepr())
        if self.pargs:
            i2n = self.instr.pargsIndexToName
            argsstrs = []
            pargs = self.pargs[0:]
            if any(arg.startswith('k') for arg in self.instr.pargsNameToIndex):
                maxi = max(i+4 for i, n in i2n.items()
                           if n.startswith('k'))
            for i, parg in enumerate(pargs, start=4):
                if i > maxi:
                    argsstrs.append("...")
                    break
                name = i2n.get(i)
                if name:
                    s = f"<b>{name}</b>:{i}=<code>{parg:.6g}</code>"
                else:
                    # s = f"<b>p{i + 4}</b>=<code>{parg:.6g}</code>"
                    s = f"<b>{i}</b>=<code>{parg:.6g}</code>"
                argsstrs.append(s)
            argsstr = " ".join(argsstrs)
            argsstr = fr'<span style="font-size:{argsfontsize};">{argsstr}</span>'
            parts.append(argsstr)
        return '<span style="font-size:12px;">∿(' + ', '.join(parts) + ')</span>'

    def _repr_html_(self) -> str:
        if config['stop_button_inside_jupyter'] and emlib.misc.inside_jupyter():
            jupytertools.displayButton("Stop", self.stop)
        return f"<p>{self._html()}</p>"

    def __repr__(self):
        playstr = _synthStatusIcon[self.playStatus()]
        parts = [f'{playstr} {self.instr.name}:{self.synthid}']
        if self.table is not None:
            parts.append(self.table._mappingRepr())
        if self.pargs:
            maxi = config['synth_repr_max_args']
            i2n = self.instr.pargsIndexToName
            maxi = max((i for i, name in i2n.items() if name.startswith("k")),
                           default=maxi)
            argsstrs = []
            pargs = self.pargs[0:]
            for i, parg in enumerate(pargs, start=0):
                if i > maxi:
                    argsstrs.append("...")
                    break
                name = i2n.get(i+4)
                if name:
                    s = f"{name}:{i+4}={parg:.6g}"
                    #s = f"p{i+4}:{name}={parg:.8g}"
                else:
                    s = f"p{i+4}={parg:.6g}"
                argsstrs.append(s)
            argsstr = " ".join(argsstrs)
            parts.append(argsstr)
        lines = ["Synth(" + ", ".join(parts) + ")"]
        # add a line for k- pargs
        return "\n".join(lines)

    @property
    def p1(self) -> float:
        """ The synth id (corresponds to the p1 value) """
        return self.synthid

    @property
    def endTime(self) -> float:
        if self.dur < 0:
            return float("inf")
        return self.startTime + self.dur

    def playStatus(self) -> str:
        """
        Returns the playing status of this synth (playing, stopped or future)

        Returns:
            'playing' if currently playing, 'stopped' if this synth has already stopped
            or 'future' if it has not started

        """
        now = time.time()
        if self.startTime > now:
            return "future"
        elif not self._playing:
            return "stopped"
        else:
            return "playing"

    def playing(self) -> bool:
        """ Is this Synth playing """
        return self.playStatus() == "playing"

    def finished(self) -> bool:
        return self.playStatus() == 'stopped'

    def tableState(self) -> Opt[Dict[str, float]]:
        if self.table is None:
            return None
        return self.table.asDict()

    def tableParams(self) -> Opt[Set[str]]:
        if self.table is None:
            return None
        return set(self.table.mapping.keys())

    def set(self, *args, delay=0., **kws) -> None:
        if not self._playing:
            logger.info("synth not playing")

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
        if self.playStatus() == 'future':
            # Can we just modify the scheduled value?
            return


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
                      mode="linear", delay=0., overtake=False) -> float:
        if not self.table:
            raise RuntimeError(
                f"{self.instr.name} (id={self.synthid}) has no parameter table")
        paramidx = self.table.paramIndex(param)
        if paramidx is None:
            raise KeyError(f"Unknown param {param} for synth {self.synthid}")
        if len(pairs)>1900:
            raise ValueError(f"pairs is too long (max. pairs = 900, got {len(pairs)/2})")
        return self.engine.automateTable(self.table.tableIndex, paramidx, pairs,
                                         mode=mode, delay=delay, overtake=overtake)

    def automatep(self, param: U[int, str], pairs: U[List[float], np.ndarray],
                  mode="linear", delay=0., overtake=False) -> float:
        if isinstance(param, str):
            pidx = self.instr.pargIndex(param)
            if not pidx:
                raise KeyError(f"parg {param} not known. "
                               f"Known pargs: {self.instr.pargsIndexToName}")
        else:
            pidx = param
        synthid = self.engine.automatep(self.p1, pidx=pidx, pairs=pairs,
                                        mode=mode, delay=delay, overtake=overtake)
        return synthid

    def stop(self, delay=0., stopParent=False) -> None:
        if self.finished():
            return
        if self.synthGroup is not None and stopParent:
            self.synthGroup.stop(delay=delay)
        else:
            self.session.unsched(self.synthid, delay=delay)


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

    def playing(self) -> bool:
        return any(s.playing() for s in self.synths)

    def finished(self) -> bool:
        return all(s.finished() for s in self.synths)

    def automateTable(self, param: str, pairs, mode="linear", delay=0.,
                      overtake=False) -> None:
        for synth in self.synths:
            if isinstance(synth, Synth):
                if synth.table and param in synth.tableParams():
                    synth.automateTable(param, pairs, mode=mode, delay=delay,
                                        overtake=overtake)
            elif isinstance(synth, SynthGroup):
                synth.automateTable(param=param, pairs=pairs, mode=mode, delay=delay,
                                    overtake=overtake)

    def namedPfields(self) -> Set[str]:
        out: Set[str] = set()
        for synth in self.synths:
            namedPargs = synth.namedPfields()
            if namedPargs:
                out.update(namedPargs)
        return out

    def automatep(self, param: U[int, str], pairs: U[List[float], np.ndarray],
                  mode="linear", delay=0., overtake=False) -> List[float]:
        eventids = []
        for synth in self.synths:
            if synth.table and param in synth.tableParams():
                eventids.append(synth.automatep(param, pairs, mode=mode, delay=delay,
                                                overtake=overtake))
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

    def _uniqueInstr(self) -> bool:
        instr0 = self.synths[0].instr
        return all(synth.instr == instr0 for synth in self.synths if synth.playing())

    def _htmlTable(self) -> Opt[str]:
        synth0 = self.synths[0]
        instr0 = synth0.instr
        if any(synth.instr != instr0 for synth in self.synths):
            return
        colnames = ["p1", "start", "dur"]
        rows = [[] for _ in self.synths]
        for row, synth in zip(rows, self.synths):
            row.append(f'{synth.synthid} <b>{_synthStatusIcon[synth.playStatus()]}</b>')
            row.append("%.3f"%(synth.startTime-time.time()))
            row.append("%.3f"%synth.dur)
        if synth0.table is not None:
            keys = list(synth0.table.mapping.keys())
            colnames.extend(synth0.table.mapping.keys())
            for row, synth in zip(rows, self.synths):
                if synth.playStatus() != 'stopped':
                    values = synth.table.array[:len(keys)]
                    for value in values:
                        row.append(f'<code>{value}</code>')
                else:
                    row.extend(["-"] * len(keys))

        if synth0.pargs:
            maxi = config['synth_repr_max_args']
            i2n = instr0.pargsIndexToName
            maxi = max((i for i, name in i2n.items() if name.startswith("k")),
                       default=maxi)
            for i, parg in enumerate(synth0.pargs):
                if i > maxi:
                    colnames.append("...")
                name = i2n.get(i+4)
                if name:
                    colnames.append(f"{i+4}:{name}")
                else:
                    colnames.append(str(i+4))
            for row, synth in zip(rows, self.synths):
                row.extend("%.5g"%parg for parg in synth.pargs[:maxi])
                if len(synth.pargs) > maxi:
                    row.append("...")

        return emlib.misc.html_table(rows, headers=colnames)

    def _repr_html_(self) -> str:
        if config['stop_button_inside_jupyter'] and emlib.misc.inside_jupyter():
            jupytertools.displayButton("Stop", self.stop)
        if self._uniqueInstr():
            instrcol = jupytertools.defaultStyle["name.color"]
            header = f'SynthGroup - <b>{len(self.synths)}</b> synths, ' \
                     f'instr: <strong style="color:{instrcol}">{self.synths[0].instr.name}</strong>'
            return header + self._htmlTable()
        parts = ['<p>SynthGroup']
        for s in self.synths:
            html = s._html()
            indent = "&nbsp" *4
            p = f'<br>{indent}{html}'
            parts.append(p)
        parts.append("</p>")
        return "".join(parts)

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

    def set(self, *args, delay=0, **kws) -> None:
        for synth in self.synths:
            synth.set(*args, delay=delay, **kws)

    def get(self, idx: U[int, str], default=None) -> List[Opt[float]]:
        """
        Get the value of a tabarg

        If a synth in this group is not playing or hasn't a tabarg
        with the given name/idx, `default` is returned for that
        slot. The returned list has the same size as the number of
        synths in this group
        """
        return [synth.get(idx, default=default) for synth in self.synths]

    def setp(self, *args, delay=0., **kws) -> None:
        for synth in self.synths:
            if synth.startTime <= time.time() + delay <= synth.endTime:
                synth.setp(*args, delay=delay, **kws)

    def tableParams(self) -> Set[str]:
        allparams = set()
        for synth in self.synths:
            params = synth.tableParams()
            if params:
                allparams.update(params)
        return allparams