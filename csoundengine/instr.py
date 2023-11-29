from __future__ import annotations
from functools import cache
from emlib import textlib, iterlib
import numpy as np
import re
import os
import textwrap

from .config import config, logger
from .errors import CsoundError
from . import csoundlib
from . import jupytertools
from . import instrtools
from ._common import EMPTYDICT, EMPTYSET
from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from .abstractrenderer import AbstractRenderer


__all__ = (
    'Instr',
)


class Instr:
    """
    An Instr is a template used to schedule a concrete instrument

    Instrs are used within a :class:`~csoundengine.session.Session` (realtime
    rendering) or a :class:`~csoundengine.offline.Renderer` (offline rendering)

    .. note::

        An Instr must be registered at the Session/Renderer before it can be used.
        See :meth:`csoundengine.instr.Instr.register` or :meth:`csoundengine.session.Session.defInstr`

    Args:
        name: the name of the instrument
        body: the body of the instr (the text **between** 'instr' end 'endin')
        args: if given, a dictionary defining default values for arguments. Can be
            init-time ('i' prefix) or performance time (with 'k' prefix).
        init: code to be initialized at the instr0 level
        preschedCallback: a function ``f(synthid, args) -> args``, called before
            a note is scheduled with
            this instrument. Can be used to allocate a table or a dict and pass
            the resulting index to the instrument as parg
        doc: some documentation describing what this instr does
        includes: a list of files which need to be included in order for this instr to work
        aliases: if given, a dict mapping arg names to real argument names. It enables
            to define named args for an instrument using any kind of name instead of
            following csound names, or use any kind of name in an instr to avoid possible
            collisions while exposing a nicer name to the outside as alias
        useDynamicPfields: if True, use pfields to implement dynamic arguments (arguments
            given as k-variables). Otherwise dynamic args are implemented as named controls,
            using a big global table

    Example
    -------

    .. code-block:: python

        s = Engine().session()
        Instr('sine', r'''
            kfreq = p5
            iamp = p6
            a0 = oscili:a(iamp, kfreq)
            outch 1, a0
        ''').register(s)
        synth = s.sched('sine', kfreq=440, iamp=0.1)
        synth.stop()

    One can create an Instr and register it at a session in one operation:

    .. code-block:: python

        s = Engine().session()
        s.defInstr('sine', r'''
            kfreq = p5
            kamp = p6
            a0 = oscili:a(kamp, kfreq)
            outch 1, a0
        ''')


    **Default Values**

    An Instr can define default values for any of its parameters and define
    aliases for its names:

    .. code-block:: python

        s = Engine().session()
        s.defInstr('sine', r'''
            kamp = p5
            kfreq = p6
            a0 = oscili:a(kamp, kfreq)
            outch 1, a0
        ''', args={'kamp': 0.1, 'kfreq': 1000}, aliases={'frequency': 'kfreq'}
        )
        # We schedule an event of sine, kamp will take the default (0.1)
        synth = s.sched('sine', kfreq=440)
        synth.set(frequency=450, delay=1)   # Use alias
        synth.stop()

    **Inline arguments**

    An inline args declaration can set both parameter name and default value:

    .. code-block:: python

        s = Engine().session()
        Instr('sine', r'''
            |iamp=0.1, kfreq=1000|
            a0 = oscili:a(iamp, kfreq)
            outch 1, a0
        ''').register(s)
        synth = s.sched('sine', kfreq=440)
        synth.stop()

    The same can be achieved via an associated table:

    .. code-block:: python

        s = Engine().session()
        Instr('sine', r'''
            a0 = oscili:a(kamp, kfreq)
            outch 1, a0
        ''', tabargs=dict(amp=0.1, freq=1000
        ).register(s)
        synth = s.sched('sine', tabargs=dict(freq=440))
        synth.stop()


    An inline syntax exists also for tables:

    .. code-block:: python

        Intr('sine', r'''
            {amp=0.1, freq=1000}
            a0 = oscili:a(kamp, kfreq)
            outch 1, a0
        ''')


    This will create a table and fill it will the given/default values,
    and generate code to read from the table and free the table after
    the event is done. Call :meth:`~csoundengine.instr.Instr.dump` to see
    the generated code:

    .. code-block:: python

        i_params = p4
        if ftexists(i_params) == 0 then
            initerror sprintf("params table (%d) does not exist", i_params)
        endif
        i__paramslen = ftlen(i_params)
        if i__paramslen < {maxidx} then
            initerror sprintf("params table is too small (size: %d, needed: {maxidx})", i__paramslen)
        endif
        kamp tab 0, i_params
        kfreq tab 1, i_params
        a0 = oscili:a(kamp, kfreq)
        outch 1, a0

    **Offline rendering**

    An Instr can also be used to define instruments for offline rendering (see
    :class:`~csoundengine.offline.Renderer`)

    .. code-block:: python

        from csoundengine import *
        renderer = Renderer(sr=44100, nchnls=2)

        instrs = [
            Instr('saw', r'''
              kmidi = p5
              outch 1, oscili:a(0.1, mtof:k(kmidi))
            '''),
            Instr('sine', r'''
              |kamp=0.1, kmidi=60|
              asig oscili kamp, mtof:k(kmidi)
              asig *= linsegr:a(0, 0.1, 1, 0.1, 0)
              outch 1, asig
            ''')
        ]

        for instr in instrs:
            instr.register(renderer)

        score = [('saw', 0,   2, 60),
                 ('sine', 1.5, 4, 67),
                 ('saw', 1.5, 4, 67.1)]

        events = [renderer.sched(ev[0], delay=ev[1], dur=ev[2], pargs=ev[3:])
                  for ev in score]

        # Offline events can be modified just like real-time events
        renderer.automate(events[0], 'kmidi', pairs=[0, 60, 2, 59])
        renderer.set(events[1], 3, 'kmidi', 67.2)
        renderer.render("out.wav")

    """

    __slots__ = (
        'name',
        'pfields',
        'aliases',
        'init',
        'id',
        'controls',
        'numchans',
        'doc',
        'pfieldIndexToName',
        'pfieldNameToIndex',
        'pfieldIndexToValue',
        'originalBody',
        'includes',
        'parsedCode',
        '_controlsDefaultValues',
        '_controlsNameToIndex',
        '_preschedCallback',
        '_argToAlias',
        '_preprocessedBody',
        '_defaultPfieldValues'
    )

    def __init__(self,
                 name: str,
                 body: str,
                 args: dict[str, float | str] | None = None,
                 init: str = '',
                 numchans: int = 1,
                 preschedCallback=None,
                 doc: str = '',
                 includes: list[str] | None = None,
                 aliases: dict[str, str] = None,
                 maxNamedArgs: int = 0,
                 useDynamicPfields: bool = None
                 ) -> None:

        assert isinstance(name, str)

        if errmsg := _checkInstr(body):
            raise CsoundError(errmsg)

        if useDynamicPfields is None:
            useDynamicPfields = config['dynamic_pfields']

        self.originalBody = body
        "original body of the instr (prior to any code generation)"

        self._controlsDefaultValues: list[float]
        self._controlsNameToIndex: dict[str, int]

        inlineargs = instrtools.parseInlineArgs(body)

        if inlineargs is not None:
            body = inlineargs.body
            args = inlineargs.args | args if args else inlineargs.args

        # At the moment we do not support mixing styles: either args passed to Instr, inline
        # args or pfields declared at the csound body

        parsedInstr = csoundlib.instrParseBody(body)

        if args:
            if useDynamicPfields:
                pfields = args
                controls = {}
            else:
                pfields = {k: v for k, v in args.items() if k.startswith('i')}
                controls = {k: v for k, v in args.items() if k.startswith('k')}
                if any(isinstance(value, str) for value in controls.values()):
                    raise ValueError(f"Dynamic controls do not accept string values, got {controls}")

            if parsedInstr.pfieldsUsed:
                minpfield = max(5, max(parsedInstr.pfieldsUsed))
            else:
                minpfield = 5

            if controls:
                self._controlsNameToIndex = {key: idx for idx, key in enumerate(controls.keys())}
                self._controlsDefaultValues = list(controls.values())

                if 0 < maxNamedArgs < len(controls):
                    raise ValueError(f"Too many named args, the maximum is {maxNamedArgs}, "
                                     f"got {controls}")
            else:
                self._controlsNameToIndex = EMPTYDICT
                self._controlsDefaultValues = []

            pargNames = list(pfields.keys())
            pargIndexes = instrtools.assignPfields(pargNames, exclude=(4,), minpfield=minpfield)
            pargsNameToIndex = dict(zip(pargNames, pargIndexes))
            pargsIndexToName = dict(zip(pargIndexes, pargNames))
            pargsIndexToValue = {pargsNameToIndex[pname]: value for pname, value in pfields.items()}

        else:
            pargsIndexToName = parsedInstr.pfieldIndexToName
            pargsIndexToValue = parsedInstr.pfieldIndexToValue or {}  # type: ignore
            pfields = parsedInstr.pfieldNameToValue

            pargsIndexToName.pop(4, None)
            pargsIndexToValue.pop(4, None)
            pfields.pop('p4', None)

            pargsNameToIndex = {pname: idx for idx, pname in pargsIndexToName.items()}
            controls = EMPTYDICT

        self.parsedCode = parsedInstr

        self.controls: dict[str, float] = controls
        "Named controls, mapping name to default value."

        self.name: str = name
        "Name of this instrument"

        self._preprocessedBody: str = textwrap.dedent(body)
        "Body after processing inline args"

        self.pfields: dict[str, float] = pfields
        """Dict mapping pfield name to default value
        
        pfield index is assigned by order, starting with p5"""

        self.init: str = init if init is not None else ''
        """Code to be initialized at the instr0 level, excluding include files"""

        self.includes: list[str] | None = includes
        """List of included files, or None"""

        self.numchans = numchans
        "Number of audio outputs of this instr"

        self.doc = doc
        "Description of this instr (optional)"

        self.id: int = self._calculateHash()
        "Unique numeric id of this instr"

        self.pfieldIndexToName: dict[int, str] = pargsIndexToName
        "Dict mapping pfield index to its name"

        self.pfieldNameToIndex: dict[str, int] = pargsNameToIndex
        "Dict mapping pfield name to its index"

        self.pfieldIndexToValue: dict[int, float] = pargsIndexToValue
        "Dict mapping pfield index to its default value"

        self.aliases = aliases or EMPTYDICT
        """Maps alias argument names to their real argument names
        
        Aliased parameters can be pfields or named controls"""

        self._argToAlias = {name: alias for alias, name in aliases.items()} if aliases else EMPTYDICT
        self._preschedCallback = preschedCallback
        self._defaultPfieldValues = list(self.pfields.values())

    def register(self, renderer: AbstractRenderer) -> None:
        """
        Register this Instr at the given session

        This is just a shortcut for ``session.register(instr)``

        Args:
            renderer: the renderer to register this Instr at


        Example
        ~~~~~~~

            >>> from csoundengine import *
            >>> s = Engine().session()
            >>> Instr('myinstr', ...).register(s)

        This is equivalent to

            >>> s.defInstr('myinstr', ...)
        """
        renderer.registerInstr(self)

    def _calculateHash(self) -> int:
        argshash = hash(frozenset(self.pfields.items())) if self.pfields else 0
        tabhash = hash(frozenset(self.controls.items())) if self.controls else 0
        return hash((self.name, self.originalBody, self.init, self.doc, self.numchans,
                     argshash, tabhash))

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, other) -> bool:
        if not isinstance(other, Instr):
            return NotImplemented
        return self.id == other.id

    def __repr__(self) -> str:
        parts = [self.name]
        if s := self._pfieldsRepr():
            parts.append(s)
        if self.controls:
            parts.append(f"tabargs={self.controls}")

        return f"Instr({', '.join(parts)})"

    def generateBody(self, renderer: AbstractRenderer = None) -> str:
        """
        Generate the actual body of this instrument

        An Instr can generate different csound code depending on the renderer.

        Args:
            renderer: the renderer for which to generate the body. If not given
                the code generated for a live session is returned

        Returns:
            the actual csound code to be used as the body of this instrument

        .. seealso:: :meth:`csoundengine.session.Session.defaultInstrBody`
        """
        if renderer:
            return renderer.generateInstrBody(self)
        from csoundengine import session
        return session.Session.defaultInstrBody(self)

    def _pfieldsRepr(self) -> str:
        pargs = self.pfieldIndexToName
        if not pargs:
            return ""
        if self.pfieldIndexToValue:
            parts = []
            for i, pname in sorted(pargs.items()):
                if i == 4:
                    continue
                if self.aliases and (alias := self._argToAlias.get(pname)) is not None:
                    pname = f"{alias}({pname})"
                parts.append(f"{pname}:{i}={self.pfieldIndexToValue.get(i, 0):.6g}")
            return ", ".join(parts)
        else:
            return ", ".join(
                    f"{pname}:{i}" for i, pname in sorted(pargs.items()) if i != 4)

    def _repr_html_(self) -> str:
        style = jupytertools.defaultPalette
        parts = [f'Instr <strong style="color:{style["name.color"]}">{self.name}</strong><br>']
        _ = jupytertools.htmlSpan
        headerfontsize = '90%'
        if self.pfields:
            indexes = [self.pfieldIndex(name) for name in self.pfields.keys()]
            indexes.sort()
            if 4 in indexes:
                indexes.remove(4)
            groups = iterlib.iterchunks(indexes, 5)
            for group in groups:
                htmls = []
                for idx in group:
                    pname = self.pfieldName(idx)
                    if self.aliases and (alias := self._argToAlias.get(pname)):
                        pname = f'{alias}({pname})'
                    # parg = _(f'p{idx}', fontsize='90%')
                    parg = f'p{idx}'
                    if pname:
                        pnamehtml = f"<b>{pname}</b>:{parg}"
                    else:
                        pnamehtml = parg
                    html = f"{pnamehtml}=<code>{self.pfieldIndexToValue.get(idx, 0):.6g}</code>"
                    html = _(html, fontsize='90%')
                    htmls.append(html)
                line = "&nbsp&nbsp&nbsp&nbsp" + ", ".join(htmls) + "<br>"
                parts.append(line)
        if self.controls:
            controlstrs = ', '.join(f'<b>{k}</b> = <code>{v}</code>' for k, v in self.controls.items())
            s = _(f'&nbsp&nbsp&nbsp&nbspControls: {controlstrs}', fontsize=headerfontsize)
            parts.append(s)
            if self.aliases:
                aliases = [f'<b>{alias}</b> → <i>{orig}</i>'
                           for alias, orig in self.aliases.items()
                           if orig in self.controls]
                parts.append(_(f"<br>&nbsp&nbsp&nbsp&nbspAliases: {', '.join(aliases)}", fontsize=headerfontsize))
        if config['jupyter_instr_repr_show_code']:
            parts.append('<hr style="width:38%;text-align:left;margin-left:0">')
            htmlorc = _(csoundlib.highlightCsoundOrc(self._preprocessedBody), fontsize=headerfontsize)
            parts.append(htmlorc)
        return "\n".join(parts)

    def dump(self) -> str:
        """
        Returns a string with the generated code of this Instr
        """
        header = f"Instr(name='{self.name}')"
        sections = ["", header]
        pargsStr = self._pfieldsRepr()
        if pargsStr:
            sections.append(pargsStr)
        if self.doc:
            sections.append(f"> doc: {self.doc}")
        if self.init:
            sections.append("> init")
            sections.append(str(self.init))
        if self.controls:
            sections.append("> table")
            sections.append(f"    {self.controls}")
        sections.append("> body")
        sections.append(self._preprocessedBody)
        return "\n".join(sections)

    def unaliasParam(self, param: str, default='') -> str:
        """
        Return the original name for parameter, if exists

        Example
        ~~~~~~~

            >>> instr = Instr('foo', r'''
            ... |kfreq=1000|
            ... ''', aliases={'frequency': 'kfreq'})
            >>> instr.unaliasParam('frequency')
            'kfreq'
        """
        if not self.aliases:
            return default
        orig = self.aliases.get(param)
        return orig if orig is not None else default

    @cache
    def controlNames(self, aliases=True, aliased=False) -> frozenset[str]:
        """
        Set of names of the controls in this instr

        Returns an empty set if this instr has no controls.
        """
        if not self.controls:
            return EMPTYSET
        names = set(self.controls.keys())
        if aliases and self.aliases:
            for name in self.controls.keys():
                if alias := self._argToAlias.get(name):
                    names.add(alias)
                    if not aliased:
                        names.remove(name)
        return frozenset(names)

    @cache
    def dynamicParamNames(self, aliases=True, aliased=False
                          ) -> frozenset[str]:
        """
        Set of all dynamic parameters accepted by this Instr

        Args:
            aliases: include aliases
            aliased: include parameters which have an alias (implies aliases)

        Returns:
            a set of the dynamic (modifiable) parameters accepted by this Instr
        """
        dynparams = self.dynamicParams(aliases=aliases, aliased=aliased)
        return frozenset(dynparams.keys())

    @cache
    def dynamicPfields(self, aliases=True, aliased=False) -> dict[str, float]:
        """
        The dynamic pfields in this instr

        A dynamic pfield is a pfield assigned to a k-variable. Such
        pfields can be modified via .set using the pwrite opcode

        Args:
            aliases: include aliases
            aliased: include parameters which have an alias (implies aliases)

        Returns:
            a dict mapping pfield name to default value.
        """
        if not self.pfields:
            return EMPTYDICT

        pfields = {name: value for name, value in self.pfields.items()
                   if name.startswith('k')}

        if aliases and self.aliases:
            for alias, realname in self.aliases.items():
                if realname in pfields:
                    pfields[alias] = pfields[realname]
                    if not aliased:
                        pfields.pop(realname, None)

        return pfields

    @cache
    def dynamicPfieldNames(self) -> frozenset[str]:
        """
        Set of dynamic pfields defined in this instr

        Dynamic pfields are pfields which have been assigned
        to a k-variable

        If this instr defines aliases for any of the dynamic
        pfields, these aliases will be included in the returned
        set

        If this instrument does not have any dynamic pfields an
        empty set will be returned. In general the returned
        set should be considered immutable
        """
        if not self.pfieldNameToIndex:
            return EMPTYSET
        pfields = [param for param in self.pfieldNameToIndex.keys()
                   if param.startswith('k')]
        if not pfields:
            return EMPTYSET

        if self.aliases:
            pfields.extend([alias for pfield in pfields
                           if (alias := self._argToAlias.get(pfield))])

        return frozenset(pfields)

    @cache
    def dynamicParams(self, aliases=True, aliased=False
                      ) -> dict[str, float]:
        """
        A dict with all dynamic parameters defined in this instr

        Dynamic parameters are not only all defined controls, but also
        any pfield assigned to a k-variable. They include aliases to
        any dynamic parameter.

        Args:
            aliases: include aliases
            aliased: include parameters which have an alias (implies aliases)

        Returns:
            a dict with all dynamic parameters and their default values. Returns an empty
            dict if this instr has no dynamic parameters.
        """
        if not self.pfields and not self.controls:
            return EMPTYDICT

        params = self.dynamicPfields(aliases=False)
        if self.controls:
            params = params | self.controls

        if aliases and self.aliases:
            params = params.copy()
            for alias, realname in self.aliases.items():
                params[alias] = params[realname]
                if not aliased:
                    del params[realname]
        return params

    def paramNames(self, aliases=True, aliased=False
                   ) -> frozenset[str]:
        """
        All parameter names
        """
        pfields = self.pfieldNames(aliases=aliases, aliased=aliased)
        return frozenset(pfields | self.controlNames()) if self.controls else pfields

    # @cache
    def pfieldNames(self, aliases=True, aliased=False
                    ) -> frozenset[str]:
        """
        The set of named pfields declared in this instrument

        Args:
            aliases: include aliases
            aliased: include parameters which have an alias (implies aliases)
        Returns:
             a set with the named pfields defined in this instr
        """
        if not self.pfieldNameToIndex:
            return EMPTYSET

        pfields = set(self.pfieldNameToIndex.keys())
        if aliases and self.aliases:
            for alias, realname in self.aliases.items():
                if realname in pfields:
                    pfields.add(alias)
                    if not aliased:
                        pfields.remove(realname)
        return frozenset(pfields)

    def paramValue(self, param: str) -> float | str | None:
        param2 = self.unaliasParam(param, param)
        defaults = self.paramDefaultValues(aliased=True)
        if param2 not in defaults:
            raise KeyError(f"Unknown parameter '{param}'. "
                           f"Possible parameters: {default.keys()}")
        return defaults[param2]

    @cache
    def paramDefaultValues(self, aliases=True, aliased=False) -> dict[str, float]:
        """
        A dict mapping named parameters to their default values
        
        Named parameters are any named pfields or controls. Also anonymous
        pfields which have an assigned default value via the 'pset' opcode
        will be included here

        Args:
            aliases: included aliases
            aliased: include parameters which have an alias

        Returns:
            a dict of named dynamic parameters to this instr and their associated
            default values

        Example
        ~~~~~~~

            >>> from csoundengine import *
            >>> s = Engine().session()
            >>> s.defInstr('test', r'''
            ... |kfreq=1000|
            ... pset 0, 0, 0, 0, 0.1, 0.5
            ... iamp = p5
            ... outch 1, oscili:a(iamp, kfreq * p6)
            ... ''')
        """
        params = {}
        if self.controls:
            params.update(self.controls)

        if self.pfieldNameToIndex:
            namedPfields = {name: self.pfieldIndexToValue.get(idx, 0.)
                            for name, idx in self.pfieldNameToIndex.items()}
            params.update(namedPfields)

        if self.aliases and aliases:
            for alias, realname in self.aliases.items():
                params[alias] = params[realname]
                if not aliased:
                    del params[realname]

        return params

    def distributeNamedParams(self, params: dict[str, float | str]
                              ) -> tuple[dict[str | int, float | str], dict[str, float]]:
        """
        Sorts params into pfields and dynamic controls

        Args:
            params: a dict mapping name to value given

        Returns:
            a tuple (pfields, dynargs) where each is a dict mapping the
            parameter to its given value
        """
        return instrtools.distributeParams(params=params,
                                           pfieldNames=self.pfieldNames(aliases=True, aliased=True),
                                           controlNames=self.controlNames(aliases=True, aliased=True))

    def pfieldName(self, index: int, alias=True) -> str:
        """
        Given the pfield index, get the name, if given

        Args:
            index: the pfield index (starts with 1)
            alias: if True, return the corresponding alias, if defined

        Returns:
            the corresponding pfield name, or an empty string if the
            index does not have an associated name
        """
        name = self.pfieldIndexToName.get(index)
        if name is None:
            logger.debug(f"Arg index {index} not used for instr {self.name}")
            return ''
        if alias and self.aliases and (name2 := self._argToAlias.get(name)):
            return name2
        return name

    # def pfieldsRegistry(self) -> dict[int, tuple[str|int, float|str]]:
    #     """
    #     dict mapping pfield index to (pfieldname: str, defaultvalue: float | str)
    #     """
    #     out = {idx: (self.pfieldName(idx), value) for idx, value in self.pfieldIndexToValue.items()}
    #     assert all(idx in out for idx in self.pfieldNameToIndex.values())
    #     return out

    @cache
    def numPfields(self) -> int:
        """
        The number of pfields in this instrument, starting with p5
        """
        n2i = self.pfieldNameToIndex
        maxkwindex = max(n2i.values())
        maxidx = max(self.pfieldIndexToValue.keys())
        maxpargs = max(maxkwindex, maxidx)
        return maxpargs - 4

    def pfieldIndex(self, name: str, default: int | None = None) -> int:
        """
        Pfield index corresponding to the given name.

        Args:
            name: the index or the name of the p-field.
            default: if the name is not known and *default* is not None, this
                value is returned as the index to indicate that the parg was not
                found (instead of raising an Exception)

        Returns:
            the index of the parg
        """
        if name[0] == 'p' and name[1:].isdigit():
            return int(name[1:])

        if self.aliases and name in self.aliases:
            name = self.aliases[name]

        if (idx := self.pfieldNameToIndex.get(name)) is not None:
            return idx
        elif default is not None:
            return default

        errormsg = (f"pfield '{name}' not known for instr '{self.name}'."
                    f"Defined named pargs: {self.pfieldNameToIndex.keys()}")
        if self.aliases:
            errormsg += f" Aliases: {self.aliases}"
        raise KeyError(errormsg)

    def parseSchedArgs(self,
                       args: list[float | str] | dict[str, float | str],
                       kws: dict[str, float | str],
                       ) -> tuple[list[float|str], dict[str, float]]:
        """
        Parse the arguments passed to sched

        Args:
            args: a list of values (starting with p5) or a dict mapping named
                param to value
            kws: a dict mapping named param to value

        Returns:
            a tuple (pfields5, dynargs), where pfields5 is a list of pfield
            values starting at p5 and dynargs is a dict of dynamic
            parameters mapping parameter name to the given value
        """
        if args is None:
            args = []

        if isinstance(args, list):
            # All pfields, starting with p5
            if not kws:
                if len(args) >= self.maxPfieldIndex() - 4:
                    pfields = args
                else:
                    defaultPfields = self.defaultPfieldValues()
                    pfields = args + defaultPfields[len(args):]
                dynargs = EMPTYDICT
            else:
                namedpfields, dynargs = self.distributeNamedParams(kws)
                pfields = self.pfieldsTranslate(args=args, kws=namedpfields)

        elif isinstance(args, dict):
            namedpfields, dynargs = self.distributeNamedParams(args)
            pfields = self.pfieldsTranslate(kws=namedpfields)
            if kws:
                dynargs = dynargs | kws
            return pfields, dynargs

        else:
            raise TypeError(f"args should be a list or a dict, got {args}")
        pfields = [p if isinstance(p, str) else float(p) for p in pfields]
        return pfields, dynargs

    @cache
    def maxPfieldIndex(self) -> int:
        n2i = self.pfieldNameToIndex
        maxkwindex = max(n2i.values()) if n2i else 3
        if self.pfieldIndexToValue:
            maxidx = max(self.pfieldIndexToValue.keys())
            return max(maxidx, maxkwindex)
        else:
            return maxkwindex

    def pfieldDefaultValue(self, pfield: str | int) -> float | str | None:
        """
        Returns the default value of a pfield

        Args:
            pfield: the name / index of the pfield

        Returns:
            the default value. Will raise an exception if the pfield is
            not known. Returns None if the pfield is known but it was
            declared without default
        """
        if isinstance(pfield, int):
            idx = pfield
        else:
            if self.aliases:
                pfield = self.aliases.get(pfield, pfield)
            idx = self.pfieldNameToIndex.get(pfield)
            if idx is None:
                raise ValueError(f"Pfield '{pfield}' not known. Named pfields: {self.pfieldNames()}")
        return self.pfieldIndexToValue.get(idx)

    def defaultPfieldValues(self) -> list[float | str]:
        """
        The default pfield values, starting with p5
        """
        return self._defaultPfieldValues

    def pfieldsTranslate(self,
                         args: Sequence[float|str] = (),
                         kws: dict[str | int, float] | None = None
                         ) -> list[float|str]:
        """
        Given pfields as values and keyword arguments, generate a list of
        values which can be passed to sched, starting with p5
        (p4 is reserved)

        Args:
            args: pfield values, starting with p5
            kws: named pfields (a name can also be 'p8' for example)

        Returns:
            a list of float values with 0 representing absent pfields

        """
        assert isinstance(args, (list, tuple))
        assert not kws or isinstance(kws, dict)
        maxidx = self.maxPfieldIndex() - 5
        if kws:
            kwsindexes = [k if isinstance(k, int) else self.pfieldIndex(k) for k in kws]
            maxidx = max(maxidx, max(kwsindexes) - 5)

        numpfields = maxidx + 1
        if not args:
            defaultvals = self.defaultPfieldValues()
            if len(defaultvals) >= numpfields:
                pargs = defaultvals.copy()
            else:
                pargs = defaultvals + [0.] * (numpfields - len(defaultvals))

        elif maxidx >= len(args):
            pargs = list(args)
            pargs.extend([0.] * (numpfields - len(args)))
            if self.pfieldIndexToValue:
                for i, v in self.pfieldIndexToValue.items():
                    pargsidx = i - 5
                    # TODO: also check for NAN
                    if pargsidx > len(args) - 1:
                        pargs[pargsidx] = v
        else:
            pargs = args if isinstance(args, list) else list(args)

        if kws:
            for idx, value in zip(kwsindexes, kws.values()):
                pargs[idx-5] = value
        return pargs

    def rec(self,
            dur: float,
            outfile: str | None = None,
            args: list[float] | dict[str, float] | None = None,
            sr: int | None = None,
            ksmps: int | None = None,
            encoding: str | None = None,
            nchnls: int = 2,
            wait=True,
            a4: int | None = None,
            delay=0.,
            **kws
            ) -> str:
        """
        Record this Instr for a given duration

        Args:
            dur: the duration of the recording
            outfile: if given, the path to the generated output.
                If not given, a temporary file will be generated.
            args: the arguments passed to the instrument (if any),
                beginning with p5 or a dict with named arguments
            sr: the sample rate -> config['rec_sr']
            ksmps: the number of samples per cycle -> config['rec_ksmps']
            encoding: the sample encoding of the rendered file, given as
                'pcmXX' or 'floatXX', where XX represent the bit-depth
                ('pcm16', 'float32', etc). If no encoding is given a suitable default for
                the sample format is chosen
            nchnls: the number of channels of the generated output.
            wait: if True, the function blocks until done, otherwise rendering
                is asynchronous
            a4: the frequency of A4 (see config['A4']
            kws: any keyword will be interpreted as a named argument of this Instr
            delay: when to schedule the instr

        Returns:
            the path of the generated soundfile

        .. seealso:: :meth:`Instr.renderSamples`

        Example
        ~~~~~~~

            >>> from csoundengine import *
            >>> from sndfileio import *
            >>> s = Engine().session()
            >>> white = s.defInstr('white', r'''
            ...   |igain=0.1|
            ...   aout = gauss:a(1) * igain
            ...   outch 1, aout
            ... ''')
            >>> samples, info = sndget(white.rec(2))
            >>> info
            samplerate : 44100
            nframes    : 88192
            channels   : 2
            encoding   : float32
            fileformat : wav
            duration   : 2.000

        """
        from csoundengine.offline import Renderer
        r = Renderer(sr=sr, nchnls=nchnls, ksmps=ksmps, a4=a4)
        r.registerInstr(self)
        r.sched(instrname=self.name,
                delay=delay,
                dur=dur,
                args=args,
                **kws)
        renderjob = r.render(outfile, wait=wait, encoding=encoding)
        return renderjob.outfile

    def renderSamples(self,
                      dur,
                      args: list[float] | dict[str, float] | None = None,
                      sr: int = 44100,
                      ksmps: int | None = None,
                      nchnls: int = 2,
                      a4: int | None = None,
                      delay=0.,
                      **kws
                      ) -> np.ndarray:
        """
        Record this instrument and return the generated samples

        Args:
            dur: the duration of the recording
            args: the args passed to this instr
            sr: the samplerate of the recording
            ksmps: the samples per cycle used
            nchnls: the number of channels
            a4: the value of a4
            delay: when to schedule this instr
            kws: any keyword will be interpreted as a named argument of this Instr


        Returns:
            the generated samples as numpy array

        .. seealso:: :meth:`Instr.rec`

        Example
        ~~~~~~~

            >>> from csoundengine import *
            >>> from sndfileio import *
            >>> s = Engine().session()
            >>> white = s.defInstr('white', r'''
            ...  |igain=0.1|
            ...  aout = gauss:a(1) * igain
            ...  outch 1, aout
            ... ''')
            # Render two seconds of white noise
            >>> samples = white.renderSamples(2)
        """
        sndfile = self.rec(dur=dur, args=args, sr=sr, ksmps=ksmps, nchnls=nchnls,
                           wait=True, a4=a4, delay=delay, **kws)
        if not os.path.exists(sndfile):
            raise RuntimeError(f"Rendering error, could not find generated soundfile ('{sndfile}')")
        import sndfileio
        samples, sr = sndfileio.sndread(sndfile)
        os.remove(sndfile)
        return samples

    def hasControls(self) -> bool:
        """
        Returns True if this instrument defines a parameters table
        """
        return bool(self.controls)

    def controlIndex(self, param: str) -> int:
        """
        Returns the index of a control parameter

        Raises KeyError if the parameter given is not defined

        Args:
            param: the parameter name

        Returns:
            the corresponding slot
        """
        if not self.hasControls():
            raise KeyError(f"This instr ({self.name}) has no named controls")
        idx = self._controlsNameToIndex.get(param)
        if idx is None:
            raise KeyError(f"Parameter {param} not known for instr {self.name}."
                           f" Known parameters: {self.controls.keys()}")
        return idx

    def overrideControls(self, d: dict[str, float] | None = None, **kws
                         ) -> list[float]:
        """
        Overrides default values for the controls in this instr

        Returns the values for all the defined slots

        Args:
            d: if given, a dictionary of the form {'argname': value}.
                Alternatively key/value pairs can be passed as keywords
            **kws: each key must match a named parameter as defined in
                the tabargs attribute

        Returns:
            A list of floats holding the new initial values of the
            parameters table. The returned list should not be modified

        Example:
            instr.overrideTable(param1=value1, param2=value2)

        """
        if not self.controls:
            raise ValueError("This instrument does not define controls")

        if d is None and not kws:
            return self._controlsDefaultValues

        out = self._controlsDefaultValues.copy()
        if d:
            for key, value in d.items():
                idx = self._controlsNameToIndex[key]
                out[idx] = value
        if kws:
            for key, value in kws.items():
                idx = self._controlsNameToIndex[key]
                out[idx] = value
        return out


def _checkInstr(instr: str) -> str:
    """
    Returns an error message if the instrument is not well-defined
    """
    lines = [line for line in (line.strip() for line in instr.splitlines()) if line]
    if not lines:
        return ''

    if re.search(r'$\s*\binstr\b', lines[0]) or re.search(r'$\s*\bendin\b', lines[-1]):
        return ("instr should be the body of the instrument,"
                " without 'instr' and 'endin")

    for i, line in enumerate(lines):
        if re.search(r"\bp4\b", line):
            return (f"The instr uses p4, but p4 is reserved for the parameters table. "
                    f"Line {i}: {line}")

    return ''


# TODO: remove this
def _namedControlsGenerateCode(controls: dict) -> str:
    """
    Generates code for an instr to read named controls

    Args:
        controls: a dict mapping control name to default value. The
            keys are valid csound k-variables

    Returns:
        the generated code
    """

    lines = [fr'''
    ; --- start generated code for dynamic args
    i__slicestart__ = p4
    i__tabnum__ chnget ".dynargsTabnum"
    if i__tabnum__ == 0 then
        initerror sprintf("Session table does not exist (p1: %f)", p1)
        goto __exit
    endif
    ''']
    idx = 0
    for key, value in controls.items():
        assert key.startswith('k')
        lines.append(f"    {key} tab i__slicestart__ + {idx}, i__tabnum__")
        idx += 1
    lines.append("    ; --- end generated code\n")
    out = textlib.stripLines(textlib.joinPreservingIndentation(lines))
    return out



