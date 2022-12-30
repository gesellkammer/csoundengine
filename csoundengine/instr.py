from __future__ import annotations
from emlib import textlib, iterlib
import re

from .config import config, logger
from .errors import CsoundError
from . import csoundlib
from . import jupytertools
from typing import Sequence


__all__ = (
    'Instr',
    'parseInlineArgs'
)


class Instr:
    """
    An Instr is a template used to schedule a concrete instrument
    within a :class:`~csoundengine.session.Session` or a :class:`~csoundengine.offline.Renderer`.

    .. note::

        An Instr must be registered at the Session or the Renderer before it can be used.
        See :meth:`csoundengine.instr.Instr.register` or :meth:`csoundengine.session.Session.defInstr`

    Args:
        name: the name of the instrument
        body: the body of the instr (the text **between** 'instr' end 'endin')
        args: if given, a dictionary defining default values for pfields
        init: code to be initialized at the instr0 level
        tabargs: An instrument can have an associated table to be able to pass
            dynamic parameters which are specific to this note (for example,
            an instrument could define a filter with a dynamic cutoff freq.)
            *tabargs* is a dict of the form: {param_name: initial_value}.
            The order of appearence will correspond to the index in the table
        preschedCallback: a function ``f(synthid, args) -> args``, called before
            a note is scheduled with
            this instrument. Can be used to allocate a table or a dict and pass
            the resulting index to the instrument as parg
        freetable: if ``True``, the associated table is freed in csound when the note
            is finished
        doc: some documentation describing what this instr does

    Example
    -------

    .. code-block:: python

        s = Engine().session()
        Instr('sine', r'''
            kfreq = p5
            kamp = p6
            a0 = oscili:a(kamp, kfreq)
            outch 1, a0
        ''').register(s)
        synth = s.sched('sine', kfreq=440, kamp=0.1)
        synth.stop()

    **Default Values**

    An Instr can define default values for any of its p-fields:

    .. code-block:: python

        s = Engine().session()
        Instr('sine', r'''
            kamp = p5
            kfreq = p6
            a0 = oscili:a(kamp, kfreq)
            outch 1, a0
        ''', args={'kamp': 0.1, 'kfreq': 1000}
        ).register(s)
        # We schedule an event of sine, kamp will take the default (0.1)
        synth = s.sched('sine', kfreq=440)
        synth.stop()

    **Inline arguments**

    An inline args declaration can set both pfield name and default value:

    .. code::

        s = Engine().session()
        Instr('sine', r'''
            |kamp=0.1, kfreq=1000|
            a0 = oscili:a(kamp, kfreq)
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

    .. code::

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

        # offline events can be modified just like real-time events
        renderer.automatep(events[0], 'kmidi', pairs=[0, 60, 2, 59])
        renderer.setp(events[1], 3, 'kmidi', 67.2)
        renderer.render("out.wav")

    """

    __slots__ = (
        'body', 'name', 'args', 'init', 'id', '_tableDefaultValues', '_tableNameToIndex',
        'tabargs', 'numchans', 'instrFreesParamTable', 'doc',
        'pargsIndexToName', 'pargsNameToIndex', 'pargsIndexToDefaultValue',
        '_numpargs', '_recproc', '_check', '_preschedCallback',
        'originalBody'
    )

    def __init__(self,
                 name: str,
                 body: str,
                 args: dict[str, float] = None,
                 init: str = None,
                 tabargs: dict[str, float] = None,
                 numchans: int = 1,
                 preschedCallback=None,
                 freetable=True,
                 doc: str = '',
                 userPargsStart=5,
                 ) -> None:

        assert isinstance(name, str)

        if errmsg := _checkInstr(body):
            raise CsoundError(errmsg)

        self.originalBody = body
        "original body of the instr (prior to any code generation)"

        self._tableDefaultValues: list[float] | None = None
        self._tableNameToIndex: dict[str, int] | None = None

        delimiters, inlineargs, body = parseInlineArgs(body)

        if delimiters == '||':
            assert not args
            args = inlineargs
        elif delimiters == '{}':
            assert not tabargs
            tabargs = inlineargs

        if tabargs:
            if any(name[0] not in 'ki' for name in tabargs.keys()):
                raise ValueError("Named parameters must start with 'i' or 'k'")
            self._tableNameToIndex = {paramname: idx for idx, paramname in
                                      enumerate(tabargs.keys())}
            defaultvals = list(tabargs.values())
            minsize = config['associated_table_min_size']
            if len(defaultvals) < minsize:
                defaultvals += [0.] * (minsize - len(defaultvals))
            self._tableDefaultValues = defaultvals
            tabcode = _tabargsGenerateCode(tabargs, freetable=freetable)
            body = textlib.joinPreservingIndentation((tabcode, body))
            needsExitLabel = True
        else:
            freetable = False
            needsExitLabel = False

        if args:
            pfields = _pfieldsMergeDeclaration(args, body, startidx=userPargsStart)
            pargsIndexToName = {i: name for i, (name, default) in pfields.items()}
            pargsDefaultValues = {i: default for i, (_, default) in pfields.items()}
            body = _updatePfieldsCode(body, pargsIndexToName)
        else:
            parsed = csoundlib.instrParseBody(body)
            pargsIndexToName = parsed.pfieldsIndexToName
            pargsDefaultValues = parsed.pfieldsDefaults or {}

        if needsExitLabel:
            body = textlib.joinPreservingIndentation((body, "__exit:"))

        self.tabargs = tabargs
        "named table args"

        self.name = name
        "name of thisinstr"

        self.body = body
        "body of the instr"

        self.args = args
        """ A dict like ``{'ibus': 1, 'kfreq': 440}``, defining default values for
        pfields. The mapping between pfield name and index is done by order, 
        starting with p5"""

        self.init = init if init else None
        """code to be initialized at the instr0 level"""

        self.numchans = numchans
        "number of audio outputs of this instr"

        self.doc = doc
        "description of this instr (optional)"

        self.id: int = self._id()
        "unique numeric id of this instr"

        self.pargsIndexToName: dict[int, str] = pargsIndexToName
        "a dict mapping parg index to its name"

        self.pargsNameToIndex: dict[str, int] = {n: i for i, n in pargsIndexToName.items()}
        "a dict mapping parg name to its index"

        self.pargsIndexToDefaultValue: dict[int, float] = pargsDefaultValues
        "a dict mapping parg index to its default value"

        self.instrFreesParamTable = freetable
        "does this instr frees its parameter table?"

        self._numpargs: int | None = None
        self._recproc = None
        self._check = config['check_pargs']
        self._preschedCallback = preschedCallback

    def _id(self) -> int:
        argshash = hash(frozenset(self.args.items())) if self.args else 0
        tabhash = hash(frozenset(self.tabargs.items())) if self.tabargs else 0
        return hash((self.name, self.body, self.init, self.doc, self.numchans,
                     argshash, tabhash))

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, other: Instr) -> bool:
        if not isinstance(other, Instr):
            return NotImplemented
        return self.id == other.id

    def __repr__(self) -> str:
        parts = [self.name]
        if s := self._pargsRepr():
            parts.append(s)
        if self.tabargs:
            parts.append(f"tabargs={self.tabargs}")

        return f"Instr({', '.join(parts)})"

    def _pargsRepr(self) -> str:
        pargs = self.pargsIndexToName
        if not pargs:
            return ""
        if self.pargsIndexToDefaultValue:
            return ", ".join(f"{pname}:{i}={self.pargsIndexToDefaultValue.get(i, 0)}"
                             for i, pname in sorted(pargs.items()) if i != 4)
        else:
            return ", ".join(
                    f"{pname}:{i}" for i, pname in sorted(pargs.items()) if i != 4)

    def _repr_html_(self) -> str:
        style = jupytertools.defaultPalette
        parts = [f'Instr <strong style="color:{style["name.color"]}">{self.name}</strong><br>']
        _ = jupytertools.htmlSpan
        if self.pargsIndexToName and len(self.pargsIndexToName) > 1:
            indexes = list(self.pargsIndexToName.keys())
            indexes.sort()
            if 4 in indexes:
                indexes.remove(4)
            groups = iterlib.split_in_chunks(indexes, 5)
            for group in groups:
                htmls = []
                for idx in group:
                    pname = self.pargsIndexToName[idx]
                    # parg = _(f'p{idx}', fontsize='90%')
                    parg = f'p{idx}'
                    html = f"<b>{pname}</b>:{parg}=" \
                           f"<code>{self.pargsIndexToDefaultValue.get(idx, 0)}</code>"
                    html = _(html, fontsize='90%')
                    htmls.append(html)
                line = "&nbsp&nbsp&nbsp&nbsp" + ", ".join(htmls) + "<br>"
                parts.append(line)
        if self.tabargs:
            parts.append(f'&nbsp&nbsp&nbsp&nbsptabargs = <code>{self.tabargs}</code>')
        if config['jupyter_instr_repr_show_code']:
            parts.append('<hr style="width:38%;text-align:left;margin-left:0">')
            htmlorc = _(csoundlib.highlightCsoundOrc(self.body), fontsize='90%')
            parts.append(htmlorc)
        return "\n".join(parts)

    def paramMode(self) -> str | None:
        """
        Returns the dynamic parameter mode, or None

        Returns one of 'parg', 'table' or None if this object does not
        define any dynamic parameters
        """
        hasDynamicPargs = any(name.startswith('k') for name in self.pargsNameToIndex.keys())
        return 'table' if self.hasParamTable() else 'parg' if hasDynamicPargs else None

    def dump(self) -> str:
        """
        Returns a string with the generated code of this Instr
        """
        header = f"Instr(name='{self.name}')"
        sections = ["", header]
        pargsStr = self._pargsRepr()
        if pargsStr:
            sections.append(pargsStr)
        if self.doc:
            sections.append(f"> doc: {self.doc}")
        if self.init:
            sections.append("> init")
            sections.append(str(self.init))
        if self._tableDefaultValues:
            sections.append("> table")
            sections.append(f"    {self.tabargs}")
        sections.append("> body")
        sections.append(self.body)
        return "\n".join(sections)

    def namedParams(self) -> dict[str, float]:
        """
        Returns named dynamic parameters and their defaults

        This method is independent of the parameter mode used (whether a param table or
        named pargs)

        Returns:
            a dict of named dynamic parameters to this instr and their associated
            default values
        """
        paramMode = self.paramMode()
        if paramMode == 'table':
            return self.tabargs
        elif paramMode == 'parg':
            return {key: self.pargsIndexToDefaultValue.get(idx)
                    for key, idx in self.pargsNameToIndex.items()}
        return {}

    def register(self, renderer) -> Instr:
        """
        Register this Instr with a Session or an offline renderer. This is the
        same as session.registerInstr(csoundinstr)

        Args:
            renderer: the name of a Session as str, the Session itself or
                an offline Renderer

        Returns:
            self. This enables a declaration like: ``instr = Instr(...).register(session)``

        Example
        =======

        .. code::

            # Create an instrument and register it at a Session and
            # at an offline Renderer
            from csoundengine import *
            session = Engine().session()
            renderer = Renderer(sr=44100)

            synth = Instr('synth', r'''
                |kmidi=60|
                outch 1, oscili:a(0.1, mtof:k(kmidi))
            ''')

            synth.register(session)
            synth.register(renderer)
        """
        if isinstance(renderer, str):
            from .engine import getEngine
            e = getEngine(renderer)
            if e is None:
                raise KeyError(f"Engine {renderer} does not exists")
            session = e.session()
            session.registerInstr(self)
        else:
            renderer.registerInstr(self)
        return self

    def pargIndex(self, parg: str) -> int:
        """
        Helper function, returns the index corresponding to the given parg.

        Args:
            parg: the index or the name of the p-field.

        Returns:
            the index of the parg
        """
        assert isinstance(parg, str)
        if (idx := self.pargsNameToIndex.get(parg)) is None:
            raise KeyError(f"parg {parg} not known. Defined named pargs: {self.pargsNameToIndex.keys()}")
        return idx

    def pargsTranslate(self, 
                       args: Sequence[float] = (), 
                       kws: dict[str | int, float] = None
                       ) -> list[float]:
        """
        Given pargs as values and keyword arguments, generate a list of
        values which can be passed to sched, starting with p5
        (p4 is reserved)

        Args:
            args: parg values, starting with p5
            kws: named pargs (a name can also be 'p8' for example)

        Returns:
            a list of float values with 0 representing absent pargs

        """
        firstp = 4  # 4=p5
        n2i = self.pargsNameToIndex
        maxkwindex = max(n2i.values())
        maxpargs = max(maxkwindex, len(args)-1)
        pargs = [0.]*(maxpargs-firstp)
        if self.pargsIndexToDefaultValue:
            for i, v in self.pargsIndexToDefaultValue.items():
                pargs[i-firstp-1] = v
        if args:
            pargs[:len(args)] = args
        if kws:
            for pname, value in kws.items():
                idx = pname if isinstance(pname, int) else _pargIndex(pname, n2i)
                pargs[idx-5] = value
        return pargs

    def asOrc(self, instrid, sr: int = None, ksmps: int = None, nchnls=2,
              a4: int = None) -> str:
        """
        Generate a csound orchestra with only this instrument defined

        Args:
            instrid: the id (instr number of name) used for this instrument
            sr: samplerate
            ksmps: ksmps
            nchnls: number of channels
            a4: freq of A4

        Returns:
            The generated csound orchestra
        """
        sr = sr or config['rec_sr']
        ksmps = ksmps or config['ksmps']
        a4 = a4 if a4 is not None else config['A4']
        if self.init is None:
            initstr = ""
        else:
            initstr = self.init
        orc = f"""
        sr = {sr}
        ksmps = {ksmps}
        nchnls = {nchnls}
        0dbfs = 1.0
        A4 = {a4}

        {initstr}

        instr {instrid}

        {self.body}

        endin

        """
        return orc

    def _numargs(self) -> int:
        if self._numpargs is None:
            self._numpargs = csoundlib.instrParseBody(self.body).numPfields()
        return self._numpargs

    def _checkArgs(self, args) -> bool:
        lenargs = 0 if args is None else len(args)
        numargs = self._numargs()
        ok = numargs == lenargs
        if not ok:
            msg = f"expected {numargs} args, got {lenargs}"
            logger.error(msg)
        return ok

    def rec(self, dur, outfile: str = None, args: list[float] = None,
            sr: int = None, ksmps: int = None, samplefmt=None, nchnls: int = 2,
            block=True, a4: int = None) -> str:
        """
        Record this Instr for a given duration

        Args:
            dur: the duration of the recording
            outfile: if given, the path to the generated output.
                If not given, a temporary file will be generated.
            args: the data. of pargs passed to the instrument (if any),
                beginning with p4
            sr: the sample rate -> config['rec_sr']
            ksmps: the number of samples per cycle -> config['rec_ksmps']
            samplefmt: one of 16, 24, 32, or 'float' -> config['rec_sample_format']
            nchnls: the number of channels of the generated output.
            block: if True, the function blocks until done, otherwise rendering
                is asynchronous
            a4: the frequency of A4 (see config['A4']

        See Also:
            :meth:`~Instr.recEvents`
        """
        event = [0., dur]
        if args:
            event.extend(args)
        return self.recEvents(events=[event], outfile=outfile, sr=sr,
                              ksmps=ksmps, samplefmt=samplefmt, nchnls=nchnls,
                              block=block, a4=a4)

    def recEvents(self, events: list[list[float]], outfile: str = None,
                  sr=44100, ksmps=64, samplefmt='float', nchnls=2,
                  block=True, a4=None
                  ) -> str:
        """
        Record the given events with this instrument.

        Args:
            events: a data. of events, where each event is the list of pargs
                passed to the instrument, as [delay, dur, p4, p5, ...]
                (p1 is omitted)
            outfile: if given, the path to the generated output. If not
                given, a temporary file will be generated.
            sr: the sample rate -> config['rec_sr']
            ksmps: the number of samples per cycle -> config['rec_ksmps']
            samplefmt: one of 16, 24, 32, or 'float' -> config['rec_sample_format']
            nchnls: the number of channels of the generated output.
            a4: the frequency of A4 (see config['A4']
            block: if True, the function blocks until done, otherwise rendering
                is asynchronous

        Returns:
            the generated output (if outfile is not given, a temp file
            is created)

        See Also:
            :meth:`~Instr.rec`
        """
        a4 = a4 or config['A4']
        sr = sr or config['rec_sr']
        ksmps = ksmps or config['rec_ksmps']
        samplefmt = samplefmt or config['rec_sample_format']
        initstr = self.init or ""
        a4 = a4 or config['A4']
        outfile, popen = csoundlib.recInstr(body=self.body,
                                            init=initstr,
                                            outfile=outfile,
                                            events=events,
                                            sr=sr,
                                            ksmps=ksmps,
                                            samplefmt=samplefmt,
                                            nchnls=nchnls,
                                            a4=a4)
        if block:
            popen.wait()
        return outfile

    def hasParamTable(self) -> bool:
        """
        Returns True if this instrument defines a parameters table
        """
        return self._tableDefaultValues is not None and len(self._tableDefaultValues) > 0

    def paramTableParamIndex(self, param: str) -> int:
        """
        Returns the index of a parameter name

        Returns -1 if the parameter was not found in the table definition
        Raises RuntimeError if this Instr does not have a parameters table
        """
        if not self.hasParamTable():
            raise RuntimeError(f"This instr ({self.name}) does not have a parameters table")
        idx = self._tableNameToIndex.get(param)
        if idx is None:
            logger.warning(f"Parameter {param} not known for instr {self.name}."
                           f" Known parameters: {self._tableNameToIndex.keys()}")
            return -1
        return idx

    def overrideTable(self, d: dict[str, float] = None, **kws) -> list[float]:
        """
        Overrides default values in the params table
        Returns the initial values

        Args:
            d: if given, a dictionary of the form {'argname': value}.
                Alternatively key/value pairs can be passed as keywords
            **kws: each key must match a named parameter as defined in
                the tabargs attribute

        Returns:
            A list of floats holding the new initial values of the
            parameters table

        Example:
            instr.overrideTable(param1=value1, param2=value2)

        """
        if self._tableDefaultValues is None:
            raise ValueError("This instrument has no associated table")
        if self._tableNameToIndex is None:
            raise ValueError("This instrument has no table mapping, so"
                             "named parameters can't be used")
        if d is None and not kws:
            return self._tableDefaultValues
        out = self._tableDefaultValues.copy()
        if d:
            for key, value in d.items():
                idx = self._tableNameToIndex[key]
                out[idx] = value
        if kws:
            for key, value in kws.items():
                idx = self._tableNameToIndex[key]
                out[idx] = value
        return out


def _checkInstr(instr: str) -> str:
    """
    Returns an error message if the instrument is not well defined
    """
    lines = [line for line in (line.strip() for line in instr.splitlines()) if line]
    errmsg = ""
    if not lines:
        return errmsg

    if "instr" in lines[0] or "endin" in lines[-1]:
        errmsg = ("instr should be the body of the instrument,"
                  " without 'instr' and 'endin")
    for i, line in enumerate(lines):
        if re.search(r"\bp4\b", line):
            errmsg = (f"The instr uses p4, but p4 is reserved for the parameters table. "
                      f"Line {i}: {line}")
            break
    return errmsg


def parseInlineArgs(body: str | list[str]
                    ) -> tuple[str, dict[str, float], str]:
    """
    Parse an instr body with a possible args declaration (see below).

    Args:
        body: the body of the instrument as a string or as a list of lines

    Returns:
        a tuple (delimiters, fields, body without fields declaration)

        Where:

        * delimiters: the string "||" or "{}", identifying the kind of inline declaration, or
            the empty string if the body does not have any inline args
        * fields: a dictionary mapping field name to default value
        * body without declaration: the body of the instrument, as a string, without
          the line declaring fields.

    .. note::
        this is not supported csound syntax, we added this ad-hoc syntax
        extension to more easily declare named pfields. In the future a possible
        solution might be an opcode:

            kamp, kfreq, kcutoff pfields 4, 0.1, 440, 2000

    Example
    =======

        >>> body = '''
        ... |ichan, kamp=0.1, kfreq=440|
        ... a0 oscili kamp, kfreq
        ... outch ichan, a0
        ... '''
        >>> delimiters, args, bodyWithoutArgs = parseInlineArgs(body)
        >>> delimiters
        '||'
        >>> args
        {'ichan': 0, 'kamp': 0.1, 'kfreq': 440}
        >>> print(bodyWithoutArgs)
        a0 oscili kamp, kfreq
        outch 1, a0
    """
    lines = body if isinstance(body, list) else body.splitlines()
    delimiters, linenum = _detectInlineArgs(lines)
    if not delimiters:
        if isinstance(body, list):
            body = "\n".join(body)
        return "", {}, body
    assert linenum is not None
    pfields = {}
    line2 = lines[linenum].strip()
    parts = line2[1:-1].split(",")
    for part in parts:
        if "=" in part:
            varname, defaultval = part.split("=")
            pfields[varname.strip()] = float(defaultval)
        else:
            pfields[part.strip()] = 0
    bodyWithoutArgs = "\n".join(lines[linenum+1:])
    return delimiters, pfields, bodyWithoutArgs


def _tabargsGenerateCode(tabargs: dict, freetable=True) -> str:
    lines: list[str] = []
    idx = 0
    maxidx = len(tabargs)
    lines.append(fr'''
    ; --- start generated table code
    iparams_ = p4
    if iparams_ == 0 || ftexists:i(iparams_) == 0 then
        initerror sprintf("Params table (%d) does not exist (p1: %f)", iparams_, p1)
        goto __exit
    endif
    iparamslen_ = ftlen(iparams_)
    if iparamslen_ < {maxidx} then
        initerror sprintf("params table too small (size: %d, needed: {maxidx})", iparamslen_)
    endif''')

    for key, value in tabargs.items():
        if key[0] == 'k':
            lines.append(f"{key} tab {idx}, iparams_")
        elif key[0] == 'i':
            lines.append(f"{key} tab_i {idx}, iparams_")
        else:
            raise ValueError(f"Named parameters should begin with k or i, got {key}")
        idx += 1
    if freetable:
        lines.append("ftfree iparams_, 1")
    lines.append("; --- end generated table code\n")
    out = textlib.joinPreservingIndentation(lines)
    out = textlib.stripLines(out)
    return out


def _pfieldsMergeDeclaration(args: dict[str, float], body: str, startidx=4
                             ) -> dict[int, tuple[str, float]]:
    """
    Given a dictionary declaring pfields and their defaults,
    merge these with the pfields declared in the body, returning
    a dictionary of the form {pindex: (name, default value)}

    Args:
        args: a dict mapping pfield name to a default value. The index
            is assigned in the order of appearence, starting with `startidx`
        body: the body of the instrument (the part between instr/endin)
        startidx: the start index for the pfields declared in `args`

    Returns:
        a dict mapping pfield index to a tuple (name, default value). pfields
        without default receive a fallback value of 0.

    Example
    =======

        >>> body = '''
        ... ichan = p5
        ... ifade, icutoff passign 6
        ... '''
        >>> _pfieldsMergeDeclaration(dict(kfreq=440, ichan=2), body)
        {4: ('kfreq', 440),
         5: ('ichan', 2),
         6: ('ifade', 0),
         7: ('icutoff', 0)}
    """
    # TODO: take pset into consideration for defaults
    parsedbody = csoundlib.instrParseBody(body)
    body_i2n = parsedbody.pfieldsIndexToName
    args_i2n = {i: n for i, n in enumerate(args.keys(), start=startidx)}
    allindexes = set(body_i2n.keys())
    allindexes.update(args_i2n.keys())
    pfields: dict[int, tuple[str, float]] = {}
    for idx in allindexes:
        body_n = body_i2n.get(idx)
        args_n = args_i2n.get(idx)
        if body_n and args_n:
            raise SyntaxError(f"pfield conflict, p{idx} is defined both"
                              f"in the body of the instrument (name: {body_n})"
                              f"and as an argument (name: {args_n}")
        if body_n:
            pfields[idx] = (body_n, 0.)
        else:
            assert args_n is not None
            pfields[idx] = (args_n, args[args_n])
    return pfields


def _updatePfieldsCode(body: str, idx2name: dict[int, str]) -> str:
    parsedCode = csoundlib.instrParseBody(body)
    newPfieldCode = _pfieldsGenerateCode(idx2name)
    return textlib.joinPreservingIndentation((newPfieldCode, "", parsedCode.body))
    # return "\n".join((newPfieldCode, "", parsedCode.body))


def _pargIndex(parg: str, pargMapping: dict[str, int]) -> int:
    idx = pargMapping.get(parg)
    if idx is None:
        # try with a k-
        if parg[0] != 'k':
            idx = pargMapping.get('k' + parg)
            if idx:
                return idx
        keys = [k for k in pargMapping.keys() if not k[0] == "p"]
        raise KeyError(f"parg '{parg}' not found for instr. "
                       f"Possible pargs: {keys} (mapping: {pargMapping})")
    assert idx > 0
    return idx


def _detectInlineArgs(lines: list[str]) -> tuple[str, int | None]:
    """
    Given a list of lines of an instrument's body, detect
    if the instrument has inline args defined, and which kind

    Inline args are defined in two ways:

    * ``|ichan, kamp=0.5, kfreq=1000|``
    * ``{ichan, kamp=0.5, kfreq=1000}``

    Args:
        lines: the body of the instrument split in lines

    Returns:
        a tuple (kind of delimiters, line number), where kind of
        delimiters will be either "||" or "{}". If no inline args are found
        the tuple ("", 0) is returned

    """
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if (line[0] == "|" and line[-1] == "|") or (line[0] == "{" and line[-1] == "}"):
            return line[0]+line[-1], i
        break
    return "", None


def _pfieldsGenerateCode(pfields: dict[int, str], strmethod='strget') -> str:
    """
    Args:
        pfields: a dict mapping p-index to name
        strmethod: if 'strget', string pargs are implemented as 'Sfoo = strget(p4)',
            otherwise just 'Sfoo = p4' is generated

    Returns:
        the generated code

    Example
    =======

        >>> print(_pfieldsGenerateCode({4: 'ichan', 5:'kfreq', '6': 'Sname'}))
        ichan = p4
        kfreq = p5
        Sname = strget(p6)

    """
    pairs = list(pfields.items())
    pairs.sort()
    lines = []
    for idx, name in pairs:
        if name[0] == 'S':
            if strmethod == 'strget':
                lines.append(f"{name} strget p{idx}")
            else:
                lines.append(f"{name} = p{idx}")
        else:
            lines.append(f"{name} = p{idx}")
    return "\n".join(lines)

