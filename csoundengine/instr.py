from __future__ import annotations
from emlib import textlib
from .config import config, logger
from .errors import CsoundError
from . import csoundlib

from typing import Dict, Optional as Opt, List, Union as U, Sequence as Seq, Tuple


class Instr:
    """
    An Instr is a template used to schedule a concrete instrument
    at a :class:`~csoundengine.session.Session` or a :class:`~csoundengine.offline.Renderer`.
    It must be registered to be used.

    Args:
        name: the name of the instrument
        body: the body of the instr (the text **between** 'instr' end 'endin')
        args: if given, a dictionary defining pfields and their defaults.
        init: code to be initialized at the instr0 level
        tabledef: An instrument can have an associated table to be able to pass
            dynamic parameters which are specific to this note (for example,
            an instrument could define a filter with a dynamic cutoff freq.)
            A tabledef is a dict of the form: {param_name: initial_value}.
            The order of appearence will correspond to the index in the table
        preschedCallback:
            a function f(synthid, args) -> args, called before a note is
            scheduled with
            this instrument. Can be used to allocate a table or a dict and pass
            the resulting index to the instrument as parg
        freetable:
            if True, the associated table is freed when the note is finished
        doc: some documentation describing what this instr does


    Attributes:
        name: the name of this Instr
        body: the body of the instr (the text inside instr xxx/endin)
        args: a dict like ``{'ibus': 1, 'kfreq': 440}``, defining pfields and
            their default values. The mapping between pfield name and index
            is done by order, starting with p5.
        init: any global code needed
        tabledef: similar to args, but not using pfields but a table. In this
            case all variables are k-type and do not need the "k" prefix
        numchans: currently not used
        freetable: if True, the Instr generates code to free the param table
        doc: some text documenting the use/purpose of this Instr


    Example
    =======

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
        ''', tabledef=dict(amp=0.1, freq=1000
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

    .. code-block:: csound

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
        'body', 'name', 'args', 'init', '_tableDefaultValues', '_tableNameToIndex',
        'tabledef', 'numchans', 'mustFreeTable', 'doc',
        'pargsIndexToName', 'pargsNameToIndex', 'pargsDefaultValues',
        '_numpargs', '_recproc', '_check', '_preschedCallback',
    )

    def __init__(self,
                 name: str,
                 body: str,
                 args: Dict[str, float] = None,
                 init: str = None,
                 tabledef: Dict[str, float] = None,
                 numchans: int = 1,
                 preschedCallback=None,
                 freetable=True,
                 doc: str = ''
                 ) -> None:

        assert isinstance(name, str)

        if errmsg := _checkInstr(body):
            raise CsoundError(errmsg)

        self._tableDefaultValues: Opt[List[float]] = None
        self._tableNameToIndex: Opt[Dict[str, int]] = None

        delimiters, inline_args, body = _parse_inline_args(body)

        if delimiters == '||':
            assert not args
            args = inline_args
        elif delimiters == '{}':
            assert not tabledef
            tabledef = inline_args

        if tabledef:
            self._tableNameToIndex = {paramname:idx for idx, paramname in
                                      enumerate(tabledef.keys())}
            defaultvals = list(tabledef.values())
            minsize = config['associated_table_min_size']
            if len(defaultvals)<minsize:
                defaultvals += [0.]*(minsize-len(defaultvals))
            self._tableDefaultValues = defaultvals
            tabcode = _tabledefGenerateCode(tabledef)
            body = textlib.joinPreservingIndentation((tabcode, body))
        else:
            freetable = False

        if args:
            pfields = _pfieldsMergeDeclaration(args, body, startidx=5)
            pargsIndexToName = {i:name for i, (name, default) in pfields.items()}
            pargsDefaultValues = {i:default for i, (_, default) in pfields.items()}
            body = _updatePfieldsCode(body, pargsIndexToName)
        else:
            parsed = csoundlib.instrParseBody(body)
            pargsIndexToName = parsed.pfieldsIndexToName
            pargsDefaultValues = parsed.pfieldsDefaults or {}

        self.tabledef = tabledef
        self.name = name
        self.body = body
        self.args = args
        self.init = init if init else None
        self.numchans = numchans
        self.doc = doc
        self._numpargs: Opt[int] = None
        self._recproc = None
        self._check = config['check_pargs']
        self._preschedCallback = preschedCallback
        self.pargsIndexToName: dict[int, str] = pargsIndexToName
        self.pargsNameToIndex: dict[str, int] = {n:i for i, n in pargsIndexToName.items()}
        self.pargsDefaultValues: dict[int, float] = pargsDefaultValues
        self.mustFreeTable = freetable

    def __repr__(self) -> str:
        parts = [self.name]
        if s := self._pargsRepr():
            parts.append(s)
        if self.tabledef:
            parts.append(f"tabargs={self.tabledef}")

        return f"Instr({', '.join(parts)})"

    def _pargsRepr(self) -> str:
        pargs = self.pargsIndexToName
        if not pargs:
            return ""
        if self.pargsDefaultValues:
            return ", ".join(f"p{i}:{pname}={self.pargsDefaultValues.get(i, 0)}"
                             for i, pname in sorted(pargs.items()) if i != 4)
        else:
            return ", ".join(
                    f"p{i}: {pname}" for i, pname in sorted(pargs.items()) if i != 4)

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
            sections.append(f"    {self.tabledef}")
        sections.append("> body")
        sections.append(self.body)
        return "\n".join(sections)

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
            from .session import getSession
            session = getSession(renderer)
            assert session is not None
            session.registerInstr(self)
        else:
            renderer.registerInstr(self)
        return self

    def pargIndex(self, parg: U[int, str]) -> int:
        """
        Helper function, returns the index corresponding to the given parg.

        Args:
            parg (int|str): the index or the name of the p-field. If the
                index is given (as int), it is returned as is

        Returns:
            the index of the parg
        """
        return parg if isinstance(parg, int) else \
            _pargIndex(parg, self.pargsNameToIndex)

    def pargsTranslate(self, args: Seq[float] = (), kws: Dict[U[str, int], float] = None
                       ) -> List[float]:
        """
        Given pargs as values and keyword arguments, generate a list of
        values which can be passed to sched, starting with p5
        (p4 is reserved)

        Args:
            *args: parg values, starting with p5
            **kws: named pargs (a name can also be 'p8' for example)

        Returns:
            a list of float values with 0 representing absent pargs

        """
        firstp = 4  # 4=p5
        n2i = self.pargsNameToIndex
        maxkwindex = max(n2i.values())
        maxpargs = max(maxkwindex, len(args)-1)
        pargs = [0.]*(maxpargs-firstp)
        if self.pargsDefaultValues:
            for i, v in self.pargsDefaultValues.items():
                pargs[i-firstp-1] = v
        if args:
            pargs[:len(pargs)] = args
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
        sr = sr or config['rec.sr']
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

    def rec(self, dur, outfile: str = None, args: List[float] = None,
            sr: int = None, ksmps: int = None, samplefmt=None, nchnls: int = 2,
            block=True, a4: int = None) -> str:
        """
        Record this Instr for a given duration

        Args:
            dur: the duration of the recording
            outfile: if given, the path to the generated soundfile.
                If not given, a temporary file will be generated.
            args: the seq. of pargs passed to the instrument (if any),
                beginning with p4
            sr: the sample rate -> config['rec.sr']
            ksmps: the number of samples per cycle -> config['rec.ksmps']
            samplefmt: one of 16, 24, 32, or 'float' -> config['rec.sample_format']
            nchnls: the number of channels of the generated soundfile.
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

    def recEvents(self, events: List[List[float]], outfile: str = None,
                  sr=44100, ksmps=64, samplefmt='float', nchnls=2,
                  block=True, a4=None
                  ) -> str:
        """
        Record the given events with this instrument.

        Args:
            events: a seq. of events, where each event is the list of pargs
                passed to the instrument, as [delay, dur, p4, p5, ...]
                (p1 is omitted)
            outfile: if given, the path to the generated soundfile. If not
                given, a temporary file will be generated.
            sr: the sample rate -> config['rec.sr']
            ksmps: the number of samples per cycle -> config['rec.ksmps']
            samplefmt: one of 16, 24, 32, or 'float' -> config['rec.sample_format']
            nchnls: the number of channels of the generated soundfile.
            a4: the frequency of A4 (see config['A4']
            block: if True, the function blocks until done, otherwise rendering
                is asynchronous

        Returns:
            the generated soundfile (if outfile is not given, a temp file
            is created)

        See Also:
            :meth:`~Instr.rec`
        """
        a4 = a4 or config['A4']
        sr = sr or config['rec.sr']
        ksmps = ksmps or config['rec.ksmps']
        samplefmt = samplefmt or config['rec.sample_format']
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

    def hasExchangeTable(self) -> bool:
        """
        Returns True if this instrument defines an exchange table
        """
        return self._tableDefaultValues is not None and len(self._tableDefaultValues)>0

    def overrideTable(self, d: Dict[str, float]=None, **kws) -> List[float]:
        """
        Overrides default values in the exchange table
        Returns the initial values

        Args:
            d: if given, a dictionary of the form {'argname': value}.
                Alternatively key/value pairs can be passed as keywords
            **kws: each key must match a named parameter as defined in
                the tabledef

        Returns:
            A list of floats holding the new initial values of the
            exchange table

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
    if "instr" in lines[0] or "endin" in lines[-1]:
        errmsg = ("instr should be the body of the instrument,"
                  " without 'instr' and 'endin")
    return errmsg


def _parse_inline_args(body: U[str, list[str]]
                       ) -> tuple[str, Opt[dict[str, float]], str]:
    """
    Parse an instr body with a possible args declaration (see below).

    Args:
        body: the body of the instrument as a string or as a list of lines

    Returns:
        a tuple (delimiters, fields, body without fields declaration)

        Where:

        * delimiters: the string "||" or "{}", identifying the kind of inline declaration
        * fields: a dictionary mapping field name to default value
        * body without declaration: the body of the instrument, as a string, without
          the line declaring fields.

        In the case that the body has no inline declaration delimiters will be an empty
        string

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
        >>> delimiters, args, body2 = _parse_inline_args(body)
        >>> delimiters
        '||'
        >>> args
        {'ichan': 0, 'kamp': 0.1, 'kfreq': 440}
        >>> print(body2)
        a0 oscili kamp, kfreq
        outch 1, a0
    """
    lines = body if isinstance(body, list) else body.splitlines()
    delimiters, linenum = _detect_inline_args(lines)
    if not delimiters:
        if isinstance(body, list):
            body = "\n".join(body)
        return "", None, body
    assert linenum is not None
    pfields = {}
    line2 = lines[linenum].strip()
    parts = line2[1:-1].split(",")
    for part in parts:
        if "=" in part:
            varname, defaultval = part.split("=")
            pfields[varname.strip()] = float(defaultval)
        else:
            pfields[part] = 0
    body2 = "\n".join(lines[linenum+1:])
    return delimiters, pfields, body2

def _tabledefGenerateCode(tabledef: dict) -> str:
    lines: List[str] = []
    idx = 0
    maxidx = len(tabledef)
    lines.append(f"""
    ; --- start generated table code
    i_params = p4
    if ftexists(i_params) == 0 then
        initerror sprintf("params table (%d) does not exist", i_params)
    endif
    i__paramslen = ftlen(i_params)
    if i__paramslen < {maxidx} then
        initerror sprintf("params table is too small (size: %d, needed: {maxidx})", i__paramslen)
    endif
    """)
    for key, value in tabledef.items():
        lines.append(f"k{key} tab {idx}, i_params")
        idx += 1
    lines.append("; --- end generated table code\n")
    return textlib.joinPreservingIndentation(lines)

def _pfieldsMergeDeclaration(args: Dict[str, float], body: str, startidx=4
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
    args_i2n = {i:n for i, n in enumerate(args.keys(), start=startidx)}
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


def _pargIndex(parg: str, pargMapping:Dict[str, int]) -> int:
    idx = pargMapping.get(parg)
    if idx is None:
        # try with a k-
        if parg[0] != 'k':
            idx = pargMapping.get('k' + parg)
            if idx:
                return idx
        keys = [k for k in pargMapping.keys() if not k[0]=="p"]
        raise KeyError(f"parg '{parg}' not found. "
                       f"Possible pargs: {keys}")
    assert idx > 0
    return idx

def _detect_inline_args(lines: List[str]) -> Tuple[str, Opt[int]]:
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
        if (line[0]=="|" and line[-1]=="|") or (line[0]=="{" and line[-1]=="}"):
            return line[0]+line[-1], i
        break
    return "", None


def _pfieldsGenerateCode(pfields: dict[int, str]) -> str:
    """
    Args:
        pfields: a dict mapping p-index to name

    Returns:
        the generated code

    Example
    =======

        >>> print(_pfieldsGenerateCode({4: 'ichan', 5:'kfreq'}))
        ichan = p4
        kfreq = p5

    """
    pairs = list(pfields.items())
    pairs.sort()
    lines = [f"{name} = p{idx}" for idx, name in pairs]
    return "\n".join(lines)