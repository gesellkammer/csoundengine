from __future__ import annotations

import functools
import re
import dataclasses
from . import internal
from._common import EMPTYDICT
import numpy as np


import typing as _t


@dataclasses.dataclass
class ParsedBlock:
    """
    A ParsedBlock represents a block (an instr, an opcode, etc) in an orchestra

    Used by :func:`parseOrc` to split an orchestra in individual blocks

    Attributes:
        kind: the kind of block ('instr', 'opcode', 'header', 'include', 'instr0')
        text: thet text of the block
        startLine: where does this block start within the parsed orchestra
        endLine: where does this block end
        name: name of the block
        attrs: some blocks need extraOptions information. Opcodes define attrs 'outargs' and
            'inargs' (corresponding to the xin and xout opcodes), header blocks have
            a 'value' attr
    """
    kind: str
    lines: list[str]
    startLine: int
    endLine: int = -1
    name: str = ''
    attrs: dict[str, str] | None = None

    def __post_init__(self):
        assert self.kind in ('instr', 'opcode', 'header', 'include', 'instr0', 'comment'), f"Unknown block kind: '{self.kind}'"
        if self.endLine == -1:
            self.endLine = self.startLine

    @property
    def text(self):
        return self.lines[0] if len(self.lines) == 1 else '\n'.join(self.lines)


@dataclasses.dataclass
class _OrcBlock:
    name: str
    startLine: int
    lines: list[str]
    endLine: int = 0
    outargs: str = ""
    inargs: str = ""


def parseOrc(code: str | list[str], keepComments=True) -> list[ParsedBlock]:
    """
    Parse orchestra code into blocks

    Each block is either an instr, an opcode, a header line, a comment
    or an instr0 line

    Example
    -------

    .. code-block:: python

        >>> from csoundengine import csoundlib
        >>> orc = r'''
        ... sr = 44100
        ... nchnls = 2
        ... ksmps = 32
        ... 0dbfs = 1
        ... seed 0
        ...
        ... opcode AddSynth,a,i[]i[]iooo
        ...  /* iFqs[], iAmps[]: arrays with frequency ratios and amplitude multipliers
        ...  iBasFreq: base frequency (hz)
        ...  iPtlIndex: partial index (first partial = index 0)
        ...  iFreqDev, iAmpDev: maximum frequency (cent) and amplitude (db) deviation */
        ...  iFqs[], iAmps[], iBasFreq, iPtlIndx, iFreqDev, iAmpDev xin
        ...  iFreq = iBasFreq * iFqs[iPtlIndx] * cent(rnd31:i(iFreqDev,0))
        ...  iAmp = iAmps[iPtlIndx] * ampdb(rnd31:i(iAmpDev,0))
        ...  aPartial poscil iAmp, iFreq
        ...  if iPtlIndx < lenarray(iFqs)-1 then
        ...   aPartial += AddSynth(iFqs,iAmps,iBasFreq,iPtlIndx+1,iFreqDev,iAmpDev)
        ...  endif
        ...  xout aPartial
        ... endop
        ...
        ... ;frequency and amplitude multipliers for 11 partials of Risset's bell
        ... giFqs[] fillarray  .56, .563, .92, .923, 1.19, 1.7, 2, 2.74, 3, 3.74, 4.07
        ... giAmps[] fillarray 1, 2/3, 1, 1.8, 8/3, 5/3, 1.46, 4/3, 4/3, 1, 4/3
        ...
        ... instr Risset_Bell
        ...  ibasfreq = p4
        ...  iamp = ampdb(p5)
        ...  ifqdev = p6 ;maximum freq deviation in cents
        ...  iampdev = p7 ;maximum amp deviation in dB
        ...  aRisset AddSynth giFqs, giAmps, ibasfreq, 0, ifqdev, iampdev
        ...  aRisset *= transeg:a(0, .01, 0, iamp/10, p3-.01, -10, 0)
        ...  out aRisset, aRisset
        ... endin
        ... ''')
        >>> csoundlib.parseOrc(orc)
        [ParsedBlock(kind='header'P, text='sr = 44100', startLine=1, endLine=1, name='sr',
                     attrs={'value': '44100'}),
         ParsedBlock(kind='header', text='ksmps = 32', startLine=2, endLine=2, name='ksmps', attrs={'value': '32'}),
         ParsedBlock(kind='header', text='nchnls = 2', startLine=3, endLine=3, name='nchnls', attrs={'value': '2'}),
         ParsedBlock(kind='header', text='0dbfs = 1', startLine=4, endLine=4, name='0dbfs', attrs={'value': '1'}),
         ParsedBlock(kind='instr0', text='seed 0', startLine=6, endLine=6, name='', attrs=None),
         ParsedBlock(kind='opcode', text='opcode AddSynth,a,i[]i[]iooo\\n iFqs[], iAmps[], iBasFreq, iPtlIndx, <...>',
                     name='AddSynth', attrs={'outargs': 'a', 'inargs': 'i[]i[]iooo'}),
         ParsedBlock(kind='comment', text=";frequency and amplitude multipliers for 11 partials of Risset's bell",
                     startLine=19, endLine=19, name='', attrs=None),
         ParsedBlock(kind='instr0', text='giFqs[] fillarray  .56, .563, .92, .923, 1.19, 1.7, 2, 2.74, 3, 3.74, 4.07', startLine=20, endLine=20, name='', attrs=None),
         ParsedBlock(kind='instr0', text='giAmps[] fillarray 1, 2/3, 1, 1.8, 8/3, 5/3, 1.46, 4/3, 4/3, 1, 4/3', startLine=21, endLine=21, name='', attrs=None),
         ParsedBlock(kind='instr', text='instr Risset_Bell\\n ibasfreq = p4\\n iamp = ampdb(p5)\\n <...>'
                     startLine=23, endLine=31, name='Risset_Bell', attrs=None)]

    """
    context = []
    blocks: list[ParsedBlock] = []
    block = _OrcBlock("", 0, [])
    lines = code if isinstance(code, list) else code.splitlines()
    for i, line in enumerate(lines):
        strippedline = line.strip()
        if not strippedline:
            continue
        if match := re.search(r"\binstr\s+(\d+|[a-zA-Z_]\w+)", line):
            context.append('instr')
            block = _OrcBlock(name=match.group(1),
                              startLine=i,
                              lines=[line])
        elif strippedline == "endin":
            assert context[-1] == "instr"
            context.pop()
            assert block.name
            block.endLine = i
            block.lines.append(line)
            blocks.append(ParsedBlock(kind='instr',
                                      lines=block.lines,
                                      startLine=block.startLine,
                                      endLine=block.endLine,
                                      name=block.name))
        elif strippedline == 'endop':
            assert context[-1] == "opcode"
            context.pop()
            block.endLine = i
            block.lines.append(line)
            blocks.append(ParsedBlock(kind='opcode',
                                      lines=block.lines,
                                      startLine=block.startLine,
                                      endLine=block.endLine,
                                      name=block.name,
                                      attrs={'outargs': block.outargs,
                                             'inargs': block.inargs}))
        elif context and context[-1] in {'instr', 'opcode'}:
            block.lines.append(line)
        elif match := re.search(r"^\s*(sr|ksmps|kr|A4|0dbfs|nchnls|nchnls_i)\s*=\s*(\d+)", line):
            blocks.append(ParsedBlock(kind='header',
                                      lines=[line],
                                      name=match.group(1),
                                      startLine=i,
                                      attrs={'value':match.group(2)}))
        elif re.search(r"^\s*(;|\/\/)", line):
            if keepComments:
                blocks.append(ParsedBlock(kind='comment',
                                          startLine=i,
                                          lines=[line]))
        elif match := re.search(r"^\s*opcode\s+(\w+)\s*,\s*([0ika\[\]]*),\s*([0ikaoOjJpP\[\]]*)", line):
            context.append('opcode')
            block = _OrcBlock(name=match.group(1),
                              startLine=i,
                              lines=[line],
                              outargs=match.group(2),
                              inargs=match.group(3))
        elif strippedline.startswith('#include'):
            blocks.append(ParsedBlock(kind='include',
                                      startLine=i,
                                      lines=[line]))
        else:
            blocks.append(ParsedBlock(kind='instr0',
                                      startLine=i,
                                      lines=[line]))
    return blocks


@dataclasses.dataclass
class ParsedInstrBody:
    """
    The result of parsing the body of an instrument

    This is used by :func:`instrParseBody`

    """
    pfieldIndexToName: dict[int, str]
    """Maps pfield index to assigned name"""

    pfieldLines: _t.Sequence[str]
    """List of lines where pfields are defined"""

    lines: _t.Sequence[str]
    """The body, split into lines"""

    pfieldIndexToValue: dict[int, float] | None = None
    "Default values of the pfields, by pfield index"

    pfieldsUsed: set[int] | None = None
    "Which pfields are accessed"

    outChannels: set[int] | None = None
    "Which output channels are used"

    @functools.cached_property
    def body(self) -> str:
        return "\n".join(self.lines)

    @functools.cached_property
    def pfieldsText(self) -> str:
        """The text containing pfield definitions"""
        return "\n".join(self.pfieldLines)

    @functools.cached_property
    def pfieldNameToIndex(self) -> dict[str, int]:
        """Maps pfield name to its index"""
        return {name: idx for idx, name in self.pfieldIndexToName.items()}

    def numPfields(self) -> int:
        """ Returns the number of pfields in this instrument """
        return 3 if not self.pfieldsUsed else max(self.pfieldsUsed)

    @functools.cached_property
    def pfieldNameToValue(self) -> dict[str, float]:
        """
        Dict mapping pfield name to default value

        If a pfield has no explicit name assigned, p## is used. If it has no explicit
        value, 0. is used

        Example
        -------

        Given a csound instr:

        >>> parsed = instrParseBody(r'''
        ... pset 0, 0, 0, 0.1, 400, 0.5
        ... iamp = p4
        ... kfreq = p5
        ... ''')
        >>> parsed.pfieldNameToValue
        {'iamp': 0.1, 'kfreq': 400, 'p6': 0.5}

        """
        if not self.pfieldNameToIndex:
            return EMPTYDICT

        if self.pfieldIndexToValue is not None:
            out1 = {(self.pfieldIndexToName.get(idx) or f"p{idx}"): value
                    for idx, value in self.pfieldIndexToValue.items()}
        else:
            out1 = {}
        if self.pfieldIndexToName is not None:
            assert self.pfieldIndexToValue is not None
            out2 = {name: self.pfieldIndexToValue.get(idx, 0.)
                    for idx, name in self.pfieldIndexToName.items()}
        else:
            out2 = {}
        out1.update(out2)
        return out1


def lastAssignmentToVariable(varname: str, lines: list[str]) -> int | None:
    """
    Line of the last assignment to a variable

    Given a piece of code (normally the body of an instrument)
    find the line in which the given variable has its **last**
    assignment

    Args:
        varname: the name of the variable
        lines: the lines which make the instrument body. We need to split
            the body into lines within the function itself and since the
            user might need to split the code anyway afterwards, we
            already ask for the lines instead.

    Returns:
        the line number of the last assignment, or None if there is no
        assignment to the given variable

    Possible matches::

        aout oscili 0.1, 1000
        aout, aout2 pan2 ...
        aout = ...
        aout=...
        aout += ...
        aout2, aout = ...


    Example
    -------

        >>> lastAssignmentToVariable("aout", r'''
        ... aout oscili:a(0.1, 1000)
        ... aout *= linen:a(...)
        ... aout = aout + 10
        ... outch 1, aout
        ... '''.splitlines())
        3
    """
    rgxs = [
        re.compile(rf'^\s*({varname})\s*(=|\*=|-=|\+=|\/=)'),
        re.compile(rf'^\s*({varname})\s*,'),
        re.compile(rf'^\s*({varname})\s+[A-Za-z]\w*'),
        re.compile(rf'^\s*(?:\w*,\s*)+\b({varname})\b')
    ]
    for i, line in enumerate(reversed(lines)):
        for rgx in rgxs:
            if rgx.search(line):
                return len(lines) - 1 - i
    return None


def locateDocstring(lines: _t.Sequence[str]) -> tuple[int | None, int]:
    """
    Locate the docstring in this instr code

    To reconstruct the docstring do::

        start, end = locatedDocstring(lines)
        if start is None:
            docstring = ''
        else:
            docstring = '\n'.join(lines[start:end])

    Args:
        lines: the code to analyze, tipically the code inside an instr
            (between instr/endin), split into lines

    Returns:
        a tuple (firstline, lastline) indicating the location of the docstring
        within the given text. firstline will be None if no docstring was found.

    """
    assert isinstance(lines, (list, tuple))
    docstringStart = None
    docstringEnd = 0
    docstringKind = ''
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if docstringStart is None:
            if re.search(r'(;|\/\/|\/\*)', line):
                docstringStart = i
                docstringKind = ';' if line[0] == ';' else line[:2]
                continue
            else:
                # Not a docstring, so stop looking
                break
        else:
            # inside docstring
            if docstringKind == '/*':
                # TODO
                pass
            elif line.startswith(docstringKind):
                docstringEnd = i+1
            else:
                break
    if docstringStart is not None and docstringEnd < docstringStart:
        docstringEnd = docstringStart + 1
    return docstringStart, docstringEnd


def firstLineWithoutComments(lines: _t.Sequence[str]) -> int | None:
    insideComment = False
    for i, line in enumerate(lines):
        if insideComment:
            if re.match(r'\s*\*\/', line):
                insideComment = False
        elif re.match(r'\s*\/\*', line):
            insideComment = True
        elif not re.match(r'\s*[;\/]', line) and line.strip():
            return i
    return None


def splitDocstring(body: str | _t.Sequence[str]) -> tuple[str, str]:
    """
    Given a docstring, split it into the docstring and the rest of the body.

    Args:
        body: the body of an instr (excluding instr/endin)

    Returns:
        a tuple (docstring, rest)
    """
    if isinstance(body, str):
        lines = body.splitlines()
    else:
        lines = body
    docstart, docend = locateDocstring(lines)
    if docstart is not None:
        docstring = '\n'.join(lines[docstart:docend])
        rest = '\n'.join(lines[docend:])
    else:
        docstring = ''
        rest = body if isinstance(body, str) else '\n'.join(lines)
    return docstring, rest


def instrGetBody(lines: list[str]) -> list[str]:
    """
    Get the body of the instrument, without 'instr' / 'endin'

    Args:
        lines: the lines of the instrument

    Returns:
        the body of the instr, split into lines
    """
    lines = internal.stripTrailingEmptyLines(lines)
    if not lines[0].lstrip().startswith('instr') or not lines[-1].rstrip().endswith('endin'):
        raise ValueError(f'Invalid instrument body: {lines}')
    lines = lines[1:-1]
    return lines


def instrParseBody(body: str | list[str]) -> ParsedInstrBody:
    """
    Parses the body of an instrument, returns pfields used, output channels, etc.

    Args:
        body (str): the body of the instr (between instr/endin)

    Returns:
        a ParsedInstrBody


    Example
    -------

        >>> from csoundengine import csoundlib
        >>> body = r'''
        ... pset 0, 0, 0, 1, 1000
        ... ibus = p4
        ... kfreq = p5
        ... a0 = busin(ibus)
        ... a1 = oscili:a(0.5, kfreq) * a0
        ... outch 1, a1
        ... '''
        >>> csoundlib.instrParseBody()
        ParsedInstrBody(pfieldsIndexToName={4: 'ibus', 5: 'kfreq'},
                        pfieldLines=['ibus = p4', ['kfreq = p5'],
                        body='\\na0 = busin(ibus)\\n
                          a1 = oscili:a(0.5, kfreq) * a0\\noutch 1, a1',
                        pfieldsDefaults={1: 0.0, 2: 0.0, 3: 0.0, 4: 1.0, 5: 1000.0},
                        pfieldsUsed={4, 5},
                        outChannels={1},
                        pfieldsNameToIndex={'ibus': 4, 'kfreq': 5})
    """
    lines = body if isinstance(body, list) else body.splitlines()
    if len(lines) == 1 and not lines[0].strip():
        return ParsedInstrBody(pfieldIndexToValue={},
                               pfieldLines=(),
                               lines=(),
                               pfieldIndexToName={})

    pfieldLines = []
    bodyLines = []
    pfieldIndexToValue = {}
    insideComment = False
    pfieldsUsed = set()
    pfieldIndexToName: dict[int, str] = {}
    outchannels: set[int] = set()
    for i, line in enumerate(lines):
        if insideComment:
            bodyLines.append(line)
            if re.match(r"\*\/", line):
                insideComment = False
            continue
        elif re.match(r"^\s*(;|\/\/)", line):
            # A line comment
            bodyLines.append(line)
            continue
        else:
            # Not inside comment
            if pfieldsInLine := re.findall(r"\bp\d+", line):
                for p in pfieldsInLine:
                    pfieldsUsed.add(int(p[1:]))

            if re.match(r"^\s*\/\*", line):
                insideComment = True
                bodyLines.append(line)
            elif re.match(r"\*\/", line) and insideComment:
                insideComment = False
                bodyLines.append(line)
            elif m := re.search(r"\bpassign\s+(\d+)", line):
                if "[" in line:
                    # array form, iarr[] passign 4, 6
                    bodyLines.append(line)
                else:
                    pfieldLines.append(line)
                    pstart = int(m.group(1))
                    argsstr, rest = line.split("passign")
                    args = argsstr.split(",")
                    for j, name in enumerate(args, start=pstart):
                        pfieldsUsed.add(j)
                        pfieldIndexToName[j] = name.strip()
            elif re.search(r"^\s*\bpset\b", line):
                s = line.strip()[4:]
                psetValues = {j: float(v) for j, v in enumerate(s.split(","), start=1)
                              if v.strip()[0].isnumeric()}
                pfieldIndexToValue.update(psetValues)
            elif m := re.search(r"^\s*\b(\w+)\s*(=|init\s)\s*p(\d+)", line):
                # 'ival = p4' / kval = p4 or 'ival init p4'
                pname = m.group(1)
                pfieldIndex = int(m.group(3))
                pfieldLines.append(line)
                pfieldIndexToName[pfieldIndex] = pname.strip()
                pfieldsUsed.add(pfieldIndex)
            else:
                if re.search(r"\bouts\s+", line):
                    outchannels.update((1, 2))
                elif re.search(r"\bout\b", line):
                    outchannels.add(1)
                elif re.search(r"\boutch\b", line):
                    args = line.strip()[5:].split(",")
                    channels = args[::2]
                    for chans in channels:
                        if chans.isnumeric():
                            outchannels.add(int(chans))
                bodyLines.append(line)

    for pidx in range(1, 4):
        pfieldIndexToValue.pop(pidx, None)
        pfieldIndexToName.pop(pidx, None)

    bodyLines = [line for line in bodyLines if line.strip()]

    return ParsedInstrBody(pfieldIndexToValue=pfieldIndexToValue,
                           pfieldIndexToName=pfieldIndexToName,
                           pfieldsUsed=pfieldsUsed,
                           outChannels=outchannels,
                           pfieldLines=pfieldLines,
                           lines=lines)


def highlightCsoundOrc(code: str, theme='') -> str:
    """
    Converts csound code to html with syntax highlighting

    Args:
        code: the code to highlight
        theme: the theme used, one of 'light', 'dark'. If not given, a default
            is used (see config['html_theme'])

    Returns:
        the corresponding html
    """
    if not theme:
        from .config import config
        theme = config['html_theme']

    import pygments
    import pygments.formatters
    if theme == 'light':
        htmlfmt = pygments.formatters.HtmlFormatter(noclasses=True, wrapcode=True)
    else:
        htmlfmt = pygments.formatters.HtmlFormatter(noclasses=True, style='fruity',
                                                    wrapcode=True)
    html = pygments.highlight(code, lexer=_pygmentsOrcLexer(), formatter=htmlfmt)
    return html


@functools.cache
def _pygmentsOrcLexer():
    import pygments.lexers.csound
    return pygments.lexers.csound.CsoundOrchestraLexer()


def isPfield(name: str) -> bool:
    """
    Is name a pfield?
    """
    return re.match(r'\bp[1-9][0-9]*\b', name) is not None


def splitScoreLine(line: str, quote=False) -> list[float | str]:
    """
    Split a score line into its tokens

    Args:
        line: the score line to split
        quote: if True, add quotation marks to strings

    Returns:
        a list of tokens
    """
    # i "instr" 1 2 3 "foo bar" 0.5 "foofi"
    kind = line[0]
    assert kind in 'ife'
    rest = line[1:]
    allparts: list[str | int | float] = [kind]
    # even parts are not quoted strings, odd parts are quoted strings
    for i, part in enumerate(rest.split('"')):
        if i % 2 == 0:
            allparts.extend(float(sub.strip()) for sub in part.split())
        else:
            allparts.append(f'"{part}"' if quote else part)
    return allparts


def normalizeNamedPfields(pfields: dict[str, float],
                          namesToIndexes: dict[str, int] | None = None
                          ) -> dict[int, float]:
    """
    Given a dict mapping pfield as str to value, return a dict mapping pfield index to value

    Args:
        pfields: a dict of the form {pfield: value} where pfield can be
        a key like 'p<n>' or a variable name which was assigned to this pfield
        (like ``ifreq = p4``
        namesToIndexes: a dict mapping variable names to indexes

    Returns:
        a dict of the form {pfieldindex: value}

    Example
    ~~~~~~~

        >>> normalizeNamedPfields({'p4': 0.5, 'ifreq': 2000}, {'ifreq': 5})
        {4: 0.5, 5: 2000}
    """
    out: dict[int, float] = {}
    for k, value in pfields.items():
        if k.startswith('p'):
            out[int(k[1:])] = value
        elif namesToIndexes:
            assert k.startswith('k') or k.startswith('i')
            idx = namesToIndexes.get(k)
            if idx is None:
                raise KeyError(f"Keyword pfield not known: {k}")
            out[idx] = value
        else:
            raise KeyError(f"Keyword pfield not known: {k}")
    return out


def fillPfields(args: _t.Sequence[float | str] | np.ndarray,
                namedpargs: dict[int, float],
                defaults: dict[int, float] | None) -> list[float | str]:
    """
    Given a set of arguments, named pfields and defaults, generates the list of pfields to be passed to csound

    Args:
        args: seq. of positional arguments, starting with p4
        namedpargs: dict mapping pfield index to values
        defaults: dict mapping pfield names to default values

    Returns:
        the pfield values, starting with p4
    """
    out: list[float | str]
    if not defaults:
        if namedpargs and not args:
            maxp = max(namedpargs.keys())
            out = [0.] * (maxp - 3)
            for idx, value in namedpargs.items():
                out[idx - 4] = value
            return out
        elif namedpargs and args:
            maxp = max(len(args) + 3, max(namedpargs.keys()))
            out = [0.] * (maxp - 3)
            for i, arg in enumerate(args):
                out[i] = arg
            for idx, value in namedpargs.items():
                out[idx-4] = value
            return out
        elif args:
            return args if isinstance(args, list) else list(args)
        else:
            # no args at all
            raise ValueError("No args or namedargs given and no default values defined")

    # with defaults
    if namedpargs and not args:
        maxp = max(max(defaults.keys()), max(namedpargs.keys()))
        out = [0.] * (maxp - 3)
        for idx, value in defaults.items():
            out[idx - 4] = value
        for idx, value in namedpargs.items():
            out[idx - 4] = value
        return out
    elif namedpargs and args:
        maxp = max(len(args)+3, max(defaults.keys()), max(namedpargs.keys()))
        out = [0.] * (maxp - 3)
        for idx, value in defaults.items():
            out[idx - 4] = value
        out[:len(args)] = args
        for idx, value in namedpargs.items():
            out[idx - 4] = value
        return out
    elif args:
        maxp = max(len(args) + 3, max(defaults.keys()))
        out = [0.] * (maxp - 3)
        for idx, value in defaults.items():
            out[idx - 4] = value
        out[:len(args)] = args
        return out
    else:
        # only defaults
        maxp = max(defaults.keys())
        out = [0.] * (maxp - 3)
        for idx, value in defaults.items():
            out[idx - 4] = value
        return out


def splitInclude(line: str) -> str:
    """
    Given an include line it splits the include path

    Example
    -------

        >>> splitInclude(r'   #include "foo/bar" ')
        foo/bar

    NB: the quotation marks are not included
    """
    match = re.search(r'#include\s+"(.+)""', line)
    if not match:
        raise ValueError("Could not parse include")
    return match.group(1)


def makeIncludeLine(include: str) -> str:
    """
    Given a path, creates the #include directive

    In particula, it checks the need for quotation marks

    Args:
        include: path to include

    Returns:

    """
    import emlib.textlib
    s = emlib.textlib.quoteIfNeeded(include.strip())
    return f'#include {s}'
