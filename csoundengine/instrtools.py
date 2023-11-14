from __future__ import annotations
from dataclasses import dataclass
from . import csoundlib
from ._common import EMPTYDICT


@dataclass
class InlineArgs:
    delimiters: str
    """The delimiter used"""

    args: dict[str, float]
    """The args in the declaration, as a dict {argname: defaultvalue}"""

    body: str
    """The body of the instrument WITHOUT the args"""

    linenum: int
    """The line number withint the original body where the inline args are placed"""

    def __post_init__(self):
        assert self.delimiters == '||' or self.delimiters == '{}'


@dataclass
class Docstring:
    shortdescr: str = ''
    longdescr: str = ''
    args: dict[str, str] | None = None


def pfieldsGenerateCode(pfields: dict[int, str],
                        strmethod='strget',
                        unifyIndentation=True) -> str:
    """
    Generate code for the given pfields

    Args:
        pfields: a dict mapping p-index to name
        strmethod: one of 'strget', 'direct'. If 'strget', string pargs are
            implemented as 'Sfoo = strget(p4)', otherwise just 'Sfoo = p4' is generated

    Returns:
        the generated code

    Example
    =======

        >>> print(pfieldsGenerateCode({4: 'ichan', 5: 'kfreq', '6': 'Sname'}))
        ichan = p4
        kfreq = p5
        Sname = strget(p6)

    """
    pairs = list(pfields.items())
    pairs.sort()
    maxwidth = max(len(name) for name in pfields.values())
    lines = []
    for idx, name in pairs:
        if unifyIndentation:
            name = name.ljust(maxwidth)

        if name[0] == 'S':
            if strmethod == 'strget':
                lines.append(f"{name} strget p{idx}")
            else:
                lines.append(f"{name} = p{idx}")
        else:
            lines.append(f"{name} = p{idx}")
    return "\n".join(lines)


def generatePfieldsCode(body: str,
                        idxToName: dict[int, str]
                        ) -> tuple[str, str, str]:
    """
    Generate pfields code

    Args:
        body: the body of the instr
        idxToName: dict mapping pfield index to name

    Returns:
        a tuple (pfieldscode, restbody, docstring)
    """
    parsedCode = csoundlib.instrParseBody(body)
    pfieldsText = pfieldsGenerateCode(idxToName)
    bodylines = parsedCode.body.splitlines()
    docstringLocation = csoundlib.locateDocstring(bodylines)
    if docstringLocation is None:
        return pfieldsText, '\n'.join(bodylines), ''
    else:
        start, end = docstringLocation
        docstring = '\n'.join(bodylines[start:end])
        rest = '\n'.join(bodylines[end:])
        return pfieldsText, rest, docstring


def detectInlineArgs(lines: list[str]) -> tuple[str, int | None]:
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


def pfieldsMergeDeclaration(args: dict[str, float],
                            body: str,
                            startidx=4
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
        >>> pfieldsMergeDeclaration(dict(kfreq=440, ichan=2), body)
        {4: ('kfreq', 440),
         5: ('ichan', 2),
         6: ('ifade', 0),
         7: ('icutoff', 0)}
    """
    parsedbody = csoundlib.instrParseBody(body)
    body_i2n = parsedbody.pfieldIndexToName
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


def assignPfields(namedargs: list[str], exclude: tuple[int, ...], minpfield=4, maxpfield=1900
                  ) -> list[int]:
    """
    Assign pfield indexes to named pfields

    Args:
        namedargs: a list of names
        exclude: pfields to exclude
        minpfield: the min. index to assign
        maxpfield: the max. index to assign

    Returns:
        a list of indexes, each index corresponds to one named argument
    """
    lastidx = minpfield
    used = set(exclude)
    indexes = []
    for arg in namedargs:
        for idx in range(lastidx, maxpfield):
            if idx not in used:
                indexes.append(idx)
                used.add(idx)
                lastidx = idx
                break
        else:
            raise ValueError("Not enough indexes to assign")
    assert len(indexes) == len(namedargs)
    assert all(idx >= 4 for idx in indexes)
    return indexes


def parseInlineArgs(body: str | list[str],
                    ) -> InlineArgs | None:
    """
    Parse an instr body with a possible args declaration (see below).

    Args:
        body: the body of the instrument as a string or as a list of lines
        allowPfields: if True, allow direct access of pfields

    Returns:
        an InlineArgs with fields

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

        >>> instrbody = '''
        ... |ichan, kamp=0.1, kfreq=440|
        ... a0 oscili kamp, kfreq
        ... outch ichan, a0
        ... '''
        >>> inlineargs = parseInlineArgs(instrbody)
        >>> inlineargs.delimiters
        '||'
        >>> args
        {'ichan': 0, 'kamp': 0.1, 'kfreq': 440}
        >>> print(bodyWithoutArgs)
        a0 oscili kamp, kfreq
        outch 1, a0
    """

    if not body:
        return None

    lines = body if isinstance(body, list) else body.splitlines()
    delimiters, linenum = detectInlineArgs(lines)

    if not delimiters:
        return None

    assert linenum is not None

    args = {}
    line2 = lines[linenum].strip()
    parts = line2[1:-1].split(",")
    for part in parts:
        if "=" in part:
            varname, defaultval = part.split("=")
            args[varname.strip()] = float(defaultval)
        else:
            args[part.strip()] = 0
    bodyWithoutArgs = "\n".join(lines[linenum+1:])
    return InlineArgs(delimiters, args=args, body=bodyWithoutArgs, linenum=linenum)


def parseDocstring(text: str | list[str]) -> Docstring | None:
    lines = text if isinstance(text, list) else text.splitlines()
    doclines: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            if not doclines:
                continue
            else:
                doclines.append('')
        elif not line.startswith(';'):
            break
        line = line.lstrip(';')
        doclines.append(line)
    if not doclines:
        return None
    docs = '\n'.join(doclines)
    import docstring_parser
    parsed = docstring_parser.parse(docs)
    if parsed.params:
        args = {param.arg_name: param.description or '' for param in parsed.params}
    else:
        args = None
    return Docstring(shortdescr=parsed.short_description or '',
                     longdescr=parsed.long_description or '',
                     args=args)


def distributeParams(params: dict[str, float],
                     pfieldNames: set[str] | frozenset[str],
                     controlNames: set[str] | frozenset[str]
                     ) -> tuple[dict[str | int, float | str], dict[str, float]]:
    """
    Sorts params into pfields and dynamic parameters

    Args:
        params: a dict mapping arg name to value given
        pfieldNames: the names of the named pfields
        controlNames: the names of the dynamic parameters

    Returns:
        a tuple (pfields, controls) where each is a dict mapping the
        parameter to its given value
    """
    if not controlNames:
        return (params, {})
    else:
        pfields = {}
        controls = {}
        for name, value in params.items():
            if isinstance(name, int) or csoundlib.isPfield(name) or name in pfieldNames:
                pfields[name] = value
            else:
                if name not in controlNames:
                    raise KeyError(f"Parameter '{name}' not known. Possible "
                                   f"dynamic arguments: {controlNames}")
                controls[name] = value
        # if pfields:
        #     assert all(pfield in pfieldNames for pfield in pfields)
        # if controls:
        #     assert all(control in controlNames for control in controls)
        return pfields, controls