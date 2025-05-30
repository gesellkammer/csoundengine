from __future__ import annotations


class CsoundError(Exception):
    pass

class TableNotFoundError(CsoundError):
    "The table does not exist"


class RenderError(Exception): ...
