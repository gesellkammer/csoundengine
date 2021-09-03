"""
This module defines some %magic / %%magic to work with an Engine or a Session

Magics defined
==============

%csound
-------

Syntax::

    %csound setengine <enginename>         : Sets the default Engine

%%csound
--------

Compile the code in this cell
"""
from __future__ import annotations
from IPython.core.getipython import get_ipython
from IPython.core.magic import Magics, magics_class, cell_magic, line_cell_magic
from IPython.display import display, HTML
from . import engine as _engine
from .config import logger, config
from . import csoundlib

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *


def _splitonce(s: str):
    parts = s.split(maxsplit=1)
    return parts if len(parts) == 2 else (parts[0], '')


@magics_class
class EngineMagics(Magics):
    """Implement magic commands for Csound."""
    def __init__(self, shell):
        super().__init__(shell)
        self.currentEngine: Optional[_engine.Engine] = None
        self._lineCommands = {
            'setengine': self._cmd_setengine
        }

    def _resolveEngine(self, line: str) -> Optional[_engine.Engine]:
        if line:
            engineName = line.strip()
            engine = _engine.getEngine(engineName)
            if engine is None:
                logger.warning(f"Engine {engineName} not known. "
                               f"Possible engines: {_engine.activeEngines()}")
                return None
            return engine
        if self.currentEngine:
            return self.currentEngine
        engines = _engine.activeEngines()
        if not engines:
            print("No active Engine. After creating an Engine, "
                  "do `%csound <enginename>` to set it as the default Engine")
            return None
        return _engine.getEngine(list(engines)[-1])

    def _cmd_setengine(self, line:str) -> None:
        engine = _engine.getEngine(line)
        if not engine:
            logger.error(f"Engine {engine} unknown. Active engines: "
                         f"{_engine.activeEngines()}")
        else:
            self.currentEngine = engine

    @line_cell_magic
    def csound(self, line, cell=None):
        """
        %csound and %%csound magics.

        ``%csound setengine <enginename>``
            Set the default engine for any subsequent magic. Otherwise the last
            created Engine will be used

        ``%%csound [<enginename>]``
            Cell magic, compiles any code within the given engine, or the default
            engine if no engine is given.

        Example
        -------

        .. code::

            from csoundengine import *
            e = Engine("foo")

            # In another cell
            %%csound
            instr 10
                kamp = p4
                kmidi = p5
                a0 oscili kamp, mtof:k(kmidi)
                aenv linsegr 0, 0.01, 1, 0.01, 0
                a0 *= aenv
                outch 1, a0
            endin

            # Schedule an A4 note lasting 20 seconds
            eventid = e.sched(10, 0, 20, [0.1, 69])

            # Change the pitch
            e.pwrite(eventid, "kmidi", 67)
        """
        # line magic
        if cell is None:
            assert line
            cmd, rest = _splitonce(line)
            func = self._lineCommands.get(cmd)
            if not func:
                logger.error(f"command {cmd} unknown. Possible commands: "
                             f"{self._lineCommands.keys()}")
                return None
            func(rest)
            return None
        # cell magic
        engine = self._resolveEngine(line)
        if not engine:
            return None
        self.currentEngine = engine
        self.currentEngine.compile(cell, block=True)
        html = csoundlib.highlightCsoundOrc(cell)
        display(HTML(html))
        return None

    @cell_magic
    def definstr(self, line: str, cell: str) -> None:
        """
        Defines a new Instr inside the current Session

        Example
        =======

        .. code-block:: python

            from csoundengine import *
            e = Engine()

        .. code-block:: csound

            %%definstr foo
            |kamp=0.5, kfreq=1000|
            a0 oscili kamp, kfreq
            aenv linsegr 0, 0.01, 1, 0.01, 0
            a0 *= aenv
            outch 1, a0

        The last block is equivalent to::

            s = e.session()
            s.defInstr("foo", r'''
                |kamp=0.5, kfreq=1000|
                a0 oscili kamp, kfreq
                aenv linsegr 0, 0.01, 1, 0.01, 0
                a0 *= aenv
                outch 1, a0
            ''')
        """
        parts = line.split()
        if len(parts) == 2:
            enginename, instrname = parts
            self.currentEngine = _engine.getEngine(enginename)
        elif len(parts) == 1:
            instrname = parts[0]
            if not self.currentEngine:
                engines = _engine.activeEngines()
                if not engines:
                    print("No active Engine. After creating an Engine, "
                          "do `%csound <enginename>` to set it as the default Engine")
                    return None
                self.currentEngine = _engine.getEngine(list(engines)[-1])
        else:
            raise SyntaxError("Syntax: %%definstr [engine] instrname")
        session = self.currentEngine.session()
        instr = session.defInstr(name=instrname, body=cell)
        display(HTML(instr._repr_html_()))


def load_ipython_extension(ip):
    if config['magics_print_info']:
        print("csoundengine.magic extension loaded")
        print("Magics available: %csound, %%csound, %%definstr")
    ip.magics_manager.register(EngineMagics)
    ip.user_ns['Engine'] = _engine.Engine
