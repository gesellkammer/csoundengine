from __future__ import annotations

import dataclasses
import functools
import os

import emlib.misc

from . import internal

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import subprocess
    from typing import Sequence



@dataclasses.dataclass
class RenderJob:
    """
    Represent an offline render process

    A RenderJob is generated each time :meth:`Renderer.render` is called.
    Each new process is appended to :attr:`Renderer.renderedJobs`. The
    last render job can be accesses via :meth:`Renderer.lastRenderJob`
    """
    outfile: str
    """The soundfile rendered / being rendererd"""

    samplerate: int
    """Samplerate of the rendered soundfile"""

    encoding: str = ''
    """Encoding of the rendered soundfile"""

    starttime: float = 0.
    """Start time of the rendered timeline"""

    endtime: float = 0.
    """Endtime of the rendered timeline"""

    process: subprocess.Popen | None = None
    """The csound subprocess used to render the soundfile"""

    @property
    def args(self) -> Sequence[str | bytes]:
        """The args used to render this job, if a process was used"""
        if not self.process:
            return []
        args = self.process.args
        from collections.abc import Iterable
        if isinstance(args, (str, bytes)):
            return [args]
        elif isinstance(args, Iterable):
            return [str(a) for a in args]
        else:
            return [str(self.args)]

    def openOutfile(self, timeout=None, appwait=True, app=''):
        """
        Open outfile in external app

        Args:
            timeout: if still rendering, timeout after this number of seconds. None
                means to wait until rendering is finished
            app: if given, use the given application. Otherwise the default
                application
            appwait: if True, wait until the external app exits before returning
                from this method
        """
        self.wait(timeout=timeout)
        emlib.misc.open_with_app(self.outfile, wait=appwait, app=app)

    def wait(self, timeout: float | None = None):
        """Wait for the render process to finish"""
        if self.process is not None:
            self.process.wait(timeout=timeout)

    def __hash__(self):
        if self.outfile and os.path.exists(self.outfile):
            return hash((internal.hashSoundfile(self.outfile), self.starttime, self.endtime))
        else:
            return id(self)

    @functools.cache
    def _repr_html_(self):
        self.wait()
        blue = internal.safeColors['blue1']

        def _(s, color=blue):
            return f'<code style="color:{color}">{s}</code>'

        if not os.path.exists(self.outfile):
            info = (f"outfile='{self.outfile}' (not found), sr={_(self.samplerate)}, "
                    f"encoding={self.encoding}, args={self.args}")
            return f'<string>RenderJob</strong>({info})'
        else:
            sndfile = self.outfile
            soundfileHtml = internal.soundfileHtml(sndfile, withHeader=False)
            info = (f"outfile='{_(self.outfile)}', sr={_(self.samplerate)}, "
                    f"encoding='{self.encoding}'")
            htmlparts = (
                f'<strong>RenderJob</strong>({info})',
                soundfileHtml
            )
            return '<br>'.join(htmlparts)
