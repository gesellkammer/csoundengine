from __future__ import annotations

from . import internal

import typing
if typing.TYPE_CHECKING:
    from .abstractrenderer import AbstractRenderer
    from typing import Sequence

class Bus:
    """
    A wrapper around a raw bus

    .. note::

        A user **never** creates a Bus directly. A Bus is created by a
        :class:`~csoundengine.session.Session` through the method
        :meth:`~csoundengine.session.Session.assignBus`.

    Args:
        kind: the bus kind, one of 'audio', 'control'
        token: the token as returned via :meth:`csoundengine.engine.Engine.assignBus`
        renderer: the renderer to which this Bus belongs
        bound: if True, the Bus object uses itself a reference. This means that the
            bus will stay alive at least as long as this object is kept alive. The
            bus might still survive the object if it is still being used by any
            instrument

    """
    def __init__(self,
                 kind: str,
                 token: int,
                 renderer: AbstractRenderer,
                 bound=True):
        self.renderer = renderer
        """The parent renderer"""

        self.token = token
        """Token as returned via :meth:`csoundengine.engine.Engine.assignBus`"""

        self.kind = kind
        """Bus kind, one of 'audio', 'control'"""

        self.bound = bound
        """Bind the bus lifetime to this object."""

    def __del__(self):
        if self.bound:
            self.renderer._releaseBus(self)

    def __format__(self, format_spec):
        return str(self.token)

    def __repr__(self) -> str:
        return f"Bus('{self.kind}', token={self.token})"

    def __int__(self):
        return self.token

    def __float__(self):
        return float(self.token)

    def set(self, value: float, delay=0.) -> None:
        """
        Set the value of this bus

        This is only valid for scalar (control) buses

        Args:
            value: the new value
            delay: when to set the value

        Example
        ~~~~~~~

        >>> from csoundengine import *
        >>> s = Engine().session()
        >>> s.defInstr('vco', r'''
        ... |ifreqbus|
        ... kfreq = busin:k(ifreqbus)
        ... outch 1, vco2:a(0.1, kfreq)
        ... ''')
        >>> freqbus = s.assignBus(value=1000)
        >>> s.sched('vco', 0, 4, ifreqbus=freqbus)
        >>> freqbus.set(500, delay=0.5)

        """
        if self.token < 0:
            raise ValueError("This Bus has been already released")
        if not self.kind == 'control':
            raise ValueError("Only control buses can be set")
        self.renderer._writeBus(bus=self, value=value, delay=delay)

    def get(self) -> float:
        """
        Get the value of the bus

        Example
        ~~~~~~~

        >>> from csoundengine import *
        >>> s = Engine().session()
        >>> s.defInstr('rms', r'''
        ... |irmsbus|
        ... asig inch 1
        ... krms = rms:k(asig)
        ... busout irmsout, krms
        ... ''')
        >>> rmsbus = s.assignBus('control')
        >>> synth = s.sched('rms', irmsbus=rmsbus.token)
        >>> while True:
        ...     rmsvalue = rmsbus.get()
        ...     print(f"Rms value: {rmsvalue}")
        ...     time.sleep(0.1)

        """
        if self.token < 0:
            raise ValueError("This Bus has been already released")
        if not self.kind == 'control':
            raise ValueError("Only control buses can be set")
        out = self.renderer._readBus(self)
        if out is None:
            raise RuntimeError(f"The renderer {self.renderer} does not support "
                               f"reading from a bus")
        return out

    def automate(self,
                 pairs: Sequence[float] | tuple[Sequence[float], Sequence[float]],
                 mode='linear', delay=0., overtake=False) -> float:
        """
        Automate this bus

        This operation is only valid for control buses. The automation is
        performed within csound and is thus assured to stay in sync

        Args:
            pairs: the automation data as a flat sequence (t0, value0, t1, value1, ...)
                Times are relative to the start of the automation event
            mode: interpolation mode, one of 'linear', 'expon(xx)', 'cos', 'smooth'.
                See the csound opcode 'interp1d' for mode information
                (https://csound-plugins.github.io/csound-plugins/opcodes/interp1d.html)
            delay: when to start the automation
            overtake: if True, the first value of pairs is replaced with the current
                value of the bus. The same effect can be achieved if the first value
                of the automation line is a nan

        Returns:
            a float representing the event id of the automation

        .. seealso:: :meth:`Engine.assignBus`, :meth:`Engine.writeBus`, :meth:`Engine.automatep`

        Example
        ~~~~~~~

        >>> from csoundengine import *
        >>> s = Engine().session()
        >>> s.defInstr('sine', r'''
        ... |ifreqbus|
        ... kfreq = busin:k(ifreqbus)
        ... outch 1, oscili:a(0.1, kfreq)
        ... ''')
        >>> freqbus = s.assignBus(value=440)
        >>> synth = s.sched('sine', ifreqbus=freqbus)
        >>> freqbus.automate([0, float('nan'), 3, 200, 5, 200])

        """
        if self.token < 0:
            raise ValueError("This Bus has been already released")
        pairs = internal.flattenAutomationData(pairs)
        return self.renderer._automateBus(self, pairs=pairs, mode=mode, delay=delay, overtake=overtake)

    def release(self) -> None:
        """
        Manually release the bus

        Normally this operation is done automatically when the object is deleted.
        """
        if self.token >= 0:
            self.renderer._releaseBus(self)
            self.token = -1
        else:
            from .config import logger
            logger.debug(f"Bus {self} already released")
