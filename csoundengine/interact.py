from __future__ import annotations

import dataclasses


import emlib.envir
from . import synth as _synth

import typing as _t
if _t.TYPE_CHECKING:
    from . import engine


def _guessStep(minval: float, maxval: float):
    diff = maxval - minval
    if diff <= 2:
        step = 0.001
    elif diff < 10:
        step = 0.01
    elif diff < 50:
        step = 0.05
    elif diff < 200:
        step = 0.1
    else:
        step = 1
    return step


def _guessRange(value: float, namehint: str = '') -> tuple[float, float]:
    if namehint:
        if "freq" in namehint and 0 < value < 12000:
            return (0, 12000)
        if ("amp" in namehint or "gain" in namehint) and 0 <= value <= 1:
            return (0, 1)
        if "cutoff" in namehint and 0 < value < 12000:
            return (0, 12000)
        if ("midi" in namehint or "pitch" in namehint) and 0 <= value <= 127:
            return (0, 127)
        if namehint == 'pan' or namehint == 'kpan' and 0 <= value <= 1:
            return (0., 1.)

    if -90 < value < 0:
        return (-100, 0)
    if value < 0:
        val0, val1 = _guessRange(-value)
        return -val1, -val0
    if value <= 1:
        return (0, 2)
    elif value < 5:
        return (0, 10)
    elif value < 50:
        return (0, 500)
    elif value < 100:
        return (0, 1000)
    else:
        return (0, value * 10)


@dataclasses.dataclass
class ParamSpec:
    name: str
    """The parameter name"""

    minvalue: float = 0.
    maxvalue: float = 1.
    step: float = -1
    startvalue: float = 0.
    widgetHint: str = 'slider'
    valuescale: str = 'linear'

    description: str = ''
    """A optional description"""

    def scale(self, x: float) -> float:
        assert 0 <= x <= 1
        if self.valuescale == 'linear':
            return (self.maxvalue - self.minvalue) * x + self.minvalue
        elif self.valuescale == 'log10':
            return (self.maxvalue - self.minvalue) * (10**(1-x)) + self.minvalue

def __post_init__(self):
        if self.step == -1:
            self.step = _guessStep(self.minvalue, self.maxvalue)


def _stepToFormat(step: float) -> str:
    if step < 0.01:
        fmt = ".3f"
    elif step < 0.1:
        fmt = ".2f"
    elif step < 1:
        fmt = ".1f"
    else:
        fmt = "d"
    return fmt


def _jupyterSlider(name: str, startvalue: float, minvalue: float, maxvalue: float,
                   callback: _t.Callable, step: float | None = None, width='80%',
                   log=False):
    import ipywidgets as ipy
    if step is None:
        step = _guessStep(minvalue, maxvalue)
    fmt = _stepToFormat(step)
    layout = ipy.Layout(width=width)
    if not log:
        s = ipy.FloatSlider(value=startvalue, min=minvalue, max=maxvalue,
                            step=step, description=name, layout=layout,
                            readout_format=fmt)
    else:

        s = ipy.FloatLogSlider(value=startvalue, min=minvalue, max=maxvalue,
                               step=step, description=name, layout=layout,
                               readout_format=fmt)
    s.observe(lambda change: callback(change['new']), names='value')
    return s


def _jupyterEntry(name: str, startvalue: float, minvalue:float, maxvalue: float,
                  callback: _t.Callable):
    import ipywidgets as ipy
    step = 0.001
    w = ipy.BoundedFloatText(value=startvalue, min=minvalue, max=maxvalue,
                             step=step, description=name)
    w.observe(lambda change: callback(change['new']), names='value')
    return w


def interact(**sliders: tuple[float, float, float, _t.Callable]):
    """
    Creates a set of interactive widgets

    Args:
        sliders: given as keywords. The key is used as the widget name. The
            value is a tuple (initialvalue, minvalue, maxvalue, callback)
            where callback has the form func(x) -> None and will be called
            with the current value of the widget

    Example
    ~~~~~~~

        from csoundengine import *
        from csoundengine.interact import *
        from IPython.display import display

        e = Engine()
        e.compile(r'''
        instr foo
          kamp = p4
          kmidi = p5
          asig vco2 lag:k(kamp, 0.1), lag:k(mtof:k(kmidi), 0.1)
          outch 1, asig
        endin
        ''')
        p1 = e.sched("foo", [0.1, 67])
        interact(kamp=(0.1, 0., 1., lambda x:e.setp(p1, 4, x),
                 kmidi=(67, 0, 127, lambda x:e.setp(p1, 5, x))
    """
    from IPython.display import display
    widgets = []
    for key, value in sliders.items():
        curvalue, minvalue, maxvalue, func = value
        s = _jupyterSlider(name=key, startvalue=curvalue, minvalue=minvalue,
                           maxvalue=maxvalue, callback=func)
        widgets.append(s)
    display(*widgets)


def guessParamSpecs(params: dict[str, float | int | str | None],
                    ranges: dict[str, tuple[float, float]] = None
                    ) -> list[ParamSpec]:
    paramspecs: list[ParamSpec] = []
    for paramname, value in params.items():
        if isinstance(value, str):
            continue
        elif value is None:
            value = 0.
        if ranges and paramname in ranges:
            minval, maxval = ranges[paramname]
        else:
            minval, maxval = _guessRange(value, paramname)
        paramspecs.append(ParamSpec(name=paramname,
                                    minvalue=minval,
                                    maxvalue=maxval,
                                    startvalue=value))
    return paramspecs


def interactSynth(synth: _synth.Synth | _synth.SynthGroup,
                  specs: list[ParamSpec] = None) -> None:
    """
    Interact with a Synth

    Args:
        synth: the synth for which to generate a UI
        specs: a list of ParamSpec
    """
    if not specs:
        dynparams = synth.dynamicParamNames(aliases=False)
        params = {param: synth.paramValue(param) for param in sorted(dynparams)}
        specs = guessParamSpecs(params=params)

    if emlib.envir.inside_jupyter():
        return _interactSynthJupyter(synth=synth, specs=specs)
    else:
        raise RuntimeError("interact is only supported inside a jupyter session at the"
                           " moment.")


def _interactSynthJupyter(synth: _synth.Synth | _synth.SynthGroup,
                          specs: list[ParamSpec],
                          stopbutton=True,
                          width='80%'
                          ) -> None:
    import ipywidgets as ipy
    from IPython.display import display
    widgets = []
    if stopbutton:
        button = ipy.Button(description="Stop")
        button.on_click(lambda *args, s=synth: s.stop())
        widgets.append(button)

    for spec in specs:
        if spec.widgetHint == 'slider':
            w = _jupyterSlider(name=spec.name,
                               startvalue=spec.startvalue,
                               minvalue=spec.minvalue,
                               maxvalue=spec.maxvalue,
                               width=width,
                               callback=lambda val, s=synth, p=spec.name: s.set(p, value=val))
        elif spec.widgetHint == 'entry':
            w = _jupyterEntry(name=spec.name,
                              startvalue=spec.startvalue,
                              minvalue=spec.minvalue,
                              maxvalue=spec.maxvalue,
                              callback=lambda val, s=synth, p=spec.name: s.set(p, value=val))
        else:
            raise ValueError(f"Widget hint not understood: {spec.widgetHint}")
        widgets.append(w)
    display(*widgets)


def interactPargs(engine: engine.Engine,
                  p1: float | str,
                  specs: dict[int|str, ParamSpec] = {},
                  **namedSpecs: ParamSpec):
    """
    Interact with pfields of an event

    Depending on the context this will create a set of sliders to
    interact with the dynamic pfields of a running event

    Example
    =======

    .. code::

        from csoundengine import *
        from csoundengine.interact import *
        e = Engine()
        e.compile(r'''
        instr 100
          kamp = p4
          kmidi = p5
          a0 vco2 kamp, lag:k(mtof:k(kmidi), 0.1)
          outch 1, a0
        ''')
        eventid = e.sched(100, args=[0.1, 67])
        interactPfields(e, eventid,
                        specs={4: ParamSpec("kamp", 0, 1),
                               5: ParamSpec("kmidi", 0, 127)})
        # This is the same:
        interactPfields(e, eventid,
                        p4=ParamSpec("kamp", 0, 1),
                        p5=ParamSpec("kmidi", 0, 127))

    """
    allspecs = {}
    if specs:
        allspecs.update(specs)
    if namedSpecs:
        allspecs.update(namedSpecs)
    if emlib.envir.inside_jupyter():
        return _jupyInteractPargs(engine=engine, p1=p1, specs=allspecs)
    else:
        raise RuntimeError("interact is only supporte inside a jupyter session at the"
                           " moment.")


def _jupyInteractPargs(engine: engine.Engine,
                       p1: float|str,
                       specs: dict[int|str, ParamSpec],
                       stopbutton=True,
                       width='80%'):
    """
    Create a jupyter interactive widget for this event

    .. note::

        This function should only be called inside a jupyter session

    Example
    =======

    .. code::

        from csoundengine import *
        e = Engine()
        e.compile(r'''
        instr 100
          kamp = p4
          kmidi = p5
          a0 vco2 kamp, lag:k(mtof:k(kmidi), 0.1)
          outch 1, a0
        endin
        ''')
        event = e.sched(100, args=[0.1, 67])
        e.interact(event, p4=ParamSpec('kamp', 0, 1){4: ParamSpec('kamp', 0, 1)})
    """
    import ipywidgets as ipy
    from IPython.display import display
    widgets = []
    if stopbutton:
        button = ipy.Button(description="Stop")
        button.on_click(lambda *args, e=engine, p1=p1: e.unsched(p1))
        widgets.append(button)

    for key, spec in specs.items():
        idx = key if isinstance(key, int) else int(key[1:])
        if isinstance(p1, str):
            p1 = engine.queryNamedInstr(p1)
        currentvalue = engine._getp(p1, idx)
        value0 = currentvalue if currentvalue is not None else spec.startvalue
        if value0 is None:
            value0 = spec.minvalue
        if spec.widgetHint == 'slider':
            w = _jupyterSlider(spec.description, startvalue=value0,
                               minvalue=spec.minvalue, maxvalue=spec.maxvalue,
                               width=width,
                               callback=lambda val, p1=p1, idx=idx: engine.setp(p1, idx, val))
        elif spec.widgetHint == 'entry':
            w = _jupyterEntry(spec.description, startvalue=value0,
                              minvalue=spec.minvalue, maxvalue=spec.maxvalue,
                              callback=lambda val, p1=p1, idx=idx: engine.setp(p1, idx, val))
        else:
            raise ValueError(f"Widget hint not understood: {spec.widgetHint}")
        widgets.append(w)
    display(*widgets)
