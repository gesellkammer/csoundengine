from __future__ import annotations
import dataclasses
import emlib.misc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import *
    from . import engine
    import ipywidgets as ipy


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


def _guessRange(value: float, namehint: str = '') -> Tuple[float, float]:
    if namehint:
        if "freq" in namehint and 0 < value < 12000:
            return (0, 12000)
        if "amp" in namehint and 0 <= value <= 1:
            return (0, 1)
        if "cutoff" in namehint and 0 < value < 12000:
            return (0, 12000)
        if "midi" in namehint and 0 <= value <= 127:
            return (0, 127)
    if value < 0:
        val0, val1 = _guessRange(-value)
        return -val1, -val0
    if value < 1:
        return (0, 10)
    elif value < 50:
        return (0, 500)
    elif value < 100:
        return (0, 1000)
    else:
        return (0, value * 10)


@dataclasses.dataclass
class ParamSpec:
    descr: str
    minvalue: float
    maxvalue: float
    step: float = -1
    startvalue: Optional[float] = None
    widgetHint: str = 'slider'

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


def _jupySlider(name:str, startvalue: float, minvalue: float, maxvalue: float,
                callback:Callable, step:float=None, width='80%', log=False):
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
    if callback:
        s.observe(lambda change:callback(change['new']), names='value')
    return s


def _jupyEntry(name: str, startvalue: float, minvalue:float, maxvalue: float,
               callback: Callable):
    import ipywidgets as ipy
    step = 0.001
    w = ipy.BoundedFloatText(value=startvalue, min=minvalue, max=maxvalue,
                             step=step, description=name)
    if callback:
        w.observe(lambda change:print(change), names='value')
    return w


def interact(**sliders: Dict[str, Tuple[float, float, float, Callable]]):
    """
    Creates a set of interactive widgets

    Args:
        sliders: given as keywords. The key is used as the widget name. The
            value is a tuple (initialvalue, minvalue, maxvalue, callback)
            where callback has the form func(x) -> None and will be called
            with the current value of the widget

    Example
    =======

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
        sliders = makeSliders(kamp=(e.getp(p1, 4, 0, 1, lambda x:e.setp(p1, 4, x),
                              kmidi=(e.getp(p1, 5, 0, 127, lambda x:e.setp(p1, 5, x))
        display(*sliders)
    """
    from IPython.display import display
    widgets = []
    for key, value in sliders.items():
        curvalue, minvalue, maxvalue, func = value
        s = _jupySlider(name=key, startvalue=curvalue, minvalue=minvalue,
                        maxvalue=maxvalue, callback=func)
        widgets.append(s)
    display(*widgets)


def interactPargs(engine: engine.Engine, p1: Union[float, str],
                  specs: Dict[Union[int, str], ParamSpec]=None,
                  **namedSpecs):
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
    if emlib.misc.inside_jupyter():
        return _jupyInteractPargs(engine=engine, p1=p1, specs=allspecs)
    else:
        raise RuntimeError("interact is only supporte inside a jupyter session at the"
                           " moment.")


def _jupyInteractPargs(engine: engine.Engine, p1: Union[float, str],
                       specs: Dict[Union[int, str], ParamSpec]=None,
                       stopbutton=True,
                       width='80%'):
    """

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
    from IPython.display import display
    import ipywidgets as ipy
    widgets = []
    if stopbutton:
        button = ipy.Button(description="Stop")
        button.on_click(lambda *args, e=engine, p1=p1: e.unsched(p1))
        widgets.append(button)

    for key, spec in specs.items():
        idx = key if isinstance(key, int) else int(key[1:])
        value0 = engine.getp(p1,idx) if spec.startvalue is None else spec.startvalue
        if spec.widgetHint == 'slider':
            w = _jupySlider(spec.descr, startvalue=value0,
                            minvalue=spec.minvalue, maxvalue=spec.maxvalue,
                            width=width,
                            callback=lambda val, p1=p1, idx=idx: engine.setp(p1, idx, val))
        elif spec.widgetHint == 'entry':
            w = _jupyEntry(spec.descr, startvalue=value0,
                           minvalue=spec.minvalue, maxvalue=spec.maxvalue,
                           callback=lambda val, p1=p1, idx=idx: engine.setp(p1, idx, val))
        else:
            raise ValueError(f"Widget hint not understood: {spec.widgetHint}")
        widgets.append(w)
    display(*widgets)

