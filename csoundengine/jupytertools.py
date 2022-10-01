from __future__ import annotations
import emlib.misc as _misc
from typing import Callable

if _misc.inside_jupyter():
    import ipywidgets as _ipywidgets
    from IPython.display import display as _ipythonDisplay


lightPalette = {
    'name.color': 'MediumSeaGreen'
}

darkPalette = lightPalette.copy()

defaultPalette = lightPalette

palettes = {
    'light': lightPalette,
    'dark': darkPalette
}


def displayButton(buttonText: str, callback: Callable[[], None]
                  ) -> None:
    """
    Create and display an html button inside a jupyter notebook

    Args:
        buttonText: the text of the button
        callback: the function to call when the button is pressed. This function
            takes no arguments and should not return anything
    """
    assert _misc.inside_jupyter(), ("This function is only available when"
                                         "running inside a jupyter notebook")
    button = _ipywidgets.Button(description=buttonText)
    output = _ipywidgets.Output()

    def clicked(b):
        with output:
            callback()

    button.on_click(clicked)
    _ipythonDisplay(button, output)


def htmlName(text: str, palette='light') -> str:
    colors = palettes.get(palette)
    return f'<strong style="color:{colors["name.color"]}">{text}</strong>'


safeColors = {
    'blue1': '#9090FF',
    'blue2': '#6666E0',
    'red1': '#FF9090',
    'red2': '#E08080',
    'green1': '#90FF90',
    'green2': '#8080E0',
    'magenta1': '#F090F0',
    'magenta2': '#E080E0',
    'cyan': '#70D0D0',
    'grey1': '#BBBBBB',
    'grey2': '#A0A0A0',
    'grey3': '#909090'
}


def htmlSpan(text, color: str = '', fontsize: str = '', italic=False, bold=False) -> str:
    if color.startswith(':'):
        color = safeColors[color[1:]]
    styleitems = {}
    if color:
        styleitems['color'] = color
    if fontsize:
        styleitems['font-size'] = fontsize
    stylestr = ";".join(f"{k}:{v}" for k, v in styleitems.items())
    text = str(text)
    if italic and bold:
        text = f'<i><b>{text}</b></i>'
    elif italic:
        text = f'<i>{text}</i>'
    elif bold:
        text = f'<b>{text}</b>'
    return f'<span style="{stylestr}">{text}</span>'