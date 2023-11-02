from __future__ import annotations
from emlib.envir import inside_jupyter
from typing import Callable

if inside_jupyter():
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

    If not inside a jupyter notebook this function will raise RuntimeError

    Args:
        buttonText: the text of the button
        callback: the function to call when the button is pressed. This function
            takes no arguments and should not return anything
    """
    if not inside_jupyter():
        raise RuntimeError("This function is only available when running inside a jupyter notebook")

    button = _ipywidgets.Button(description=buttonText)
    output = _ipywidgets.Output()

    def clicked(b):
        with output:
            callback()

    button.on_click(clicked)
    _ipythonDisplay(button, output)


def htmlName(text: str, palette='light') -> str:
    """A name as html

    It will use the colors as defined in the palette
    """
    colors = palettes.get(palette)
    if not colors:
        raise ValueError(f"palette {palette} not known. Possible palettes: {palettes.keys()}")
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


def htmlSpan(text, color='', fontsize='', italic=False, bold=False, tag='span') -> str:
    """
    Create a html span with the given attributes

    Args:
        text: the text inside the span
        color: a valid css color. If it starts with ':', one of the 'safe' colors as
            defined in `safeColors` (blue1, blue2, red1, red2, green1, green2, cyan, grey1, grey2,
            grey3, magenta1, magenta2). For example, ':blue1' will use the blue1 safe color
        fontsize: a valid size, such as '12px'.
        italic: if True, use italic text
        bold: if True, use bold text
        tag: the tag to use (span by default, can be code, for example)

    Returns:
        the resulting html

    """
    text = str(text)
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
    return f'<{tag} style="{stylestr}">{text}</{tag}>'
