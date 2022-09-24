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



