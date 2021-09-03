import emlib.misc

if emlib.misc.inside_jupyter():
    import ipywidgets as _ipywidgets
    from IPython.display import display as _ipythonDisplay


lightStyle = {
    'name.color': 'MediumSeaGreen'
}

darkStyle = lightStyle.copy()

defaultStyle = lightStyle

colorStyles = {
    'light': lightStyle,
    'dark': darkStyle
}


def displayButton(buttonText: str, callback):
    """
    Create and display an html button inside a jupyter notebook

    Args:
        buttonText: the text of the button
        callback: the function to call when the button is pressed. This function
            takes no arguments and should not return anything
    """
    assert emlib.misc.inside_jupyter()
    button = _ipywidgets.Button(description=buttonText)
    output = _ipywidgets.Output()

    def clicked(b):
        with output:
            callback()

    button.on_click(clicked)
    _ipythonDisplay(button, output)



def htmlName(text: str) -> str:
    style = defaultStyle
    return f'<strong style="color:{style["name.color"]}">{text}</strong>'



