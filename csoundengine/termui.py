"""
Utilities for user interfaces at the terminal
"""
from __future__ import annotations
import time


def waitWithAnimation(label: str, waittime: float, dt=0.1) -> None:
    """
    Show a waiting animation at the terminal (blocking)

    Args:
        label: a label to show
        waittime: the total time to wait
        dt: interval at which the animation is updated
    """
    # Uses progressbar2
    import progressbar
    if not label[-1].isspace():
        label += ' '
    widgets = [label, progressbar.AnimatedMarker(markers='◢◣◤◥')]
    bar = progressbar.ProgressBar(widgets=widgets)
    for i in bar((i for i in range(int(waittime / dt)))):
        time.sleep(dt)

