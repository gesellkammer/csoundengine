"""
Utilities for user interfaces at the terminal
"""
from __future__ import annotations
import time


def waitWithAnimation(waittime: float, dt=0.1) -> None:
    """
    Show a waiting animation at the terminal (blocking)

    Args:
        waittime: the total time to wait
        dt: interval at which the animation is updated
    """
    # Uses progressbar2
    import progressbar
    widgets = ['Restarting ', progressbar.AnimatedMarker(markers='◢◣◤◥')]
    bar = progressbar.ProgressBar(widgets=widgets)
    for i in bar((i for i in range(int(waittime / dt)))):
        time.sleep(dt)

