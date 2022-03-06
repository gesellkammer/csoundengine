import time


def waitWithAnimation(waittime: float, dt=0.1) -> None:
    import progressbar
    widgets = ['Restarting ', progressbar.AnimatedMarker(markers='◢◣◤◥')]
    bar = progressbar.ProgressBar(widgets=widgets)
    for i in bar((i for i in range(int(waittime / dt)))):
        time.sleep(dt)