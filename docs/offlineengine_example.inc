.. code-block:: python

    from csoundengine import OfflineEngine
    import random

    engine = OfflineEngine()

    engine.compile(r'''
    instr vco
        pset 0, 0, 0, 263, 0.1, 1, 1000, 0.1, 0.5
        ifreq = p4
        iamp = p5
        ktransp = p6
        kcutoff = p7
        ifade = p8
        ifilterq = p9
        kfreq = ifreq * ktransp
        aout = vco2:a(iamp, kfreq)
        iresonance = 0.6
        idistortion = 0.
        aout = lpf18(aout, kcutoff, iresonance, idistortion)
        aout *= linsegr:a(0, ifade, 1, ifade, 0)
        outch 1, aout
    endin
    ''')

    f0 = 130
    for factor in [1, 2.76, 5.3, 5.4, 8.93]:

        freq = f0 * factor
        # with unique=True a fractional p1 is generated, this makes it
        # possible to automate the event later
        # NB: csoundengine parses instr code and is aware of names given to pfields
        event = engine.sched('vco', engine.now, 8, unique=True,
                            ifreq=freq, iamp=0.1, kcutoff=freq*3, ifade=freq/f0* 0.5, ifilterq=0.8)
        transp = random.uniform(0.92, 1.08)
        # start automation of p6 at time=2. This case, modify the transposition factor,
        # from 1 to a random value between 0.92 and 1.08
        engine.automatep(event, 6, (0, 1, 5, transp), delay=2)
        # The pfield can also be given by name. With overtake=True the value given
        # to the pfield at init is used as the starting point
        engine.automatep(event, 'kcutoff', (0, 0, 3, freq*1.1), overtake=True, delay=2 + factor*0.3)

    # Advance time to actually perform the events. We give some extra time
    # to account for fade outs.
    engine.perform(extratime=5)
    engine.stop()

.. image:: assets/offlineengine-example1.png

.. raw:: html

    <audio controls="controls">
      <source src="_static/offlineengine-example1.mp3" type="audio/mpeg">
      Your browser does not support the <code>audio</code> element.
    </audio>
