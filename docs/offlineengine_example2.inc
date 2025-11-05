.. code-block:: python

    from csoundengine.offline import OfflineEngine
    e = OfflineEngine(outfile="pitchtrack.wav")
    e.compile('''
    instr pitchtrack
        isndtable, iouttable passign 4
        kfreq init 100

        achannels[] = loscilx(1, 1, isndtable, 4, 1)
        a0 = achannels[0]
        icutoff = 2000
        afeat = butlp(a0, icutoff)
        afeat = butlp(afeat, icutoff)
        fsig = pvsanal(afeat, 1024, 256, 1024, 0)
        fsig2 = pvsanal(a0, 1024, 256, 1024, 0)
        kflatness = pow(pvsflatness:k(fsig2, 50, 10000), 1.5)
        kpeakyness = pvsmagsumn(fsig2, 30, 40) / max(0.0000001, pvsmagsum(fsig2))
        knoisyness = sqrt(kflatness*kflatness + pow(1-kpeakyness, 2))
        kfreq1, kamp pvspitch fsig, ampdb(-30)
        afreq2, alock plltrack afeat, 0.06, 18, 0.3, 30, 900
        kfreq2 = downsamp(afreq2, ksmps)

        knoisegate = schmitt(knoisyness, 0.5, 0.4)
        if knoisegate < 0.5 then
            kfreq = abs(kfreq1 - kfreq) < abs(kfreq2 - kfreq) ? kfreq1 : kfreq2
        endif
        kfreq = lagud(kfreq, 0.01, 0.05)
        kcount = eventcycles()
        kidx = kcount * 3 + 1
        ktime = eventtime()
        knoisegate *= kfreq > 0 ? 1 : 0

        tabw ktime, kidx, iouttable
        tabw kfreq, kidx+1, iouttable
        tabw knoisegate, kidx+2, iouttable
        tabw kcount, 0, iouttable

        aenv = lagud:a(interp(1 - knoisegate), 0.03, 0.01)
        aenv *= aenv
        asig = buzz(aenv, kfreq, 7, -1)
        outch 1, delay(a0, 1024 / sr)
        outch 2, asig * interp(kamp) * 1.2
    endin
    ''')

    sndtab = e.readSoundfile("../snd/voiceover-fragment-48k.flac")
    sndinfo = e.tableInfo(sndtab)
    duration = sndinfo.duration
    numcycles = int(duration * e.sr / e.ksmps)
    numcols = 3  # The instrument saves 3 columns of information per cycle (times, freqs, locks)
    freqtab = e.makeEmptyTable((numcycles + 100) * numcols)
    e.sched('pitchtrack', 0, duration + 0.1, sndtab, freqtab)
    e.perform()

    # Now retrieve information, accesses the table created before
    outarr = e.tableData(freqtab)
    # The first index keeps the number of cycles analyzed
    datalen = int(outarr[0])
    times = outarr[1:1+datalen*numcols:numcols]
    freqs = outarr[2:1+datalen*numcols:numcols]
    gate = outarr[3:1+datalen*numcols:numcols]

.. code-block:: python

    import matplotlib.pyplot as plt
    import csoundengine.plotting

    samples = e.tableData(sndtab)
    sr = e.tableInfo(sndtab).sr

    fig, ax = plt.subplots(1, 1, figsize=(24, 8))
    csoundengine.plotting.plotSpectrogram(samples, sr, axes=ax, maxfreq=2000)
    freqs2 = freqs * (gate < 0.5)

    freqs2[freqs2 > 1500] = 0
    freqs2[freqs2 == 0] = float('nan')
    ax.plot(times - 0.02, freqs2)
