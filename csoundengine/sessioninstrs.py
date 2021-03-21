from .instr import Instr

builtinInstrs = [
    Instr('.sinegliss', body="""
        ioutchan, iAmp, iPitchStart, iPitchEnd passign 5
        kmidi linseg iPitchStart, p3, iPitchEnd
        kfreq = mtof:k(kmidi)
        aenv linsegr 0, 0.01, 1, 0.05, 0
        a0 oscili iAmp, kfreq
        a0 *= aenv
        outch ioutchan, a0
        """),
    Instr('.sine', body="""
        ;                amp  midi pos  chan
        pset 0, 0, 0, 0, 0.5, 60,  0.5, 0
        kamp = p5
        kmidi = p6
        kpos = p7
        ichan = p8
        aenv linsegr 0, 0.04, 1, 0.08, 0
        a0 oscil interp(kamp), mtof:k(kmidi)
        a0 *= aenv
        aL, aR pan2 a0, kpos
        outch ichan, aL, ichan+1, aR
        """),
    Instr('.playSample',
          body=r"""
        pset 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, -1
        isndtab, iloop, istart, ifade, igaingroup passign 5
        kchan = p10
        kspeed = p11
        kgain = p12
        kpan = p13

        ifade = ifade < 0 ? 0.05 : ifade
        igaingroup = limit(igaingroup, 0, 100)
        inumouts = ftchnls(isndtab)
        inumsamples = nsamp(isndtab)
        isr = ftsr(isndtab)
        idur = inumsamples / isr
        ixfade = 0.005
        know init istart
        ksubgain = table:k(igaingroup, gi__subgains)

        if inumouts == 0 then
            ; not a gen1 table, fail
            initerror sprintf("Table %d was not generated via gen1", isndtab)
        endif

        kidx init 0
        aenv = linsegr:a(0, ifade, 1, ifade, 0)
        aenv *= a(ksubgain*kgain)
        if inumouts == 1 then
            ; a1 loscil3 1, kspeed, isndtab, 1, iloop
            ; asig1[,asig2] flooper2 kamp, kpitch, kloopstart, kloopend, kcrossfade, ifn \
            ; [, istart, imode, ifenv, iskip]
            a1 flooper2 1, kspeed, istart, idur, ixfade, isndtab, istart
            a1 *= aenv
            kpan = kpan == -1 ? 0 : kpan
            aL, aR pan2 a1, kpan
            outch kchan, aL, kchan+1, aR
        elseif inumouts == 2 then
            a1, a2 loscil3 1, kspeed, isndtab, 1, iloop
            a1 *= aenv
            a2 *= aenv
            kpan = kpan == -1 ? 0.5 : kpan
            aL, aR _panstereo a1, a2, kpan
            outch kchan, aL, kchan+1, aR
        else
            ; 4: cubic interpolation
            aouts[] loscilx 1, kspeed, isndtab, 4, 0, iloop
            aouts *= aenv
            ichan = p(10)
            ichans[] genarray ichan, ichan+inumouts-1
            poly0 inumouts, "outch", ichans, aouts
        endif   
        ionecycle = ksmps/sr
        know += ionecycle * kspeed
        imaxtime = idur - ifade - ionecycle
        if iloop == 0 && know >= imaxtime then
            turnoff
        endif
        """),
    Instr('.playbuf', body="""
        pset 0, 0, 0, 0, 0, 1, 1, 0
        itabnum, ioutchan, igain, iloop passign 5
        inumsamps ftlen itabnum
        idur = inumsamps / ftsr(itabnum)

        if (iloop == 1) then
            iPitch = 1
            iCrossfade = 0.050
            aSig flooper2 igain, iPitch, 0, idur, iCrossfade, itabnum
        else
            iReadfreq = sr / inumsamps; frequency of reading the buffer
            aSig poscil3 igain, iReadfreq, itabnum
            iatt = 1/kr
            irel = 3*iatt
            aEnv linsegr 0, iatt, 1, idur-iatt*2, 1, iatt, 0
            aSig *= aEnv
            if timeinsts() > idur then
                turnoff
            endif
        endif
        outch ioutchan, aSig
        """)
]