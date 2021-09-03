from .instr import Instr

builtinInstrs = [
    Instr('.sine', body="""
        |kamp=0.5, kmidi=60, kpos=0.5, ichan=0|
        aenv linsegr 0, 0.04, 1, 0.08, 0
        a0 oscil interp(kamp), mtof:k(kmidi)
        a0 *= aenv
        aL, aR pan2 a0, kpos
        outch ichan, aL, ichan+1, aR
        """),
    Instr('.playSample',
          body=r"""
        |isndtab=0, iloop=0, istart=0, ifade=0, igaingroup=0, kchan=1, kspeed=1, kgain=1, kpan=-1|
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
            a1 flooper2 1, kspeed, istart, idur, ixfade, isndtab, istart
            a1 *= aenv
            kpan = kpan == -1 ? 0 : kpan
            aL, aR pan2 a1, kpan
            outch kchan, aL, kchan+1, aR
        elseif inumouts == 2 then
            a1, a2 loscil3 1, kspeed, isndtab, 1, iloop
            a1 *= aenv
            a2 *= aenv
            kpan = kpan < 0 ? 0.5 : kpan
            kL, kR _panweights kpan
            a1 *= kL
            a2 *= kR
            outch kchan, a1, kchan+1, a2
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
        |itabnum=0, ioutchan=1, igain=1, iloop=0|
        inumsamps ftlen itabnum
        idur = inumsamps / ftsr(itabnum)

        if (iloop == 1) then
            iPitch = 1
            iCrossfade = 0.050
            aSig flooper2 igain, iPitch, 0, idur, iCrossfade, itabnum
            outch ioutchan, aSig
        else
            iReadfreq = sr / inumsamps; frequency of reading the buffer
            aSig poscil3 igain, iReadfreq, itabnum
            iatt = 1/kr
            irel = 3*iatt
            aEnv linsegr 0, iatt, 1, idur-iatt*2, 1, iatt, 0
            aSig *= aEnv
            outch ioutchan, aSig
            if timeinsts() > idur then
                turnoff
            endif
        endif
        """)
]