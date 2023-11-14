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
    Instr('.testAudio', body=r'''
        |imode=0, iperiod=1, igain=0.1|
        kchan init -1
        if imode == 0 then
            prints "\nTestaudio: pink noise mode\n"
            a0 pinker
            a0 *= igain
        elseif imode == 1 then
            prints "\nTestaudio: sine tone mode\n"
            a0 oscili igain, 1000
        else
            initerror sprintf("testaudio: imode %d unknown", imode)
        endif
        kswitch metro 1/iperiod
        kchan = (kchan + kswitch) % nchnls
        outch kchan+1, a0
        if kswitch == 1 then
            println "Channel: %d / %d", kchan+1, nchnls
        endif
    '''),
    Instr('.playSample', body=r"""
        |isndtab=0, istart=0, ifadein=0, ifadeout=0, kchan=1, kspeed=1, kgain=1, kpan=0.5, ixfade=-1|
        ; Play a sample loaded via GEN01
        ; Args:
        ;   istart: the start time within the sample
        ;   ifade: fade in / out
        ;   kchan: output channel
        ;   kspeed: playback speed
        ;   kgain: gain
        ;   kpan: pan position, between 0-1. Use -1 to use default, which is 0 for mono and 0.5 for stereo
        ;   ixfade: crossfade time, if negative no looping
        iloop = ixfade >= 0 ? 1 : 0
        ionecycle = ksmps/sr
        ifadein = max(ifadein, ionecycle)
        ifadeout = max(ifadeout, ionecycle)
        inumouts = ftchnls(isndtab)
        inumsamples = nsamp(isndtab)
        isr = ftsr(isndtab)
        
        if isr <= 0 then
            initerror sprintf("Could not determine sr of table %d", isndtab)
        endif
        idur = inumsamples / isr
        
        know init istart
        if inumouts == 0 then
            ; not a gen1 table, fail
            initerror sprintf("Table %d was not generated via gen1", isndtab)
        endif

        kidx init 0
        aenv = linsegr:a(0, ifadein, 1, ifadeout, 0)
        aenv *= interp(kgain)
        
        if inumouts == 1 then
            ; a1 flooper2 1, kspeed, istart, idur, ixfade, isndtab, istart
            a1 flooper2 1, kspeed, istart, idur, ixfade, isndtab, istart
            a1 *= aenv
            aL, aR pan2 a1, kpan
            outch kchan, aL, kchan+1, aR
        elseif inumouts == 2 then
            a1, a2 flooper2 1, kspeed, istart, idur, ixfade, isndtab, istart
            ; a1, a2 loscil3 1, ispeed*kspeed, isndtab, 1, iloop
            a1 *= aenv
            a2 *= aenv
            kL, kR _panweights kpan
            if kL != 1 then
                a1 *= kL
                a2 *= kR
            endif
            outch kchan, a1, kchan+1, a2
        else
            ; TODO: test samplerate speed compensation
            ; 4: cubic interpolation
            ; 1: ibas, base frequency
            ; NB: loscilx has no crossfade
            aouts[] loscilx 1, kspeed, isndtab, 4, 1, iloop
            aouts *= aenv
            kidx = 0
            while kidx < inumouts do
                outch kchan+kidx, aouts[kidx]
                kchan += 1
            od
        endif   
        know += ionecycle * kspeed
        imaxtime = idur - ifadeout - ionecycle
        if iloop == 0 && know >= imaxtime then
            turnoff
        endif   
    """, aliases={'speed': 'kspeed', 'gain': 'kgain', 'pan': 'kpan'}),
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
        """),
    Instr('.playPartials', body=r'''
        |ifn, iskip=-1, inumrows=0, inumcols=0, kspeed=1, kloop=0, kminfreq=0, kmaxfreq=0, iflags=0, istart=0, istop=0, kfreqscale=1, ichan=1, kbwscale=1., kgain=1., iposition=0.|
        ifade = 0.02
        kplayhead init istart
        
        if iskip == -1 then    
            iskip      tab_i 0, ifn
            inumrows   tab_i 1, ifn
            inumcols   tab_i 2, ifn
        endif
        
        it0 = tab_i(iskip, ifn)
        it1 = tab_i(iskip+inumcols, ifn)
        idt = it1 - it0
        inumpartials = (inumcols-1) / 3 
        
        imaxrow = inumrows - 2
        it = ksmps / sr
        idur = imaxrow * idt
        istop = istop > 0 ? istop : idur
        
        ; prints "skip: %d, numcols: %d, numrows: %d, idt: %f \n", iskip, inumcols, inumrows, idt
        
        krow = kplayhead / idt
        
        ; each row has the format frametime, freq0, amp0, bandwidth0, freq1, amp1, bandwidth1, ...
        kF[] getrowlin krow, ifn, inumcols, iskip, 1, 0, 3
        kA[] getrowlin krow, ifn, inumcols, iskip, 2, 0, 3
        kB[] getrowlin krow, ifn, inumcols, iskip, 3, 0, 3
        kSel[] init inumcols / 3
        
        if kmaxfreq > 0 || kminfreq > 0 then
          kmaxfreq = kmaxfreq > 0 ? kmaxfreq : sr / 2
          kSel cmp kminfreq, "<=", kF, "<=", kmaxfreq
          kA *= kSel
        endif
        
        aout beadsynt kF, kA, kB, -1, iflags, kfreqscale, kbwscale
        
        aenv cossegr 0, ifade, 1, ifade, 0
        if kgain != 1 then
            aenv *= kgain
        endif
        aout *= aenv
        
        if iposition == 0  then
            outch ichan, aout
        else
            aL, aR pan2 aout, iposition
            outch ichan, aL, ichan+1, aR
        endif
        
        kplayhead += ksmps/sr * kspeed
        if kplayhead >= istop then
          kplayhead = istart
          if kloop == 0 && release() == 0 then  
            turnoff
          endif
        endif  
    '''),
    Instr('.dummy', body="")
]

builtinInstrIndex = {instr.name: instr for instr in builtinInstrs}