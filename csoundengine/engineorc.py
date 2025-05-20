from __future__ import annotations

from functools import cache
from string import Template

from . import internal
import typing as _t

# In all templates we make the difference between substitutions which
# are constant (with the form ${subst} and substitutions which respond
# to the configuration of a specific engine / offline renderer ({subst})

_orc = r'''

${includes}

gi__responses       ftgen  ${responses}, 0, ${numTokens}, -2, 0
gi__tokenToInstrnum ftgen ${tokenToInstrnum}, 0, ${maxNumInstrs}, -2, 0
gi__soundfontIndexes dict_new "str:float"
gi__soundfontIndexCounter init 1000
gi__builtinInstrs dict_new "str:float", "notifyDealloc", ${notifyDealloc}

chn_k "_soundfontPresetCount", 3

; ---------------------------------
;          builtin-instruments
; ---------------------------------

opcode _panweights, kk, k
    kpos xin
    kampL = bpf:k(kpos, 0, 1.4142, 0.5, 1, 1, 0)
    kampR = bpf:k(kpos, 0, 0,      0.5, 1, 1, 1.4142)
    xout kampL, kampR; inum nstrnum Sname
endop

opcode namedinstrtofrac, i, S
    ; similar to nametoinstrnum but takes the fractional part into account
    ; given "foo.003", if foo is instr 35, then this returns 35.003
    Sname xin
    indx strindex Sname, "."
    if indx < 0 then
        ip1 = nametoinstrnum(Sname)
    else
        Sbase = strsub(Sname, 0, indx)
        Sfrac = strsub(Sname, indx)
        ip1 = nametoinstrnum(Sbase)
        if ip1 > 0 then
            ip1 += strtod(Sfrac)
        endif
    endif
    xout ip1
endop

opcode sfloadonce, i, S
    Spath xin
    iidx dict_get gi__soundfontIndexes, Spath, -1
    if (iidx == -1) then
        iidx sfload Spath
        dict_set gi__soundfontIndexes, Spath, iidx
    endif
    xout iidx
endop

opcode sfpresetindex, i, Sii
    Spath, ibank, ipresetnum xin
    isf sfloadonce Spath
    Skey sprintf "SFIDX:%d:%d:%d", isf, ibank, ipresetnum
    iidx dict_get gi__soundfontIndexes, Skey, -1
    if iidx == -1 then
        iidx chnget "_soundfontPresetCount"
        chnset iidx+1, "_soundfontPresetCount"
        i0 sfpreset ipresetnum, ibank, isf, iidx
        if iidx != i0 then
            prints "???: iidx = %d, i0 = %d\n", iidx, i0
        endif
        dict_set gi__soundfontIndexes, Skey, i0
    endif
    xout iidx
endop

opcode sendsync, 0, io
    ; send a sync message back, to be used by instruments scheduled
    ; via schedSync
    itoken, ivalue xin
    tabw_i ivalue, itoken, gi__responses
    outvalue "__sync__", itoken
endop


instr ${notifyDealloc}
    outvalue "__dealloc__", p4
    turnoff
endin

instr ${notifyDeallocOsc}
    ; p4=p1, p5=port
    OSCsend 1, "127.0.0.1", p5, "/dealloc", "d", p4
    turnoff
endin

instr ${pingback}
    ; p4=token
    outvalue "__sync__", p4
endin

instr ${turnoff}
    iwhich = p4
    imode = p5   ; -1: turnoff future, otherwise it is the mode passed to turnoff2
    if qnan:i(iwhich) == 0 then
        if imode < 0 then
            turnoff3 iwhich
        else
            turnoff2_i iwhich, imode, 1
        endif
    else
        Swhich = p4
        if imode < 0 then
            turnoff3 Swhich
        else
            turnoff2_i Swhich, imode, 1
        endif
    endif
endin

instr ${print}
    Smsg = p4
    prints "csoundengine: '%s'\n", Smsg
endin

instr ${nstrnum}
    itoken = p4
    Sname = p5
    inum nametoinstrnum Sname
    tabw_i inum, itoken, gi__responses
    outvalue "__sync__", itoken
endin

instr ${tabwrite}
    itab = p4
    iidx = p5
    ival = p6
    tabw_i ival, iidx, itab
endin

instr ${chnset}
    Schn = p4
    ival = p5
    chnset ival, Schn
endin

instr ${initDynamicControls}
    pset 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    iindex = p4
    inumitems = p5
    itabnum chnget ".dynargsTabnum"
    ivalues[] passign 6, 6+inumitems
    copya2ftab ivalues, itabnum, iindex
    turnoff
endin

; can make a table with data or just empty of a given size
; data must be smaller than 2000 which is the current limit
; for pfields
instr ${maketable}
    ; args:
    ;   itoken: the token used to sync with the engine
    ;   itabnum: the table number (can be 0 to let csound assign a number)
    ;   ilen: the size of the table
    ;   iempty: should the table be empty or should it be filled with data?
    ;           in this case, the rest p-args should containt ilen datapoints
    itoken = p4
    itabnum = p5
    ilen = p6
    iempty = p7
    isr = p8
    inumchannels = p9
    if (iempty == 1) then
        itabnum ftgen itabnum, 0, -ilen, -2, 0
    else
        iValues[] passign 10, 10+ilen
        itabnum ftgen itabnum, 0, -ilen, -2, iValues
    endif
    if isr > 0 then
        ftsetparams itabnum, isr, inumchannels
    endif
    ; notify host that token is ready (if asked to)
    if itoken > 0 then
        tabw_i itabnum, itoken, gi__responses
        outvalue "__sync__", itoken
    endif
endin

instr ${automatePargViaPargs}
    ipindex = p5
    imode = p6;  interpolation method
    iovertake = p7
    ilenpairs = p8
    ip1string = p9
    if ip1string == 0 then
        ip1 = p4
    else
        Sp1 = p4
        ip1 = namedinstrtofrac(Sp1)
    endif

    ; special case: simple line, two pairs
    if ilenpairs == 4 && p10 == 0 && iovertake == 0 then
        iy0 = p11
        ix1 = p12
        iy1 = p13
        ky linseg iy0, ix1, iy1
        goto end
    endif

    ipairs[] passign 10, 10+ilenpairs
    iXs[] slicearray ipairs, 0, ilenpairs-1, 2
    iYs[] slicearray ipairs, 1, ilenpairs-1, 2
    Sinterpmethod = strget(imode)

    if iovertake == 1 then
        icurrval pread ip1, ipindex, -1
        iYs[0] = icurrval
    endif

    kt = eventtime()
    kidx bisect kt, iXs
    ky interp1d kidx, iYs, Sinterpmethod
end:
    pwrite ip1, ipindex, ky
endin

instr ${automateTableViaPargs}
    iargtab = p4
    iargidx = p5
    imode = p6
    iovertake = p7
    ilenpairs = p8

    if ftexists:i(iargtab) == 0 then
        initerror sprintf("automateTableViaPargs: table %d does not exist", iargtab)
        goto exit
    endif

    ipairs[] passign 9, 9+ilenpairs
    iXs[] slicearray ipairs, 0, ilenpairs-1, 2
    iYs[] slicearray ipairs, 1, ilenpairs-1, 2
    Sinterpmethod = strget(imode)

    if iovertake == 1 then
        icurrval = tab_i(iargidx, iargtab)
        iYs[0] = icurrval
    endif

    Smode = strget(imode)

    kt timeinsts
    kidx bisect kt, iXs
    ky interp1d kidx, iYs, Smode

    if ftexists:k(iargtab) == 1 then
        tablew ky, iargidx, iargtab
    else
        throwerror "warning", sprintf("Table (%d) was freed during automation, stopping automation", iargtab)
        turnoff
    endif
exit:
endin

instr ${pread}
    itoken = p4
    ip1 = p5
    ipnum = p6
    inotify = p7
    ival pread ip1, ipnum
    tabw_i ival, itoken, gi__responses
    if inotify == 1 then
        outvalue "__sync__", itoken
    endif
endin

instr ${uniqinstance}
    itoken = p4
    ip1    = p5
    iuniq uniqinstance ip1
    tabw_i iuniq, itoken, gi__responses
    outvalue "__sync__", itoken
endin

instr ${freetable}
    ifn = p4
    ftfree ifn, 0
endin

instr ${playsndfile}
    Spath  = p4
    kgain  = p5
    kspeed = p6
    ichan  = p7
    ifade  = p8
    idur = filelen(Spath)
    know init 0
    iwsize = 4   ; cubic interpolation
    aouts[] diskin2 Spath, kspeed, 0, 0, 0, iwsize
    inumouts = lenarray(aouts)
    ichans[] genarray ichan, ichan+inumouts-1
    aenv linsegr 0, ifade, 1, ifade, 0
    aenv *= interp(kgain)
    aouts = aouts * aenv
    kchan = 0
    while kchan < inumouts do
      outch ichan + kchan, aouts[kchan]
      kchan += 1
    od
    know += ksmps/sr * kspeed
    if know >= idur && release() == 0 then
      turnoff
    endif
endin

instr ${readSndfile}
    itoken = p4
    Spath = strget(p5)
    itab = p6
    ichan = p7
    iskiptime = p8
    itime0 = p9
    itab2 ftgen itab, 0, 0, -1, Spath, iskiptime, 0, ichan
    ; OSCsend 1, "127.0.0.1", 9990, "/sync", "f", itime0
    if itoken > 0 then
        tabw_i itab2, itoken, gi__responses
        outvalue "__sync__", itoken
    endif
    turnoff
endin

instr ${playgen1}
    ;             4     5      6       7     8     9          11
    ;             gain, speed, source, chan, fade, starttime, lagtime
    pset 0, 0, 0, 1,    1,     1,      1,    0.05, 0,         0.01
    kgain = p4
    kspeed = p5
    itabnum, ichan, ifade, ioffset, igaingroup, ilagtime passign 6
    ifade = max(ifade, ksmps/sr*2)
    inumsamples = nsamp(itabnum)
    itabsr = ftsr(itabnum)
    if itabsr <= 0 then
        prints "Could not determine sr for table %d, ftsr returned: %f, using sr: %d", itabnum, itabsr, sr
        itabsr = sr
    endif
    istartframe = ioffset * itabsr
    idur = inumsamples / itabsr
    kplayhead init ioffset
    iperiod = ksmps/sr
    kperiod = iperiod * kspeed
    kplayhead += kperiod

    if (release() == 0) && (kplayhead >= (idur-ifade-kperiod)) then
        turnoff
    endif

    ; ar[] loscilx xamp, kcps, ifn, iwsize, ibas, istrt
    aouts[] loscilx 1, kspeed, itabnum, 4, 1, istartframe
    inumouts = lenarray(aouts)
    aenv = linsegr:a(0, ifade, 1, ifade, 0)
    again = lag:a(a(kgain), ilagtime)
    aenv *= again
    aouts = aouts * aenv

    kchan = 0
    while kchan < inumouts do
        outch kchan+ichan, aouts[kchan]
        kchan += 1
    od

endin

instr ${ftsetparams}
    itabnum, isr, inumchannels passign 4
    ftsetparams itabnum, isr, inumchannels
endin

instr ${pwrite}
    itype = p4
    if itype == 1 then
        Sp1 = p5
        ip1 = namedinstrtofrac(Sp1)
    else
        ip1 = p5
    endif
    inumpairs = p6

    if inumpairs == 1 then
        pwrite ip1, p7, p8
    elseif inumpairs == 2 then
        pwrite ip1, p7, p8, p9, p10
    elseif inumpairs == 3 then
        pwrite ip1, p7, p8, p9, p10, p11, p12
    elseif inumpairs == 4 then
        pwrite ip1, p7, p8, p9, p10, p11, p12, p13, p14
    elseif inumpairs == 5 then
        pwrite ip1, p7, p8, p9, p10, p11, p12, p13, p14, 15, p16
    else
        initerror sprintf("Max. pairs is 5, got %d", inumpairs)
    endif
endin


instr ${testaudio}
    pset 0, 0, 0, 0, 1, 1, 1
    imode = p4
    iperiod = p5
    igain = p6
    iecho = p7

    kchan init -1
    if imode == 0 then
        if iecho == 1 then
            prints "\nTestaudio: pink noise mode\n"
        endif
        a0 pinker
        a0 *= igain
    elseif imode == 1 then
        if iecho == 1 then
            prints "\nTestaudio: sine tone mode\n"
        endif
        a0 oscili igain, 1000
    else
        initerror sprintf("testaudio: imode %d unknown", imode)
    endif
    kswitch metro 1/iperiod
    kchan = (kchan + kswitch) % nchnls
    outch kchan+1, a0
    if iecho * kswitch == 1 then
        println "Channel: %d / %d", kchan+1, nchnls
    endif
endin

instr ${tableInfo}
    itabnum, itok1, itok2, itok3, itok4 passign 4
    if itabnum == 0 || ftexists:i(itabnum) == 0 then
        tabw_i -1, itok1, gi__responses
        prints "tableInfo: warning: Table %d does not exist\n", itabnum
    else
        isr ftsr itabnum
        ichnls ftchnls itabnum
        inumframes nsamp itabnum
        ilen ftlen itabnum
        tabw_i isr, itok1, gi__responses
        tabw_i ichnls, itok2, gi__responses
        tabw_i inumframes, itok3, gi__responses
        tabw_i ilen, itok4, gi__responses
    endif
    outvalue "__sync__", itok1
endin

instr ${sfPresetAssignIndex}
    ; assign an index to a soundfont preset
    ipath, ibank, ipresetnum, iidx passign 4
    Spath strget ipath
    isf sfloadonce Spath
    i0 sfpreset ipresetnum, ibank, isf, iidx
endin

instr ${soundfontPlay}
    kpitch = p4
    kamp = p5
    ipresetidx, ivel, ichan passign 6
    inote = int(p4)
    aL, aR sfplay3 ivel, inote, kamp/16384, mtof:k(kpitch), ipresetidx, 1
    aenv linsegr 0, 0.01, 1, 0.1, 0
    kfinished_  trigger detectsilence:k(aL, 0.0001, 0.05), 0.5, 0
    if kfinished_ == 0  then
      aL *= aenv
      aR *= aenv
      outch ichan, aL, ichan+1, aR
    else
      turnoff
    endif
endin


instr ${dummy_post}
    ; this instrument is only here to prevent a crash
    ; when named instruments and numbered instruments
    ; are mixed in separate calls to compile
endin


chnset 1, "_soundfontPresetCount"

; ----------------------

${globalcode}

'''

_busOrc = r'''

gi__numControlBuses = ${numControlBuses}
gi__numAudioBuses = ${numAudioBuses}

#define _BUSUNSET #${BUSUNSET}#
#define _BUSKIND_AUDIO   #0#
#define _BUSKIND_CONTROL #1#

; The actual buses
ga__buses[] init gi__numAudioBuses
gi__bustable ftgen 0, 0, gi__numControlBuses, -2, 0

; This table keeps track of the number of references a bus has
gi__busrefs  ftgen 0, 0, ${numAudioBuses}, -2, 0
gi__busrefsk ftgen 0, 0, ${numControlBuses}, -2, 0

; A pool of bus indexes
gi__buspool  pool_gen ${numAudioBuses}
gi__buspoolk pool_gen ${numControlBuses}

; A dict mapping bustoken to bus number, used for both audio and scalar buses
gi__bustoken2num  dict_new "int:float"

; A dict mapping bustoken to bus kind (0=control, 1=audio). NB: tokens are unique,
; independently of the kind. No two buses of any kind should share a token
gi__bustoken2kind dict_new "int:float"

chn_k "_busTokenCount", 3

opcode _busnew, i, ii
    itoken, ikind xin
    if itoken < 0 then
        itoken chnget "_busTokenCount"
        chnset itoken+1, "_busTokenCount"
    endif
    ipool = ikind == $$_BUSKIND_AUDIO ? gi__buspool : gi__buspoolk
    ibus pool_pop ipool, -1
    if ibus == -1 then
        initerror "_busnew failed, out of buses"
    endif
    dict_set gi__bustoken2num, itoken, ibus
    dict_set gi__bustoken2kind, itoken, ikind
    xout ibus
endop

opcode _bususe, i, ii
    itoken, ikind xin
    ibus dict_get gi__bustoken2num, itoken, -1
    if ibus < 0 then
        ; initerror sprintf("Bus not found (token %d)\n", itoken)
        ibus _busnew itoken, ikind
        prints "Bus not found (token %d, kind=%d). Assigned bus %d\n", itoken, ikind, ibus
    else
        ikind2 dict_get gi__bustoken2kind, itoken, -1
        if ikind != ikind2 then
            initerror sprintf("Bus kind mismatch, asked for %d but the bus seems to be of kind %d", ikind, ikind2)
        endif
    endif

    if ikind == $$_BUSKIND_CONTROL && ibus >= gi__numControlBuses then
        initerror sprintf("Invalid control bus (%d) for token %d", ibus, itoken)
    elseif ikind == $$_BUSKIND_AUDIO && ibus >= gi__numAudioBuses then
        initerror sprintf("Invalid audio bus (%d) for token %d", ibus, itoken)
    endif

    itab = ikind == $$_BUSKIND_AUDIO ? gi__busrefs : gi__busrefsk
    tabw_i tab_i(ibus, itab) + 1, ibus, itab
    atstop ${busrelease}, 0, 0, itoken
    xout ibus
endop

opcode _busaddref, 0, ii
    ibus, ikind xin
    itab = ikind == $$_BUSKIND_AUDIO ? gi__busrefs : gi__busrefsk
    irefs tab_i ibus, itab
    tabw_i irefs+1, ibus, itab
endop

opcode _busget, i, ii
    ; like _bususe but does not add a reference, only gets the bus index
    itoken, ikind  xin
    ibus dict_get gi__bustoken2num, itoken, -1
    if ibus < 0 then
        initerror sprintf("Bus not found (token: %d)", itoken)
    endif
    if ikind == $$_BUSKIND_CONTROL then
        if ibus >= gi__numControlBuses then
            initerror sprintf("Invalid control bus (%d) for token %d", ibus, itoken)
        endif
    ; it is an audio bus
    elseif ibus >= gi__numAudioBuses then
        initerror sprintf("Invalid audio bus (%d) for token %d", ibus, itoken)
    endif
    xout ibus
endop

opcode busassign, i, So
    /* Create a bus, returns a token pointing to that bus

    Args:
      Skind: "a" / "k"
      ipersist: if non-zero, adds an extra-reference to this bus in order
        to make it peristent. To release such a bus use busrelease

    Returns:
        itoken: the token pointing to the newly assigned bus

    */
    Skind, ipersist xin
    ikind = strcmp(Skind, "a") == 0 ? $$_BUSKIND_AUDIO : $$_BUSKIND_CONTROL
    ; generate a new token
    itoken chnget "_busTokenCount"
    chnset itoken+1, "_busTokenCount"

    ; assign a bus to the new token
    ibus _busnew itoken, ikind

    ; use the bus during the lifetime of this event, add an extra reference
    ; if asked to persist the bus
    itab = ikind == $$_BUSKIND_AUDIO ? gi__busrefs : gi__busrefsk
    irefs tab_i ibus, itab
    irefs += ipersist == 0 ? 1 : 2
    tabw_i irefs, ibus, itab
    atstop ${busrelease}, 0, 0, itoken
    xout itoken
endop

opcode busin, i, io
    itoken, idefault xin
    ibus = _bususe(itoken, $$_BUSKIND_CONTROL)
    ival tab_i ibus, gi__bustable
    if ival == $$_BUSUNSET then
        tabw_i idefault, ibus, gi__bustable
    endif
    ival tab_i ibus, gi__bustable
    xout ival
endop

opcode busin, a, i
    itoken xin
    ibus = _bususe(itoken, $$_BUSKIND_AUDIO)
    aout = ga__buses[ibus]
    xout aout
endop

opcode busin, k, io
    itoken, idefault xin
    ibus = _bususe(itoken, $$_BUSKIND_CONTROL)
    ival tab_i ibus, gi__bustable
    if ival == $$_BUSUNSET then
        tabw_i idefault, ibus, gi__bustable
    endif
    kval tab ibus, gi__bustable
    xout kval
endop

opcode busout, 0, ii
    itoken, isig xin
    ibus = _busget(itoken, $$_BUSKIND_CONTROL)
    tabw_i isig, ibus, gi__bustable
endop

opcode busout, 0, ik
    itoken, ksig xin
    ibus = _bususe(itoken, $$_BUSKIND_CONTROL)
    tabw ksig, ibus, gi__bustable
endop

opcode busout, 0, ia
    itoken, asig xin
    ibus = _bususe(itoken, $$_BUSKIND_AUDIO)
    ga__buses[ibus] = asig
endop

opcode busmix, 0, ia
    itoken, asig xin
    ibus = _bususe(itoken, $$_BUSKIND_AUDIO)
    ga__buses[ibus] = ga__buses[ibus] + asig
endop

; This instr MUST come before any other instrs using buses for
; offline rendering to work
instr ${busassign}
    ; query the index of a bus / create a bus if not assigned
    ; args:
    ;  isynctoken: the synctoken to return the bus index. if 0, no
    ;    callback is scheduled
    ;  ibustoken: the bus token
    ;  iassign: if 1, assign a bus if no bus is found for this bustoken
    ;  ikind: 0: audio bus, 1: scalar bus
    isynctoken = p4
    ibustoken  = p5
    ikind      = p6
    iaddref    = p7
    ivalue     = p8
    ibus dict_get gi__bustoken2num, ibustoken, -1

    if ibus == -1 then
        ibus = _busnew(ibustoken, ikind)
    else
        goto __exit
    endif

    if ikind == $$_BUSKIND_CONTROL then
        ; a new control bus, set default value
        tabw_i ivalue, ibus, gi__bustable
    endif

    if iaddref == 1 then
        _busaddref(ibus, ikind)
    endif

__exit:
    if isynctoken > 0 then
        tabw_i ibus, isynctoken, gi__responses
        outvalue "__sync__", isynctoken
    endif
    turnoff
endin

instr ${automateBusViaPargs}
    itoken        = p4
    iinterpmethod = p5
    iovertake     = p6
    ilenpairs     = p7

    ipairs[] passign 8, 8+ilenpairs
    iXs[] slicearray ipairs, 0, ilenpairs-1, 2
    iYs[] slicearray ipairs, 1, ilenpairs-1, 2
    Sinterpmethod = strget(iinterpmethod)

    ibus = _busget(itoken, $$_BUSKIND_CONTROL)

    if iovertake == 1 || qnan:i(iYs[0]) == 1 then
        iYs[0] = tab_i(ibus, gi__bustable)
    endif

    kt timeinsts
    kidx bisect kt, iXs
    ky interp1d kidx, iYs, Sinterpmethod

    tabw ky, ibus, gi__bustable
endin

instr ${busaddref}
    itoken = p4
    ikind = p5
    ibus = _busget(itoken, ikind)
    _busaddref(ibus, ikind)
endin

instr ${busrelease}  ; release audio bus
    itoken = p4
    ikind dict_get gi__bustoken2kind, itoken, -1
    ibus dict_get gi__bustoken2num, itoken, -1
    if ibus < 0 then
        initerror sprintf("itoken %d has no bus assigned to it", itoken)
        goto __exit
    endif

    if ikind < 0 then
        initerror sprintf("Invalid kind for bus token %d", itoken)
        goto __exit
    endif
    ;; --- TODO
    if ikind == $$_BUSKIND_AUDIO then
        ; ------ audio bus ------
        irefs tab_i ibus, gi__busrefs
        if irefs <= 1 then
            if pool_isfull:i(gi__buspool) == 1 then
                initerror "Audio bus pool is full!"
                goto __exit
            endif
            pool_push gi__buspool, ibus
            dict_del gi__bustoken2num, itoken
            dict_del gi__bustoken2kind, itoken
            tabw_i 0, ibus, gi__busrefs
        else
            tabw_i irefs-1, ibus, gi__busrefs
        endif
    else
        ; ------ control bus ------
        irefs tab_i ibus, gi__busrefsk
        if irefs <= 1 then
            if pool_isfull:i(gi__buspoolk) == 1 then
                initerror "Control bus pool is full!"
                goto __exit
            endif

            pool_push gi__buspoolk, ibus
            dict_del gi__bustoken2num, itoken
            dict_del gi__bustoken2kind, itoken
            tabw_i 0, ibus, gi__busrefsk
            tabw_i $$_BUSUNSET, ibus, gi__bustable
        else
            tabw_i irefs-1, ibus, gi__busrefsk
        endif
    endif
__exit:
endin

instr ${busoutk}
    itoken = p4
    ivalue = p5
    ibus = _busget(itoken, 1)
    tabw_i ivalue, ibus, gi__bustable
    turnoff
endin

instr ${clearbuses_post}
    ; Use masked version of zeroarray
    zeroarray ga__buses, gi__busrefs
endin

chnset 0, "_busTokenCount"
ftset gi__bustable, $$_BUSUNSET

'''


# Constants
CONSTS = {
    'numTokens': 2000,  # Number of tokens to be allocated for responses
    'postProcInstrnum': 10000,   # Post processing instrument number
    'reservedTablesStart': 300,  # Tables managed by csoundengine start here. ftgen tables start at 100
    'reservedInstrsStart': 1,
    'userInstrsStart': 100,
    'sessionInstrsStart': 400,
    'numReservedTables': 2000,   # Size of a pool of table numbers
    'maxNumInstrs': 9999,
    'BUSUNSET': -999999999,

}


UNSET_VALUE = float("-inf")
BUSKIND_AUDIO = 0
BUSKIND_CONTROL = 1


_tableNames = ['responses', 'tokenToInstrnum']

BUILTIN_TABLES = {name:i for i, name in enumerate(_tableNames, start=1)}


@cache
def makeOrcHeader(sr: int,
                  ksmps: int,
                  nchnls: int,
                  nchnls_i: int,
                  a4: float,
                  zerodbfs: float = 1.) -> str:
    return f"""
sr = {sr}
ksmps = {ksmps}
nchnls = {nchnls}
nchnls_i = {nchnls_i}
a4 = {a4}
0dbfs = {zerodbfs}
"""

@cache
def makeOrc(globalcode: str = "",
            includestr: str = "",
            ) -> tuple[str, dict[str, int]]:
    """
    Create an Engine's orchestra

    Returns:
        the orchestra and a dict mapping builtin instr names to their
        instr number

    """
    orcproto = _orc
    template = Template(orcproto)
    instrs = internal.assignInstrNumbers(orcproto,
                                         startInstr=CONSTS['reservedInstrsStart'],
                                         postInstr=CONSTS['postProcInstrnum'])
    subs: dict[str, _t.Any] = {name: f"{num}  /* {name} */"
                            for name, num in instrs.items()}
    subs.update(BUILTIN_TABLES)
    subs.update(CONSTS)
    orc = template.substitute(
            globalcode=globalcode,
            includes=includestr,
            **subs
    )
    return orc, instrs


@cache
def makeBusOrc(numAudioBuses: int, numControlBuses: int, startInstr: int, postInstr: int
               ) -> tuple[str, dict[str, int]]:
    template = Template(_busOrc)
    instrs = internal.assignInstrNumbers(_busOrc, startInstr, postInstr)
    subs: dict[str, _t.Any] = {name: f"{num}  /* {name} */"
                            for name, num in instrs.items()}
    subs.update(BUILTIN_TABLES)
    subs.update(CONSTS)
    orc = template.substitute(numAudioBuses=numAudioBuses, numControlBuses=numControlBuses, **subs)
    return orc, instrs


# def busSupportCode(numAudioBuses: int,
#                    numControlBuses: int,
#                    postInstrNum: int,
#                    startInstr: int
#                    ) -> tuple[str, dict[str, int]]:
#     """
#     Generates bus support code

#     Args:
#         numAudioBuses: the number of audio buses
#         numControlBuses: the number of control buses
#         postInstrNum: the starting instr number for *post* instrs. A post
#             instr should be run at the end of the evaluation chain. Such
#             instruments should have a _post in their name
#         startInstr: start instrument number for non-post instruments
#     """
#     instrnums = internal.assignInstrNumbers(_busOrc,
#                                             startInstr=startInstr,
#                                             postInstrNum=postInstrNum)
#     subs: dict[str, _t.Any] = {name: f"{num}  /* {name} */"
#                             for name, num in instrnums.items()}
#     subs['clearbuses'] = f'{postInstrNum} /* clearbuses */'
#     orc = Template(_busOrc).substitute(numAudioBuses=numAudioBuses,
#                                        numControlBuses=numControlBuses,
#                                        BUSUNSET=CONSTS['BUSUNSET'],
#                                        **subs)
#     return orc, instrnums
