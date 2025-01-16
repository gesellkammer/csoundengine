<CsoundSynthesizer>
<CsOptions>
</CsOptions>

<CsInstruments>

sr     = 44100
ksmps  = 64
0dbfs  = 1
A4     = 442
nchnls = 2
; ----- global code


gi__dynargsNumSlots = 10000
gi__dynargsSliceSize = 16

gi__tokenToDynargsSlot dict_new "int:float"
gi__dynargsSlotsPool pool_gen 1, gi__dynargsNumSlots
gi__dynargsTableSize = gi__dynargsNumSlots * gi__dynargsSliceSize
gi__dynargsTable ftgen 0, 0, gi__dynargsTableSize, -2, 0

gi__soundfontIndexes dict_new "str:float"
gi__soundfontIndexCounter init 1000

; numtokens = 1000
gi__responses   ftgen  0, 0, 1000, -2, 0

chn_k "_soundfontPresetCount", 3

instr 3000
endin

opcode _assignControlSlot, i, i
    itoken xin
    iprevslot dict_get gi__tokenToDynargsSlot, itoken, -1
    islot pool_pop gi__dynargsSlotsPool, -1
    if islot == -1 then
        initerror "Control slots pool is empty"
    elseif iprevslot >= 0 then
        prints sprintf("Warning: the token %d has already an assigned slot %d. New slot: %d\n", itoken, iprevslot, islot)
    endif
    dict_set gi__tokenToDynargsSlot, itoken, islot
    xout islot    
endop

opcode _getControlSlot, i, i
    itoken xin
    islot dict_get gi__tokenToDynargsSlot, itoken
    if islot <= 0 then
        ; islot = _assignControlSlot(itoken)
        initerror sprintf("Slot not found for token %d", itoken)
    endif
    xout islot
endop

instr _releaseDynargsToken
    ; to be called at end of the instr (using atstop) with dur=0
    itoken = p4
    islot dict_get gi__tokenToDynargsSlot, itoken
    if islot <= 0 then
        initerror sprintf("itoken %d has no slot assigned to it", itoken)
    endif
    pool_push gi__dynargsSlotsPool, islot
    dict_del gi__tokenToDynargsSlot, itoken
endin

opcode _panweights, kk, k
    kpos xin   
    kampL = bpf:k(kpos, 0, 1.4142, 0.5, 1, 1, 0)
    kampR = bpf:k(kpos, 0, 0,      0.5, 1, 1, 1.4142)
    xout kampL, kampR
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



instr _stop
    ; turnoff inum (match instr number exactly, allow release)
    inum = p4
    turnoff2_i inum, 4, 1
    turnoff
endin

instr _setControl
    itoken = p4
    iindex = p5
    ivalue = p6
    islot = _getControlSlot(itoken)
    if islot <= 0 then
        initerror sprintf("Control slot not assigned for token %d", itoken)
    endif
    iindex0 = islot * gi__dynargsSliceSize
    tabw_i ivalue, iindex0 + iindex, gi__dynargsTable
endin

instr _initDynamicControls
    ; to be called with dur 0
    pset 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    itoken = p4
    inumitems = p5
    islot = _assignControlSlot(itoken)
    ivalues[] passign 6, 6+inumitems
    iindex = islot * gi__dynargsSliceSize
    copya2ftab ivalues, gi__dynargsTable, iindex
endin

instr _automateControlViaPargs
    itoken = p4
    iparamindex = p5
    imode = p6
    iovertake = p7
    ilenpairs = p8

    islot = _getControlSlot(itoken)
    if islot <= 0 then
        initerror sprintf("No control slot assigned to token %d", itoken)
    endif

    iindex0 = islot * gi__dynargsSliceSize
    iabsindex = iindex0 + iparamindex

    ipairs[] passign 9, 9+ilenpairs
    iXs[] slicearray ipairs, 0, ilenpairs-1, 2
    iYs[] slicearray ipairs, 1, ilenpairs-1, 2
    Sinterpmethod = strget(imode)

    if iovertake == 1 then
        iYs[0] = tab_i(iabsindex, gi__dynargsTable)
    endif

    kt timeinsts   ;; TODO: use eventtime
    kidx bisect kt, iXs
    ky interp1d kidx, iYs, Sinterpmethod
    tabw ky, iabsindex, gi__dynargsTable
endin

instr _automatePargViaPargs
    ip1 = p4
    ipindex = p5
    imode = p6;  interpolation method
    iovertake = p7
    ilenpairs = p8

    ; special case: simple line, two pairs
    if ilenpairs == 4 && p9 == 0 && iovertake == 0 then
        iy0 = p10
        ix1 = p11
        iy1 = p12
        ky linseg iy0, ix1, iy1
        goto end 
    endif

    ipairs[] passign 9, 9+ilenpairs
    iXs[] slicearray ipairs, 0, ilenpairs-1, 2
    iYs[] slicearray ipairs, 1, ilenpairs-1, 2
    Sinterpmethod = strget(imode)

    if iovertake == 1 then
        icurrval pread ip1, ipindex, -1
        iYs[0] = icurrval
    endif

    kt timeinsts
    kidx bisect kt, iXs
    ky interp1d kidx, iYs, Sinterpmethod

end:
    pwrite ip1, ipindex, ky
endin

instr _pwrite
  ip1 = p4
  ipindex = p5
  ivalue = p6
  pwrite ip1, ipindex, ivalue
  turnoff
endin

instr _chnset
    ; to be called with dur 0
    Schn = p4
    ival = p5
    chnset ival, Schn
endin

; -------------------- end prelude -----------------




gi__numControlBuses = 10000
gi__numAudioBuses = 1000

#define _BUSUNSET #-999999999#
#define _BUSKIND_AUDIO   #0#
#define _BUSKIND_CONTROL #1#

; The actual buses
ga__buses[] init gi__numAudioBuses
gi__bustable ftgen 0, 0, gi__numControlBuses, -2, 0

; This table keeps track of the number of references a bus has
gi__busrefs ftgen 0, 0, 1000, -2, 0
gi__busrefsk ftgen 0, 0, 10000, -2, 0

; A pool of bus indexes
gi__buspool pool_gen 1000
gi__buspoolk pool_gen 10000

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
    ipool = ikind == $_BUSKIND_AUDIO ? gi__buspool : gi__buspoolk
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

    if ikind == $_BUSKIND_CONTROL && ibus >= gi__numControlBuses then
        initerror sprintf("Invalid control bus (%d) for token %d", ibus, itoken)
    elseif ikind == $_BUSKIND_AUDIO && ibus >= gi__numAudioBuses then
        initerror sprintf("Invalid audio bus (%d) for token %d", ibus, itoken)
    endif

    itab = ikind == $_BUSKIND_AUDIO ? gi__busrefs : gi__busrefsk
    tabw_i tab_i(ibus, itab) + 1, ibus, itab
    atstop 24  /* busrelease */, 0, 0, itoken
    xout ibus
endop

opcode _busaddref, 0, ii
    ibus, ikind xin
    itab = ikind == $_BUSKIND_AUDIO ? gi__busrefs : gi__busrefsk
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
    if ikind == $_BUSKIND_CONTROL && ibus >= gi__numControlBuses then
        initerror sprintf("Invalid control bus (%d) for token %d", ibus, itoken)
    elseif ikind == $_BUSKIND_AUDIO && ibus >= gi__numAudioBuses then
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
    ikind = strcmp(Skind, "a") == 0 ? $_BUSKIND_AUDIO : $_BUSKIND_CONTROL
    ; generate a new token
    itoken chnget "_busTokenCount"
    chnset itoken+1, "_busTokenCount"

    ; assign a bus to the new token
    ibus _busnew itoken, ikind

    ; use the bus during the lifetime of this event, add an extra reference
    ; if asked to persist the bus
    itab = ikind == $_BUSKIND_AUDIO ? gi__busrefs : gi__busrefsk
    irefs tab_i ibus, itab
    irefs += ipersist == 0 ? 1 : 2
    tabw_i irefs, ibus, itab
    atstop 24  /* busrelease */, 0, 0, itoken
    xout itoken
endop

opcode busin, i, io
    itoken, idefault xin
    ibus = _bususe(itoken, $_BUSKIND_CONTROL)
    ival tab_i ibus, gi__bustable
    if ival == $_BUSUNSET then
        tabw_i idefault, ibus, gi__bustable
    endif
    ival tab_i ibus, gi__bustable
    xout ival
endop

opcode busin, a, i
    itoken xin
    ibus = _bususe(itoken, $_BUSKIND_AUDIO)
    aout = ga__buses[ibus]
    xout aout
endop

opcode busin, k, io
    itoken, idefault xin
    ibus = _bususe(itoken, $_BUSKIND_CONTROL)
    ival tab_i ibus, gi__bustable
    if ival == $_BUSUNSET then
        tabw_i idefault, ibus, gi__bustable
    endif
    kval tab ibus, gi__bustable
    xout kval
endop

opcode busout, 0, ii
    itoken, isig xin
    ibus = _busget(itoken, $_BUSKIND_CONTROL)
    tabw_i isig, ibus, gi__bustable
endop

opcode busout, 0, ik
    itoken, ksig xin
    ibus = _bususe(itoken, $_BUSKIND_CONTROL)
    tabw ksig, ibus, gi__bustable
endop

opcode busout, 0, ia
    itoken, asig xin
    ibus = _bususe(itoken, $_BUSKIND_AUDIO)
    ga__buses[ibus] = asig
endop

opcode busmix, 0, ia
    itoken, asig xin
    ibus = _bususe(itoken, $_BUSKIND_AUDIO)
    ga__buses[ibus] = ga__buses[ibus] + asig
endop

; This instr MUST come before any other instrs using buses for
; offline rendering to work
instr 20  /* busassign */
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

    if ikind == $_BUSKIND_CONTROL then
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

instr 21  /* automateBusViaPargs */
    itoken        = p4
    iinterpmethod = p5
    iovertake     = p6
    ilenpairs     = p7

    ipairs[] passign 8, 8+ilenpairs
    iXs[] slicearray ipairs, 0, ilenpairs-1, 2
    iYs[] slicearray ipairs, 1, ilenpairs-1, 2
    Sinterpmethod = strget(iinterpmethod)

    ibus = _busget(itoken, $_BUSKIND_CONTROL)

    if iovertake == 1 || qnan:i(iYs[0]) == 1 then
        iYs[0] = tab_i(ibus, gi__bustable)
    endif

    kt timeinsts
    kidx bisect kt, iXs
    ky interp1d kidx, iYs, Sinterpmethod

    tabw ky, ibus, gi__bustable
endin

instr 22  /* busaddref */
    itoken = p4
    ikind = p5
    ibus = _busget(itoken, ikind)
    _busaddref(ibus, ikind)
endin

instr 23  /* busdump */
    itoken = p4
    ibus dict_get gi__bustoken2num, itoken, -1
    if ibus < 0 then
        initerror sprintf("itoken %d has no bus assigned to it", itoken)
    endif
    ikind dict_get gi__bustoken2kind, itoken
    irefstable = ikind == $_BUSKIND_AUDIO ? gi__busrefs : gi__busrefsk
    irefs tab_i ibus, irefstable
    if ikind == $_BUSKIND_CONTROL then
        ivalue = tab_i(ibus, gi__bustable)
        prints "Bus token=%d, bus=%d, kind=k, value=%f, refs=%d\n", itoken, ibus, ivalue, irefs
    else
        prints "Bus token=%d, bus=%d, kind=a, refs=%d\n", itoken, ibus, irefs
    endif
endin

instr 24  /* busrelease */  ; release audio bus
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

    if ikind == $_BUSKIND_AUDIO then
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
            tabw_i $_BUSUNSET, ibus, gi__bustable
        else
            tabw_i irefs-1, ibus, gi__busrefsk
        endif
    endif
__exit:
endin

instr 25  /* busoutk */
    itoken = p4
    ivalue = p5
    ibus = _busget(itoken, 1)
    tabw_i ivalue, ibus, gi__bustable
    turnoff
endin

instr 2446  /* clearbuses_post */
    ; Use masked version of zeroarray
    zeroarray ga__buses, gi__busrefs
    ; zeroarray ga__buses
endin

chnset 0, "_busTokenCount"
ftset gi__bustable, $_BUSUNSET


chnset 1, ".dynargsTabnum"
; ----- end global code

instr 50  ; sin
  imidi = p5
  ifreq = mtof:i(imidi)
  print imidi, ifreq
  a0 oscili 0.1, mtof:i(imidi)
  a0 *= linsegr:a(0, 0.1, 1, 0.1, 0)
  outch 1, a0
endin

</CsInstruments>

<CsScore>

C 0.0     ; Disable carry
f 1 0 -160000 -2 0
; i 2447 0 -1
i 50.0001 0.0 4.0 -1 60.0
i 50.0002 1.0 3.5 -1 60.5

</CsScore>
</CsoundSynthesizer>
