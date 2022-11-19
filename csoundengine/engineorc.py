from __future__ import annotations
from string import Template
from functools import cache
from typing import List, Dict, Any, Tuple
import re

# In all templates we make the difference between substitutions which
# are constant (with the form ${subst} and substitutions which respond
# to the configuration of a specific engine / offline renderer ({subst})

_orc = r'''
sr     = ${sr}
ksmps  = ${ksmps}
nchnls = ${nchnls}
nchnls_i = ${nchnls_i}
0dbfs  = 1
A4     = ${a4}

${includes}

gi__subgains        ftgen  ${subgains},  0, 100, -2, 0
gi__responses       ftgen  ${responses}, 0, ${numtokens}, -2, 0
gi__tokenToInstrnum ftgen ${tokenToInstrnum}, 0, ${maxNumInstrs}, -2, 0
gi__soundfontIndexes dict_new "str:float"
gi__soundfontIndexCounter init 1000

chn_k "_soundfontPresetCount", 3

${globalcode}

; ---------------------------------
;          builtin-instruments
; ---------------------------------

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

opcode sfPresetIndex, i, Sii
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

instr ${notifyDealloc}
    outvalue, "__dealloc__", p4
endin

instr ${pingback}
    itoken = p4
    outvalue "__sync__", itoken
endin

instr ${turnoff}
    iwhich = p4
    imode = p5
    turnoff2_i iwhich, imode, 1
endin

instr ${turnoff_future}
    iwhich = p4
    turnoff3 iwhich
endin

instr ${nstrnumsync}
    itoken = p4
    Sname = p5
    ; inum nstrnum Sname
    inum nametoinstrnum Sname
    if itoken > 0 then
        tabw_i inum, itoken, gi__responses
        outvalue "__sync__", itoken
    endif
endin

instr ${nstrnum}
    itoken = p4
    Sname = p5
    ; inum nstrnum Sname
    inum nametoinstrnum Sname
    tabw_i inum, itoken, gi__responses
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
        itabnum ftgen itabnum, 0, ilen, -2, 0
    else
        iValues[] passign 10, 10+ilen
        itabnum ftgen itabnum, 0, ilen, -2, iValues
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
    ip1 = p4
    ipindex = p5
    imode = p6;  interpolation method
    iovertake = p7
    ilenpairs = p8
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
    pwrite ip1, ipindex, ky
endin

instr ${automatePargViaTable}
    ip1 = p4
    ipindex = p5
    itabpairs = p6  ; a table containing flat pairs t0, y0, t1, y1, ...
    imode = p7;  interpolation method
    iovertake = p8
    Sinterpmethod = strget(imode)
    ftfree itabpairs, 1
    
    if iovertake == 1 then
        icurrval pread ip1, ipindex, -1
        tabw_i icurrval, 1, itabpairs
    endif

    kt timeinsts
    kidx bisect kt, itabpairs, 2, 0
    ky interp1d kidx, itabpairs, Sinterpmethod, 2, 1
    pwrite ip1, ipindex, ky
endin

instr ${automateTableViaTable}
    iargtab = p4
    iargidx = p5
    idatatab = p6
    imode = p7
    idatastep = p8
    idataoffset = p9
    iovertake = p10
    
    ftfree idatatab, 1
    Smode = strget(imode)
    
    if iargtab == 0 || ftexists:i(iargtab) == 0 then
        initerror sprintf("Instr table %d does not exist", iargtab)
        goto exit
    endif
    if idatatab == 0 || ftexists:i(idatatab) == 0 then
        initerror sprintf("Automation table %d does not exist", iargtab)
        goto exit
    endif

    if ftlen(iargtab) <= iargidx then
        initerror sprintf("Table too small (%d <= %d)", ftlen(iargtab), iargidx)
        goto exit
    endif
    
    if iovertake == 1 then
        icurrval = tab_i(iargidx, iargtab)
        tabw_i icurrval, 1, idatatab
    endif

    kt timeinsts
    kidx bisect kt, idatatab, idatastep, 0
    ky interp1d kidx, idatatab, Smode, idatastep, idataoffset
    if ftexists:k(iargtab) == 1 then
        tablew ky, iargidx, iargtab
    else
        throwerror "warning", sprintf("dest table (%d) was freed, stopping automation", \ 
                                      iargtab)
        turnoff
    endif
exit:
endin

instr ${uniqinstance}
    itoken = p4
    ip1    = p5
    iuniq uniqinstance ip1
    tabw_i iuniq, itoken, gi__responses
    outvalue "__sync__", itoken
endin

instr ${scheduniq}
    itoken = p4
    ip1 = p5
    idelay = p6
    idur = p7
    inumargs = p8
    ipargs[] passign 9, 9+inumargs
    iuniq uniqinstance ip1
    iargs[] init 3+inumargs
    iargs[0] = iuniq
    iargs[1] = idelay
    iargs[2] = idur
    setslice iargs, ipargs, 3
    schedule iargs
    tabw_i iuniq, itoken, gi__responses
    outvalue "__sync__", itoken
endin

instr ${freetable}
    ifn = p4
    idelay = p5
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
    poly0 inumouts, "outch", ichans, aouts
    know += 1/kspeed
    if know >= idur then
        turnoff
    endif
endin

instr ${readSndfile}
    itoken = p4
    Spath = strget(p5)
    itab = p6
    ichan = p7
    itab2 ftgen itab, 0, 0, -1, Spath, 0, 0, ichan
    if itoken > 0 then
        tabw_i itab2, itoken, gi__responses
        outvalue "__sync__", itoken
    endif
endin

instr ${playgen1}
    ;             4     5      6       7     8     9          10        11
    ;             gain, speed, source, chan, fade, starttime, gaingroup lagtime
    pset 0, 0, 0, 1,    1,     1,      1,    0.05, 0,         0,        0.01
    kgain = p4
    kspeed = p5
    itabnum, ichan, ifade, ioffset, igaingroup, ilagtime passign 6
    ifade = max(ifade, ksmps/sr*2)
    inumsamples = nsamp(itabnum)
    itabsr = ftsr(itabnum)
    if itabsr <= 0 then
        initerror sprintf("Could not determine sr for table %d", itabnum)
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
    kgain *= table:k(igaingroup, gi__subgains)
    again = lag:a(a(kgain), ilagtime)
    aenv *= again
    aouts = aouts * aenv
    
    kchan = 0
    while kchan < inumouts do
        outch kchan+ichan, aouts[kchan]
        kchan += 1
    od
    
endin

instr ${strset}
    Sstr = p4
    idx  = p5
    strset idx, Sstr
endin

instr ${ftsetparams}
    itabnum, isr, inumchannels passign 4
    ftsetparams itabnum, isr, inumchannels
endin

instr ${pwrite}
    ip1 = p4
    inumpairs = p5
    if inumpairs == 1 then
        pwrite ip1, p(6), p(7)
    elseif inumpairs == 2 then
        pwrite ip1, p(6), p(7), p(8), p(9)
    elseif inumpairs == 3 then
        pwrite ip1, p(6), p(7), p(8), p(9), p(10), p(11)
    elseif inumpairs == 4 then
        pwrite ip1, p(6), p(7), p(8), p(9), p(10), p(11), p(12), p(13)
    elseif inumpairs == 5 then
        pwrite ip1, p(6), p(7), p(8), p(9), p(10), p(11), p(12), p(13), p(14), p(15)
    else
        initerror sprintf("Max. pairs is 5, got %d", inumpairs)
    endif
endin

instr ${pread}
    itoken = p4
    ip1 = p5
    ipindex = p6
    inotify = p7
    ival pread ip1, ipindex
    tabw_i ival, itoken, gi__responses
    if inotify == 1 then
        outvalue "__sync__", itoken
    endif
endin

instr ${preadmany}
    ; reads multiple pfields, puts them in a table
    itoken, ip1, iouttab, ioffset, inumpfields passign 4
    ipfields[] passign 9, inumpfields
    ivalues[] pread ip1, ipfields
    i0 = 0
    while i0 < inumpfields do
        tabw_i ivalues[i0], ioffset+i0, iouttab
        i0 += 1
    od
    if itoken > 0 then
        outvalue "__sync__", itoken
    endif
endin

instr ${testaudio}
    pset 0, 0, 0, 0, 1, 1
    imode = p4
    iperiod = p5
    igain = p6
    
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


ftset gi__subgains, 1
chnset 1, "_soundfontPresetCount"   

'''

_busOrc = r'''

#define _BUSUNSET #${BUSUNSET}#

; The actual buses
ga__buses[]   init ${numAudioBuses}
gi__bustable ftgen 0, 0, ${numControlBuses}, -2, 0
; gk__buses[]   init ${numControlBuses}

; This table keeps track of the number of references a bus has
gi__busrefs ftgen 0, 0, ${numAudioBuses}, -2, 0
gi__busrefsk ftgen 0, 0, ${numControlBuses}, -2, 0

; A pool of bus indexes
gi__buspool pool_gen ${numAudioBuses}
gi__buspoolk pool_gen ${numControlBuses} 

; A dict mapping bustoken to bus number, used for both audio and scalar buses
gi__bustoken2num dict_new "int:float"

chn_k "_busTokenCount", 3

opcode _busnew, i, ii
    itoken, ikind xin
    if itoken < 0 then
        itoken chnget "_busTokenCount"
        chnset itoken+1, "_busTokenCount"
    endif
    ipool = ikind == 0 ? gi__buspool : gi__buspoolk
    ibus pool_pop ipool, -1
    if ibus == -1 then
        initerror "busassign failed, out of buses"
    endif    
    dict_set gi__bustoken2num, itoken, ibus
    xout ibus
endop

opcode _bususe, i, ii
    itoken, ikind xin
    ibus dict_get gi__bustoken2num, itoken, -1
    if ibus < 0 then
        initerror sprintf("Bus not found (token %d)\n", itoken)
    else
        itab = ikind == 0 ? gi__busrefs : gi__busrefsk
        irefs tab_i ibus, itab
        tabw_i irefs+1, ibus, itab
        atstop ${busrelease}, 0, 0, itoken, ikind
    endif
    xout ibus
endop

opcode _busaddref, 0, ii
    ibus, ikind xin
    itab = ikind == 0 ? gi__busrefs : gi__busrefsk
    irefs tab_i ibus, itab
    tabw_i irefs+1, ibus, itab
endop

opcode _busget, i, ii
    itoken, ikind xin
    ibus dict_get gi__bustoken2num, itoken, -1
    if ibus == -1 then
        initerror sprintf("Bus not found (token: %d)", itoken)
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
    ikind = strcmp(Skind, "a") == 0 ? 0 : 1  ; 0=audio, 1=k
    ; generate a new token
    itoken chnget "_busTokenCount"
    chnset itoken+1, "_busTokenCount"
    
    ; assign a bus to the new token
    ibus _busnew itoken, ikind
    
    ; use the bus during the lifetime of this event, add an extra reference
    ; if asked to persist the bus
    itab = ikind == 0 ? gi__busrefs : gi__busrefsk
    irefs tab_i ibus, itab
    irefs += ipersist == 0 ? 1 : 2
    tabw_i irefs, ibus, itab
    atstop ${busrelease}, 0, 0, itoken, ikind
    xout itoken
endop

opcode busin, a, i
    itoken xin
    ibus = _bususe(itoken, 0)
    aout = ga__buses[ibus]
    xout aout
endop

opcode busin, k, io
    itoken, idefault xin
    ibus = _bususe(itoken, 1)
    ival tab_i ibus, gi__bustable
    if ival == $$_BUSUNSET then
        tabw_i idefault, ibus, gi__bustable
    endif
    kval tab ibus, gi__bustable
    xout kval
endop

opcode busout, 0, ii
    itoken, isig xin
    ibus = _busget(itoken, 1)
    tabw_i isig, ibus, gi__bustable
endop

opcode busout, 0, ik
    itoken, ksig xin
    ibus = _bususe(itoken, 1)
    tabw ksig, ibus, gi__bustable
endop

opcode busout, 0, ia
    itoken, asig xin
    ibus = _bususe(itoken, 0)
    ga__buses[ibus] = asig
endop

opcode busmix, 0, ia
    itoken, asig xin
    ibus = _bususe(itoken, 0)
    ga__buses[ibus] = ga__buses[ibus] + asig
endop

instr ${busaddref}
    itoken = p4
    ikind = p5
    ibus = _busget(itoken, ikind)
    _busaddref(ibus, ikind)
endin

instr ${busrelease}  ; release audio bus
    itoken = p4
    ikind = p5
    ibus dict_get gi__bustoken2num, itoken, -99999999
    if ibus == -99999999 then
        initerror sprintf("itoken %d has no bus assigned to it", itoken)
    endif
    
    ; ikind == 0: audio, 1: scalar
    if ikind == 0 then
        irefs tab_i ibus, gi__busrefs
        if irefs <= 1 then
            if pool_isfull:i(gi__buspool) == 1 then
                initerror "Bus pool is full!"
            endif
            pool_push gi__buspool, ibus
            dict_del gi__bustoken2num, itoken
            tabw_i 0, ibus, gi__busrefs
        else   
            tabw_i irefs-1, ibus, gi__busrefs
        endif
    else
        irefs tab_i ibus, gi__busrefsk
        if irefs <= 1 then
            if pool_isfull:i(gi__buspoolk) == 1 then
                initerror "Bus pool is full!"
            endif
            pool_push gi__buspoolk, ibus
            dict_del gi__bustoken2num, itoken
            tabw_i 0, ibus, gi__busrefsk
            tabw_i $$_BUSUNSET, ibus, gi__bustable
        else   
            tabw_i irefs-1, ibus, gi__busrefsk
        endif
    endif
endin

instr ${busindex}
    ; query the index of a bus / create a bus if not assigned
    ; args: 
    ;  isynctoken: the synctoken
    ;  ibustoken: the bus token
    ;  iassign: if 1, assign a bus if no bus is found for this bustoken
    ;  ikind: 0: audio bus, 1: scalar bus
    isynctoken = p4
    ibustoken = p5
    ikind = p6
    iaddref = p7
    ibus dict_get gi__bustoken2num, ibustoken, -1
    if ibus == -1 then
        ibus = _busnew(ibustoken, ikind)
    endif
    if iaddref == 1 then
        _busaddref(ibus, ikind)
    endif
    tabw_i ibus, isynctoken, gi__responses
    outvalue "__sync__", isynctoken
endin



instr ${busoutk}
    itoken = p4
    ivalue = p5
    ibus = _busget(itoken, 1)
    tabw_i ivalue, ibus, gi__bustable
endin

instr ${clearbuses_post}
    ; Use masked version of zeroarray
    zeroarray ga__buses, gi__busrefs
    ; zeroarray ga__buses
endin

chnset 0, "_busTokenCount"
ftset gi__bustable, $$_BUSUNSET

'''


def _extractInstrNames(s:str) -> list[str]:
    return [match.group(1) for line in s.splitlines()
            if (match:=re.search(r"\binstr\s+\$\{(\w+)\}", line))]


# Constants
CONSTS = {
    'numtokens': 1000,
    'eventMaxSize': 1999,
    'highestInstrnum': 11500,
    'postProcInstrnum': 11000,
    'reservedTablesStart': 300,
    'reservedInstrsStart': 1,
    'userInstrsStart': 100,
    'sessionInstrsStart': 500,
    'numReservedTables': 2000,
    'maxNumInstrs': 10000,
    'BUSUNSET': -999999999
}

_tableNames = ['responses', 'subgains', 'tokenToInstrnum']

BUILTIN_TABLES = {name:i for i, name in enumerate(_tableNames, start=1)}


@cache
def assignBuiltinInstrs(orc: str, startInstr: int, postInstrNum: int) -> Dict[str, int]:
    """
    Given an orc with quotes instr. names, assign numbers to each instr
    """
    names = _extractInstrNames(orc)
    preInstrs = [name for name in names if not name.endswith('_post')]
    postInstrs = [name for name in names if name.endswith('_post')]

    instrs = {name: i for i, name in enumerate(preInstrs, start=startInstr)}
    for i, name in enumerate(postInstrs):
        instrs[name] = postInstrNum + i
    return instrs


def _joinOrc(busSupport=True) -> str:
    parts = [_orc]
    if busSupport:
        parts.append(_busOrc)
    orc = "\n".join(parts)
    return orc


@cache
def makeOrc(sr:int, ksmps:int, nchnls:int, nchnls_i:int,
            a4:float, globalcode:str="", includestr:str="",
            numAudioBuses:int=0, numControlBuses:int=0
            ) -> Tuple[str, Dict[str, int]]:
    """
    Create an Engine's orchestra

    Returns:
        the orchestra and a dict mapping builtin instr names to their
        instr number

    """
    withBusSupport = numAudioBuses > 0 or numControlBuses > 0
    orcproto = _joinOrc(busSupport=withBusSupport)
    template = Template(orcproto)
    instrs = assignBuiltinInstrs(orcproto,
                                 startInstr=CONSTS['reservedInstrsStart'],
                                 postInstrNum=CONSTS['postProcInstrnum'])
    subs: Dict[str, Any] = {name:f"{num}  /* {name} */"
                            for name, num in instrs.items()}
    subs.update(BUILTIN_TABLES)
    subs.update(CONSTS)
    orc = template.substitute(
            sr=sr,
            ksmps=ksmps,
            nchnls=nchnls,
            nchnls_i=nchnls_i,
            a4=a4,
            globalcode=globalcode,
            includes=includestr,
            numAudioBuses=numAudioBuses,
            numControlBuses=numControlBuses,
            **subs
    )
    return orc, instrs


def busSupportCode(numAudioBuses:int,
                   numControlBuses:int,
                   postInstrNum:int,
                   startInstr:int
                   ) -> Tuple[str, Dict[str, int]]:
    instrnums = assignBuiltinInstrs(_busOrc, startInstr)
    subs: Dict[str, Any] = {name: f"{num}  /* {name} */"
                            for name, num in instrnums.items()}
    subs['clearbuses'] = f'{postInstrNum} /* clearbuses */'
    orc = Template(_busOrc).substitute(numAudioBuses=numAudioBuses,
                                       numControlBuses=numControlBuses,
                                       BUSUNSET=CONSTS['BUSUNSET'],
                                       **subs)
    return orc, instrnums
