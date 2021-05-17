from __future__ import annotations
from string import Template
from functools import lru_cache
from typing import List, Dict, Any
import re

# In all templates we make the difference between substitutions which
# are constant (with the form ${subst} and substitutions which respond
# to the configuration of a specific engine / offline renderer ({subst})

_orc = r"""
sr     = ${sr}
ksmps  = ${ksmps}
nchnls = ${nchnls}
nchnls_i = ${nchnls_i}
0dbfs  = 1
A4     = ${a4}

${includes}

gi__responses ftgen  ${responses}, 0, ${numtokens}, -2, 0
gi__subgains  ftgen  ${subgains},  0, 100, -2, 1
gi__tokenToInstrnum ftgen ${tokenToInstrnum}, 0, ${maxNumInstrs}, -2, 0

${globalcode}

; ---------------------------------
;          builtin-instruments
; ---------------------------------

instr __init
    ftset gi__subgains, 1
    ;; prints "<<< Init done >>>\n"
    turnoff
endin

opcode _panstereo, aa, aak
    ; kpos: 0-1
    a0, a1, kpos xin
    imax = 1.4142
    kamp0 = bpf:k(kpos, 0, imax, 0.5, 1, 1, 0)
    kamp1 = bpf:k(kpos, 0, 0,    0.5, 1, 1, imax)
    a0 *= kamp0
    a1 *= kamp1
    xout a0, a1
endop

instr _notifyDealloc
    outvalue "__dealloc__", p4
    turnoff
endin

instr ${turnoff}
    iwhich = p4
    turnoff2 iwhich, 4, 1
    turnoff
endin

instr ${nstrnum}
    itoken = p4
    Sname = p5
    inum nstrnum Sname
    if itoken > 0 then
        tabw_i inum, itoken, gi__responses
        outvalue "__sync__", itoken
    endif
    turnoff
endin
     
instr ${tabwrite}
    itab = p4
    iidx = p5
    ival = p6
    tabw_i ival, iidx, itab
    turnoff
endin

instr ${chnset}
    Schn = p4
    ival = p5
    chnset ival, Schn
    turnoff
endin

instr ${pingback}
    itoken = p4
    outvalue "__sync__", itoken
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
        ifn ftgen itabnum, 0, ilen, -2, 0
    else
        iValues[] passign 10, 10+ilen
        ifn ftgen itabnum, 0, ilen, -2, iValues
    endif
    if isr > 0 then
        ftsetparams ifn, isr, inumchannels
    endif 
    ; notify host that token is ready
    if itoken > 0 then
        tabw_i ifn, itoken, gi__responses
        outvalue "__sync__", itoken
    endif
    turnoff
endin

instr ${automatePargViaTable}
    ip1 = p4
    ipindex = p5
    itabpairs = p6  ; a table containing flat pairs t0, y0, t1, y1, ...
    imode = p7;  interpolation method
    Sinterpmethod = strget(imode)
    ftfree itabpairs, 1

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
    Smode = strget(imode)
    idatastep = p8
    idataoffset = p9
    ftfree idatatab, 1

    if ftexists:i(iargtab) == 0 then
        initerror sprintf("Instr table %d doesn't exist", iargtab)
    endif
    if ftexists:i(idatatab) == 0 then
        initerror sprintf("Automation table %d doesn't exist", iargtab)
    endif

    if ftlen(iargtab) <= iargidx then
        initerror sprintf("Table too small (%d <= %d)", ftlen(iargtab), iargidx)
    endif

    kt timeinsts
    kidx bisect kt, idatatab, idatastep, 0
    ky interp1d kidx, idatatab, Smode, idatastep, idataoffset
    if ftexists:k(iargtab) == 0 then
        throwerror "warning", sprintf("dest table (%d) was freed, stopping", iargtab)
        turnoff
    endif
    tablew ky, iargidx, iargtab
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
    turnoff
endin

instr ${freetable}
    ifn = p4
    idelay = p5
    ftfree ifn, 0
    turnoff
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
    ; prints ">>> Spath: %s, itab: %d, itab2: %d \n", Spath, itab, itab2
    tabw_i itab2, itoken, gi__responses
    outvalue "__sync__", itoken
endin

instr ${playgen1}
    ;             4     5      6       7     8     9          10        11
    ;             gain, speed, tabnum, chan, fade, starttime, gaingroup lagtime
    pset 0, 0, 0, 1,    1,     1,      1,    0.05, 0,         0,        0.01
    kgain = p4
    kspeed = p5
    ksampsplayed = 0
    itabnum, ichan, ifade, ioffset, igaingroup, ilagtime passign 6
    ifade = max(ifade, ksmps/sr*2)
    inumsamples = nsamp(itabnum)
    itabsr = ftsr(itabnum)
    if itabsr <= 0 then
        initerror sprintf("Could not determine sr for table %d", itabnum)
    endif
    istartframe = ioffset * itabsr
    ksampsplayed += ksmps * kspeed
    ; ar[] loscilx xamp, kcps, ifn, iwsize, ibas, istrt
    aouts[] loscilx 1, kspeed, itabnum, 4, 1, istartframe
    aenv = linsegr:a(0, ifade, 1, ifade, 0)
    kgain *= table:k(igaingroup, gi__subgains)
    again = lag:a(a(kgain), ilagtime)
    aenv *= again
    aouts = aouts * aenv
    inumouts = lenarray(aouts)
    kchan = 0
    while kchan < inumouts do
        outch kchan+ichan, aouts[kchan]
        kchan += 1
    od
    if ksampsplayed >= inumsamples then
        turnoff
    endif
endin

instr ${strset}
    Sstr = p4
    idx  = p5
    strset idx, Sstr
    turnoff
endin

instr ${ftsetparams}
    itabnum, isr, inumchannels passign 4
    ftsetparams itabnum, isr, inumchannels
    turnoff
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
    turnoff
endin

instr ${testaudio}
    pset 0, 0, 0, 0
    imode passign 4
    kchan init -1
    if imode == 0 then
        prints "Testaudio: pink noise mode\n"
        a0 pinker
    elseif imode == 1 then
        prints "Testaudio: sine tone mode\n"
        a0 oscili 0.1, 1000
    endif
    kswitch metro 1
    kchan = (kchan + kswitch) % nchnls
    outch kchan+1, a0
    if kswitch == 1 then
        println "Channel: %d", kchan
    endif
endin

instr ${tableInfo}
    itabnum, itok1, itok2, itok3, itok4 passign 4
    if ftexists(itabnum) == 0 then
        initerror sprintf("Table %d does not exist", itabnum)
    endif
    isr ftsr itabnum
    ichnls ftchnls itabnum
    inumframes nsamp itabnum
    ilen ftlen itabnum
    
    tabw_i isr, itok1, gi__responses
    tabw_i ichnls, itok2, gi__responses
    tabw_i inumframes, itok3, gi__responses
    tabw_i ilen, itok4, gi__responses
    outvalue "__sync__", itok1
endin

schedule "__init", 0, 1

"""

_busOrc = r'''

; The actual buses
ga__buses[]   init ${numAudioBuses}

; This table keeps track of the number of references a bus has
gi__busrefs ftgen 0, 0, ${numAudioBuses}, -2, 0

; A pool of bus indexes
gi__buspool pool_gen ${numAudioBuses}

; A dict mapping bustoken to bus number
gi__bustoken2num dict_new "int:float"

chn_k "_busTokenCount", 3

instr __businit
    chnset 0, "_busTokenCount"
    turnoff
endin

instr ${clearbuses}
    zeroarray ga__buses
endin

opcode busassign, i, j
    itoken xin
    if itoken == -1 then
        itoken chnget "_busTokenCount"
    endif
    ibus pool_pop gi__buspool, -1
    if ibus == -1 then
        initerror "busassign failed, out of buses"
    endif
    dict_set gi__bustoken2num, itoken, ibus
    chnset itoken+1, "_busTokenCount"
    xout ibus
endop

instr _busassign
    itoken = p4
    ibus busassign itoken
    turnoff
endin

instr _busrelease
    itoken = p4
    ibus dict_get gi__bustoken2num, itoken, -1
    if ibus == -1 then
        initerror sprintf("itoken %d has no bus assigned to it", itoken)
    endif
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
    turnoff    
endin

opcode _bususe, i, i
    itoken xin
    ibus dict_get gi__bustoken2num, itoken, -1
    if ibus == -1 then
        ibus = busassign(itoken)
    endif
    irefs tab_i ibus, gi__busrefs
    tabw_i irefs+1, ibus, gi__busrefs
    atstop "_busrelease", 0, 0.1, itoken
    xout ibus
endop

opcode busin, a, i
    itoken xin
    ibus = _bususe(itoken)
    xout ga__buses[ibus]
endop

opcode busout, 0, ia
    itoken, asig xin
    ibus = _bususe(itoken)
    ga__buses[ibus] = asig
endop

opcode busmix, 0, ia
    itoken, asig xin
    ibus = _bususe(itoken)
    ga__buses[ibus] = ga__buses[ibus] + asig
endop

schedule "__businit", 0, 1
'''


def _extractInstrNames(s:str) -> List[str]:
    names = []
    for line in s.splitlines():
        if match := re.search(r"\binstr\s+\$\{(\w+)\}", line):
            instrname = match.group(1)
            names.append(instrname)
    return names


_instrNames = _extractInstrNames(_orc)

# Constants
CONSTS = {
    'numtokens': 1000,
    'eventMaxSize': 1999,
    'highestInstrnum': 11500,
    'postProcInstrnum': 11000,
    'reservedTablesStart': 500,
    'reservedInstrsStart': 500,
    'numReservedTables': 2000,
    'maxNumInstrs': 10000
}

_tableNames = ['responses', 'subgains', 'tokenToInstrnum']

BUILTIN_INSTRS = {k:i for i, k in enumerate(_instrNames, start=CONSTS['reservedInstrsStart'])}
BUILTIN_INSTRS['clearbuses'] = CONSTS['postProcInstrnum']
BUILTIN_TABLES = {name:i for i, name in enumerate(_tableNames, start=1)}


@lru_cache(maxsize=0)
def orcTemplate(busSupport=True) -> Template:
    parts = [_orc]
    if busSupport:
        parts.append(_busOrc)
    orc = "\n".join(parts)
    subs: Dict[str, Any] = {name:f"{num} ; {name}"
                            for name, num in BUILTIN_INSTRS.items()}
    subs.update(BUILTIN_TABLES)
    subs.update(CONSTS)
    return Template(Template(orc).safe_substitute(**subs))



def makeOrc(sr:int, ksmps:int, nchnls:int, nchnls_i:int,
            backend:str, a4:float, globalcode:str="", includestr:str="",
            numAudioBuses:int=32):
    template = orcTemplate(busSupport=numAudioBuses>0)
    return template.substitute(
            sr=sr,
            ksmps=ksmps,
            nchnls=nchnls,
            nchnls_i=nchnls_i,
            backend=backend,
            a4=a4,
            globalcode=globalcode,
            includes=includestr,
            numAudioBuses=numAudioBuses)


def busSupportCode(numAudioBuses:int, clearBusesInstrnum:int) -> str:
    return Template(_busOrc).substitute(numAudioBuses=numAudioBuses,
                                        clearbuses=f'{clearBusesInstrnum} ;  clearbuses')
