from __future__ import annotations
from string import Template


_orc = r"""
sr     = {sr}
ksmps  = {ksmps}
nchnls = {nchnls}
nchnls_i = {nchnls_i}
0dbfs  = 1
A4     = {a4}

{includes}

gi__responses ftgen  ${responses}, 0, ${numtokens}, -2, 0
gi__subgains  ftgen  ${subgains},  0, 100, -2, 1
ga__buses[] init {numAudioBuses}
gi__busrefs ftgen ${busrefs}, 0, {numAudioBuses}, -2, 0

{globalcode}


; ---------------------------------
;          builtin-instruments
; ---------------------------------

instr _init
    ftset gi__subgains, 1
    prints "<<< Init done >>>\n"
    turnoff
endin

instr ${highestInstrnum}
    ; this is used to prevent a crash when an opcode is defined as part 
    ; of globalcode, and later on an instr is defined with a high instrnum
    turnoff
endin

instr ${cleanbuses}
    zeroarray ga__buses
endin

opcode _bususe, 0, i
    iidx xin
    ; prints "using bus %d \n", iidx
    ival tab_i iidx, gi__busrefs
    tabw_i ival+1, iidx, gi__busrefs
    atstop "_busdec", 0, -1, iidx
endop

instr _busdec
    iidx = p4
    ival tab_i iidx, gi__busrefs
    tabw_i ival-1, iidx, gi__busrefs
endin

opcode busnew, i, 0
    iidx ftfind gi__busrefs, 0
    if iidx < 0 then
        initerror "Out of buses"
    endif
    xout iidx    
endop

opcode busmix, 0, ia
    iidx, asig xin
    _bususe(iidx)
    ga__buses[iidx] = ga__buses[iidx] + asig
endop

opcode busout, 0, ia
    iidx, asig xin
    _bususe(iidx)
    ga__buses[iidx] = asig
endop

opcode busin, a, i
    iidx xin
    _bususe(iidx)
    xout ga__buses[iidx]
endop

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

; this is used to fill a table with the given pargs
instr ${filltable}
    itoken, ifn, itabidx, ilen passign 4
    iArgs[] passign 8, 8+ilen
    copya2ftab iArgs, ifn, itabidx
    if itoken > 0 then
        outvalue "__sync__", itoken
    endif
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
    prints ">>> instr maketable\n"

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

instr ${automate}
    iargtab = p4
    iargidx = p5
    inumpairs = p6
    imode = p7   ; 0 = linear, 1 = cos, 2 = lag
    iparam = p8
    idataidx = 9

    if ftexists:i(iargtab) == 0 then
        initerror sprintf("Table %d doesn't exist", iargtab)
    endif
    if ftlen(iargtab) <= iargidx then
        initerror sprintf("Table too small (%d <= %d)", ftlen(iargtab), iargidx)
    endif

    ilen = inumpairs*2
    iValues[] passign idataidx, idataidx+ilen
    ; WTF: slicearray's end index is inclusive
    iXs[] slicearray iValues, 0, ilen-1, 2
    iYs[] slicearray iValues, 1, ilen-1, 2
    kt timeinsts
    if imode == 0 then
        ky bpf kt, iXs, iYs
    elseif imode == 1 then
        ky bpfcos kt, iXs, iYs
    elseif imode == 2 then
        ky bpf kt, iXs, iYs
        ilagtime = max(iparam, 0.05)
        ky = lag(ky, ilagtime)
    else
        throwerror "init", sprintf("imode %d not supported", imode)
    endif
    if ftexists:k(iargtab) == 0 then
        throwerror "warning", \
            sprintf("automate: dest table (%d) was freed, stopping", iargtab)
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
    ichan  = p6
    ifade  = p8
    idur = filelen(Spath)
    know init 0
    iwsize = ispeed == 1 ? 1 : 4
    aouts[] diskin2 Spath, kspeed, 0, 0, 0, iwsize
    inumouts = lenarray(aouts)
    ichans[] genarray ichan, ichan+inumouts-1
    aenv linsegr 0, ifade, 1, ifade, 0
    aenv *= intrp(kgain)
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
    itab ftgen 0, 0, -1, Spath, 0, 0, ichan
    tabw_i itab, itoken, gi__responses
    outvalue "__sync__", itoken
endin

instr ${playgen1}
    ;             4     5      6       7     8     9          10
    ;             gain, speed, tabnum, chan, fade, starttime, gaingroup
    pset 0, 0, 0, 0,    1,     1,      1,    0.05, 0,         0
    kgain = p4
    kspeed = p5
    ksampsplayed = 0
    itabnum, ichan, ifade, ioffset, igaingroup passign 6
    inumsamples = nsamp(itabnum)
    itabsr = ftsr(itabnum)
    istartframe = ioffset * itabsr
    ksampsplayed += ksmps * kspeed
    ; ar[] loscilx xamp, kcps, ifn, iwsize, ibas, istrt
    aouts[] loscilx kgain, kspeed, itabnum, 4, 1, istartframe
    aenv = linsegr:a(0, ifade, 1, ifade, 0)
    ksubgain = table:k(igaingroup, gi__subgains)
    aenv *= a(ksubgain)
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
        ; prints "Testaudio: pink noise mode\n"
        a0 pinker
    elseif imode == 1 then
        ; prints "Testaudio: sine tone mode\n"
        a0 oscili 0.1, 1000
    endif
    kswitch metro 1
    kchan = (kchan + kswitch) % nchnls
    outch kchan+1, a0
    if kswitch == 1 then
        println "Channel: %d", kchan
    endif
endin

schedule "_init", 0, 1

"""

_instrNames = [
    'turnoff',
    'chnset',
    'filltable',
    'freetable',
    'maketable',
    'playsndfile',
    'uniqinstance',
    'scheduniq',
    'strset',
    'playgen1',
    'ftsetparams',
    'pwrite',
    'automate',
    'testaudio',
    'tabwrite',
    'automateTableViaTable',
    'automatePargViaTable',
    'readSndfile',
    'pingback',
    'nstrnum'
]

# Constants
CONSTS = {
    'numtokens': 1000,
    'eventMaxSize': 1999,
    'highestInstrnum': 11500,
    'postProcInstrnum': 11000,
    'reservedTablesStart': 500,
    'reservedInstrsStart': 500,
    'numReservedTables': 2000,
}

_tableNames = ['responses', 'subgains', 'busrefs']

BUILTIN_INSTRS = {k:i for i, k in enumerate(_instrNames, start=CONSTS['reservedInstrsStart'])}
BUILTIN_INSTRS['cleanbuses'] = CONSTS['postProcInstrnum']
BUILTIN_TABLES = {name:i for i, name in enumerate(_tableNames, start=1)}


ORC_TEMPLATE = Template(_orc).safe_substitute(
            **{name: f"{num} ; {name}" for name, num in BUILTIN_INSTRS.items()},
            **BUILTIN_TABLES,
            **CONSTS)
