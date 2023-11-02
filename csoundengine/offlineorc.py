from __future__ import annotations
from functools import cache
import textwrap
from string import Template


_prelude = r'''

gi__dynargsNumSlots = ${controlNumSlots}
gi__dynargsSliceSize = ${controlArgsPerInstr}

gi__tokenToDynargsSlot dict_new "int:float"
gi__dynargsSlotsPool pool_gen 1, gi__dynargsNumSlots
gi__dynargsTableSize = gi__dynargsNumSlots * gi__dynargsSliceSize
gi__dynargsTable ftgen 0, 0, gi__dynargsTableSize, -2, 0

gi__soundfontIndexes dict_new "str:float"
gi__soundfontIndexCounter init 1000

; numtokens = 1000
gi__responses   ftgen  0, 0, 1000, -2, 0

gi__subgains    ftgen 0, 0, 100, -2, 0
ftset gi__subgains, 1

chn_k "_soundfontPresetCount", 3

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
'''

# The order of the named instruments is relevant.

_offlineOrc = r'''

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

'''


# @cache
def prelude(controlNumSlots: int,
            controlArgsPerInstr: int,
            ) -> str:
    """
    Dedented version of _prelude

    """
    return Template(_prelude).substitute(
        controlNumSlots=controlNumSlots,
        controlArgsPerInstr=controlArgsPerInstr,
    )


@cache
def orchestra() -> str:
    """
    Dedented version of _offlineOrc
    """
    return textwrap.dedent(_offlineOrc)
