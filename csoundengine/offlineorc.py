from __future__ import annotations
from functools import cache
import textwrap


_prelude = r'''
gi__soundfontIndexes dict_new "str:float"
gi__soundfontIndexCounter init 1000
; maxNumInstrs  = 10000
gi__tokenToInstrnum ftgen 0, 0, 10000, -2, 0

; numtokens = 1000
gi__responses   ftgen  0, 0, 1000, -2, 0

gi__subgains    ftgen 0, 0, 100, -2, 0
ftset gi__subgains, 1


chn_k "_soundfontPresetCount", 3

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

_offlineOrc = r'''

instr _stop
    ; turnoff inum (match instr number exactly, allow release)
    inum = p4
    turnoff2_i inum, 4, 1
    turnoff
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

instr _automatePargViaTable
  ; automates a parg from a table
  ip1 = p4
  ipindex = p5
  itabpairs = p6  ; a table containing flat pairs t0, y0, t1, y1, ...
  imode = p7;  interpolation method
  Sinterpmethod = strget(imode)
  if ftexists:i(itabpairs) == 0 then
    initerror sprintf("Table with pairs %d does not exists", itabpairs)
  endif 
  ftfree itabpairs, 1

  kt timeinsts
  kidx bisect kt, itabpairs, 2, 0
  ky interp1d kidx, itabpairs, Sinterpmethod, 2, 1
  pwrite ip1, ipindex, ky
endin 

instr _automateTableViaTable
  ; automates a slot within a table from another table
  itabnum = p4
  ipindex = p5
  itabpairs = p6
  imode = p7
  Sinterpmethod = strget(imode)
  if ftexists:i(itabpairs) == 0 then
    initerror sprintf("Table with pairs %d does not exists", itabpairs)
  endif 
  ftfree itabpairs, 1
  kt timeinsts
  kidx bisect kt, itabpairs, 2, 0
  ky interp1d kidx, itabpairs, Sinterpmethod, 2, 1
  tabw ky, ipindex, itabnum
endin 

instr _pwrite
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

'''


@cache
def prelude() -> str:
    """
    Dedented version of _prelude
    """
    return textwrap.dedent(_prelude)


@cache
def orchestra() -> str:
    """
    Dedented version of _offlineOrc
    """
    return textwrap.dedent(_offlineOrc)
