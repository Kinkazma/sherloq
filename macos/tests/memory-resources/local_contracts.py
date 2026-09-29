from pathlib import Path
import sys,json,importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core import pixel_stats,bit_planes,defect_pixels,memory_resources as mr
from gui.sherloq_app.core.bounded_ops import local_result

def reference(name):
 spec=importlib.util.spec_from_file_location('gui.sherloq_app.core._before_'+name,Path(__file__).with_name(name+'_before.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old={n:reference(n) for n in ('pixel_stats','bit_planes','defect_pixels')};rng=np.random.default_rng(677);rows=[]
# 513+ pixels crosses real adapter tile boundaries; ties and isolated spots.
im=rng.integers(0,256,(545,567,3),dtype=np.uint8);im[500:540,495:540]=127;im[511,511]=255;im[513,513]=0
for mode in ('min','avg','max'):
 for inclusive in (False,True):
  params=(mode,inclusive);a=old['pixel_stats'].StatsEngine(im).compute(params);b=local_result(im,lambda roi:pixel_stats.StatsEngine(roi)._compute(params))[0]
  assert np.array_equal(a,b);rows.append(['stats',params])
for channel in range(5):
 for filtering in range(3):
  for bit in (0,7):
   params=(channel,bit,filtering);a=old['bit_planes'].PlanesEngine(im).compute(params);b=local_result(im,lambda roi:bit_planes.PlanesEngine(roi)._compute(params),halo=1 if filtering else 0)[0]
   assert np.array_equal(a,b),(params,np.count_nonzero(a!=b));rows.append(['planes',params])
for radius in (1,2):
 for kind in (0,1,2):
  for mode in (0,1,2):
   params=((radius,7,60,kind,True),mode);a=old['defect_pixels'].DefectEngine(im).compute(params)
   def one(roi):
    out,flags,_,median=defect_pixels.DefectEngine(roi)._compute(params);return out,flags,median
   result,flags,median=local_result(im,one,halo=radius,dtypes=(np.uint8,)*3)
   for x,y in ((a[0],result),(a[1],flags),(a[3],median)):assert np.array_equal(x,y),(params,np.count_nonzero(x!=y))
   assert a[2]==np.count_nonzero(np.any(flags,axis=2));rows.append(['defects',params])
# Public entry point dispatch, not only helper calls.
limit=mr.MEMORY.limit;mr.MEMORY.limit=64*mr.MiB
try:
 im=rng.integers(0,256,(2501,2503,3),dtype=np.uint8)
 for name,cls,params in [('pixel_stats','StatsEngine',('avg',True)),('bit_planes','PlanesEngine',(4,2,2)),('defect_pixels','DefectEngine',((2,10,40,0,True),2))]:
  current=getattr(globals()[name],cls)(im).compute(params);expected=getattr(old[name],cls)(im).compute(params)
  if name=='defect_pixels':assert all(np.array_equal(a,b) for a,b in zip(current,expected));assert isinstance(current[0],np.memmap)
  else:assert np.array_equal(current,expected) and isinstance(current,np.memmap)
finally:mr.MEMORY.limit=limit
r=dict(passed=True,exact_cases=len(rows),public_dispatch=3)
Path(__file__).with_name('local-results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
