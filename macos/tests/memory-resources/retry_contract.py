from pathlib import Path
import sys,json
import cv2 as cv,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core import gradient,pixel_stats,bit_planes,defect_pixels,color_spaces,noise,interactive,minmax,adjust,memory_resources as mr
rng=np.random.default_rng(752);im=rng.integers(0,256,(101,107,3),dtype=np.uint8);n=0
for name,cls,p in [('gradient','GradientEngine',(95,3,True,False)),('pixel_stats','StatsEngine',('avg',True)),('bit_planes','PlanesEngine',(4,2,2)),('defect_pixels','DefectEngine',((2,10,40,0,True),2)),('color_spaces','SpaceEngine',('cmyk',2)),('noise','NoiseEngine',(1,3,20,False,False,0)),('interactive','EchoEngine',(3,85,True)),('minmax','MinMaxEngine',(4,3,1,4)),('adjust','AdjustEngine',(10,-8,80,12,8,4,120,245,12,0,3,True))]:
 cls=getattr(globals()[name],cls);expected=cls(im)._compute(p)
 for kind in ('numpy','opencv'):
  engine=cls(im)
  def fail(*a):
   if kind=='numpy':raise MemoryError('Injected allocation failure')
   err=cv.error('Injected native allocation failure');err.code=cv.Error.StsNoMem;raise err
  engine._compute=fail
  out=engine.compute(p)
  aa=out if isinstance(out,tuple) else (out,);bb=expected if isinstance(expected,tuple) else (expected,)
  assert all(np.array_equal(a,b) for a,b in zip(aa,bb)),(name,kind)
  assert mr.MEMORY.used==0;n+=1
# Other errors must not be hidden, and a failed fallback is never retried.
budget=mr.MemoryCoordinator(128*mr.MiB,available=lambda:8*mr.GiB);calls=[]
def invalid():raise ValueError('actual bug')
try:budget.execute(1,64*mr.MiB,invalid,lambda:calls.append(1))
except ValueError:pass
else:raise AssertionError('error suppressed')
assert not calls
r=dict(passed=True,public_allocation_failure_recovery_cases=n,unrelated_errors_preserved=True,claims_released=True)
Path(__file__).with_name('retry-results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
