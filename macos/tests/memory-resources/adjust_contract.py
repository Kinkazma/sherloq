from pathlib import Path
import sys,json,importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core.adjust_bounded import compute
from gui.sherloq_app.core.adjust import AdjustEngine
from gui.sherloq_app.core import memory_resources as mr
spec=importlib.util.spec_from_file_location('gui.sherloq_app.core._adjust_before',Path(__file__).with_name('adjust_before.py'));old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
rng=np.random.default_rng(31);n=0
for shape in [(1,13),(48,48),(48,47),(47,48),(545,567)]:
 im=rng.integers(0,256,(*shape,3),dtype=np.uint8)
 for equalize in range(6):
  for threshold in (0,70,255):
   p=(10,-8,80,12,8,4,120,245,12,threshold,equalize,True)
   a=old.AdjustEngine(im).compute(p);b=compute(im,p)
   assert np.array_equal(a,b),(shape,p,np.count_nonzero(a!=b),np.max(abs(a.astype(int)-b)));n+=1
limit=mr.MEMORY.limit;mr.MEMORY.limit=80*mr.MiB
try:
 im=rng.integers(0,256,(1601,1603,3),dtype=np.uint8);p=(10,-8,80,12,8,4,120,245,12,0,3,True)
 a=AdjustEngine(im).compute(p);b=old.AdjustEngine(im).compute(p)
 assert isinstance(a,np.memmap) and np.array_equal(a,b)
finally:mr.MEMORY.limit=limit
r=dict(passed=True,exact_cases=n,public_dispatch=1);Path(__file__).with_name('adjust-results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
