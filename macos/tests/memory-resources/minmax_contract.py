from pathlib import Path
import sys,json,importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core import minmax,memory_resources as mr
from gui.sherloq_app.core.minmax_bounded import compute
spec=importlib.util.spec_from_file_location('gui.sherloq_app.core._old_minmax',Path(__file__).with_name('minmax_before.py'));old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
rng=np.random.default_rng(91);n=0
for shape in [(2,17),(33,49),(545,567)]:
 im=rng.integers(0,256,(*shape,3),dtype=np.uint8)
 for channel in range(5):
  for radius in (0,1,4):
   for low,high in [(0,1),(3,3),(4,0)]:
    p=(channel,low,high,radius);a=old.MinMaxEngine(im).compute(p);b=compute(im,p)
    assert all(np.array_equal(x,y) for x,y in zip(a,b)),(shape,p,[np.count_nonzero(x!=y) for x,y in zip(a,b)]);n+=1
limit=mr.MEMORY.limit;mr.MEMORY.limit=64*mr.MiB
try:
 im=rng.integers(0,256,(1601,1603,3),dtype=np.uint8);p=(4,3,1,4)
 a=minmax.MinMaxEngine(im).compute(p);b=old.MinMaxEngine(im).compute(p)
 assert isinstance(a[0],np.memmap) and all(np.array_equal(x,y) for x,y in zip(a,b))
finally:mr.MEMORY.limit=limit
r=dict(passed=True,exact_cases=n,public_dispatch=1);Path(__file__).with_name('minmax-results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
