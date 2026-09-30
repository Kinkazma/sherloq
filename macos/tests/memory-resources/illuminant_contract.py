from pathlib import Path
import sys,json,importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core.illuminant import IlluminantEngine
from gui.sherloq_app.core.illuminant_bounded import compute
from gui.sherloq_app.core import memory_resources as mr
spec=importlib.util.spec_from_file_location('gui.sherloq_app.core._illuminant_before',Path(__file__).with_name('illuminant_before.py'));old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
rng=np.random.default_rng(911);n=0
for shape in [(13,19),(545,567)]:
 image=rng.integers(0,256,(*shape,3),dtype=np.uint8)
 image[:10,:10]=0;image[-10:,-10:]=255
 engine=IlluminantEngine(image);ref=old.IlluminantEngine(image)
 for block in (32,64,128,256):
  for method in (0,1,2):
   for linear in (False,True):
    for exclude in (False,True):
     for mode in (0,1,2):
      p=((block,method,linear,exclude),mode);actual=compute(engine,p);expected=ref.compute(p)
      assert all(np.array_equal(a,b) for a,b in zip((actual[0],*actual[1]),(expected[0],*expected[1]))),(shape,p)
      n+=1
 print(shape,'passed',flush=True)
# Dispatch through the public engine, with both RAM/bounded caches reusable.
image=rng.integers(0,256,(1601,1603,3),dtype=np.uint8);p=((32,1,True,True),0)
limit=mr.MEMORY.limit;mr.MEMORY.limit=80*mr.MiB
try:engine=IlluminantEngine(image);a=engine.compute(p)
finally:mr.MEMORY.limit=limit
b=old.IlluminantEngine(image).compute(p)
assert isinstance(a[0],np.memmap)
assert all(np.array_equal(x,y) for x,y in zip((a[0],*a[1]),(b[0],*b[1])))
engine.renders.clear();assert np.array_equal(engine._compute(p)[0],a[0])
r=dict(passed=True,exact_cases=n,public_dispatch=True,shared_caches=True)
Path(__file__).with_name('illuminant-results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
