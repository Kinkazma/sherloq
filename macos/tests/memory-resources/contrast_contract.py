from pathlib import Path
import sys,json,importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core.contrast import ContrastEngine
from gui.sherloq_app.core.contrast_bounded import compute
from gui.sherloq_app.core import memory_resources as mr
spec=importlib.util.spec_from_file_location('gui.sherloq_app.core._contrast_before',Path(__file__).with_name('contrast_before.py'));old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
rng=np.random.default_rng(91);n=0
for shape in [(1,1),(1,257),(258,1),(32,64),(128,256),(545,567)]:
 for kind in ('noise','flat','edges'):
  image=rng.integers(0,256,(*shape,3),dtype=np.uint8)
  if kind=='flat':image[:]=64
  if kind=='edges':image[:shape[0]//2]=0;image[shape[0]//2:]=255
  engine=ContrastEngine(image);ref=old.ContrastEngine(image)
  for block in (32,64,128,256):
   for mode in (0,1,2):
    p=(block,mode);a=compute(engine,p);b=ref.compute(p)
    assert np.array_equal(a,b),(shape,kind,p,'display',np.count_nonzero(a!=b))
    assert all(np.array_equal(x,y) for x,y in zip(engine.maps.get(block),ref.maps.get(block))),(shape,kind,p,'maps')
    n+=1
 print(shape,'passed',flush=True)
image=rng.integers(0,256,(1601,1603,3),dtype=np.uint8);limit=mr.MEMORY.limit;mr.MEMORY.limit=80*mr.MiB
try:a=ContrastEngine(image).compute((32,2))
finally:mr.MEMORY.limit=limit
assert isinstance(a,np.memmap) and np.array_equal(a,old.ContrastEngine(image).compute((32,2)))
r=dict(passed=True,exact_cases=n,public_dispatch=True)
Path(__file__).with_name('contrast-results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
