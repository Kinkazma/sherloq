import sys,json,time,importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core.gradient import GradientEngine
from gui.sherloq_app.core.gradient_bounded import compute
from gui.sherloq_app.core import memory_resources as mr
spec=importlib.util.spec_from_file_location('gui.sherloq_app.core._gradient_reference',Path(__file__).with_name('gradient_before.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
rng=np.random.default_rng(10);cases=[]
for shape in ((5,7),(133,149),(579,637)):
 for kind in ('noise','flat','ramp'):
  im=rng.integers(0,256,(*shape,3),dtype=np.uint8) if kind=='noise' else np.full((*shape,3),127,np.uint8) if kind=='flat' else np.broadcast_to((np.arange(shape[1])%256)[None,:,None],(*shape,3)).astype(np.uint8)
  ref=module.GradientEngine(im)
  for mode in range(4):
   for equalize in (False,True):
    params=(95,mode,mode%2==1,equalize);a=ref.compute(params);b=compute(im,params)
    assert np.array_equal(a,b),(shape,kind,params,np.max(np.abs(a.astype(int)-b)),np.count_nonzero(a!=b))
    cases.append((shape,kind,params))
# Actual engine dispatch forced through the shared budget, with same result.
limit=mr.MEMORY.limit;mr.MEMORY.limit=64*mr.MiB
try:
 im=rng.integers(0,256,(1201,1301,3),dtype=np.uint8);params=(80,3,True,True)
 actual=GradientEngine(im).compute(params);expected=module.GradientEngine(im).compute(params)
 assert np.array_equal(actual,expected) and isinstance(actual,np.memmap)
finally:mr.MEMORY.limit=limit
r=dict(passed=True,exact_cases=len(cases),engine_dispatch=True)
Path(__file__).with_name('gradient-results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
