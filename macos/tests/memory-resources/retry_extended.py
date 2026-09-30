from pathlib import Path
import sys,json
import cv2 as cv,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core.interactive import WaveletEngine
from gui.sherloq_app.core.wavelet_blocking import WaveletBlockingEngine
from gui.sherloq_app.core.illuminant import IlluminantEngine
from gui.sherloq_app.core.contrast import ContrastEngine
from gui.sherloq_app.core import memory_resources as mr
rng=np.random.default_rng(94);image=rng.integers(0,256,(129,135,3),dtype=np.uint8);n=0

def equal(a,b):
 if isinstance(a,tuple):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return np.array_equal(a,b)
for create,params in [(lambda:WaveletEngine(image),('db8',20,3,'soft')),(lambda:WaveletBlockingEngine('',image),8),(lambda:IlluminantEngine(image),((32,1,True,True),0)),(lambda:ContrastEngine(image),(32,2))]:
 expected=create()._compute(params)
 for kind in ('numpy','opencv'):
  engine=create()
  def fail(*a):
   if kind=='numpy':raise MemoryError('Injected allocation failure')
   error=cv.error('Injected allocation failure');error.code=cv.Error.StsNoMem;raise error
  engine._compute=fail;actual=engine.compute(params)
  assert equal(actual,expected);assert mr.MEMORY.used==0;n+=1
r=dict(passed=True,public_recovery_cases=n,claims_released=True)
Path(__file__).with_name('retry-extended-results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
