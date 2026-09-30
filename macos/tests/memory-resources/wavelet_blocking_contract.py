from pathlib import Path
import sys,json,importlib.util,tempfile,time,resource
import numpy as np,cv2 as cv
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core.wavelet_blocking import WaveletBlockingEngine
from gui.sherloq_app.core.wavelet_blocking_bounded import compute
from gui.sherloq_app.core import memory_resources as mr
spec=importlib.util.spec_from_file_location('gui.sherloq_app.core._wavelet_blocking_before',Path(__file__).with_name('wavelet_blocking_before.py'));old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
rng=np.random.default_rng(801);n=0
with tempfile.TemporaryDirectory() as temp:
 for shape in [(1,1),(17,19),(256,256),(545,567)]:
  im=rng.integers(0,256,(*shape,3),dtype=np.uint8);path=Path(temp)/'image.png';cv.imwrite(str(path),im)
  for filename in (str(path),''):
   for block in (1,3,8):
    reference=old.WaveletBlockingEngine(filename,im);engine=WaveletBlockingEngine(filename,im)
    if block>min(reference.detail().shape):continue
    a=reference.compute(block);b=compute(engine,block)
    assert np.array_equal(reference.detail(),engine.detail()),(shape,block,'detail')
    assert all(np.array_equal(x,y) for x,y in zip(a,b)),(shape,block,'outputs',[np.max(abs(x.astype(float)-y)) for x,y in zip(a,b)])
    assert reference.source_mode==engine.source_mode;n+=1
 limit=mr.MEMORY.limit;mr.MEMORY.limit=128*mr.MiB
 try:
  im=rng.integers(0,256,(1601,1603,3),dtype=np.uint8);cv.imwrite(str(path),im)
  a=WaveletBlockingEngine(str(path),im).compute(8);b=old.WaveletBlockingEngine(str(path),im).compute(8)
  assert isinstance(a[0],np.memmap) and all(np.array_equal(x,y) for x,y in zip(a,b))
 finally:mr.MEMORY.limit=limit
r=dict(passed=True,exact_cases=n,coefficients_exact=True,public_dispatch=True)
Path(__file__).with_name('wavelet-blocking-results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
