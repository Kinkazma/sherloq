from pathlib import Path
import sys,json,importlib.util
import numpy as np,pywt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core.interactive import WaveletEngine
from gui.sherloq_app.core.bounded_wavelets import compute,decompose,reconstruct
from gui.sherloq_app.core import memory_resources as mr
spec=importlib.util.spec_from_file_location('gui.sherloq_app.core._threshold_before',Path(__file__).with_name('interactive_before_wavelet.py'));old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
rng=np.random.default_rng(71);n=0;fields=0
for shape in [(33,45),(545,567)]:
 im=rng.integers(0,256,(*shape,3),dtype=np.uint8)
 for wavelet in ('haar','db2','db8','sym4','coif3','bior3.5','rbio2.2','dmey'):
  store=mr.TemporaryArrays(32*mr.MiB);a=pywt.wavedec2(im[:,:,0],wavelet);b=decompose(im[:,:,0],pywt.Wavelet(wavelet),store)
  assert len(a)==len(b)
  for x,y in zip(a,b):
   if isinstance(x,tuple):assert all(np.array_equal(p,q) for p,q in zip(x,y)),(shape,wavelet,'detail')
   else:assert np.array_equal(x,y),(shape,wavelet,'approximation')
  expected=pywt.waverec2(a,wavelet);actual=reconstruct(b,pywt.Wavelet(wavelet),store)
  assert np.array_equal(expected,actual),(shape,wavelet,'reconstruction',np.max(abs(expected-actual)));fields+=1
  ref=old.WaveletEngine(im);engine=WaveletEngine(im)
  for threshold,level,mode in [(0,0,'soft'),(10,1,'soft'),(10,3,'hard'),(30,30,'garrote'),(90,3,'greater'),(90,3,'less')]:
   p=(wavelet,threshold,level,mode);x=ref.compute(p);y=compute(engine,p)
   assert np.array_equal(x,y),(shape,p,np.count_nonzero(x!=y));n+=1
  print(shape,wavelet,'passed',flush=True)
# Retained transforms and prefixes are shared across RAM/bounded paths.
p=('db8',30,3,'soft');engine=WaveletEngine(im);a=compute(engine,p);engine.results.clear();b=engine._compute(p)
assert np.array_equal(a,b);engine.results.clear();c=compute(engine,('db8',10,3,'hard'))
assert np.array_equal(c,old.WaveletEngine(im).compute(('db8',10,3,'hard')))
limit=mr.MEMORY.limit;mr.MEMORY.limit=80*mr.MiB
try:
 im=rng.integers(0,256,(1601,1603,3),dtype=np.uint8);p=('db8',20,3,'soft');a=WaveletEngine(im).compute(p);b=old.WaveletEngine(im).compute(p)
 assert isinstance(a,np.memmap) and np.array_equal(a,b)
finally:mr.MEMORY.limit=limit
r=dict(passed=True,exact_cases=n,exact_global_transforms=fields,public_dispatch=True)
Path(__file__).with_name('wavelets-results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
