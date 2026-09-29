from pathlib import Path
import ctypes as ct,sys,time,json,importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core import dense_copy as dc
from gui.sherloq_app.core.gradient import GradientEngine
spec=importlib.util.spec_from_file_location('gui.sherloq_app.core._oldgradient',Path(__file__).with_name('gradient_before.py'));old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
candidate=dc.library();reference=ct.CDLL(str(Path(__file__).with_name('legacy_bridge.dylib')))
for name in ('sherloq_patchmatch_metric',):
 getattr(reference,name).argtypes=getattr(candidate,name).argtypes;getattr(reference,name).restype=ct.c_int
rng=np.random.default_rng(713);a=rng.random((211,251,128),dtype=np.float32);b=rng.random(a.shape,dtype=np.float32);mask=np.ones(a.shape[:2],np.uint8);times={'reference':[],'candidate':[]};expected=None
for iteration in range(8):
 for label,lib in (('reference',reference),('candidate',candidate)) if iteration%2==0 else (('candidate',candidate),('reference',reference)):
  dc._LIB=lib;t=time.perf_counter();result=dc.field(a,b,mask,500,5,iterations=3);elapsed=time.perf_counter()-t
  if expected is None:expected=result
  assert all(np.array_equal(x,y) for x,y in zip(expected,result))
  if iteration:times[label].append(elapsed)
dc._LIB=reference
im=rng.integers(0,256,(800,1100,3),dtype=np.uint8);gtimes={'reference':[],'candidate':[]};expected=None
for iteration in range(12):
 for label,cls in (('reference',old.GradientEngine),('candidate',GradientEngine)) if iteration%2==0 else (('candidate',GradientEngine),('reference',old.GradientEngine)):
  t=time.perf_counter();result=cls(im).compute((95,3,True,False));elapsed=time.perf_counter()-t
  if expected is None:expected=result
  assert np.array_equal(expected,result)
  if iteration:gtimes[label].append(elapsed)
r=dict(passed=True,patchmatch_seconds=times,gradient_seconds=gtimes,patchmatch_median_ratio=float(np.median(times['candidate'])/np.median(times['reference'])),gradient_median_ratio=float(np.median(gtimes['candidate'])/np.median(gtimes['reference'])))
Path(__file__).with_name('fast-path-results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
