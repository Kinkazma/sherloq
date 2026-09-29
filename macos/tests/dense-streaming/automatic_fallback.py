from pathlib import Path
import ctypes as ct,sys,time,json,resource,gc
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core import dense_copy as dc,dense_streaming as ds
from gui.sherloq_app.core.dense_parallel import BUDGET
from gui.sherloq_app.core.memory_resources import MEMORY,MiB
from gui.sherloq_app.core.cache_budget import GLOBAL_CACHE_BUDGET
old=dc.library();lib=ct.CDLL(str(Path(__file__).with_name('candidate.dylib')))
for name in ('sherloq_dense_features','sherloq_patchmatch','sherloq_patchmatch_gap','sherloq_patchmatch_metric'):
 getattr(lib,name).argtypes=getattr(old,name).argtypes;getattr(lib,name).restype=ct.c_int
dc._LIB=lib;ds.native.cache_clear();rng=np.random.default_rng(41)
im=rng.integers(0,256,(2001,2201,3),dtype=np.uint8);im[1500:1670,1800:1970]=im[150:320,160:330]
args=('PatchMatch SIFT',10000,4000,5,.06,(),False,(8,1,False,2.),lambda:False,lambda *a:None)
t=time.perf_counter();reference=dc.DenseCopyEngine(im).analyze(*args,backend='metal',geometry=('Affine',3.,5));normal=time.perf_counter()-t
print(dict(stage='ram_reference',seconds=normal),flush=True)
GLOBAL_CACHE_BUDGET.clear();gc.collect();limit=BUDGET.limit;BUDGET.limit=512*MiB
try:
 mode,reserved=ds.memory_plan(*im.shape[:2],1,8,False,'metal');assert mode=='mapped',(mode,reserved)
 engine=dc.DenseCopyEngine(im);t=time.perf_counter();actual=engine.analyze(*args,backend='metal',geometry=('Affine',3.,5));elapsed=time.perf_counter()-t
 for key in ('points','pairs','members','dense_count','dense_consistent_count','candidate_comparisons'):assert np.array_equal(reference[key],actual[key]),key
 max_error=0.
 for a,b in zip(reference['dense_maps'],actual['dense_maps']):
  for key in ('targets','distances_squared','consistent_mask'):assert np.array_equal(a[key],b[key]),key
  max_error=max(max_error,float(np.max(abs(a['coherence_error']-b['coherence_error']))))
 assert max_error<1e-4,max_error
 assert any(key[0][-1]=='mapped' for key in engine.fields.items)
 assert BUDGET.used==MEMORY.used==0
 r=dict(passed=True,image_pixels=im.shape[0]*im.shape[1],all_pixel_matching=True,automatic_mode=mode,reservation_bytes=reserved,outputs_exact=True,coherence_max_error=max_error,normal_seconds=normal,mapped_seconds=elapsed,iterations=1)
 Path(__file__).with_name('automatic-fallback-results.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
finally:BUDGET.limit=limit
