"""Full 96 MP noisy domain, one PatchMatch iteration, automatic RAM planning."""
from pathlib import Path
import sys,ctypes as ct,json,time,resource
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core import dense_copy as dc,dense_streaming as ds
from gui.sherloq_app.core.dense_parallel import BUDGET
from gui.sherloq_app.core.memory_resources import MEMORY,resident_memory
old=dc.library();candidate=ct.CDLL(str(Path(__file__).with_name('candidate.dylib')))
for name in ('sherloq_dense_features','sherloq_patchmatch','sherloq_patchmatch_gap','sherloq_patchmatch_metric'):
 getattr(candidate,name).argtypes=getattr(old,name).argtypes;getattr(candidate,name).restype=ct.c_int
dc._LIB=candidate;ds.native.cache_clear();rng=np.random.default_rng(1024);shape=(8000,12000);im=np.empty((*shape,3),np.uint8)
for y in range(0,len(im),128):im[y:y+128]=rng.integers(0,256,im[y:y+128].shape,dtype=np.uint8)
im[6500:6884,11000:11384]=im[600:984,300:684]
mode,reservation=ds.memory_plan(*shape,1,8,False,'metal');assert mode in ('compact','mapped'),mode
start=time.perf_counter()
def progress(n,text):
 r=dict(progress=n,message=text,seconds=time.perf_counter()-start,resident=resident_memory(),peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 Path(__file__).with_name('huge-full-progress.json').write_text(json.dumps(r));print(r,flush=True)
engine=dc.DenseCopyEngine(im)
r=engine.analyze('PatchMatch SIFT',20000,20000,5,.06,(),False,(8,1,False,2.),lambda:False,progress,backend='metal',geometry=('Affine',3.,5))
assert BUDGET.used==MEMORY.used==0
assert len(r['dense_maps'])==1 and r['dense_maps'][0]['targets'].shape==(7976,11976)
report=dict(passed=True,image_pixels=shape[0]*shape[1],all_pixel_matching=True,automatic_mode=mode,reservation=reservation,iterations=1,extended_profile=False,seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,comparisons=r['candidate_comparisons'],matched_count=r['dense_count'],coherent_count=r['dense_consistent_count'],display_links=len(r['pairs']))
Path(__file__).with_name('huge-full-results.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
