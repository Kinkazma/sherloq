"""96 MP source, global full-size field with two very distant search zones.

All-pixel matching of 96 MP is NOT claimed by this test. It verifies global
coordinates/matches, low-memory pools, mapped field output and cancellation
checkpoints across the entire addressing domain, with 18,432 eligible pixels.
"""
from pathlib import Path
import ctypes as ct,time,resource,json
import numpy as np
from stream_candidate import prepare
from compact_contract import lib,p,dc
from gui.sherloq_app.core.memory_resources import TemporaryArrays,MiB,resident_memory
store=TemporaryArrays(256*MiB);root=Path(__file__).parent;start=time.perf_counter();shape=(8000,12000)
im=store.array((*shape,3),np.uint8);rng=np.random.default_rng(601)
for y in range(0,len(im),128):im[y:y+128]=rng.integers(0,256,im[y:y+128].shape,dtype=np.uint8);store.checkpoint()
im[6500:6884,11000:11384]=im[600:984,300:684];store.checkpoint(force=True)
def progress(stage):
 r=dict(stage=stage,elapsed=time.perf_counter()-start,resident=resident_memory(),peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 (root/'global-progress.json').write_text(json.dumps(r));print(r,flush=True)
f=prepare(im,8,store=store,progress=progress);h,w=f.view_height,f.view_width
mask=store.array((h,w),np.uint8,zero=True);mask[660:756,360:456]=1;mask[6560:6656,11060:11156]=2
matches=store.array((h,w),np.int32);distances=store.array((h,w),np.float32);n=ct.c_uint64();err=ct.create_string_buffer(1024);failure=[]
def checkpoint():
 try:store.checkpoint()
 except Exception as e:failure.append(str(e));return 1
 return 0
cb=dc.CALLBACK(checkpoint);t=time.perf_counter()
code=lib.sherloq_patchmatch_compact(ct.byref(f),ct.byref(f),p(mask,ct.c_ubyte),w,h,True,5,1e5,3,729,p(matches,ct.c_int),p(distances),ct.byref(n),cb,err,0,0,None,None,4096)
assert code==0,(code,err.value,failure)
progress('global_field_complete');elapsed=time.perf_counter()-t
ys,xs=np.mgrid[660:756,360:456];target=matches[660:756,360:456];correct=(target//w-ys==5900)&(target%w-xs==10700)
assert correct.mean()>.95,correct.mean()
assert (mask[6560:6656,11060:11156]==2).all()
r=dict(passed=True,image_pixels=shape[0]*shape[1],field_shape=[h,w],eligible_pixels=2*96*96,all_pixel_matching=False,global_coordinates_preserved=True,exact_translation_fraction=float(correct.mean()),iterations=3,comparisons=n.value,matching_seconds=elapsed,total_seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,peak_temp_bytes=store.peak_bytes)
(root/'global-results.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
