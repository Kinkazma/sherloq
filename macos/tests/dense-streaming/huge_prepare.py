from pathlib import Path
import sys,time,json,resource,gc
import numpy as np
from stream_candidate import prepare
from compact_contract import lib,p,unpack,Compact
from gui.sherloq_app.core.memory_resources import TemporaryArrays,resident_memory,MiB
import ctypes as ct
root=Path(__file__).parent;start=time.perf_counter();store=TemporaryArrays(256*MiB)
shape=(8000,12000);im=store.array((*shape,3),np.uint8);rng=np.random.default_rng(601)
for y in range(0,shape[0],128):im[y:y+128]=rng.integers(0,256,im[y:y+128].shape,dtype=np.uint8);store.checkpoint()
# A distant copied patch crosses preparation tiles; keep original pixels.
im[6500:6884,11000:11384]=im[600:984,300:684];store.checkpoint(force=True)
def progress(stage):
 r=dict(stage=stage,elapsed=time.perf_counter()-start,resident=resident_memory(),peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,disk=store.bytes)
 (root/'huge-progress.json').write_text(json.dumps(r));print(r,flush=True)
f=prepare(im,8,store=store,progress=progress)
# Reconstruct distant descriptors and verify finite data; full NNF is separate.
samples=[(600+100,300+100),(6500+100,11000+100),(100,100),(7900,11900)]
a=np.empty((len(samples),128),np.float32)
for i,(y,x) in enumerate(samples):lib.sherloq_sift_unpack(ct.byref(f),y*f.view_width+x,1,p(a[i]))
assert np.isfinite(a).all()
progress('prepared')
r=dict(image_pixels=shape[0]*shape[1],shape=shape,prepared=True,full_matching_complete=False,seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,peak_temp_bytes=store.peak_bytes,compact_bytes=sum(x.nbytes for x in f.owners),expanded_descriptor_bytes=f.view_width*f.view_height*128*4,copy_descriptor_max_error=float(np.max(abs(a[0]-a[1]))))
(root/'huge-results.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
