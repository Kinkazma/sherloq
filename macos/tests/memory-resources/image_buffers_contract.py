from pathlib import Path
import sys,json,tempfile,hashlib,time,resource
import numpy as np,cv2 as cv
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core.image_buffers import resize_rgb,image_sha256
from gui.sherloq_app.core.memory_resources import TemporaryArrays,MiB,resident_memory
rng=np.random.default_rng(98);n=0
for shape in ((1,17),(545,567),(1200,1800)):
 image=rng.integers(0,256,(*shape,3),dtype=np.uint8)
 for source in (image,image[:,::2],image[::-1,:,::-1]):
  assert image_sha256(source)==hashlib.sha256(source.tobytes()).hexdigest()
  for target in ((1024,1024),(55,30),(567,545)):
   a=cv.resize(source[:,:,::-1],target);b=resize_rgb(source,target)
   assert np.array_equal(a,b),(shape,target);n+=1
# Read-only mappings returned by existing subprocess workers are supported.
with tempfile.TemporaryFile() as f:
 f.truncate(128*MiB);rw=np.memmap(f,np.uint8,'r+',shape=(128*MiB,));rw[:]=19;rw.flush();del rw
 ro=np.memmap(f,np.uint8,'r',shape=(128*MiB,));store=TemporaryArrays();store.watch(ro)
 assert int(ro.sum())==19*len(ro);before=resident_memory();store.checkpoint(force=True);released=before-resident_memory()
 assert (ro[::4096]==19).all() and released>96*MiB,released
image=rng.integers(0,256,(8000,12000,3),dtype=np.uint8)
t=time.perf_counter();a=resize_rgb(image,(1024,1024));seconds=time.perf_counter()-t
b=cv.resize(image[:,:,::-1],(1024,1024));assert np.array_equal(a,b)
assert image_sha256(image)==hashlib.sha256(memoryview(image)).hexdigest()
r=dict(passed=True,exact_resize_cases=n,large_pixels=96000000,large_resize_bit_exact=True,sha256_exact=True,readonly_pages_released=released,resize_seconds=seconds)
Path(__file__).with_name('image-buffers-results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
