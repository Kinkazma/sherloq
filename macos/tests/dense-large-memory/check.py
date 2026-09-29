import sys,json,importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core import metal_dense as new
from gui.sherloq_app.core import dense_copy as dc
from gui.sherloq_app.core import sift_frames as sf
from gui.sherloq_app.core.dense_memory import workspace_bytes
from gui.sherloq_app.core.dense_parallel import BUDGET
spec=importlib.util.spec_from_file_location('gui.sherloq_app.core._memory_baseline',Path(__file__).with_name('metal_dense_before.py'));old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
rng=np.random.default_rng(773);cases=[]
for shape in ((91,113),(180,231),(512,768)):
 im=rng.integers(0,256,(*shape,3),dtype=np.uint8)
 for patch,flip in ((6,False),(8,False),(8,True),(12,True)):
  a=old.features_sift(im,patch,flip);b=new.features_sift(im,patch,flip)
  assert a[2]==b[2]
  for x,y in zip(a[:2],b[:2]):assert np.array_equal(x,y),(shape,patch,flip,np.max(abs(x-y)))
  assert (b[0] is b[1])== (not flip)
  for x in b[:2]:
   expected=sf.quarter_turn_frame(x)
   actual=x.copy();output=sf.quarter_turn_frame(actual,inplace=True)
   assert np.shares_memory(output,actual) and np.array_equal(expected,output)
  cases.append(dict(shape=shape,patch=patch,flip=flip))
 for target in (6,10,12):
  for flip in (False,True):
   a=old.features_sift(im,8,False);b=old.features_sift(im,target,flip)
   shift=1.5*max(8,target);h,w=shape[0]-3*max(8,target),shape[1]-3*max(8,target)
   x,y=int(shift-a[2]),int(shift-b[2])
   expected=(np.ascontiguousarray(a[0][x:x+h,x:x+w]),np.ascontiguousarray(b[1][y:y+h,y:y+w]))
   actual=new.paired_sift(im,8,target,flip)
   for x,y in zip(expected,actual):assert np.array_equal(x,y),(shape,target,flip)
   # The resulting PatchMatch field also stays identical, not only descriptors.
   if shape==(180,231):
    mask=np.ones((h,w),np.uint8)
    aa=dc.field(*expected,mask,120,5,iterations=2)
    bb=dc.field(*actual[:2],mask,120,5,iterations=2)
    assert all(np.array_equal(x,y) for x,y in zip(aa,bb))
   cases.append(dict(shape=shape,target=target,flip=flip,paired=True))
size=(3510,4387)
reservations=[dict(flip=flip,target=target,bytes=workspace_bytes(*size,1,8,flip,'metal',target,True)) for flip in (False,True) for target in (6,8,10,12)]
assert all(x['bytes']<=BUDGET.limit for x in reservations)
assert max(x['bytes'] for x in reservations)*2>BUDGET.limit  # Large SIFT passes serialize.
r=dict(passed=True,descriptor_cases=len(cases),bit_exact=True,field_cases=6,full_size_reservations=reservations,workspace_limit=BUDGET.limit,cases=cases)
Path(__file__).with_name('results.json').write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k not in ('cases','full_size_reservations')},flush=True)
