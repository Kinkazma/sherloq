from pathlib import Path
import sys,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core.dense_links import sample_unique_links
rng=np.random.default_rng(74)
def reference(targets,squared,selected,limit):
 rows=np.flatnonzero(selected);target=targets.ravel()[rows];ordered=np.column_stack((np.minimum(rows,target),np.maximum(rows,target)))
 order=np.argsort(squared.ravel()[rows],kind='stable');_,unique=np.unique(ordered[order],axis=0,return_index=True);rows=np.sort(rows[order[unique]]);count=len(rows)
 if count>limit:rows=rows[np.linspace(0,count-1,limit,dtype=int)]
 return rows,count
cases=0
for size in (1,2,99,1001,65537,200003):
 for kind in ('random','reciprocal','ties','empty'):
  target=rng.integers(0,size,size=size,dtype=np.int32);distance=rng.random(size,dtype=np.float32);selected=rng.random(size)>.3
  if kind in ('reciprocal','ties'):
   idx=np.arange(size//2)*2;target[idx]=idx+1;target[idx+1]=idx
  if kind=='ties':distance[:]=.5
  if kind=='empty':selected[:]=False
  for cap in (1,37,size+1):
   a=reference(target,distance,selected,cap);b=sample_unique_links(target,distance,selected,cap,block_size=71)
   assert a[1]==b[1] and np.array_equal(a[0],b[0]),(size,kind,cap)
   cases+=1
size=1_000_000;targets=rng.integers(0,size,size=size,dtype=np.int32);distance=rng.random(size,dtype=np.float32);selected=np.ones(size,bool)
t=time.perf_counter();a=reference(targets,distance,selected,20000);old=time.perf_counter()-t
t=time.perf_counter();b=sample_unique_links(targets,distance,selected,20000);new=time.perf_counter()-t
assert a[1]==b[1] and np.array_equal(a[0],b[0])
r=dict(passed=True,exact_cases=cases,benchmark_pixels=size,old_seconds=old,new_seconds=new,speedup=old/new,bounded_batch=65536)
Path(__file__).with_name('dense-links-results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
