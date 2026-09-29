from pathlib import Path
import sys,gc,threading,time,weakref,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core import memory_resources as mr
from gui.sherloq_app.core.cache_budget import CacheBudget,ArrayCache
from gui.sherloq_app.core.jpeg_curve import Cancelled
store=mr.TemporaryArrays(32*mr.MiB)
a=store.array((64,128,3),np.uint8);a[:]=np.arange(3);view=np.asarray(a)[4:9];mapped=weakref.ref(a._mmap);size=a.nbytes
store.checkpoint(force=True);del a;gc.collect()
assert store.bytes==size and mapped() is not None and (view==np.arange(3)).all()
del view;gc.collect();assert store.bytes==0 and mapped() is None
assert not mr._storage_pending
# Full image coverage; halo boundaries preserve a finite-support operator.
im=np.arange(31*49).reshape(31,49);out=np.empty_like(im);visits=np.zeros_like(im)
for dst,src,crop in mr.tiles(im.shape,(9,11),3):out[dst]=im[src][crop];visits[dst]+=1
assert np.array_equal(im,out) and (visits==1).all()
# Concurrent reservations never overcommit, release on exception and cancel.
budget=mr.MemoryCoordinator(64*mr.MiB,available=lambda:8*mr.GiB)
plan=budget.plan(256*mr.MiB,40*mr.MiB,100*mr.MiB);assert plan.mode=='bounded'
started=threading.Event();released=threading.Event();errors=[]
def other():
 started.set()
 try:
  with budget.claim(plan):raise RuntimeError('injected')
 except RuntimeError:pass
 released.set()
with budget.claim(plan):
 t=threading.Thread(target=other);t.start();started.wait();time.sleep(.12)
 assert not released.is_set()
 try:
  with budget.claim(plan,cancel=lambda:True):pass
 except Cancelled:pass
 else:raise AssertionError('cancellation ignored')
t.join();assert budget.used==0 and budget.peak==40*mr.MiB
# Host pressure evicts oldest cache without mutating externally retained views.
cache_budget=CacheBudget(1000);c=ArrayCache(1,budget=cache_budget)
x=np.arange(100,dtype=np.uint8);c.put('one',x);c.put('two',x.copy());c.get('one');cache_budget.trim(100)
assert c.get('two') is None and np.array_equal(x,np.arange(100)) and c.get('one') is x
# Capacity changes before admission select the bounded alternative.
free=[8*mr.GiB];b=mr.MemoryCoordinator(256*mr.MiB,available=lambda:free[0]);p=b.plan(200*mr.MiB,32*mr.MiB)
free[0]=min(2*mr.GiB,mr.physical_memory()//16)+64*mr.MiB
with b.claim(p) as actual:assert actual.mode=='bounded' and actual.reservation==32*mr.MiB
# Darwin must actually release resident pages, not only mark them reclaimable.
base=mr.resident_memory();large=store.array((128*mr.MiB,),np.uint8);large[:]=19
written=mr.resident_memory();store.checkpoint(force=True);flushed=mr.resident_memory()
assert written-flushed>96*mr.MiB,(base,written,flushed)
assert (large[::4096]==19).all();del large;gc.collect()
r=dict(passed=True,released_resident_bytes=written-flushed,available=mr.available_memory(),resident=mr.resident_memory(),mapped_views_lifetime=True,pressure_lru=True,cancel_and_exception_release=True,admission_replan=True)
Path(__file__).with_name('results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
# A later job reusing a cached descriptor must manage the borrowed mapping too.
owner=mr.TemporaryArrays();a=owner.array((128*mr.MiB,),np.uint8);a[:]=42
borrower=mr.TemporaryArrays();borrower.watch(np.asarray(a)[::2]);before=mr.resident_memory();borrower.checkpoint(force=True)
assert before-mr.resident_memory()>96*mr.MiB
assert (a[::8192]==42).all();del a;gc.collect()
