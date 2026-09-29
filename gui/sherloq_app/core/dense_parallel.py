"""Independent zone jobs, deterministic collection and a shared workspace budget."""
import os
from contextlib import contextmanager
from threading import Condition
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .cloning import check


class WorkspaceBudget:
    def __init__(self):
        from .memory_resources import MEMORY
        # Shared host-aware admission below is the effective cap. This ceiling
        # no longer rejects a field simply because it exceeds 16/24 GiB.
        self.limit=MEMORY.limit
        self.used=0;self.peak=0;self.active=0;self.max_active=0;self.condition=Condition()
    @contextmanager
    def claim(self,size,cancel):
        if size>self.limit:raise ValueError('Cette zone dépasse le budget mémoire du mode combiné ; réduire la zone ou le rayon dense.')
        with self.condition:
            while self.used+size>self.limit:
                check(cancel);self.condition.wait(.05)
            check(cancel);self.used+=size;self.peak=max(self.peak,self.used);self.active+=1;self.max_active=max(self.max_active,self.active)
        try:yield
        finally:
            with self.condition:self.used-=size;self.active-=1;self.condition.notify_all()


BUDGET=WorkspaceBudget()


def _job_shape(engine,zone,patch):
    h,w=engine.image.shape[:2]
    if zone:
        xy=np.concatenate(zone);lo=np.maximum(0,np.floor(xy.min(0))-3*patch).astype(int);hi=np.minimum([w,h],np.ceil(xy.max(0))+3*patch+1).astype(int)
        w,h=map(int,np.maximum(0,hi-lo))
    return h,w


def _plans(engine,algorithm,regions,compare,options,backend,target_patch,quarter,cap=None):
    from .dense_streaming import memory_plan
    jobs=[regions] if compare or not regions else [(region,) for region in regions]
    support=max(options[0],target_patch or options[0])
    shapes=[_job_shape(engine,zone,support) for zone in jobs]
    plans=[memory_plan(h,w,int(algorithm=='PatchMatch SIFT'),options[0],options[2],backend,target_patch,quarter,cap) for h,w in shapes]
    minimum=max(256*1024**2+h*w*2+max(h,w)*2048 for h,w in shapes)
    return plans,minimum


@contextmanager
def _admission(engine,algorithm,regions,compare,options,backend,target_patch,quarter,cancel):
    from .memory_resources import MEMORY,MemoryPlan
    plans,minimum=_plans(engine,algorithm,regions,compare,options,backend,target_patch,quarter)
    estimate=max(size for mode,size in plans)
    # Never raise the old descriptor cap to force a raw allocation. Choose a
    # compact/global provider, then reserve its actual bounded working set.
    with BUDGET.claim(estimate,cancel):
        with MEMORY.claim(MemoryPlan('selected',estimate,estimate,0,min(estimate,minimum)),cancel) as admitted:
            if admitted.reservation<estimate:
                plans,_=_plans(engine,algorithm,regions,compare,options,backend,target_patch,quarter,admitted.reservation)
            yield tuple(mode for mode,size in plans)


def serial_fields(engine,algorithm,limit,radius,minimum,threshold,regions,compare,options,cancel,progress,geometry,radii,gap,excluded,guides,backend,quarter_turn,target_patch):
    with _admission(engine,algorithm,regions,compare,options,backend,target_patch,quarter_turn,cancel) as modes:
        return engine.analyze(algorithm,limit,radius,minimum,threshold,regions,compare,options,cancel,progress,geometry,radii,gap,excluded,guides,workers=1,backend=backend,quarter_turn=quarter_turn,target_patch=target_patch,storage_modes=modes)


def parallel_fields(engine,algorithm,limit,radius,minimum,threshold,regions,compare,options,cancel,progress,geometry,radii,gap,excluded,guides,workers,backend="cpu",quarter_turn=False,target_patch=None):
    from .dense_copy import DenseCopyEngine
    from threading import Lock,Event
    jobs=[regions] if compare or not regions else [(region,) for region in regions]
    # Engines are reused only by their own serial job slot. Cache data obeys the
    # existing process-wide RAM budget; no QWidget or image file enters a worker.
    key=(algorithm,regions,compare,quarter_turn,target_patch)
    if getattr(engine,'_parallel_key',None)!=key:
        engine._parallel_key=key;engine._parallel_engines=[DenseCopyEngine(engine.image) for _ in jobs]
    abort=Event();stopped=lambda:cancel() or abort.is_set();lock=Lock();done=[0]*len(jobs)
    def task(index):
        child=engine._parallel_engines[index];zone=jobs[index]
        before=child.counts.copy()
        def updated(n,text):
            with lock:
                done[index]=n;progress(int(sum(done)/len(done)),text)
        with _admission(child,algorithm,zone,compare,options,backend,target_patch,quarter_turn,stopped) as modes:
            result=child.analyze(algorithm,max(1,limit//len(jobs)),radius if compare or radii is None else radii[index],minimum,threshold,zone,compare,options,stopped,updated,geometry=geometry,gap=gap,excluded=excluded,guides=guides,workers=1,backend=backend,quarter_turn=quarter_turn,target_patch=target_patch,storage_modes=modes)
        updated(100,'Zone dense conservée')
        return result,{name:child.counts[name]-before[name] for name in before}
    results=[]
    with ThreadPoolExecutor(max_workers=min(workers,len(jobs)),thread_name_prefix='cm2-'+algorithm.split()[-1]) as pool:
        futures=[pool.submit(task,i) for i in range(len(jobs))]
        try:
            for future in futures:results.append(future.result())
        except BaseException:
            abort.set()
            for future in futures:future.cancel()
            raise
    # Fixed zone order, same sampling per zone as the serial engine.
    points=[];pairs=[];members=[];maps=[];offset=0;full=consistent=evaluated=0
    for i,(r,counts) in enumerate(results):
        for name,value in counts.items():engine.counts[name]+=value
        points.append(r['points']);pair=r['pairs'].copy();pair[:,:2]+=offset;pairs.append(pair)
        m=np.zeros((len(r['points']),max(1,len(regions))),bool)
        if compare:m=r['members']
        else:m[:,i]=True
        members.append(m);offset+=len(r['points'])
        maps.extend({**field,'zone':i} for field in r['dense_maps'])
        full+=r['dense_count'];consistent+=r['dense_consistent_count'];evaluated+=r['candidate_comparisons']
    points=np.concatenate(points);pairs=np.concatenate(pairs)
    if len(jobs)>1 and len(pairs):
        endpoints=points[pairs[:,:2].astype(int),:2];swap=(endpoints[:,0,0]>endpoints[:,1,0])|((endpoints[:,0,0]==endpoints[:,1,0])&(endpoints[:,0,1]>endpoints[:,1,1]));endpoints[swap]=endpoints[swap,::-1]
        _,unique=np.unique(endpoints.reshape(-1,4),axis=0,return_index=True);pairs=pairs[np.sort(unique)]
    return dict(points=points,pairs=pairs,members=np.concatenate(members),dense_count=full,dense_consistent_count=consistent,dense_maps=maps,candidate_comparisons=evaluated)
