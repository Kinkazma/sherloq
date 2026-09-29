from pathlib import Path
import ctypes as ct,sys,json,time,gc
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core import dense_copy as dc,dense_streaming as ds
old=dc.library();lib=old if '--production' in sys.argv else ct.CDLL(str(Path(__file__).with_name('candidate.dylib')))
for name in ('sherloq_dense_features','sherloq_patchmatch','sherloq_patchmatch_gap','sherloq_patchmatch_metric'):
 getattr(lib,name).argtypes=getattr(old,name).argtypes;getattr(lib,name).restype=ct.c_int
dc._LIB=lib;ds.native.cache_clear()
rng=np.random.default_rng(769);im=rng.integers(0,256,(161,233,3),dtype=np.uint8);im[90:140,160:210]=im[20:70,20:70]
regions=(((10.,10.),(80.,10.),(80.,80.),(10.,80.)),((150.,80.),(220.,80.),(220.,150.),(150.,150.)))
excluded=(((30.,40.),(40.,40.),(40.,50.),(30.,50.)),)
cases=[]
def run(algorithm,flip,quarter,target,zone_mode,storage,backend='metal'):
 zones=() if zone_mode=='all' else regions
 return dc.DenseCopyEngine(im).analyze(algorithm,5000,400,5,.4,zones,zone_mode=='compare',(8,2,flip,2.),lambda:False,lambda *a:None,geometry=('Affine',3.,5),excluded=excluded,backend=backend,quarter_turn=quarter,target_patch=target,storage_modes=(storage,)*(2 if zone_mode=='search' else 1))
for algorithm,flip,quarter,target in [('PatchMatch Zernike',False,False,8),('PatchMatch Zernike',True,False,8),('PatchMatch SIFT',False,False,8),('PatchMatch SIFT',True,False,8),('PatchMatch SIFT',True,True,8),('PatchMatch SIFT',True,True,6),('PatchMatch SIFT',False,True,10)]:
 for mode in ('all','search','compare'):
  a=run(algorithm,flip,quarter,target,mode,'ram')
  for storage in ('compact','mapped'):
   b=run(algorithm,flip,quarter,target,mode,storage)
   for key in ('points','pairs','members','dense_count','dense_consistent_count','candidate_comparisons'):
    assert np.array_equal(a[key],b[key]),(algorithm,flip,quarter,target,mode,storage,key)
   delta=0.
   for x,y in zip(a['dense_maps'],b['dense_maps']):
    for key in ('consistent_mask','targets','distances_squared'):assert np.array_equal(x[key],y[key]),(algorithm,mode,storage,key)
    delta=max(delta,float(np.max(abs(x['coherence_error']-y['coherence_error']))))
   assert delta<1e-4,delta
   cases.append(dict(algorithm=algorithm,flip=flip,quarter=quarter,target=target,mode=mode,storage=storage,outputs_exact=True,coherence_error_max=delta))
  print(cases[-1],flush=True)
# CPU checkbox / reference preparation remains a supported bounded path.
for algorithm in dc.METHODS:
 a=run(algorithm,True,False,8,'all','ram','cpu');b=run(algorithm,True,False,8,'all','mapped','cpu')
 for key in ('points','pairs','members','dense_count','dense_consistent_count','candidate_comparisons'):assert np.array_equal(a[key],b[key]),(algorithm,'cpu',key)
 cases.append(dict(algorithm=algorithm,backend='cpu',outputs_exact=True))
# Public admission, serial and parallel preserve ordinary operation.
from gui.sherloq_app.core.dense_parallel import BUDGET
from gui.sherloq_app.core.memory_resources import MEMORY
for workers in (1,2):
 result=dc.DenseCopyEngine(im).analyze('PatchMatch SIFT',5000,400,5,.4,regions,False,(8,2,False,2.),lambda:False,lambda *a:None,workers=workers,backend='metal')
 assert BUDGET.used==MEMORY.used==0
r=dict(passed=True,cases=cases,public_serial_parallel=True)
Path(__file__).with_name('integration-production-results.json' if '--production' in sys.argv else 'integration-results.json').write_text(json.dumps(r,indent=2)+'\n');print('Passed',len(cases),'cases',flush=True)
