from pathlib import Path
import sys,json,time,resource,importlib.util,gc
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core import pixel_stats,bit_planes,defect_pixels,color_spaces,noise,interactive,minmax,adjust,memory_resources as mr
name=sys.argv[1];cls,params={'adjust':('AdjustEngine',(10,-8,80,12,8,4,120,245,12,0,3,True)),'color_spaces':('SpaceEngine',('cmyk',2)),'noise':('NoiseEngine',(1,3,20,False,False,0)),'interactive':('EchoEngine',(3,85,True)),'minmax':('MinMaxEngine',(4,3,1,4)),'pixel_stats':('StatsEngine',('avg',True)),'bit_planes':('PlanesEngine',(4,2,2)),'defect_pixels':('DefectEngine',((2,10,40,0,True),2))}[name]
spec=importlib.util.spec_from_file_location('gui.sherloq_app.core._huge_ref_'+name,Path(__file__).with_name(name+'_before.py'));reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
rng=np.random.default_rng(109);shape=(8000,12000);im=np.empty((*shape,3),np.uint8)
for y in range(0,len(im),128):im[y:y+128]=rng.integers(0,256,im[y:y+128].shape,dtype=np.uint8)
im[510:550,510:550]=127;im[512,512]=255;im[514,514]=0
original_limit=mr.MEMORY.limit;mr.MEMORY.limit=(384 if name=='adjust' else 64)*mr.MiB
try:
 t=time.perf_counter();actual=getattr(globals()[name],cls)(im).compute(params);elapsed=time.perf_counter()-t;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
finally:mr.MEMORY.limit=original_limit
t=time.perf_counter();expected=getattr(reference,cls)(im).compute(params);reference_seconds=time.perf_counter()-t
if not isinstance(actual,tuple):actual=(actual,);expected=(expected,)
for a,b in zip(actual,expected):
 if isinstance(a,np.ndarray):
  for y in range(0,len(a),128):assert np.array_equal(a[y:y+128],b[y:y+128])
 else:assert a==b
r=dict(passed=True,tool=name,image_pixels=shape[0]*shape[1],bit_exact=True,bounded_seconds=elapsed,reference_seconds=reference_seconds,bounded_peak_rss_bytes=peak)
Path(__file__).with_name('huge-'+name+'-results.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
