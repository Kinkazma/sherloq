from pathlib import Path
import sys,json,time,importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core import gradient,pixel_stats,bit_planes,defect_pixels,color_spaces,noise,interactive,minmax
rng=np.random.default_rng(791);image=rng.integers(0,256,(721,1093,3),dtype=np.uint8);results={}
for name,cls,params in [('gradient','GradientEngine',(95,3,True,False)),('pixel_stats','StatsEngine',('avg',True)),('bit_planes','PlanesEngine',(4,2,2)),('defect_pixels','DefectEngine',((2,10,40,0,True),2)),('color_spaces','SpaceEngine',('cmyk',2)),('noise','NoiseEngine',(1,3,20,False,False,0)),('interactive','EchoEngine',(3,85,True)),('minmax','MinMaxEngine',(4,3,1,4))]:
 spec=importlib.util.spec_from_file_location('gui.sherloq_app.core._fast_'+name,Path(__file__).with_name(name+'_before.py'));old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 classes=[getattr(old,cls),getattr(globals()[name],cls)];times=[[],[]];expected=None
 for iteration in range(17):
  for i in ((0,1) if iteration%2 else (1,0)):
   engine=classes[i](image);t=time.perf_counter();out=engine.compute(params);dt=time.perf_counter()-t
   if iteration>2:times[i].append(dt)
   if expected is None:expected=out
   aa=out if isinstance(out,tuple) else (out,);bb=expected if isinstance(expected,tuple) else (expected,)
   assert all(np.array_equal(a,b) for a,b in zip(aa,bb)),name
 results[name]=dict(old_seconds=float(np.median(times[0])),current_seconds=float(np.median(times[1])),ratio=float(np.median(times[1])/np.median(times[0])),outputs_exact=True)
 print(name,results[name],flush=True)
Path(__file__).with_name('fast-local-results.json').write_text(json.dumps(results,indent=2)+'\n')
