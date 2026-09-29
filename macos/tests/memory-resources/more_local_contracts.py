from pathlib import Path
import sys,json,importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core import noise,color_spaces,interactive,memory_resources as mr
from gui.sherloq_app.core.noise_bounded import compute as bounded_noise
from gui.sherloq_app.core.echo_bounded import compute as bounded_echo
from gui.sherloq_app.core.bounded_ops import local_result

def reference(name):
 spec=importlib.util.spec_from_file_location('gui.sherloq_app.core._before_'+name,Path(__file__).with_name(name+'_before.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old={n:reference(n) for n in ('noise','color_spaces','interactive')};rng=np.random.default_rng(681);rows=[]
im=rng.integers(0,256,(545,567,3),dtype=np.uint8);im[500:540,495:540]=127;im[511,511]=255;im[513,513]=0
for space,count in [('rgb',3),('cmyk',4),('gray',4),('hsv',3),('hls',3),('ycrcb',3),('xyz',3),('lab',3),('luv',3)]:
 for i in range(count):
  p=(space,i);a=old['color_spaces'].SpaceEngine(im).compute(p);b=local_result(im,lambda roi:color_spaces.SpaceEngine(roi)._compute(p))[0]
  assert np.array_equal(a,b),(p,np.count_nonzero(a!=b));rows.append(['space',p])
print('Color spaces passed',flush=True)
for mode in range(5):
 for gray in (False,True):
  for denoised,levels in [(True,0),(False,0),(False,170)]:
   p=(mode,2,20,gray,denoised,levels);a=old['noise'].NoiseEngine(im,backend='cpu').compute(p);b=bounded_noise(noise.NoiseEngine(im,backend='cpu'),p)
   assert np.array_equal(a,b),(p,np.count_nonzero(a!=b),np.max(np.abs(a.astype(int)-b)));rows.append(['noise',p])
 print('Noise',mode,'passed',flush=True)
for radius in (1,2,4,5,10,15):
 for gray in (False,True):
  p=(radius,85,gray);a=old['interactive'].EchoEngine(im,backend='cpu').compute(p);b=bounded_echo(im,p)
  assert np.array_equal(a,b),(p,np.count_nonzero(a!=b),np.max(np.abs(a.astype(int)-b)));rows.append(['echo',p])
print('Echo passed',flush=True)
limit=mr.MEMORY.limit;mr.MEMORY.limit=64*mr.MiB
try:
 im=rng.integers(0,256,(1601,1603,3),dtype=np.uint8)
 for module,cls,p in [('color_spaces','SpaceEngine',('cmyk',2)),('noise','NoiseEngine',(1,3,20,False,False,0)),('interactive','EchoEngine',(3,85,True))]:
  a=getattr(globals()[module],cls)(im).compute(p);b=getattr(old[module],cls)(im).compute(p)
  assert isinstance(a,np.memmap) and np.array_equal(a,b),module
finally:mr.MEMORY.limit=limit
r=dict(passed=True,exact_cases=len(rows),public_dispatch=3);Path(__file__).with_name('more-local-results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
