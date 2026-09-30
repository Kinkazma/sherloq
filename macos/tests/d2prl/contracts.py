"""Checkpoint/postprocessing/region/public result contracts, no runtime calibration."""
from pathlib import Path
import sys,json
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'source'))
import cv2 as cv,numpy as np
from skimage import morphology
from gui.sherloq_app.core.d2prl import postprocess
from gui.sherloq_app.core import clone_detectors as core
# Literal upstream postprocessing, float32 input as returned by torch.
def reference(raw):
 union,mt,ms=raw
 mask=morphology.remove_small_objects(np.rint(union)>.5,min_size=500)
 mt=mt.copy();ms=ms.copy();mt[mt>0]=1;ms[ms>0]=1
 both=morphology.remove_small_objects(mt+ms>0,min_size=500)
 mt=mt*both;ms=both-mt
 filtered=cv.filter2D(mt-ms,-1,np.ones([50,50]),borderType=cv.BORDER_CONSTANT)
 mt=(filtered>0)*both;ms=1.*both-1.*mt
 return mask,mt,ms
rng=np.random.default_rng(20260930)
cases=0
for h,w in [(32,35),(448,448),(77,131)]:
 for style in range(4):
  raw=rng.random((3,h,w),dtype=np.float32)
  if style==0:raw[:]=0
  if style==1:raw[:,h//3:,:w//2]=0
  if style==2:raw[1:]*=1e-7  # role threshold must remain >0, not >0.5
  got=postprocess(raw);expected=reference(raw)
  for a,b in zip(got,expected):assert np.array_equal(a,b)
  cases+=1
# Exercise public engine dispatch with a stub model, preserving independent zones.
import gui.sherloq_app.core.d2prl as engine
seen=[]
def fake(crop,loaded,progress):
 seen.append(crop.copy());raw=np.zeros((3,448,448),np.float32);raw[:,100:250,130:330]=.8
 m,t,s=postprocess(raw)
 return dict(raw=raw,map=raw[0],mask=m,target=t,source=s)
engine.predict=fake
class Models:
 def get(self,*args):return dict(device='cpu',weights={}),False
im=rng.integers(0,256,(121,311,3),dtype=np.uint8)
regions=[[(0,0),(90,0),(90,120),(0,120)],[(200,0),(310,0),(310,120),(200,120)]]
r=core.analyze(im,dict(variant='D2PRL',regions=regions),'cpu',Models())
assert len(seen)==2 and np.array_equal(seen[0],im[:,:91]) and np.array_equal(seen[1],im[:,200:])
assert not r['mask'][:,91:200].any() and not r['analyzed'][:,91:200].any()
assert r['raw_probabilities'].shape==(2,3,448,448)
assert all(k in r for k in ['source','target','map','mask'])
assert r['metadata']['channel_order']==['union_probability','target_residual','source_residual']
try:core.analyze(im,dict(variant='D2PRL',regions=regions,compare=True),'cpu',Models())
except ValueError:pass
else:raise AssertionError('Segmentation must not fake a between-zones comparison')
report=dict(passed=True,postprocess_cases=cases,independent_regions=2,role_threshold=0)
print(report);(R/'tests/d2prl/contracts.json').write_text(json.dumps(report,indent=2))
