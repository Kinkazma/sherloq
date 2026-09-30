import sys,json
from pathlib import Path
import numpy as np,cv2 as cv
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'source'))
from gui.sherloq_app.core.d2prl_regions import regions,RegionCache
from gui.sherloq_app.core.automatic_clones import render,D2PRL_SOURCE
from gui.sherloq_app.core.clone_corroboration import context_counts,votes,unique_envelopes
from gui.sherloq_app.core.d2prl import postprocess,refilter
raw=np.zeros((3,448,448),np.float32);raw[0,30:70,30:70]=1;raw[0,40:55,40:55]=0;raw[0,120:125,120:125]=1
mask,t,s=postprocess(raw)
r=dict(mask=mask,target=t,source=s,analyzed=np.ones_like(mask),raw_probabilities=raw[None],metadata=dict(boxes=[[0,0,448,448]],zones=[{}]))
rows=regions(r)
assert len(rows)==1 and rows[0]['source']==D2PRL_SOURCE
count=context_counts((448,448,3),rows)
assert np.array_equal(count,mask) and not count[45,45]
assert np.array_equal(votes((448,448,3),rows),mask)
assert len(regions(r,0))==2 and np.array_equal(r['mask'],mask)
assert np.array_equal(context_counts((448,448,3),rows+rows),mask)
image=np.full((448,448,3),100,np.uint8)
for mode in ('biomes','overlay'):
 out=render((image,rows,(),mode,.7));assert np.array_equal(out[mask==0],image[mask==0])
cache=RegionCache();a=cache(r,500);assert cache(r,500) is a;assert len(cache(r,0))==2
assert len(unique_envelopes(rows))==1
print(json.dumps(dict(passed=True,exact_mask_holes=True,one_vote=True,refilter_without_inference=True)))
# Cached real D2PRL outputs from the two supplied TIFFs: projection must neither
# fill holes nor invent convex envelopes, including an originally empty mask.
folder=R/'tests/d2prl/user-images-round2'
if (folder/'results.json').exists():
 reports=json.loads((folder/'results.json').read_text())
 for i,report in enumerate(reports,1):
  with np.load(folder/f'image-{i}-maps.npz',allow_pickle=False) as z:real={k:z[k] for k in z.files}
  real['metadata']=report['metadata']
  for minimum in (500,0):
   from gui.sherloq_app.core.d2prl import refilter
   expected=real['mask'] if minimum==500 else refilter((real,minimum))['mask']
   assert np.array_equal(context_counts((*expected.shape,3),regions(real,minimum)),expected)
 print('Cached real TIFF masks: exact at 500 and 0 pixels.')
# Excluded panels affect output scope, including a later post-filter update.
from unittest.mock import patch
from gui.sherloq_app.core.clone_detectors import analyze
class Model:
 def get(self,*_):return dict(device='cpu',weights={}),False
fake=dict(raw=raw,map=raw[0],mask=mask,target=t,source=s)
with patch('gui.sherloq_app.core.d2prl.predict',return_value=fake):
 result=analyze(image,dict(variant='D2PRL',excluded=[[[30,30],[39,30],[39,39],[30,39]]]),'cpu',Model())
 assert not result['mask'][30:40,30:40].any()
 assert not refilter((result,0))['mask'][30:40,30:40].any()
print('Excluded zones preserved before and after refiltering.')
