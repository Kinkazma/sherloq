import numpy as np
from compact_contract import md,dc
from gui.sherloq_app.core.memory_resources import TemporaryArrays,tiles

def prepare(image,patch=8,flip=False,*,tile=512,gpu=True,store=None):
 store=store or TemporaryArrays();shape=(*image.shape[:2],12);a=store.array(shape);b=store.array(shape) if flip else a
 for dst,src,crop in tiles(image.shape,(tile,tile),3*patch+1):
  pixels=np.ascontiguousarray(image[src])
  first,second,_=md.features(pixels,patch,flip) if gpu else dc.features(pixels,0,patch,flip)
  a[dst]=first[crop]
  if flip:b[dst]=second[crop]
  store.checkpoint()
 store.checkpoint(force=True);return a,b,0.

if __name__=='__main__':
 import json
 from pathlib import Path
 rng=np.random.default_rng(89);rows=[]
 for shape in ((83,101),(205,291),(531,679)):
  im=rng.integers(0,256,(*shape,3),dtype=np.uint8)
  for patch in (6,8,12):
   for flip in (False,True):
    ref=md.features(im,patch,flip);got=prepare(im,patch,flip,tile=113)
    assert all(np.array_equal(x,y) for x,y in zip(ref,got)),(shape,patch,flip,max(float(np.max(abs(x-y))) for x,y in zip(ref[:2],got[:2])))
    rows.append(dict(shape=shape,patch=patch,flip=flip,exact=True))
 Path(__file__).with_name('zernike-results.json').write_text(json.dumps(rows,indent=2)+'\n');print(len(rows),'exact Zernike cases')
