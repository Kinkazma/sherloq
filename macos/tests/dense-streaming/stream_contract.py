import numpy as np,json,time
from pathlib import Path
from compact_contract import prepare as full,unpack
from stream_candidate import prepare
rng=np.random.default_rng(832);rows=[]
for shape in ((83,97),(205,271),(563,727)):
 im=rng.integers(0,256,(*shape,3),dtype=np.uint8)
 for patch,flip,quarter in ((6,False,False),(8,True,True),(12,True,False)):
  a=full(im,patch,flip,quarter);t=time.perf_counter()
  b=prepare(im,patch,flip,quarter,tile=113,stripe_bytes=65536)
  x=unpack(a);y=unpack(b)
  assert np.array_equal(x,y),(shape,patch,flip,np.max(abs(x-y)),np.count_nonzero(x!=y))
  rows.append(dict(shape=shape,patch=patch,flip=flip,quarter=quarter,exact=True,seconds=time.perf_counter()-t));print(rows[-1],flush=True)
Path(__file__).with_name('stream-contract-results.json').write_text(json.dumps(rows,indent=2)+'\n')
