"""Redistributable deterministic inputs; no photo, external file or network."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
import cv2 as cv
import numpy as np
from gui.sherloq_app.core.ela import ElaEngine
from gui.sherloq_app.core.jpeg import compress_jpg
from gui.sherloq_app.core.utility import mat2img
rng=np.random.default_rng(741);cases=[]
for shape in [(31,37,3),(128,129,3),(256,256,3)]:
 image=rng.integers(0,256,shape,dtype=np.uint8);engine=ElaEngine(image)
 for q in [10,75,95]:
  compressed=compress_jpg(image,q)
  expected=cv.absdiff(image,compressed)
  assert np.any(compressed.astype(np.int16)<image.astype(np.int16))
  started=time.perf_counter();actual=engine.base(q,True);cold=time.perf_counter()-started
  started=time.perf_counter();warm=engine.base(q,True);hot=time.perf_counter()-started
  assert np.array_equal(actual,expected);assert warm is actual
  expected_float=cv.sqrt(cv.absdiff(image.astype(np.float32)/255,compressed.astype(np.float32)/255))*255
  assert np.array_equal(engine.base(q,False),expected_float)
  assert np.array_equal(engine.compute((q,1,0,True,False)),expected)
  cases.append({'shape':shape,'quality':q,'exact':True,'cold_seconds':cold,'cached_seconds':hot})
image=rng.integers(0,256,(23,25,3),dtype=np.uint8);view=image[::2,::2]
qimage=mat2img(view);saved=qimage.copy();image[:]=0
assert qimage==saved,'Non-contiguous input must be owned'
print(json.dumps({'passed':True,'cases':cases,'noncontiguous_qimage_ownership':True,'limitations':'Synthetic ELA and memory ownership only; timings are not full-analysis speedups.'},indent=2))
