"""Check the worst full-resolution descriptor preparations, without rerunning NNF."""
import sys,time,json,resource,gc
from pathlib import Path
import cv2 as cv,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core.metal_dense import paired_sift,features_sift
from gui.sherloq_app.core.sift_frames import quarter_turn_frame,orientation_diversity
from gui.sherloq_app.core.dense_parallel import BUDGET
from gui.sherloq_app.core.dense_memory import workspace_bytes
image=cv.imread(sys.argv[1]);assert image is not None
rows=[]
for target in (6,8,10,12):
 start=time.monotonic();estimate=workspace_bytes(*image.shape[:2],1,8,True,'metal',target,True)
 with BUDGET.claim(estimate,lambda:False):
  a,b,shift=features_sift(image,8,True) if target==8 else paired_sift(image,8,target,True)
  assert a.shape==b.shape and a is not b and a.flags.c_contiguous and b.flags.c_contiguous
  quarter_turn_frame(a,inplace=True);quarter_turn_frame(b,inplace=True)
  valid=orientation_diversity(a)&orientation_diversity(b)
  for field in (a,b):
   flat=field.reshape(-1,128)
   for i in range(0,len(flat),8192):assert np.isfinite(flat[i:i+8192]).all()
  r=dict(target=target,shape=a.shape,shift=shift,seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,reserved_bytes=estimate,eligible=int(valid.sum()))
  print(r,flush=True);rows.append(r);del a,b,field,flat,valid
 gc.collect()
assert BUDGET.used==0
Path(__file__).with_name('full-descriptors-results.json').write_text(json.dumps(dict(passed=True,source_shape=image.shape,cases=rows,full_resolution=True,all_mirror_scales=True,matching=False),indent=2)+'\n')
