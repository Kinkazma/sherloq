"""Qualify CPU keypoints/YOLO + MPS embeddings/LightGlue before enabling."""
import os,sys,time,json,socket,gc,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
os.environ.update(YOLO_CONFIG_DIR=str(ROOT/'integration/clone_detectors/yolo-config'),YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',NO_ALBUMENTATIONS_UPDATE='1',PYTORCH_MPS_FAST_MATH='0')
def deny(*a,**k):raise RuntimeError('No network')
socket.socket.connect=deny;socket.create_connection=deny
import torch,numpy as np,cv2 as cv
torch.set_num_threads(8)
from gui.sherloq_app.core.forgeryscope_adapter import load,predict
from gui.sherloq_app.core.clone_detectors import FORGERYSCOPE
records=[]
public=cv.imread(str(ROOT/'third_party/research/clone_detectors/04_mgcfdn/image/84t.tif'));public=cv.resize(public,(256,192));rng=np.random.default_rng(807)
fixtures=[('identical',public,public),('rotation',public,cv.rotate(public,cv.ROTATE_180)),('unrelated',public,rng.integers(0,256,public.shape,np.uint8))]
references={}
for device in ('cpu','mps'):
 loaded=load(device)
 for name,m in [(n,loaded[n]) for n in ('micro','blot')]:
  for label,im,im2 in fixtures:
   inputs=[np.ascontiguousarray(a[:,:,::-1]) for a in (im,im2)]
   def execute():return m._match_features(*(m._prepare_image(a).to(m.device) for a in inputs))
   execute();torch.mps.synchronize();start=time.perf_counter();r=execute();torch.mps.synchronize();elapsed=time.perf_counter()-start
   key=(name,label)
   if device=='cpu':references[key]=(r,elapsed)
   else:
    a,ct=references[key];equal=all(np.array_equal(a[i],r[i]) for i in (0,1));delta=float(np.max(np.abs(a[2]-r[2]))) if a[2] is not None and r[2] is not None and a[2].shape==r[2].shape else 0 if a[2] is r[2] is None else None
    row=dict(component=name,fixture=label,points_exact=equal,score_max_abs=delta,cpu_seconds=ct,hybrid_seconds=elapsed,speedup=ct/elapsed);records.append(row);print(row,flush=True)
 # Whole pipeline includes geometric masks and candidate metadata.
 for variant in FORGERYSCOPE:
  if variant=='Forgeryscope Auto':continue # Manual panels have no inferred type; dedicated Auto tests.
  im=np.zeros((208,560,3),np.uint8);im[8:200,8:264]=public;im[8:200,296:552]=public
  label='Microscopy' if 'microscopie' in variant else 'Blots';panels=[(label,1.,8,8,264,200),(label,1.,296,8,552,200)]
  r=predict(im,loaded,variant,panels);key=(variant,'complete')
  if device=='cpu':references[key]=r
  else:
   a=references[key];same=all(np.array_equal(a[n],r[n]) for n in ('map','mask','candidates'));row=dict(profile=variant,pipeline_arrays_exact=same,status=r['metadata']['status']);records.append(row);print(row,flush=True)
 del loaded;gc.collect();torch.mps.empty_cache()
(ROOT/'tests/clone-detectors/forgery-hybrid-results.json').write_text(json.dumps(records,indent=2));assert all(r.get('points_exact',True) and (r.get('score_max_abs',0) is not None and r.get('score_max_abs',0)<=1e-4) and r.get('pipeline_arrays_exact',True) for r in records)
