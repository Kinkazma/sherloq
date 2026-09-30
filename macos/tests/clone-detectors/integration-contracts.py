import sys,os,json,time,socket,gc
from pathlib import Path
import cv2 as cv,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
os.environ.update(YOLO_CONFIG_DIR=str(ROOT/'integration/clone_detectors/yolo-config'),YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',NO_ALBUMENTATIONS_UPDATE='1')
def deny(*a,**k):raise RuntimeError('Network forbidden in integration test')
socket.socket.connect=deny;socket.create_connection=deny
import torch
torch.set_num_threads(8)
from gui.sherloq_app.core.clone_detectors import Models,analyze,SEGMENTERS,FORGERYSCOPE
records=[];models=Models();rng=np.random.default_rng(731)
# Both rectangles contain identical material; backgrounds are distinct and excluded.
im=rng.integers(0,256,(192,416,3),np.uint8);im[16:176,240:400]=im[16:176,16:176]
zones=(((16,16),(175,16),(175,175),(16,175)),((240,16),(399,16),(399,175),(240,175)))
for variant in (() if os.environ.get('FORGERYSCOPE_ONLY') else SEGMENTERS):
 p=dict(variant=variant,regions=zones,selection_present=True,compare=False);t=time.perf_counter();a=analyze(im,p,'cpu',models)
 assert np.array_equal(a['raw_probabilities'][0],a['raw_probabilities'][1]),variant
 assert a['mask'].shape==im.shape[:2] and not a['analyzed'][:,176:240].any()
 b=analyze(im,p,'mps',models);delta=float(np.max(np.abs(a['raw_probabilities']-b['raw_probabilities'])))
 # D2PRL: explicit user acceptance of measured CPU/MPS divergence, 2026-09-30.
 # Dedicated numerical/quality evidence lives in tests/d2prl/. Other limits stay.
 if variant!='D2PRL':assert delta <= 1e-4, (variant,delta)
 else:assert np.isfinite(b['raw_probabilities']).all()
 rec=dict(variant=variant,seconds=time.perf_counter()-t,device=b['metadata']['device'],mps_max_abs=delta,mask_difference=int((a['mask']!=b['mask']).sum()),independent_regions_exact=True)
 records.append(rec);print(rec,flush=True);(ROOT/'tests/clone-detectors/integration-results.json').write_text(json.dumps(records,indent=2))
models.clear()
for variant in FORGERYSCOPE:
 p=dict(variant=variant,regions=zones,selection_present=True,compare=True);t=time.perf_counter();r=analyze(im,p,'cpu',models)
 assert not r['mask'][:,176:240].any() and not r['candidates'][:,176:240].any()
 rec=dict(variant=variant,seconds=time.perf_counter()-t,status=r['metadata']['status'],mask_pixels=int(r['mask'].sum()),candidate_pixels=int(r['candidates'].sum()),zones=r['metadata']['zones']);records.append(rec);print({k:v for k,v in rec.items() if k!='zones'},flush=True)
 (ROOT/'tests/clone-detectors/integration-results.json').write_text(json.dumps(records,indent=2))
for params in (dict(variant=SEGMENTERS[0],regions=zones,compare=True),dict(variant=SEGMENTERS[0],regions=[],selection_present=True)):
 try:analyze(im,params,'cpu',models)
 except ValueError:pass
 else:raise AssertionError('Invalid mode accepted')
models.clear();print('ALL CONTRACTS PASSED',flush=True)
