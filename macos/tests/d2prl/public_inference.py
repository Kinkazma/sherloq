from pathlib import Path
import sys,time,json,resource
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'source'))
import torch,cv2 as cv,numpy as np
from gui.sherloq_app.core.clone_detectors import Models,analyze
torch.set_num_threads(8)
device=sys.argv[1] if len(sys.argv)>1 else 'mps'
image=cv.imread(str(R/'third_party/research/clone_detectors/01_d2prl/testpic/1.jpg'))
start=time.perf_counter()
def progress(n,t):
 if n%20==0 or n==t:print(n,t,round(time.perf_counter()-start,2),flush=True)
r=analyze(image,dict(variant='D2PRL'),device,Models(),progress)
seconds=time.perf_counter()-start
raw=r['raw_probabilities'][0]
reference=R/f'tests/d2prl/probe-{device}-448-40.npz'
diff=None
if reference.exists():
 with np.load(reference) as ref:
  diff=[float(np.abs(raw[i]-ref[str(i)][0,0]).max()) for i in range(3)]
gt_bgr=cv.imread(str(R/'third_party/research/clone_detectors/01_d2prl/output/1_gt.png'))
gt=(gt_bgr[:,:,1]>127)|(gt_bgr[:,:,2]>127)
print(gt.shape,r['mask'].shape,flush=True)
mask=r['mask'].astype(bool)
if gt.shape!=mask.shape:gt=cv.resize(gt.astype(np.uint8),(mask.shape[1],mask.shape[0]),interpolation=cv.INTER_NEAREST)>0
report=dict(passed=all(np.isfinite(a).all() for k,a in r.items() if k!='metadata'),seconds=seconds,device=device,reference_max_errors=diff,union_pixels=int(mask.sum()),ground_truth_pixels=int(gt.sum()),iou=float((mask&gt).sum()/max(1,(mask|gt).sum())),peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,metadata=r['metadata'])
print(json.dumps(report),flush=True)
(R/f'tests/d2prl/public-{device}.json').write_text(json.dumps(report,indent=2))
np.savez(R/f'tests/d2prl/public-{device}.npz',**{k:v for k,v in r.items() if k!='metadata'})
cv.imwrite(str(R/f'tests/d2prl/mask-{device}.png'),r['mask']*255)
