from pathlib import Path
import sys,time,json
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R/'source'),str(R/'integration/clone_detectors')]
import torch,numpy as np
from sherloq_clone_models.d2prl.deep_PM import PatchMatch
from gui.sherloq_app.core.d2prl_ops import evaluate_batched
torch.set_num_threads(8);torch.manual_seed(22);reports=[]
for device in ['cpu','mps']:
 for side,k in [(32,5),(64,13),(448,13)]:
  p=PatchMatch(1,50,side,1)
  x=(torch.rand(1,k,side,side)*side-side/2).to(device);y=x.flip(-1).clone()
  for channels,ev in [(36,p.evaluate_ZM),(96,p.evaluate_CNN)]:
   f=torch.rand(1,channels,side,side,dtype=torch.float16,device=device)
   with torch.inference_mode():
    t=time.perf_counter();old=ev(f,x,y)
    if device=='mps':torch.mps.synchronize()
    oldt=time.perf_counter()-t;t=time.perf_counter();new=evaluate_batched(ev,f,x,y)
    if device=='mps':torch.mps.synchronize()
    newt=time.perf_counter()-t
   error=max(float((a-b).abs().max()) for a,b in zip(old,new))
   row=dict(device=device,side=side,candidates=k,channels=channels,maximum_error=error,old_seconds=oldt,new_seconds=newt)
   print(row,flush=True);reports.append(row)
(R/'tests/d2prl/evaluator.json').write_text(json.dumps(reports,indent=2))
assert all(r['maximum_error']==0 for r in reports)
