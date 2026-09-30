from pathlib import Path
import sys,time,json,resource,faulthandler
faulthandler.enable()
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R/'source'),str(R/'integration/clone_detectors')]
import torch,cv2 as cv,numpy as np
from sherloq_clone_models.d2prl.models_D2PRL import DPM
from torchvision import transforms as T
device=sys.argv[1];side=int(sys.argv[2]) if len(sys.argv)>2 else 448;iters=int(sys.argv[3]) if len(sys.argv)>3 else 40
torch.set_num_threads(8);torch.manual_seed(22)
start=time.perf_counter();model=DPM(side,1,iters,R/'third_party/research/clone_detectors/01_d2prl')
s=torch.load(R/'models/external/clone_detectors/01_d2prl/d2prl.pth',map_location='cpu',weights_only=True)
print(model.load_state_dict(s,strict=True),flush=True);del s
model.eval().to(device)
image=cv.imread(str(R/'third_party/research/clone_detectors/01_d2prl/testpic/1.jpg'))
x=T.Compose([T.ToPILImage(),T.ToTensor(),T.Resize((side,side))])(cv.cvtColor(image,cv.COLOR_BGR2RGB))[None].to(device)
print('loaded',device,round(time.perf_counter()-start,3),flush=True)
count=[0]
def progress(*args):
 count[0]+=1
 if count[0]%20==0:print('eval',count[0],round(time.perf_counter()-start,2),flush=True)
model.patchmatch.evaluate_CNN.register_forward_hook(progress)
begin=time.perf_counter()
with torch.inference_mode():out=model(x)
if device=='mps':torch.mps.synchronize()
a=[t.cpu().numpy() for t in out]
report=dict(device=device,side=side,iterations=iters,seconds=time.perf_counter()-begin,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,arrays=[dict(shape=v.shape,finite=bool(np.isfinite(v).all()),min=float(v.min()),max=float(v.max()),positive=int((v>.5).sum())) for v in a])
print(json.dumps(report),flush=True)
(R/f'tests/d2prl/probe-{device}-{side}-{iters}.json').write_text(json.dumps(report,indent=2))
np.savez(R/f'tests/d2prl/probe-{device}-{side}-{iters}.npz',**{str(i):v for i,v in enumerate(a)})
