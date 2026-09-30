"""Separate arithmetic drift from device-specific random number sequences."""
from pathlib import Path
import sys,json,time
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R/'source'),str(R/'integration/clone_detectors')]
import torch,cv2 as cv,numpy as np
from gui.sherloq_app.core.d2prl import load,predict
torch.set_num_threads(8);loaded=load('mps')
from sherloq_clone_models.d2prl import deep_PM
class Proxy:
 def __getattr__(self,name):return getattr(torch,name)
 def rand(self,shape,device):return torch.rand(shape,device='cpu').to(device)
deep_PM.torch=Proxy()
im=cv.imread(str(R/'third_party/research/clone_detectors/01_d2prl/testpic/1.jpg'))
start=time.perf_counter();r=predict(im,loaded);seconds=time.perf_counter()-start
np.savez(R/'tests/d2prl/shared-random-mps.npz',**r)
print('done',seconds,flush=True)
