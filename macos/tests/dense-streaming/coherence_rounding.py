from pathlib import Path
import sys,importlib.util,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core.dense_copy import coherent_mask
spec=importlib.util.spec_from_file_location('gui.sherloq_app.core._old_dense_rounding',Path(__file__).with_name('dense_copy_before_integration.py'));old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
h,w=1500,2500;y,x=np.mgrid[:h,:w];targets=((y+400)*w+x+1200).astype(np.int32);valid=(y+400<h)&(x+1200<w);targets[~valid]=-1;squared=np.zeros((h,w),np.float32)
a=old.coherent_mask(targets,squared,1.,0.,6,5);b=coherent_mask(targets,squared,1.,0.,6,5);interior=np.s_[10:1050,10:1250]
assert np.all(b[1][interior]==0) and np.all(b[0][interior])
r=dict(passed=True,exact_translation=True,old_spurious_residual_max=float(a[1][interior].max()),old_rejected_exact_pixels=int(np.count_nonzero(~a[0][interior])),new_spurious_residual_max=float(b[1][interior].max()))
Path(__file__).with_name('coherence-rounding-results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
