from pathlib import Path
import sys,time,json
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'source'))
import torch,cv2 as cv,numpy as np
from gui.sherloq_app.core.clone_detectors import Models,analyze
from gui.sherloq_app.tools.tampering.clone_detectors import render
torch.set_num_threads(8);models=Models();records=[]
for i,filename in enumerate(sys.argv[1:],1):
 image=cv.imread(filename);assert image is not None,filename
 t=time.perf_counter();r=analyze(image,dict(variant='D2PRL'),'mps',models,lambda n,d: print(f'image {i}: {n}/{d}',flush=True) if n%20==0 else None)
 row=dict(image_id=i,shape=list(image.shape),seconds=time.perf_counter()-t,mask_pixels=int(r['mask'].sum()),mask_fraction=float(r['mask'].mean()),metadata=r['metadata'])
 records.append(row);print(json.dumps(row),flush=True)
 folder=R/'tests/d2prl/user-images';folder.mkdir(exist_ok=True)
 np.savez_compressed(folder/f'image-{i}-maps.npz',**{k:v for k,v in r.items() if k!='metadata'})
 # Preview only: full result arrays above retain original image dimensions.
 scaled={k:cv.resize(v,(max(1,round(image.shape[1]*min(1,1600/max(image.shape[:2])))),max(1,round(image.shape[0]*min(1,1600/max(image.shape[:2]))))),interpolation=cv.INTER_NEAREST if k in ('mask','analyzed','candidates') else cv.INTER_LINEAR) for k,v in r.items() if k in ('map','mask','source','target','analyzed','candidates')}
 thumb=cv.resize(image,(scaled['map'].shape[1],scaled['map'].shape[0]))
 for view in ['Carte','Masque','Superposition','Source','Cible']:
  cv.imwrite(str(folder/f'image-{i}-{view}.jpg'),render((thumb,scaled,view)))
 (folder/'results.json').write_text(json.dumps(records,indent=2))
