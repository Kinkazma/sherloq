"""Pinned Adaptive CFA inference, exact block geometry and bounded working memory."""
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
WEIGHTS={'Original':'pretrained.pt','JPEG 95':'adapted_to_j95_database.pt','Sans JPEG':'adapted_to_nojpeg_database.pt'}


def load(variant,device):
    import torch
    from ..vendor.adaptive_cfa.structure import FullNet
    path=ROOT/'models/external'/WEIGHTS[variant]
    model=FullNet();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
    return model.eval().to(device)


def predict(image,model,block=32,tile=512,progress=lambda *a:None):
    import torch
    if block<8 or block%2:raise ValueError('CFA block size must be even and at least 8.')
    h,w=image.shape[:2];h-=h%2;w-=w%2;ny,nx=(h-8)//block,(w-8)//block
    if min(ny,nx)<1:raise ValueError(f'Adaptive CFA needs at least {block+8} pixels per side.')
    # Four-pixel spatial halo; core and tile starts remain aligned to CFA parity
    # and block pooling. No rescaling, no recompression and no artificial padding.
    device=next(model.parameters()).device;step=max(1,tile//block) if tile else max(ny,nx);prob=np.empty((4,4,ny,nx),np.float32)
    total=((ny+step-1)//step)*((nx+step-1)//step);done=0
    with torch.inference_mode():
        for y in range(0,ny,step):
            for x in range(0,nx,step):
                hh,ww=min(step,ny-y),min(step,nx-x)
                pixels=image[y*block:(y+hh)*block+8,x*block:(x+ww)*block+8,::-1].copy()
                if not tile:pixels=image[:h,:w,::-1].copy()
                # Published input conversion: normalize float before casting float32.
                rgb=torch.from_numpy((pixels.astype(np.float64)/255).transpose(2,0,1)).float()[None].to(device)
                prob[:,:,y:y+hh,x:x+ww]=model(rgb,block).cpu().numpy();done+=1;progress(done,total)
    prob=np.exp(prob)
    if not np.isfinite(prob).all():raise ValueError('Adaptive CFA returned non-finite values.')
    aligned=prob.copy();aligned[:,1]=prob[[1,0,3,2],1];aligned[:,2]=prob[[2,3,0,1],2];aligned[:,3]=prob[[3,2,1,0],3]
    grids=np.mean(aligned,axis=1);best=int(np.argmax(np.mean(grids,axis=(1,2))));local=np.argmax(grids,axis=0).astype(np.uint8)
    confidence=np.clip(1-np.max(grids,axis=0),0,1);confidence[local==best]=1;suspicion=1-confidence
    return dict(probabilities=prob,grids=grids,local_grid=local,suspicion=suspicion,
                metadata=dict(best_grid=best,block=block,origin=[4,4],valid_shape=[ny*block,nx*block],image_shape=list(image.shape[:2]),tile=tile))
