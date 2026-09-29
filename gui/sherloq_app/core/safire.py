"""SAFIRE multisource inference with streamed prompt masks and local weights."""
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4]

def load(device):
    import torch
    from ..vendor.safire.segment_anything import sam_model_registry
    from ..vendor.safire.networks.safire_model import AdaptorSAM
    base=sam_model_registry['vit_b_adaptor'](checkpoint='')
    model=AdaptorSAM(base.image_encoder,base.mask_decoder,base.prompt_encoder)
    state=torch.load(ROOT/'models/external/safire.pth',map_location='cpu',weights_only=True)['model'];state={k.removeprefix('module.'):v for k,v in state.items()}
    model.load_state_dict(state,strict=True)
    return model.eval().to(device)

def cluster(features,kind='kmeans',groups=3,eps=.2,minimum=1,seed=1701):
    """CPU clustering, fixed seed; empty clusters retain their previous centre."""
    import torch
    if not len(features):return np.empty(0,np.int64)
    if kind=='dbscan':
        from sklearn.cluster import DBSCAN
        return DBSCAN(eps=eps,min_samples=minimum).fit_predict(features).astype(np.int64)
    x=torch.from_numpy(np.asarray(features,np.float32));n=min(groups,len(x));rng=np.random.RandomState(seed);centres=x[rng.choice(len(x),n,replace=False)].clone()
    for _ in range(1001):
        a=x[:,None]/x[:,None].norm(dim=-1,keepdim=True).clamp_min(1e-12);b=centres[None]/centres[None].norm(dim=-1,keepdim=True).clamp_min(1e-12)
        labels=(1-(a*b).sum(-1)).argmin(1);old=centres.clone()
        for i in range(n):
            selected=x[labels==i]
            if len(selected):centres[i]=selected.mean(0)
        shift=((centres-old).square().sum(1).sqrt()).sum()
        if shift.square()<1e-4:break
    return labels.numpy()

def predict(image,model,side=16,groups=3,kind='kmeans',eps=.2,minimum=1,binary=False,progress=lambda *a:None,cache=None):
    import cv2 as cv,torch
    import torch.nn.functional as F
    from ..vendor.safire.segment_anything.predictor_safire import SamPredictor
    from ..vendor.safire.segment_anything.utils.amg import build_point_grid
    if side not in (4,8,16,24,32) or not 1<=groups<=16:raise ValueError('Invalid SAFIRE sampling settings.')
    import hashlib
    cache_key=(hashlib.sha256(image.tobytes()).hexdigest(),side,str(model.device))
    reused=cache is not None and cache.get('key')==cache_key
    if reused:
        points,features,ious,low_masks,areas,valid_ids=cache['value']
    else:
        predictor=SamPredictor(model);points=build_point_grid(side)*1024;features=[];ious=[];low_masks=[];areas=[];valid_ids=[]
        with torch.inference_mode():
            predictor.set_image(cv.resize(image[:,:,::-1],(1024,1024)));progress(1,len(points)+1)
            for lo in range(0,len(points),16):
                coords=torch.as_tensor(points[lo:lo+16],dtype=torch.float32,device=model.device);labels=torch.ones((len(coords),1),dtype=torch.int,device=model.device)
                masks,confidence,low=predictor.predict_torch(coords[:,None],labels,multimask_output=False,return_logits=True)
                for i in range(len(coords)):
                    # Same 1024->64 nearest-neighbour sampling as the published code.
                    selection=masks[i,0,::16,::16]>0
                    if not selection.any():continue
                    avg=predictor.features[0,:,selection].mean(1).cpu().numpy()
                    if not np.isfinite(avg).all():continue
                    features.append(avg);ious.append(float(confidence[i,0]));low_masks.append(low[i,0].cpu().numpy());areas.append(int((masks[i,0]>0).sum()));valid_ids.append(lo+i)
                progress(lo+len(coords)+1,len(points)+1)
            predictor.reset_image()
        if not features:raise ValueError('SAFIRE produced no nonempty region. Try a denser point grid.')
        if cache is not None:
            cache.clear();cache.update(key=cache_key,value=(points,features,ious,low_masks,areas,valid_ids))
    features=np.stack(features);assignments=cluster(features,kind,2 if binary else groups,eps,minimum)
    chosen=[max(np.flatnonzero(assignments==label),key=lambda i:ious[i]) for label in np.unique(assignments) if label>=0]
    if not chosen:raise ValueError('SAFIRE clustering produced no supported source region. Adjust the grouping settings.')
    if len(chosen)>64:raise ValueError('More than 64 source groups. Increase DBSCAN distance or use k-means.')
    # Upstream multi-source channel order follows the original prompt order.
    chosen=sorted(chosen)
    with torch.inference_mode():
        logits=F.interpolate(torch.from_numpy(np.stack([low_masks[i] for i in chosen]))[:,None],size=(1024,1024),mode='bilinear',align_corners=False)[:,0]
        probabilities=torch.softmax(logits,dim=0).numpy()
        if binary:
            if len(chosen)>1:
                front,back=(0,1) if areas[chosen[0]]<=areas[chosen[1]] else (1,0)
                suspicion=(torch.sigmoid(logits[front])+(1-torch.sigmoid(logits[back])))/2
            else:suspicion=torch.sigmoid(logits[0])
            output=suspicion.numpy()
        else:output=probabilities.max(0)
    return dict(map=output,source_probabilities=probabilities,source_labels=probabilities.argmax(0).astype(np.uint8),prompt_features=features,prompt_confidence=np.asarray(ious,np.float32),prompt_clusters=assignments,points=points[np.asarray(valid_ids)].astype(np.float32),
        metadata=dict(image_shape=list(image.shape[:2]),analysis_shape=[1024,1024],sources=len(chosen),chosen_prompt_ids=[valid_ids[i] for i in chosen],valid_prompts=len(valid_ids),total_prompts=len(points),binary=binary,cluster=kind,seed=1701,points_per_batch=16,proposals_reused=reused))
