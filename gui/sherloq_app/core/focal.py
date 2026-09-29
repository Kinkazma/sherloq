"""FOCAL published ViT-L+HRNet ensemble; two-cluster partition, not authenticity."""
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4]

def load(device):
    import torch
    from ..vendor.focal.models.vit import FOCAL_ViT
    from ..vendor.focal.models.hrnet import FOCAL_HRNet
    models=[]
    for name,constructor in (('FOCAL_ViT',FOCAL_ViT),('FOCAL_HRNet',FOCAL_HRNet)):
        model=constructor();state=torch.load(ROOT/'models/external'/f'{name}_weights.pth',map_location='cpu',weights_only=True);state={k.removeprefix('module.'):v for k,v in state.items()}
        # The released files also contain a two-class convolution absent from
        # both published feature-extractor classes and their forward methods.
        # Validate exactly these unused tensors; all inference tensors are strict.
        extras=set(state)-set(model.state_dict());channels=256 if name=='FOCAL_ViT' else 32
        if extras!={'fc.weight','fc.bias'} or state['fc.weight'].shape!=(2,channels,3,3) or state['fc.bias'].shape!=(2,):raise ValueError('Unexpected FOCAL checkpoint extras.')
        for key in extras:del state[key]
        model.load_state_dict(state,strict=True);models.append(model.eval().to(device))
    return torch.nn.ModuleList(models)

def predict(image,models):
    import torch,cv2 as cv
    import torch.nn.functional as F
    from torch_kmeans import KMeans
    from torch_kmeans.utils.distances import CosineSimilarity
    device=next(models.parameters()).device
    pixels=(cv.resize(image[:,:,::-1],(1024,1024)).astype(float)/255).astype(np.float32)
    rgb=torch.from_numpy(pixels.transpose(2,0,1))[None].to(device)
    with torch.inference_mode():
        first=models[0](rgb);h,w=first.shape[-2:];features=[F.normalize(first.permute(0,2,3,1),dim=3)]
        for model in models[1:]:features.append(F.normalize(F.interpolate(model(rgb),(h,w)).permute(0,2,3,1),dim=3))
        combined=torch.cat(features,dim=3).flatten(1,2).cpu()
        # Identical CPU clustering and fixed seed for both feature backends.
        result=KMeans(verbose=False,n_clusters=2,distance=CosineSimilarity,seed=123)(x=combined,k=2);labels=result.labels[0]
        if labels.sum()>(1-labels).sum():labels=1-labels
        output=labels.reshape(h,w).numpy().astype(np.float32)
    return dict(map=output,features=combined[0].numpy(),metadata=dict(image_shape=list(image.shape[:2]),analysis_shape=[1024,1024],native_shape=[h,w],seed=123,clusters=2,smallest_region_is_candidate=True,unused_checkpoint_heads=['fc.weight','fc.bias']))
