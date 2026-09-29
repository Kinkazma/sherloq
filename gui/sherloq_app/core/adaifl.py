"""Pinned AdaIFL inference and deterministic per-image token routing."""
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4]

def load(device):
    import torch
    from ..vendor.adaifl.models.net import AdaIFL
    model=AdaIFL();state=torch.load(ROOT/'models/external/AdaIFL_v0.pth',weights_only=True,map_location='cpu');state={k.removeprefix('module.'):v for k,v in state.items()};model.load_state_dict(state,strict=True)
    return model.eval().to(device)

def predict(image,model):
    import torch
    from torchvision import transforms
    device=next(model.parameters()).device
    transform=transforms.Compose([transforms.ToPILImage(),transforms.Resize((1024,1024)),transforms.ToTensor()]);rgb=transform(image[:,:,::-1].copy())[None].to(device)
    torch.manual_seed(1701)
    with torch.inference_mode():output=torch.sigmoid(model(rgb))[0,0].cpu().numpy()
    return dict(map=output,mask=(output>.5).astype(np.uint8),metadata=dict(image_shape=list(image.shape[:2]),analysis_shape=[1024,1024],native_shape=list(output.shape),seed=1701,threshold=.5))
