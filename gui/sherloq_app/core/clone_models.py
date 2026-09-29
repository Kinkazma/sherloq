"""Pinned local clone-detector assets and strict model loading (worker only)."""
from pathlib import Path
from functools import lru_cache
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'integration/clone_detectors'
WEIGHTS=ROOT/'models/external/clone_detectors'
@lru_cache(1)
def inventory():return {r['path']:r['sha256'] for r in json.loads((BASE/'files.sha256.json').read_text())['files']}
def verified(path):
    path=Path(path).resolve();relative=str(path.relative_to(ROOT))
    if not path.is_file():raise FileNotFoundError('Poids ou source manquant : '+relative)
    expected=inventory().get(relative)
    if expected is None:raise ValueError('Ressource non inventoriée : '+relative)
    with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
    if actual!=expected:raise ValueError('Empreinte incorrecte : '+relative)
    return path,actual
@lru_cache(1)
def runtime():
    for row in json.loads((BASE/'runtime-manifest.json').read_text()):
        if hashlib.sha256((ROOT/row['target']).read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Adaptateur local modifié : '+row['target'])
    sys.path.insert(0,str(BASE))

def load_segmentation(variant,device):
    import torch
    from importlib import import_module
    runtime()
    if variant.startswith('CMSeg-Net'):
        from sherloq_clone_models.cmsegnet.model import UnetMobilenetV2
        filename='addnoise' if 'addnoise' in variant else 'generalization'
        path,sha=verified(WEIGHTS/'03_cmsegnet/predict_folder/weight'/(filename+'.pth'))
        model=UnetMobilenetV2(pretrained=False);side=512;kind='sigmoid'
    else:
        names={'MGCFDN 16×16':('MGCFDN','MGCFDN_16x16.pth',1),'MGCFDN TNT 16×16':('TNT_MGCFDN','TNT_MGCFDN_16x16.pth',1),'MGCFDN VIG 16×16':('VIG_MGCFDN','VIG_MGCFDN_16x16.pth',1),'MGCFDN':('MGCFDN','MGCFDN.pth',1),'MGCFDN source/cible':('MGCFDN','MGCFDN_ST.pth',3),'MGCFDN EffNet 16×16':('EffNet_MGCFDN','EffNet_MGCFDN_16x16.pth',1),'MGCFDN MPDN 16×16':('MPDN_MGCFDN','MPDN_MGCFDN_16x16.pth',1)}
        module,filename,channels=names[variant];path,sha=verified(WEIGHTS/'04_mgcfdn/weight'/filename)
        model=import_module('sherloq_clone_models.mgcfdn.model.'+module).MultiGranularityConsistencyForgeryDetectionNet(out_channel=channels)
        if variant=='MGCFDN 16×16':
            from functools import partial
            model.visual_feature_extractor.forward=partial(model.visual_feature_extractor.forward,out_size=(16,16))
        side=256;kind='softmax' if channels==3 else 'sigmoid'
    state=torch.load(path,map_location='cpu',weights_only=True);model.load_state_dict(state,strict=True)
    return dict(model=model.eval().to(device),side=side,kind=kind,device=device,weights={str(path.relative_to(ROOT)):sha})

def segment(image,loaded):
    import torch,cv2 as cv
    from PIL import Image
    from torchvision import transforms as T
    side=loaded['side'];x=T.Compose([T.Resize((side,side)),T.ToTensor()])(Image.fromarray(image[:,:,::-1])).unsqueeze(0).to(loaded['device'])
    with torch.inference_mode():
        logits=loaded['model'](x);p=(logits.softmax(1) if loaded['kind']=='softmax' else logits.sigmoid())[0].cpu().numpy()
    if not __import__('numpy').isfinite(p).all():raise ValueError('Sortie non finie du modèle.')
    return p
