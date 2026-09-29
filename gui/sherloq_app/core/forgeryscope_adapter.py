"""Local public Forgeryscope profile; no competition-only checkpoints implied."""
import importlib,json,tempfile,os
from pathlib import Path
import numpy as np
from .clone_models import ROOT,BASE,WEIGHTS,verified,runtime

def local_yolo(path):
    import torch,yaml
    from ultralytics import YOLO, settings
    settings.update({'sync': False})
    path,sha=verified(path)
    records=json.loads((BASE/'checkpoint-inspection.json').read_text())
    record=next(r for r in records if r['path']==str(path.relative_to(ROOT)))
    symbols=record['unsafe_globals'];actual=torch.serialization.get_unsafe_globals_in_checkpoint(path)
    if sorted(actual)!=sorted(symbols):raise ValueError('Classes inattendues dans le checkpoint YOLO.')
    allowed=[]
    for symbol in symbols:
        if not symbol.startswith(('torch.nn.modules.','ultralytics.nn.')):raise ValueError('Classe YOLO non autorisée.')
        mod,cls=symbol.rsplit('.',1);allowed.append(getattr(importlib.import_module(mod),cls))
    with torch.serialization.safe_globals(allowed):raw=torch.load(path,map_location='cpu',weights_only=True)
    network=raw['model'].float();scale=network.yaml['scale']
    with tempfile.TemporaryDirectory(prefix='sherloq-yolo-') as folder:
        config=Path(folder)/('yolo11'+scale+'.yaml');config.write_text(yaml.safe_dump(network.yaml))
        model=YOLO(str(config),task='detect')
        model.model.load_state_dict(network.state_dict(),strict=True);model.model.names=network.names
    return model

def local_lightglue(features,**kwargs):
    import torch
    from ..vendor.lightglue.lightglue import LightGlue
    kind=features;model=LightGlue(features=None,input_dim=128,add_scale_ori=kind=='sift',flash=False,**kwargs)
    filename='sift_lightglue.pth' if kind=='sift' else 'aliked_lightglue.pth'
    path,sha=verified(ROOT/'models/external'/filename);state=torch.load(path,map_location='cpu',weights_only=True)
    for i in range(9):state={k.replace(f'self_attn.{i}',f'transformers.{i}.self_attn').replace(f'cross_attn.{i}',f'transformers.{i}.cross_attn'):v for k,v in state.items()}
    if 'confidence_thresholds' not in state:state['confidence_thresholds']=model.confidence_thresholds
    model.load_state_dict(state,strict=True);return model

class ReferenceExtractor:
    """Exact CPU features, bounded device cache, reused across panel pairs/flips."""
    def __init__(self, extractor, device, budget=64 * 1024**2):
        from collections import OrderedDict
        self.extractor = extractor.cpu()
        self.device = device
        self.cache = OrderedDict()
        self.bytes = 0
        self.budget = budget
        self.computations = 0
    def extract(self, image, **preprocess):
        import hashlib
        from ..vendor.lightglue.utils import map_tensor
        image = image.cpu().contiguous()
        key = (tuple(image.shape), str(image.dtype), repr(sorted(preprocess.items())),
               hashlib.sha256(image.numpy().tobytes()).digest())
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key][0]
        features = self.extractor.extract(image, **preprocess)
        features = map_tensor(features, lambda tensor: tensor.to(self.device))
        self.computations += 1
        sizes = []
        map_tensor(features, lambda tensor: sizes.append(tensor.numel() * tensor.element_size()) or tensor)
        size = sum(sizes)
        if size <= self.budget:
            while self.cache and self.bytes + size > self.budget:
                _, (_, old_size) = self.cache.popitem(last=False)
                self.bytes -= old_size
            self.cache[key] = (features, size)
            self.bytes += size
        return features

def load(device):
    runtime();folder=WEIGHTS/'02_forgeryscope';hashes={}
    for name in ('yolo_panel_extractor.pt','yolo_lane_extractor.pt','aliked_wblot.pth','wblot_duplicate_embedder.ckpt','wblot_overlap_embedder.ckpt','wblot_lane_embedder.ckpt','micro_overlap_embedder.ckpt'):
        path,sha=verified(folder/name);hashes[str(path.relative_to(ROOT))]=sha
    for name in ('aliked-n16.pth','aliked_lightglue.pth','sift_lightglue.pth'):
        path,sha=verified(ROOT/'models/external'/name);hashes[str(path.relative_to(ROOT))]=sha
    os.environ['FORGERYSCOPE_CACHE_DIR']=str(folder)
    from sherloq_clone_models.forgeryscope import Embedder,PanelExtractor
    from sherloq_clone_models.forgeryscope.matcher.lightglue import LightGlueOverlap
    panel=PanelExtractor(device='cpu',conf_threshold=.7,iou_threshold=.4);panel.EXCLUDED_LABELS={'Graphs','Flow Cytometry','Body Imaging'}
    lane=local_yolo(folder/'yolo_lane_extractor.pt');lane.to('cpu')
    embeds={n:Embedder(n,device=device,verbose=False) for n in ('wblot_duplicate_embedder','wblot_overlap_embedder','wblot_lane_embedder','micro_overlap_embedder')}
    micro=LightGlueOverlap(max_keypoints=4096,matcher_features='sift',device='cpu',depth_confidence=.9,width_confidence=.9)
    blot=LightGlueOverlap(max_keypoints=512,matcher_features='aliked',device='cpu',depth_confidence=-1,width_confidence=-1,estimator_method='MAGSAC',reprojThreshold=3.)
    import torch
    state=torch.load(folder/'aliked_wblot.pth',map_location='cpu',weights_only=True)['model']
    blot.extractor.load_state_dict({k.removeprefix('extractor.'):v for k,v in state.items() if k.startswith('extractor.')},strict=True)
    weights={k.removeprefix('matcher.'):v for k,v in state.items() if k.startswith('matcher.')}
    for i in range(9):weights={k.replace(f'self_attn.{i}',f'transformers.{i}.self_attn').replace(f'cross_attn.{i}',f'transformers.{i}.cross_attn'):v for k,v in weights.items()}
    if 'confidence_thresholds' not in weights:weights['confidence_thresholds']=blot.matcher.confidence_thresholds
    blot.matcher.load_state_dict(weights,strict=True)
    for matcher in (micro, blot):
        matcher.extractor = ReferenceExtractor(matcher.extractor, device)
        matcher.matcher.to(device)
        matcher.device = torch.device(device)
    return dict(panel=panel,lane=lane,embeds=embeds,micro=micro,blot=blot,device=device,weights=hashes,engines=dict(panel_detector='cpu',lane_detector='cpu',keypoints='cpu',embeddings=device,lightglue=device,geometry='cpu'))

def predict(image,loaded,profile,panels=None,progress=lambda *args: None,excluded_boxes=()):
    """Search auto panels, or compare exactly two explicitly supplied panels.

    Embedding-only full-box fallbacks are exported as candidates, separately
    from geometrically supported masks. No cross-job/ROI matching is possible.
    """
    import pandas as pd
    from sherloq_clone_models.forgeryscope import Embedder,PanelExtractor
    from sherloq_clone_models.forgeryscope.matcher.geometry import get_intersections
    from sherloq_clone_models.forgeryscope.matcher.lightglue import create_duplicate_masks
    from sherloq_clone_models.forgeryscope.matcher.lane import find_lanes_in_blot_panels,create_lane_match_masks
    rgb=np.ascontiguousarray(image[:,:,::-1]);manual=panels is not None
    if panels is None:panels=loaded['panel'].extract_panels(rgb)
    label='Microscopy' if 'microscopie' in profile else 'Blots'
    panels=[tuple(p) for p in panels if p[0]==label]
    # Automatic search can disable a subimage inside its enclosing ROI. Never
    # pass a panel touching excluded pixels to embeddings or geometric matching.
    rejected=[]
    if excluded_boxes:
        def touches(p):
            return any(min(p[4],x1)>max(p[2],x0) and min(p[5],y1)>max(p[3],y0)
                       for x0,y0,x1,y1 in excluded_boxes)
        rejected=[p for p in panels if touches(p)]
        panels=[p for p in panels if not touches(p)]
    if len(panels)>64:raise ValueError('Plus de 64 panneaux : sélectionner des zones plus petites.')
    crops=PanelExtractor.crop_panels(rgb,panels);mask=np.zeros(image.shape[:2],np.uint8);candidates=np.zeros_like(mask);details=[]
    meta=dict(panels=panels,profile=profile,public_simplified=True,status='no_panels' if not panels else 'empty',comparisons=[],analysis='panel comparisons',manual_panels=manual)
    if excluded_boxes:meta.update(excluded_boxes=list(excluded_boxes),excluded_panels=rejected)
    if not panels or (len(panels)<2 and 'pistes' not in profile):
        if panels:meta['status']='insufficient_panels'
        return dict(mask=mask,map=mask.astype(np.float32),candidates=candidates,metadata=meta)
    embeds=loaded['embeds'];similar=[]
    if 'pistes' in profile:
        lanes=find_lanes_in_blot_panels(panels,list(range(len(panels))),crops,loaded['lane'],embeds['wblot_lane_embedder'],similarity_threshold=.65,overlap_threshold=5)
        masks=create_lane_match_masks(image.shape,lanes['best_matches'],lanes=lanes['lanes'])
        for m in masks:candidates|=m.astype(np.uint8)
        meta.update(lanes=len(lanes['lanes']),lane_matches=len(lanes['best_matches']),status='candidates' if candidates.any() else 'empty',mask_semantics='embedding-supported lane candidates; no geometric verification')
        return dict(mask=mask,map=mask.astype(np.float32),candidates=candidates,metadata=meta)
    models=[('micro_overlap_embedder',.58)] if label=='Microscopy' else [('wblot_duplicate_embedder',.84)] if 'complets' in profile else [('wblot_overlap_embedder',.85)]
    intersections=set(get_intersections(panels,margin=10))
    for name,threshold in models:
        vectors=np.concatenate([embeds[name].get_embedding_batch(crops[i:i+8]).cpu().numpy() for i in range(0,len(crops),8)])
        for i,j,score in Embedder.find_similar_pairs(vectors,threshold):
            if (i,j) in intersections or (j,i) in intersections:continue
            similar.append((label,float(score),i,j))
    meta['embedding_candidates']=[dict(label=l,score=s,panel0=i,panel1=j) for l,s,i,j in similar]
    if len(similar)>256:raise ValueError('Plus de 256 comparaisons de panneaux : réduire les zones.')
    df=pd.DataFrame(similar,columns=['label','score','idx1','idx2'])
    if len(df):
        def matches():
            for index in range(len(df)):
                progress(index, len(df))
                yield from create_duplicate_masks(rgb,panels,crops,df.iloc[index:index+1],loaded['micro'],loaded['blot'],to_bbox_micro=False,to_bbox_blot=True,fallback_for_wblot=True,test_transforms_micro=True,test_transforms_blot=False)
        for r in matches():
            m=r['match_result'];supported='fallback' not in m and m['inliers']>=8 and m['mean_match_score']>=.73
            union=(r['mask0']|r['mask1']).astype(np.uint8)
            if supported:mask|=union
            else:candidates|=union
            details.append(dict(panel0=int(r['panel_id0']),panel1=int(r['panel_id1']),supported=bool(supported),inliers=int(m['inliers']),score=float(m['mean_match_score']),matrix=m['H'].tolist() if 'H' in m else None,polygon0=np.asarray(r['poly_coords0']).tolist(),polygon1=np.asarray(r['poly_coords1']).tolist(),fallback=m.get('fallback')))
    meta.update(comparisons=details,status='ok' if mask.any() else 'candidates' if candidates.any() else 'empty',mask_semantics='geometric overlap envelopes, not pixel segmentation')
    return dict(mask=mask,map=mask.astype(np.float32),candidates=candidates,metadata=meta)
