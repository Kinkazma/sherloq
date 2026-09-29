"""Whole-image segmentation and independent rectangular clone-analysis jobs."""
import gc,time
import numpy as np
import cv2 as cv
from .clone_models import ROOT,load_segmentation,segment
SEGMENTERS=('CMSeg-Net generalization','CMSeg-Net addnoise','MGCFDN','MGCFDN source/cible','MGCFDN 16×16','MGCFDN EffNet 16×16','MGCFDN MPDN 16×16','MGCFDN TNT 16×16','MGCFDN VIG 16×16')
FORGERYSCOPE=('Forgeryscope microscopie','Forgeryscope blots complets','Forgeryscope chevauchements','Forgeryscope pistes')
VARIANTS=SEGMENTERS+FORGERYSCOPE

def boxes_for(shape,regions):
    h,w=shape[:2];boxes=[]
    for polygon in regions:
        p=np.asarray(polygon,float)
        if p.shape!=(4,2) or not np.isfinite(p).all():raise ValueError('Ces détecteurs utilisent des zones rectangulaires.')
        x0,y0=np.floor(p.min(0)).astype(int);x1,y1=np.ceil(p.max(0)).astype(int)+1
        corners={(float(x),float(y)) for x,y in p}
        if corners!={(x,y) for x in (p[:,0].min(),p[:,0].max()) for y in (p[:,1].min(),p[:,1].max())}:raise ValueError('La zone doit être un rectangle aligné sur l’image.')
        x0=max(0,x0);y0=max(0,y0);x1=min(w,x1);y1=min(h,y1)
        if x1-x0<8 or y1-y0<8:raise ValueError('Chaque zone doit mesurer au moins 8 × 8 pixels.')
        boxes.append(tuple(map(int,(x0,y0,x1,y1))))
    return boxes or [(0,0,w,h)]

class Models:
    def __init__(self):self.key=None;self.value=None;self.signatures=()
    def clear(self):
        self.key=None;self.value=None;self.signatures=();gc.collect()
        import torch
        if torch.backends.mps.is_available():torch.mps.empty_cache()
    def get(self,variant,requested):
        import torch
        # VIG's graph neighbour ranking amplified MPS differences (~0.12).
        # Forgeryscope keeps YOLO/keypoint extraction exact on CPU; neural matching uses MPS.
        device='cpu' if requested=='cpu' or variant=='MGCFDN VIG 16×16' or not torch.backends.mps.is_available() else 'mps'
        key=('forgeryscope' if variant in FORGERYSCOPE else variant,device,'float32')
        signature=lambda paths:tuple((p,(ROOT/p).stat().st_size,(ROOT/p).stat().st_mtime_ns) for p in paths)
        if key==self.key and signature(self.value['weights'])==self.signatures:return self.value,True
        self.clear()
        if variant in FORGERYSCOPE:
            from .forgeryscope_adapter import load
            value=load(device)
        else:value=load_segmentation(variant,device)
        self.value=value;self.key=key;self.signatures=signature(value['weights']);return value,False

def analyze(image,params,requested,models,progress=lambda *x:None):
    variant=params['variant'];regions=params.get('regions',());compare=bool(params.get('compare',False))
    if variant not in VARIANTS:raise ValueError('Détecteur inconnu ou poids non disponibles.')
    if requested not in ('cpu','mps'):raise ValueError('Device invalide.')
    if params.get('selection_present') and not regions:raise ValueError('Aucune zone active.')
    if compare and (variant not in FORGERYSCOPE or len(regions)!=2):raise ValueError('Comparer exige deux zones et un profil Forgeryscope.')
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:raise ValueError('Image BGR 8 bits attendue.')
    if image.shape[0]*image.shape[1]>64_000_000:raise ValueError('Image au-delà du budget de 64 mégapixels.')
    boxes=boxes_for(image.shape,regions);loaded,reused=models.get(variant,requested)
    excluded=boxes_for(image.shape,params['excluded']) if params.get('excluded') else []
    if excluded and (compare or variant not in FORGERYSCOPE):raise ValueError('Exclusions require independent Forgeryscope search.')
    allowed=np.ones(image.shape[:2],np.uint8)
    for x0,y0,x1,y1 in excluded:allowed[y0:y1,x0:x1]=0
    score=np.zeros(image.shape[:2],np.float32);mask=np.zeros(image.shape[:2],np.uint8);candidates=np.zeros_like(mask);analyzed=np.zeros_like(mask);source=np.zeros_like(score);target=np.zeros_like(score);raw=[];metadata=[]
    if compare:
        # Context outside the two panels is never passed to either matcher or embedder.
        label='Microscopy' if 'microscopie' in variant else 'Blots';panels=[(label,1.,*b) for b in boxes]
        work=np.zeros_like(image)
        for x0,y0,x1,y1 in boxes:work[y0:y1,x0:x1]=image[y0:y1,x0:x1];analyzed[y0:y1,x0:x1]=1
        from .forgeryscope_adapter import predict
        r=predict(work,loaded,variant,panels,progress);score=r['map']*analyzed;mask=r['mask']*analyzed;candidates=r['candidates']*analyzed;metadata.append(r['metadata'])
    else:
        for index,(x0,y0,x1,y1) in enumerate(boxes):
            crop=np.ascontiguousarray(image[y0:y1,x0:x1]);analyzed[y0:y1,x0:x1]=1
            if variant in FORGERYSCOPE:
                from .forgeryscope_adapter import predict
                local_excluded=[(max(a,x0)-x0,max(b,y0)-y0,min(c,x1)-x0,min(d,y1)-y0) for a,b,c,d in excluded if min(c,x1)>max(a,x0) and min(d,y1)>max(b,y0)]
                if local_excluded:
                    crop=crop.copy();crop[allowed[y0:y1,x0:x1]==0]=0
                extra={'excluded_boxes':local_excluded} if local_excluded else {}
                r=predict(crop,loaded,variant,progress=progress,**extra);value=r['map'];binary=r['mask'];candidates[y0:y1,x0:x1]|=r['candidates'];metadata.append(dict(origin=[x0,y0],**r['metadata']))
            else:
                p=segment(crop,loaded);raw.append(p)
                size=(x1-x0,y1-y0)
                if p.shape[0]==3:
                    # Official test_st_core.py: 0 = target, 1 = source, 2 = background.
                    value=cv.resize(p[0]+p[1],size,interpolation=cv.INTER_LINEAR);binary=cv.resize(((p[0]>=.5)|(p[1]>=.5)).astype(np.uint8),size,interpolation=cv.INTER_NEAREST)
                    source[y0:y1,x0:x1]=np.maximum(source[y0:y1,x0:x1],cv.resize(p[1],size));target[y0:y1,x0:x1]=np.maximum(target[y0:y1,x0:x1],cv.resize(p[0],size))
                else:value=cv.resize(p[0],size,interpolation=cv.INTER_LINEAR);binary=cv.resize((p[0]>.5).astype(np.uint8),size,interpolation=cv.INTER_NEAREST)
                metadata.append(dict(origin=[x0,y0],analysis_shape=list(p.shape),status='ok' if binary.any() else 'empty'))
            score[y0:y1,x0:x1]=np.maximum(score[y0:y1,x0:x1],value);mask[y0:y1,x0:x1]|=binary;progress(index+1,len(boxes))
    score*=allowed;mask*=allowed;candidates*=allowed;analyzed*=allowed
    result=dict(map=score,mask=mask,candidates=candidates,analyzed=analyzed)
    if raw:result['raw_probabilities']=np.stack(raw)
    if variant=='MGCFDN source/cible':result.update(source=source,target=target)
    result['metadata']=dict(method='clone_detectors',variant=variant,device=loaded['device'],requested_device=requested,weights=loaded['weights'],model_reused=reused,engines=loaded.get('engines',{'segmentation':loaded['device']}),boxes=boxes,compare=compare,zones=metadata,threshold=.5,probability_interpolation='bilinear',mask_interpolation='nearest',status='ok' if mask.any() else 'candidates' if candidates.any() else 'no_panels' if all(m['status']=='no_panels' for m in metadata) else 'insufficient_panels' if all(m['status'] in ('no_panels','insufficient_panels') for m in metadata) else 'empty',radius_supported=False,segmentation=variant in SEGMENTERS)
    if variant=='MGCFDN source/cible':result['metadata']['channel_order']=['target','source','background']
    if excluded:result['metadata']['excluded_boxes']=excluded
    return result
