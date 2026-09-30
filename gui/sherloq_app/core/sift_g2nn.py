"""SIFT/G2NN classical adaptation of the luc_pub research procedure.

Independent implementation. Original profile: contrast .001, G2NN .7,
4x extraction below mean side 1024, unique centres, affine RANSAC >=24.
SHERLOQ spatial constraints precede exact neighbour selection; unlike the
upstream unconstrained FLANN search, no forbidden neighbour can win the ratio.
"""
import cv2 as cv
import os
import numpy as np
from scipy.spatial import cKDTree
from .cloning import check,pack_keypoints

NAME='SIFT + G2NN + RANSAC'

def matching_backend(desc,members,force_cpu=False):
    """Use the exact accelerator for the measured large-search workload.

    Small jobs retain CPU startup latency. The environment override is for
    reproducible benchmarks; the UI CPU switch always takes precedence.
    """
    mode=os.environ.get('SHERLOQ_G2NN_METAL','auto')
    if force_cpu or mode=='0':return 'cpu'
    if mode!='1' and np.max(np.count_nonzero(members,axis=0),initial=0)<20000:return 'cpu'
    if not exact_mps_eligible(desc):return 'cpu'
    from .learned_copy import device_for
    return device_for(False)

def _extract_global(image,limit,regions,excluded,cancel):
    from .cloning2 import polygon_mask,memberships
    gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY);gray=cv.normalize(gray,None,0,255,cv.NORM_MINMAX)
    scale=4 if sum(gray.shape)/2<1024 else 1
    if gray.size*scale*scale>32_000_000:raise ValueError('SIFT G2NN : image trop grande ; sélectionner une image plus petite.')
    check(cancel);mask=polygon_mask(gray.shape,regions,excluded)
    enlarged=cv.resize(gray,None,fx=scale,fy=scale,interpolation=cv.INTER_CUBIC)
    if mask is not None:mask=cv.resize(mask,(enlarged.shape[1],enlarged.shape[0]),interpolation=cv.INTER_NEAREST)
    kp,desc=cv.SIFT_create(nfeatures=limit*2,contrastThreshold=.001).detectAndCompute(enlarged,mask)
    check(cancel)
    if desc is None:return np.empty((0,7),np.float32),np.empty((0,128),np.float32),np.empty((0,max(1,len(regions))),bool),0
    for k in kp:k.pt=(k.pt[0]/scale,k.pt[1]/scale);k.size/=scale
    points=pack_keypoints(kp);_,ids=np.unique(np.round(points[:,:2],6),axis=0,return_index=True);ids=np.sort(ids)
    ids=ids[np.argsort(-points[ids,4],kind='stable')[:limit]];points=points[ids];desc=desc[ids]
    return points,desc,memberships(points,gray.shape,regions),len(kp)

def extract(image,limit,regions,excluded,cancel):
    """Each ROI has its own contrast normalization, scale and point budget.

    Adding an enclosing ROI must not change features of an existing child.
    Exact global behaviour is preserved when there are no selected regions.
    Coordinates and keypoint sizes are returned in original image pixels.
    """
    if not regions:return _extract_global(image,limit,(),excluded,cancel)
    points=[];descriptors=[];membership=[];total=0
    h,w=image.shape[:2]
    for zone,polygon in enumerate(regions):
        check(cancel)
        xy=np.asarray(polygon);lo=np.maximum(0,np.floor(xy.min(0))).astype(int)
        hi=np.minimum([w,h],np.ceil(xy.max(0))+1).astype(int)
        x,y=lo;right,bottom=hi
        if right<=x or bottom<=y:continue
        local=tuple(tuple((px-x,py-y) for px,py in poly) for poly in (polygon,))
        exclusions=tuple(tuple((px-x,py-y) for px,py in poly) for poly in excluded)
        p,d,_,count=_extract_global(image[y:bottom,x:right],limit,local,exclusions,cancel)
        p[:,:2]+=lo;members=np.zeros((len(p),len(regions)),bool);members[:,zone]=True
        points.append(p);descriptors.append(d);membership.append(members);total+=count
    if not points:return np.empty((0,7),np.float32),np.empty((0,128),np.float32),np.empty((0,len(regions)),bool),0
    return np.concatenate(points),np.concatenate(descriptors),np.concatenate(membership),total

class ExactSiftNeighbours:
    """Integer-valued OpenCV SIFT on MPS, with stable index tie breaking.

    Norm bounds keep every integer product/sum exactly representable in float32.
    Other descriptor formats stay on the CPU matcher. No approximate ranking.
    """
    def __init__(self,training):
        import torch
        self.torch=torch
        self.train=torch.as_tensor(training,device='mps')
        self.norm=(self.train*self.train).sum(1)

    def query(self,queries,allowed,k):
        torch=self.torch
        q=torch.as_tensor(queries,device='mps')
        squared=(q*q).sum(1)[:,None]+self.norm[None,:]-2*(q@self.train.T)
        squared.masked_fill_(~torch.as_tensor(allowed,device='mps'),float('inf'))
        indices=[];values=[];rows=torch.arange(len(q),device='mps')
        for _ in range(k):
            value,index=torch.min(squared,dim=1)
            values.append(value);indices.append(index)
            squared[rows,index]=float('inf')
        return torch.stack(indices,1).cpu().numpy(),torch.stack(values,1).cpu().numpy()


def exact_mps_eligible(desc):
    # Bounds apply to this specific quantized SIFT representation only.
    return (desc.dtype==np.float32 and np.isfinite(desc).all() and
            np.array_equal(desc,np.floor(desc)) and np.max(np.abs(desc),initial=0)<=255 and
            np.max(np.sum(desc*desc,axis=1),initial=0)<2**21)

def match(points,desc,members,radius,minimum,ratio,compare,cancel,progress,radii=None,gap=(0.,0.),axes=None,backend="cpu",variants=None):
    """Exact OpenCV top-k, independently per search region before union.

    The enclosing rectangle must not suppress an inner rectangle's ratio.
    Masks enforce spatial constraints before neighbor selection. Query batches
    bound temporary storage and cancellation latency; no FLANN approximation.
    """
    from .auto_zones import compact_points
    from .cloning2 import MAX_PAIRS
    coords=compact_points(points[:,:2],axes);gap=np.asarray(gap);pairs={};evaluated=0
    if not len(points):return np.empty((0,4),np.float64),0
    jobs=[]
    if compare:
        if members.shape[1]!=2:raise ValueError('Compare requires exactly two zones.')
        jobs=[(np.flatnonzero(members[:,0]),np.flatnonzero(members[:,1]),radius,gap),
              (np.flatnonzero(members[:,1]),np.flatnonzero(members[:,0]),radius,-gap)]
    else:
        for zone in range(members.shape[1]):
            ids=np.flatnonzero(members[:,zone])
            jobs.append((ids,ids,min(radius,radii[zone]) if radii is not None else radius,np.zeros(2)))
    if variants is not None:
        # Cross-frame only, before nearest-neighbour selection. Same-position
        # copies are still excluded by the original spatial minimum below.
        jobs=[(q[variants[q]==v],t[variants[t]!=v],r,g) for q,t,r,g in jobs for v in (False,True)]
    total=sum(len(q) for q,t,r,g in jobs);done=0
    matcher=cv.BFMatcher(cv.NORM_L2SQR)
    gpu=backend=="mps" and exact_mps_eligible(desc)
    for query,train,maximum,shift in jobs:
        if len(train)<2:done+=len(query);continue
        training=np.ascontiguousarray(desc[train],np.float32)
        accelerator=ExactSiftNeighbours(training) if gpu else None
        batch=256 if gpu else 64
        for start in range(0,len(query),batch):
            check(cancel);qs=query[start:start+batch]
            delta=coords[train][None,:,:]-coords[qs][:,None,:]
            if compare:delta-=shift
            distance=np.linalg.norm(delta,axis=2)
            allowed=(distance>=minimum)&(distance<=maximum)&(qs[:,None]!=train[None,:])
            evaluated+=int(allowed.sum())
            if accelerator is not None:
                positions,values=accelerator.query(np.ascontiguousarray(desc[qs],np.float32),allowed,min(4,len(train)))
                found=[positions[row][np.isfinite(values[row])] for row in range(len(qs))]
            else:
                matches=matcher.knnMatch(np.ascontiguousarray(desc[qs],np.float32),training,k=min(4,len(train)),mask=allowed.astype(np.uint8),compactResult=False)
                found=[np.array([m.trainIdx for m in neighbors],dtype=int) for neighbors in matches]
            for row,local in enumerate(found):
                if len(local)<2:continue
                i=int(qs[row]);js=train[local]
                # Restore the reference NumPy score/ratio arithmetic on the few
                # selected neighbors; SIFT's integer descriptors make L2SQR exact.
                d=np.linalg.norm(desc[js]-desc[i],axis=1);rank=np.lexsort((js,d));cutoff=None
                for j in range(len(rank)-1):
                    if d[rank[j+1]]>d[rank[0]]/ratio:cutoff=j+1;break
                if cutoff is None:continue
                for k in rank[:cutoff]:
                    key=tuple(sorted((i,int(js[k]))));pairs[key]=(key[0],key[1],float(d[k]/(512*np.sqrt(2))),float(distance[row,local[k]]))
            if len(pairs)>MAX_PAIRS:raise ValueError('Trop de correspondances G2NN ; réduire les zones ou le nombre de points.')
            done+=len(qs);progress(15+60*done//max(1,total),'Voisins G2NN par zone')
    return np.array([pairs[k] for k in sorted(pairs)],np.float64).reshape(-1,4),evaluated

def filter_models(groups,models):
    selected=[];accepted=[]
    for group,model in zip(groups,models):
        s=np.linalg.svd(np.asarray(model['matrix'])[:2,:2],compute_uv=False)
        scale=float(np.sqrt(s.prod()))
        if .25<=scale<=4 and s[-1]>0 and s[0]/s[-1]<=3.1:selected.append(group);accepted.append(model)
    return tuple(selected),tuple(accepted)
