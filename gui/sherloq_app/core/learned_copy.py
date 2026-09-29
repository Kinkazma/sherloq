"""Pinned learned features and spatially sparse LightGlue adaptation.

No automatic network/weights loading. Descriptor candidates are selected by
geometry before learned cross attention and assignment. Independent ROI jobs
remain independent through every learned layer.
"""
from pathlib import Path
from threading import RLock
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import numpy as np
from .cloning import check
ROOT=Path(__file__).resolve().parents[4]
LOCK=RLock();MODELS={}
INFERENCE=ThreadPoolExecutor(max_workers=1,thread_name_prefix='sherloq-learned')


def inference_thread(function):
    """Own PyTorch's thread in Python, not Qt (checkpoint hook destructor safety)."""
    @wraps(function)
    def call(*args,**kwargs):
        return INFERENCE.submit(function,*args,**kwargs).result()
    return call


def device_for(cpu):
    import torch
    return 'cpu' if cpu or not torch.backends.mps.is_available() else 'mps'


def xfeat_model(device):
    key=('xfeat',device)
    if key not in MODELS:
        from ..vendor.xfeat.modules.xfeat import XFeat
        MODELS[key]=XFeat(str(ROOT/'models/external/xfeat.pt'),device=device)
    return MODELS[key]


@inference_thread
def extract(image,limit,regions,cpu=False,cancel=lambda:False,excluded=()):
    import torch
    from .cloning2 import polygon_mask,memberships
    device=device_for(cpu);check(cancel)
    with LOCK,torch.inference_mode():
        model=xfeat_model(device);mask=polygon_mask(image.shape[:2],regions,excluded)
        result=model.detectAndCompute(image,top_k=limit,mask=mask)[0]
        packed=np.zeros((len(result['keypoints']),7),np.float64);packed[:,:2]=result['keypoints'].cpu().numpy();packed[:,2]=8;packed[:,3]=-1;packed[:,4]=result['scores'].cpu().numpy();packed[:,6]=-1
        descriptors=result['descriptors'].cpu().numpy();check(cancel)
        order=np.lexsort((packed[:,0],packed[:,1]));packed=packed[order];descriptors=descriptors[order]
        return packed,descriptors,memberships(packed,image.shape[:2],regions),len(packed)


def lighter_model(device):
    import torch
    key=('lighter',device)
    if key not in MODELS:
        from ..vendor.lightglue.lightglue import LightGlue
        from ..vendor.xfeat.modules.model import XFeatModel
        raw=torch.load(ROOT/'models/external/xfeat-lighterglue.pt',map_location='cpu',weights_only=True)
        if any(not k.startswith(('matcher.','extractor.model.net.')) for k in raw):raise ValueError('Unexpected LighterGlue checkpoint keys.')
        # The official inference loads standalone xfeat.pt; the additional
        # extractor carried in this checkpoint is validated, but not substituted.
        XFeatModel().load_state_dict({k.removeprefix('extractor.model.net.'):v for k,v in raw.items() if k.startswith('extractor.model.net.')},strict=True)
        model=LightGlue(features=None,input_dim=64,descriptor_dim=96,n_layers=6,num_heads=1,flash=False,depth_confidence=-1,width_confidence=-1)
        state={k.removeprefix('matcher.'):v for k,v in raw.items() if k.startswith('matcher.')}
        for i in range(6):state={k.replace(f'self_attn.{i}',f'transformers.{i}.self_attn').replace(f'cross_attn.{i}',f'transformers.{i}.cross_attn'):v for k,v in state.items()}
        # Deterministic non-learned buffer introduced by newer LightGlue.
        state['confidence_thresholds']=model.confidence_thresholds
        model.load_state_dict(state,strict=True);MODELS[key]=model.eval().to(device)
    return MODELS[key]


@inference_thread
def extract_lightglue(image,limit,regions,cpu,kind,cancel=lambda:False,excluded=()):
    import torch
    from .cloning2 import polygon_mask,memberships
    from ..vendor.lightglue.utils import ImagePreprocessor
    device='cpu' if kind=='sift' else device_for(cpu);key=('extract',kind,device)
    with LOCK,torch.inference_mode():
        check(cancel)
        if key not in MODELS:
            if kind=='sift':
                from ..vendor.lightglue.sift import SIFT
                model=SIFT(max_num_keypoints=None)
            else:
                from ..vendor.lightglue.aliked import ALIKED
                model=ALIKED(model_name=kind,max_num_keypoints=-1)
            MODELS[key]=model.eval().to(device)
        model=MODELS[key];mask=polygon_mask(image.shape[:2],regions,excluded)
        rgb=torch.from_numpy(image[:,:,::-1].copy()).permute(2,0,1)[None].float()/255
        # Common CPU resampling ensures that both devices receive identical pixels.
        resized,scales=ImagePreprocessor(resize=None if kind=='sift' else 1024)(rgb)
        data={'image':resized.to(device)}
        if mask is not None:
            if kind=='sift':data['mask']=mask
            else:
                # Map resized pixel centres through the same convention as keypoints.
                h,w=resized.shape[-2:];xs=np.clip(np.rint((np.arange(w)+.5)/float(scales[0])-.5).astype(int),0,image.shape[1]-1);ys=np.clip(np.rint((np.arange(h)+.5)/float(scales[1])-.5).astype(int),0,image.shape[0]-1)
                data['mask']=torch.from_numpy(mask[ys[:,None],xs]>0)[None].to(device)
        r={k:v[0].cpu().numpy() for k,v in model(data).items()};check(cancel)
        packed=np.zeros((len(r['keypoints']),7),np.float64);packed[:,:2]=(r['keypoints'].astype(np.float64)+.5)/scales.numpy()-.5
        packed[:,2]=r['scales'] if kind=='sift' else 8;packed[:,3]=np.rad2deg(r['oris']) if kind=='sift' else -1;packed[:,4]=r['keypoint_scores'];packed[:,6]=-1
        member=memberships(packed,image.shape[:2],regions);eligible=member.any(1)
        if excluded:
            xy=np.rint(packed[:,:2]).astype(int);xy[:,0]=np.clip(xy[:,0],0,image.shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,image.shape[0]-1)
            eligible&=mask[xy[:,1],xy[:,0]]>0
        ids=np.flatnonzero(eligible);total=len(ids)
        ids=ids[np.argsort(-packed[ids,4],kind='stable')[:limit]]
        ids=ids[np.lexsort((packed[ids,0],packed[ids,1]))]
        return packed[ids],r['descriptors'][ids],member[ids],total


def lightglue_model(device,kind):
    import torch
    key=('lightglue',kind,device)
    if key not in MODELS:
        from ..vendor.lightglue.lightglue import LightGlue
        model=LightGlue(features=None,input_dim=128,add_scale_ori=kind=='sift',flash=False,depth_confidence=-1,width_confidence=-1)
        state=torch.load(ROOT/'models/external'/f'{kind}_lightglue.pth',map_location='cpu',weights_only=True)
        for i in range(model.conf.n_layers):state={k.replace(f'self_attn.{i}',f'transformers.{i}.self_attn').replace(f'cross_attn.{i}',f'transformers.{i}.cross_attn'):v for k,v in state.items()}
        if 'confidence_thresholds' not in state:state['confidence_thresholds']=model.confidence_thresholds
        model.load_state_dict(state,strict=True);MODELS[key]=model.eval().to(device)
    return MODELS[key]


def edge_logits(a,b,ia,ib,cancel=lambda:False):
    """Dot products only at candidate edges, bounded gather temporaries."""
    import torch
    pieces=[]
    for offset in range(0,len(ia),16384):
        check(cancel);ja=ia[offset:offset+16384];jb=ib[offset:offset+16384]
        pieces.append((a[:,ja]*b[:,jb]).sum(-1))
    return torch.cat(pieces,dim=-1)


def reduction_plan(indices,size):
    """Fixed-order padded rows avoid Metal's atomic scatter-add rounding drift."""
    import torch
    if indices.device.type!='mps':return None
    ids=indices.cpu().numpy();order=np.argsort(ids,kind='stable');counts=np.bincount(ids,minlength=size);starts=np.r_[0,np.cumsum(counts)];plan=[]
    for lo in range(0,size,32):
        n=counts[lo:lo+32];width=int(n.max(initial=0))
        if not width:continue
        offsets=np.arange(width)[None];valid=offsets<n[:,None];positions=np.minimum(starts[lo:lo+len(n),None]+offsets,len(order)-1)
        plan.append((lo,torch.as_tensor(order[positions],device=indices.device),torch.as_tensor(valid,device=indices.device)))
    return plan


def grouped_log_softmax(logits,indices,size,plan=None):
    import torch
    index=indices[None].expand(logits.shape[0],-1);maximum=logits.new_full((logits.shape[0],size),-float('inf'));maximum.scatter_reduce_(1,index,logits,reduce='amax',include_self=True)
    centered=logits-maximum.gather(1,index);sums=logits.new_zeros((logits.shape[0],size));values=centered.exp()
    if plan is None:sums.scatter_add_(1,index,values)
    else:
        for lo,edges,valid in plan:sums[:,lo:lo+len(edges)]=(values[:,edges]*valid).sum(-1)
    return centered-sums.gather(1,index).log()


def accumulate(weights,values,queries,targets,size,cancel,plan=None):
    import torch
    out=values.new_zeros((values.shape[0],size,values.shape[-1]))
    if plan is not None:
        for lo,edges,valid in plan:
            check(cancel);out[:,lo:lo+len(edges)]=(values[:,targets[edges]]*(weights[:,edges]*valid)[...,None]).sum(-2)
        return out
    for offset in range(0,len(queries),8192):
        check(cancel);q=queries[offset:offset+8192];t=targets[offset:offset+8192];v=values[:,t]*weights[:,offset:offset+8192,None]
        out.scatter_add_(1,q[None,:,None].expand_as(v),v)
    return out


def sparse_cross(block,x0,x1,ia,ib,cancel,plans=(None,None)):
    q0,q1=block.to_qk(x0),block.to_qk(x1);v0,v1=block.to_v(x0),block.to_v(x1)
    q0,q1,v0,v1=[v[0].unflatten(-1,(block.heads,-1)).transpose(0,1) for v in (q0,q1,v0,v1)]
    logits=edge_logits(q0*block.scale**.5,q1*block.scale**.5,ia,ib,cancel)
    m0=accumulate(grouped_log_softmax(logits,ia,x0.shape[1],plans[0]).exp(),v1,ia,ib,x0.shape[1],cancel,plans[0])
    m1=accumulate(grouped_log_softmax(logits,ib,x1.shape[1],plans[1]).exp(),v0,ib,ia,x1.shape[1],cancel,plans[1])
    import torch
    m0,m1=[block.to_out(v.transpose(0,1).flatten(-2)[None]) for v in (m0,m1)]
    return x0+block.ffn(torch.cat((x0,m0),-1)),x1+block.ffn(torch.cat((x1,m1),-1))


def scores(model,k0,k1,d0,d1,edges,image_size,cancel=lambda:False,scale_ori=None):
    import torch
    import torch.nn.functional as F
    from ..vendor.lightglue.lightglue import normalize_keypoints
    device=next(model.parameters()).device;ia=torch.as_tensor(edges[:,0],device=device);ib=torch.as_tensor(edges[:,1],device=device)
    plans=(reduction_plan(ia,len(k0)),reduction_plan(ib,len(k1)))
    keypoints=[normalize_keypoints(torch.as_tensor(k,device=device,dtype=torch.float32)[None],image_size) for k in (k0,k1)]
    if model.conf.add_scale_ori:
        if scale_ori is None:raise ValueError('SIFT scales and orientations are required.')
        keypoints=[torch.cat((k,torch.as_tensor(s,device=device,dtype=torch.float32)[None]),-1) for k,s in zip(keypoints,scale_ori)]
    x0,x1=[model.input_proj(torch.as_tensor(d,device=device)[None]) for d in (d0,d1)]
    e0,e1=[model.posenc(k) for k in keypoints]
    for layer in model.transformers:
        check(cancel);x0,x1=layer.self_attn(x0,e0),layer.self_attn(x1,e1);x0,x1=sparse_cross(layer.cross_attn,x0,x1,ia,ib,cancel,plans)
    assignment=model.log_assignment[-1];a,b=[assignment.final_proj(x)[0][None]/model.conf.descriptor_dim**.25 for x in (x0,x1)]
    logits=edge_logits(a,b,ia,ib,cancel);z0,z1=[F.logsigmoid(assignment.matchability(x)).flatten() for x in (x0,x1)]
    logscore=grouped_log_softmax(logits,ia,len(k0),plans[0])+grouped_log_softmax(logits,ib,len(k1),plans[1])+z0[ia]+z1[ib]
    return logscore.exp()[0].cpu().numpy()


@inference_thread
def match(points,desc,members,radius,minimum,threshold,compare,cpu,image_shape,cancel=lambda:False,progress=lambda *x:None,kind='xfeat',radii=None,gap=(0.,0.),axes=None):
    import torch
    from scipy.spatial import cKDTree
    jobs=[(np.flatnonzero(members[:,0]),np.flatnonzero(members[:,1]))] if compare else [(np.flatnonzero(members[:,i]),np.flatnonzero(members[:,i])) for i in range(members.shape[1])]
    parts=[];evaluated=0;model=None
    with LOCK,torch.inference_mode():
        for zone,(a,b) in enumerate(jobs):
            check(cancel)
            if not len(a) or not len(b):continue
            if max(len(a),len(b))>6000:raise ValueError('Learned matching supports at most 6000 points per zone. Reduce the point limit or select smaller zones.')
            xy0,xy1=points[a,:2],points[b,:2]
            from .auto_zones import compact_points
            search_xy0=compact_points(xy0,axes)
            search_xy1=xy1-np.asarray(gap) if compare else compact_points(xy1,axes)
            search_radius=radius if compare or radii is None else radii[zone]
            tree=cKDTree(search_xy1);edges=[];count=0
            for i,point in enumerate(search_xy0):
                if i%128==0:check(cancel)
                candidates=tree.query_ball_point(point,search_radius)
                j=np.asarray(candidates,np.int64);length=np.linalg.norm(search_xy1[j]-search_xy0[i],axis=1);j=j[(length>=minimum)&(length<=search_radius)&(b[j]!=a[i])]
                if len(j):edges.append(np.column_stack((np.full(len(j),i),j)));count+=len(j)
                if count>4_000_000:raise ValueError('More than four million learned candidates. Reduce spacing, zones or point limit.')
            if not edges:continue
            edges=np.concatenate(edges);evaluated+=len(edges)
            if model is None:model=lighter_model(device_for(cpu)) if kind=='xfeat' else lightglue_model(device_for(cpu),kind)
            progress(25+int(50*zone/max(1,len(jobs))),'Matching learned spatial candidates')
            so=[np.column_stack((points[ids,2],np.deg2rad(points[ids,3]))) for ids in (a,b)] if kind=='sift' else None
            confidence=scores(model,xy0,xy1,desc[a],desc[b],edges,(image_shape[1],image_shape[0]),cancel,so)
            # Mutual best edge, with stable first-index tie handling.
            order=np.lexsort((edges[:,1],edges[:,0],-confidence));ranked=edges[order]
            _,i0=np.unique(ranked[:,0],return_index=True);_,i1=np.unique(ranked[:,1],return_index=True);selected=np.intersect1d(i0,i1);indices=order[selected];indices=indices[confidence[indices]>=1-threshold]
            ids=np.column_stack((a[edges[indices,0]],b[edges[indices,1]]));length=np.linalg.norm(search_xy0[edges[indices,0]]-search_xy1[edges[indices,1]],axis=1)
            parts.append(np.column_stack((np.min(ids,axis=1),np.max(ids,axis=1),1-confidence[indices],length)))
    pairs=np.concatenate(parts) if parts else np.empty((0,4),np.float64)
    if len(pairs):
        order=np.argsort(pairs[:,2],kind='stable');_,unique=np.unique(pairs[order,:2],axis=0,return_index=True);pairs=pairs[order[unique]]
    return pairs,evaluated
