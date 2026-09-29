"""ARM64 dense IPOL descriptors + bounded, deterministic PatchMatch.

The nearest-neighbor field is approximate; masks/radii are hard constraints.
No resizing. Display sampling is separate from the full field computation.
"""
import ctypes as ct
from pathlib import Path
import cv2 as cv
import numpy as np
from .cloning import check
from .jpeg_curve import Cancelled
from .interactive import ArrayCache

METHODS=('PatchMatch Zernike','PatchMatch SIFT')
CALLBACK=ct.CFUNCTYPE(ct.c_int)
_LIB=None


def library():
    global _LIB
    if _LIB is None:
        path=Path(__file__).resolve().parents[4]/'native/runtime/libsherloq_patchmatch.dylib'
        lib=ct.CDLL(str(path));floatp=ct.POINTER(ct.c_float);intp=ct.POINTER(ct.c_int)
        lib.sherloq_dense_features.argtypes=[floatp,ct.c_int,ct.c_int,ct.c_int,ct.c_int,ct.c_int,floatp,floatp,ct.c_char_p]
        lib.sherloq_dense_features.restype=ct.c_int
        lib.sherloq_patchmatch.argtypes=[floatp,floatp,ct.POINTER(ct.c_ubyte),ct.c_int,ct.c_int,ct.c_int,ct.c_int,ct.c_float,ct.c_float,ct.c_int,ct.c_uint32,intp,floatp,ct.POINTER(ct.c_uint64),CALLBACK,ct.c_char_p]
        lib.sherloq_patchmatch.restype=ct.c_int
        lib.sherloq_patchmatch_gap.argtypes=lib.sherloq_patchmatch.argtypes+[ct.c_float,ct.c_float]
        lib.sherloq_patchmatch_gap.restype=ct.c_int
        lib.sherloq_patchmatch_metric.argtypes=lib.sherloq_patchmatch_gap.argtypes+[floatp,floatp]
        lib.sherloq_patchmatch_metric.restype=ct.c_int;_LIB=lib
    return _LIB


def ptr(a,ctype=ct.c_float):return a.ctypes.data_as(ct.POINTER(ctype))


def features(image,method,patch=8,flip=False):
    if method==1 and patch<3:raise ValueError('Dense SIFT requires a patch size of at least 3.')
    h,w=image.shape[:2];border=3*patch if method==1 else 0
    if min(h,w)<=3*patch:raise ValueError(f'Dense descriptors need dimensions above {3*patch} pixels.')
    dims=128 if method else 12;shape=(h-border,w-border,dims)
    # Includes bridge temporaries and a margin; refuse rather than swap blindly.
    estimated=np.prod(shape)*4*(4 if flip else 3)+h*w*32
    if estimated>16*1024**3:raise ValueError('Dense descriptor working set exceeds 16 GiB. Select smaller regions or use Zernike.')
    gray=image.astype(np.float32).sum(axis=2)*np.float32(1/np.sqrt(np.float32(3)))
    a=np.empty(shape,np.float32);b=np.empty_like(a) if flip else None;error=ct.create_string_buffer(1024)
    code=library().sherloq_dense_features(ptr(gray),w,h,method,patch,int(flip),ptr(a),ptr(b) if b is not None else None,error)
    if code:raise ValueError(error.value.decode())
    for field in (a,b):
        if field is not None:field/=np.maximum(np.linalg.norm(field,axis=2,keepdims=True),1e-12)
    return a,b if b is not None else a,1.5*patch if method else 0.


def field(first,second,mask,radius,minimum,iterations=8,compare=False,seed=729,cancel=lambda:False,gap=(0.,0.),axes=None):
    check(cancel);h,w,d=first.shape
    if second.shape!=first.shape or mask.shape!=(h,w):raise ValueError('Dense field shape mismatch.')
    mask=np.ascontiguousarray(mask,np.uint8);matches=np.empty((h,w),np.int32);distances=np.empty((h,w),np.float32)
    comparisons=ct.c_uint64();error=ct.create_string_buffer(1024);callback=CALLBACK(lambda:int(cancel()))
    if axes is not None:axes=tuple(np.ascontiguousarray(a,np.float32) for a in axes)
    status=library().sherloq_patchmatch_metric(ptr(first),ptr(second),ptr(mask,ct.c_ubyte),w,h,d,int(compare),minimum,radius,iterations,seed,ptr(matches,ct.c_int),ptr(distances),ct.byref(comparisons),callback,error,*gap,ptr(axes[0]) if axes is not None else None,ptr(axes[1]) if axes is not None else None)
    if status==1:raise Cancelled()
    if status:raise ValueError(error.value.decode())
    check(cancel);return matches,distances,comparisons.value


def coherent_mask(targets,squared,threshold,error_threshold,radius,minimum):
    """Full-field local affine residual before any display sampling.

    Closed-form least squares on a symmetric disk. Only complete valid disks
    contribute, so invalid/out-of-zone matches cannot fabricate support.
    """
    valid=(targets>=0)&(squared<=threshold*threshold)
    yy,xx=np.mgrid[-radius:radius+1,-radius:radius+1]
    kernel=((xx*xx+yy*yy)<=radius*radius).astype(np.float64)
    count=kernel.sum();moment=(kernel*xx*xx).sum();h,w=targets.shape
    complete=cv.filter2D(valid.astype(np.float64),-1,kernel,borderType=cv.BORDER_CONSTANT)>count-.5
    residual=np.zeros((h,w),np.float64)
    for axis in (0,1):
        if axis==0:delta=(targets%w-np.arange(w)[None,:]).astype(np.float64)
        else:delta=(targets//w-np.arange(h)[:,None]).astype(np.float64)
        delta[~valid]=0
        total=cv.filter2D(delta,-1,kernel,borderType=cv.BORDER_CONSTANT)
        residual+=cv.filter2D(delta*delta,-1,kernel,borderType=cv.BORDER_CONSTANT)-total*total/count
        for coords in (xx,yy):
            linear=cv.filter2D(delta,-1,kernel*coords,borderType=cv.BORDER_CONSTANT)
            residual-=linear*linear/moment
    error=np.sqrt(np.maximum(residual,0)/count)
    selected=complete&(error<=error_threshold)
    n,labels,stats,_=cv.connectedComponentsWithStats(selected.astype(np.uint8),connectivity=4)
    sizes=stats[:,cv.CC_STAT_AREA];keep=sizes>=minimum;keep[0]=False
    selected=keep[labels]
    return selected,error.astype(np.float32)


class DenseCopyEngine:
    def __init__(self,image):
        self.image=image;self.descriptors=ArrayCache(2048);self.fields=ArrayCache(512);self.consistency=ArrayCache(256);self.counts=dict(detections=0,matchings=0)

    def analyze(self,algorithm,limit,radius,minimum,threshold,regions,compare,options,cancel,progress,geometry=None,radii=None,gap=(0.,0.),excluded=(),guides=(),workers=1,backend="cpu",quarter_turn=False,target_patch=None):
        if workers>1:
            from .dense_parallel import parallel_fields
            return parallel_fields(self,algorithm,limit,radius,minimum,threshold,regions,compare,options,cancel,progress,geometry,radii,gap,excluded,guides,workers,backend,quarter_turn,target_patch)
        from .cloning2 import polygon_mask
        from .auto_zones import compact_axes,compact_points
        axes=compact_axes(self.image.shape,guides)
        method=METHODS.index(algorithm);patch,iterations,flip,texture=options
        target_patch=patch if target_patch is None else int(target_patch)
        if target_patch!=patch and (method!=1 or not 3<=target_patch<=32 or (target_patch-patch)%2):
            raise ValueError('Paired dense SIFT bins need equal parity and sizes from 3 to 32.')
        support_patch=max(patch,target_patch)
        if compare and len(regions)!=2:raise ValueError('Compare requires two zones.')
        jobs=[regions] if compare or not regions else [(region,) for region in regions]
        all_pairs=[];all_points=[];all_members=[];full_count=0;consistent_count=0;comparisons=0;maps=[];offset=0
        for job_index,job_regions in enumerate(jobs):
            job_radius=radius if compare or radii is None else radii[job_index]
            check(cancel);h,w=self.image.shape[:2]
            if job_regions:
                xy=np.concatenate([np.asarray(r) for r in job_regions]);x0,y0=np.maximum(0,np.floor(xy.min(axis=0))-3*support_patch).astype(int);x1,y1=np.minimum([w,h],np.ceil(xy.max(axis=0))+3*support_patch+1).astype(int)
            else:x0=y0=0;x1=w;y1=h
            crop=self.image[y0:y1,x0:x1];key=(method,patch,flip,x0,y0,x1,y1,backend,quarter_turn,target_patch)
            shift=1.5*support_patch if method else 0.;border=3*support_patch if method else 0
            dh,dw=crop.shape[0]-border,crop.shape[1]-border
            if min(crop.shape[:2])<=3*support_patch:raise ValueError(f'Dense descriptors need dimensions above {3*support_patch} pixels.')
            fkey=(key,job_regions,compare,job_radius,minimum,iterations,texture,gap,excluded,guides);value=self.fields.get(fkey)
            if value is None:
                local=tuple(tuple((x-x0,y-y0) for x,y in p) for p in job_regions)
                allowed=np.ones((dh,dw),np.uint8) if not local else np.zeros((dh,dw),np.uint8)
                index=np.rint(shift+np.arange(dh)).astype(int);columns=np.rint(shift+np.arange(dw)).astype(int)
                for j,p in enumerate(local):allowed |= (polygon_mask(crop.shape[:2],(p,))[np.ix_(index,columns)]>0).astype(np.uint8)*(1<<j)
                if excluded:
                    local_excluded=tuple(tuple((x-x0,y-y0) for x,y in p) for p in excluded)
                    exclusion_mask=polygon_mask(crop.shape[:2],local_excluded)
                    allowed[exclusion_mask[np.ix_(index,columns)]>0]=0
                if texture>0:
                    gray=crop.astype(np.float32).mean(axis=2);size=2*patch+1
                    deviation=np.sqrt(np.maximum(cv.boxFilter(gray*gray,-1,(size,size))-cv.boxFilter(gray,-1,(size,size))**2,0))
                    allowed[deviation[np.ix_(index,columns)]<texture]=0
                check(cancel)
                eligible=job_radius>=minimum and np.count_nonzero(allowed)>=2 and (not compare or (np.any(allowed&1) and np.any(allowed&2)))
                if eligible:
                    cached=self.descriptors.get(key)
                    if cached is None:
                        progress(5,'Computing dense descriptors')
                        if backend=='metal':
                            from .metal_dense import features as metal_zernike,features_sift
                            metal_features=features_sift if method else metal_zernike
                            compute=lambda size,mirror:metal_features(crop,size,mirror)
                        else:compute=lambda size,mirror:features(crop,method,size,mirror)
                        cached=compute(patch,flip if target_patch==patch else False)
                        if target_patch!=patch:
                            check(cancel)
                            first,_,source_shift=cached
                            _,second,target_shift=compute(target_patch,flip)
                            source_offset=int(shift-source_shift);target_offset=int(shift-target_shift)
                            cached=(np.ascontiguousarray(first[source_offset:source_offset+dh,source_offset:source_offset+dw]),
                                    np.ascontiguousarray(second[target_offset:target_offset+dh,target_offset:target_offset+dw]),shift)
                        if quarter_turn:
                            if method!=1:raise ValueError('Quarter-turn frame is only available for SIFT.')
                            from .sift_frames import quarter_turn_frame
                            first,second,center=cached
                            transformed=quarter_turn_frame(first,cancel)
                            cached=(transformed,transformed if first is second else quarter_turn_frame(second,cancel),center)
                        check(cancel);self.descriptors.put(key,cached);self.counts['detections']+=1
                    a,b,_=cached
                    if quarter_turn:
                        from .sift_frames import orientation_diversity
                        diverse=orientation_diversity(a,cancel)
                        if b is not a:diverse &= orientation_diversity(b,cancel)
                        allowed[~diverse]=0
                    progress(30,'Computing bounded dense correspondences');targets,squared,count=field(a,b,allowed,job_radius,minimum,iterations,compare,cancel=cancel,gap=gap,axes=None if axes is None else (np.interp(np.arange(dw)+shift+x0,np.arange(w),axes[0]),np.interp(np.arange(dh)+shift+y0,np.arange(h),axes[1])))
                else:
                    targets=np.full((dh,dw),-1,np.int32);squared=np.full((dh,dw),np.inf,np.float32);count=0
                value=(targets,squared,count,allowed);self.fields.put(fkey,value);self.counts['matchings']+=1
            targets,squared,count,allowed=value;comparisons+=count
            selection=(targets>=0)&(squared<=threshold*threshold)
            full_count+=int(selection.sum())
            consistency_error=None
            if geometry and geometry[0]!='None':
                ckey=(fkey,threshold,geometry[1],geometry[2],min(6,patch));coherence=self.consistency.get(ckey)
                if coherence is None:
                    progress(70,'Checking the full dense field');coherence=coherent_mask(targets,squared,threshold,geometry[1],min(6,patch),geometry[2]);check(cancel);self.consistency.put(ckey,coherence)
                selection,consistency_error=coherence
            rows=np.flatnonzero(selection)
            # Remove duplicate undirected links, keeping the smaller distance.
            target=targets.ravel()[rows];ordered=np.column_stack((np.minimum(rows,target),np.maximum(rows,target)))
            order=np.argsort(squared.ravel()[rows],kind='stable');_,unique=np.unique(ordered[order],axis=0,return_index=True);rows=rows[order[unique]]
            consistent_count+=len(rows);maps.append(dict(consistent_mask=selection,coherence_error=consistency_error,origin=(int(x0),int(y0)),shift=shift,targets=targets,distances_squared=squared,zone=job_index))
            # Explicit display cap distributed across all requested zones.
            cap=max(1,limit//len(jobs));rows=np.sort(rows)
            if len(rows)>cap:rows=rows[np.linspace(0,len(rows)-1,cap,dtype=int)]
            target=targets.ravel()[rows];ids=np.unique(np.r_[rows,target]);points=np.zeros((len(ids),7),np.float32)
            points[:,0]=ids%dw+shift+x0;points[:,1]=ids//dw+shift+y0;points[:,2]=patch*(3 if method else 2)
            pa=np.searchsorted(ids,rows);pb=np.searchsorted(ids,target);delta=compact_points(points[pa,:2],axes)-compact_points(points[pb,:2],axes)
            if compare and np.any(gap):
                forward=(allowed.ravel()[rows]&1)>0
                delta+=np.where(forward[:,None],np.asarray(gap),-np.asarray(gap))
            length=np.linalg.norm(delta,axis=1)
            all_pairs.append(np.column_stack((pa+offset,pb+offset,np.sqrt(squared.ravel()[rows]),length)));all_points.append(points)
            members=np.zeros((len(ids),max(1,len(regions))),bool)
            if compare:
                masks=allowed.ravel()[ids];members[:,0]=(masks&1)>0;members[:,1]=(masks&2)>0
            else:members[:,job_index]=True
            all_members.append(members);offset+=len(points)
        points=np.concatenate(all_points);pairs=np.concatenate(all_pairs)
        if len(jobs)>1 and len(pairs):
            # The enclosing zone overlaps its children. Do not double-count
            # identical undirected correspondences already found in a child.
            endpoints=points[pairs[:,:2].astype(int),:2]
            swap=(endpoints[:,0,0]>endpoints[:,1,0])|((endpoints[:,0,0]==endpoints[:,1,0])&(endpoints[:,0,1]>endpoints[:,1,1]))
            endpoints[swap]=endpoints[swap,::-1]
            _,unique=np.unique(endpoints.reshape(-1,4),axis=0,return_index=True)
            pairs=pairs[np.sort(unique)]
        return dict(points=points,pairs=pairs,members=np.concatenate(all_members),dense_count=full_count,dense_consistent_count=consistent_count,dense_maps=maps,candidate_comparisons=comparisons)
