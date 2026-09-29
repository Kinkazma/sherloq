"""Content-matched JPEG Ghost curve deficits; exploratory, not probabilities."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import median_filter
from .ela_biomes import check


def score_curves(content, curves, supported, cancel=lambda:False):
    """Compare normalized quality curves with distant, similar-content cells.

    A 3-quality median rejects a single quality spike. A fixed .10 normalized
    error floor prevents tiny numerical/profile differences from exploding.
    Positive scores indicate lower recompression loss than comparable cells.
    """
    shape=supported.shape;n=supported.size;nq=curves.shape[2]
    c=content.reshape(n,6);p=curves.reshape(n,nq);ids=np.flatnonzero(supported)
    score=np.zeros(n,np.float32);quality=np.zeros(n,np.int16);counts=np.zeros(n,np.int32)
    if len(ids)<17:return score.reshape(shape),quality.reshape(shape),counts.reshape(shape)
    center=np.median(c[ids],axis=0)
    scale=np.maximum(1.4826*np.median(np.abs(c[ids]-center),axis=0),[16,.25,.25,.25,8,.25])
    normalized=(c-center)/scale/np.sqrt(6)
    tree=cKDTree(normalized[ids]);coords=np.column_stack([a.ravel() for a in np.indices(shape)])
    for lo in range(0,n,64):
        check(cancel);hi=min(n,lo+64)
        distance,indices=tree.query(normalized[lo:hi],k=min(128,len(ids)),workers=1)
        peers=ids[indices]
        allowed=(distance<=.5)&(np.max(np.abs(coords[peers]-coords[lo:hi,None]),axis=2)>max(2,min(shape)//5))
        allowed&=np.cumsum(allowed,axis=1)<=64
        count=allowed.sum(1);good=(count>=16)&supported.ravel()[lo:hi];counts[lo:hi]=np.where(good,count,0)
        if not good.any():continue
        values=p[peers[good]].copy();values[~allowed[good]]=np.nan
        ref=np.nanmedian(values,axis=1);mad=1.4826*np.nanmedian(np.abs(values-ref[:,None]),axis=1)
        z=np.maximum(0,(ref-p[lo:hi][good])/np.maximum(mad,.10))
        stable=median_filter(z,size=(1,3),mode='constant',cval=0)
        # Endpoints are not a quality-local contrast; retain Q31..99 only.
        stable[:,[0,-1]]=0
        index=stable.argmax(1);rows=np.flatnonzero(good)+lo
        score[rows]=stable[np.arange(len(rows)),index];quality[rows]=index+30
    return score.reshape(shape),quality.reshape(shape),counts.reshape(shape)


def aggregate(maps, block, shape, dx=0, dy=0):
    """Area-exact registration of normalized 16px cells into original ELA cells.

    Fractional edges use the integral of a piecewise constant 16px map. Wrapped
    or incomplete last cells remain unsupported, never extrapolated evidence.
    """
    rows,cols=shape
    y=(np.arange(rows+1)*block+dy)/16
    x=(np.arange(cols+1)*block+dx)/16
    valid=(y[1:,None]<=maps.shape[0])&(x[None,1:]<=maps.shape[1])
    y=np.minimum(y,maps.shape[0]);x=np.minimum(x,maps.shape[1])
    iy=np.floor(y).astype(int);ix=np.floor(x).astype(int)
    fy=y-iy;fx=x-ix
    integ=np.pad(maps.cumsum(0).cumsum(1),((1,0),(1,0),(0,0)))
    jy=np.minimum(iy+1,maps.shape[0]);jx=np.minimum(ix+1,maps.shape[1])
    corners=(integ[iy[:,None],ix]*(1-fy[:,None,None])*(1-fx[None,:,None])+
             integ[jy[:,None],ix]*fy[:,None,None]*(1-fx[None,:,None])+
             integ[iy[:,None],jx]*(1-fy[:,None,None])*fx[None,:,None]+
             integ[jy[:,None],jx]*fy[:,None,None]*fx[None,:,None])
    return (corners[1:,1:]-corners[:-1,1:]-corners[1:,:-1]+corners[:-1,:-1])/(block/16)**2,valid


class GhostBiomeEngine:
    def __init__(self,image):
        from .ghost_maps import GhostEngine
        from .cache_budget import ArrayCache
        self.engine=GhostEngine(image,workers=4);self.cache=ArrayCache(64)

    def compute(self,base,all_grids=False,cancel=lambda:False):
        from .jpeg_curve import Cancelled as GhostCancelled
        from .ela_biomes import Cancelled
        block=base['metadata']['block'];key=(block,bool(all_grids))
        check(cancel);cached=self.cache.get(key)
        if cached is not None:return cached
        shape=base['score'].shape;best=np.zeros(shape,np.float32)
        quality=np.zeros(shape,np.int16);phase=np.zeros((*shape,2),np.uint8)
        counts=np.zeros(shape,np.int32);curves=np.zeros((*shape,71),np.float32)
        supported=np.zeros(shape,bool)
        try:
            for dy in range(8 if all_grids else 1):
                for dx in range(8 if all_grids else 1):
                    check(cancel)
                    maps=self.engine.maps((30,100,1,dx,dy,True,False),cancel)
                    current,valid=aggregate(maps,block,shape,dx,dy)
                    score,q,count=score_curves(base['content'],current,base['supported']&valid,cancel)
                    update=score>best
                    best[update]=score[update];quality[update]=q[update]
                    phase[update]=(dx,dy);counts[update]=count[update];curves[update]=current[update]
                    supported|=count>=16
                    if dx==dy==0:
                        curves[~update]=current[~update];counts[~update]=count[~update]
        except GhostCancelled:raise Cancelled() from None
        result=dict(ghost_score=best,ghost_quality=quality,ghost_phase=phase,
                    ghost_peer_count=counts,ghost_curves=curves,ghost_supported=supported)
        check(cancel);self.cache.put(key,result);return result
