"""Exploratory ELA biome analysis; scores are not calibrated probabilities.

Compare raw linear residual profiles to spatially separate patches with similar
image content. JPEG encoding always uses original pixel dimensions. Cache one
feature set; threshold, support and display changes never re-encode the image.
"""
from time import perf_counter
import hashlib
import cv2 as cv
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import median_filter
from .jpeg import compress_jpg
from .jpeg_quality import read_quantization, table_estimate
from .utility import create_lut

# Shared preset for this panel and the complete automatic analysis.
DEFAULT_BLOCK = 32
DEFAULT_THRESHOLD = 2.0
DEFAULT_MINIMUM_CELLS = 3

PROFILE_NAMES = ['luminance', 'chroma_red', 'chroma_blue', 'grain_1px', 'grain_2px']


class Cancelled(Exception):
    pass


def check(cancel):
    if cancel():
        raise Cancelled()


def reference_quality(filename, requested=0):
    if not 0 <= requested <= 100:
        raise ValueError("La qualité doit être comprise entre 0 (auto) et 100.")
    if requested:
        return int(requested), 'manual', None
    if filename:
        try:
            tables = read_quantization(filename)
            estimate = table_estimate(*tables) if tables else None
            if estimate is not None:
                _, deviation, distance = estimate
                # The closest conventional table, without the old heuristic
                # subtracting table deviation from the nominal quality number.
                return max(1, int(np.argmin(distance))), 'jpeg_tables', float(deviation)
        except (OSError, ValueError, AttributeError, TypeError):
            pass
    return 75, 'fallback_no_usable_jpeg_tables', None


def qualities(q):
    return [q,q+5,q+10] if q<=5 else [q-10,q-5,q] if q>=96 else [q-5,q,q+5]


def tiles(array, block, rows, cols):
    return array[:rows*block, :cols*block].reshape(rows, block, cols, block, *array.shape[2:]).swapaxes(1, 2).reshape(rows*cols, block, block, *array.shape[2:])


def describe(image, compressed, block, cancel, *, background=False):
    """Bounded stripe temporaries, complete cells only; no padded evidence."""
    h, w = image.shape[:2]; rows, cols = h//block, w//block
    content, profiles, usable, middle = [], [], [], []
    for row in range(rows):
        check(cancel)
        # Halo makes spatial derivatives independent of artificial stripe edges.
        y = row*block; lo=max(0,y-8); hi=min(h,y+block+8)
        orig = image[lo:hi].astype(np.float32)
        ycc = cv.cvtColor(orig, cv.COLOR_BGR2YCrCb)
        gray = ycc[:,:,0]
        coarse = cv.GaussianBlur(gray,(0,0),1.5)
        dx=cv.Sobel(coarse,cv.CV_32F,1,0,ksize=3)/8
        dy=cv.Sobel(coarse,cv.CV_32F,0,1,ksize=3)/8
        grad=cv.magnitude(dx,dy)
        lap=np.abs(cv.Laplacian(coarse,cv.CV_32F))
        chroma=cv.GaussianBlur(cv.magnitude(ycc[:,:,1]-.5,ycc[:,:,2]-.5),(0,0),1.5)
        cg=cv.magnitude(cv.Sobel(chroma,cv.CV_32F,1,0)/8,cv.Sobel(chroma,cv.CV_32F,0,1)/8)
        sl=slice(y-lo,y-lo+block)
        g=tiles(gray[sl],block,1,cols); edge=tiles(grad[sl],block,1,cols)
        c=tiles(chroma[sl],block,1,cols)
        axes=(1,2)
        content.append(np.stack([g.mean(axes),np.log1p(tiles(coarse[sl],block,1,cols).std(axes)),np.log1p(edge.mean(axes)),
            np.log1p(tiles(lap[sl],block,1,cols).mean(axes)),c.mean(axes),np.log1p(tiles(cg[sl],block,1,cols).mean(axes))],axis=1))
        # Ignore the strongest 30% of edges when measuring residual energy.
        quiet=edge<=np.quantile(edge,.7,axis=axes)[:,None,None]
        valid=(g>3)&(g<252)&quiet
        count=valid.sum(axes);usable.append(count>=block*block*.35)
        # Float conversion precedes subtraction: negative differences survive.
        difference=orig-compressed[lo:hi].astype(np.float32)
        if background:
            from .ela_background import mid_profile
            levels=cv.cvtColor(np.abs(difference),cv.COLOR_BGR2GRAY)
            middle.append(mid_profile(tiles(levels[sl],block,1,cols),valid))
        residual=cv.cvtColor(difference,cv.COLOR_BGR2YCrCb)
        residual[:,:,1:]-=.5  # OpenCV float chroma offset, not a signal.
        r=tiles(residual[sl],block,1,cols)
        energy=np.sqrt((r*r*valid[:,:,:,None]).sum(axes)/np.maximum(count,1)[:,None])
        luminance=np.abs(r[:,:,:,0]);corr=[]
        for lag in (1,2):
            aa=[];bb=[];vv=[]
            for a,b,v in [(luminance[:,:,lag:],luminance[:,:,:-lag],valid[:,:,lag:]&valid[:,:,:-lag]),
                          (luminance[:,lag:,:],luminance[:,:-lag,:],valid[:,lag:,:]&valid[:,:-lag,:])]:
                aa.append(a.reshape(cols,-1));bb.append(b.reshape(cols,-1));vv.append(v.reshape(cols,-1))
            a=np.concatenate(aa,1);b=np.concatenate(bb,1);v=np.concatenate(vv,1)
            n=np.maximum(v.sum(1),1);ma=(a*v).sum(1)/n;mb=(b*v).sum(1)/n
            a=a-ma[:,None];b=b-mb[:,None]
            covariance=(a*b*v).sum(1);denom=np.sqrt((a*a*v).sum(1)*(b*b*v).sum(1))
            corr.append(np.divide(covariance,denom,out=np.zeros_like(covariance),where=denom>1e-6))
        profiles.append(np.column_stack([np.log1p(energy),*corr]))
    result=(np.concatenate(content).astype(np.float32),np.concatenate(profiles).astype(np.float32),np.concatenate(usable))
    return (*result,np.concatenate(middle)) if background else result


def peer_scores(content, profiles, usable, shape, cancel):
    """Robust, leave-neighbourhood-out reference; unsupported cells stay unknown."""
    n=len(content); nq=profiles.shape[1]
    scores=np.zeros((n,nq),np.float32); signed=np.zeros((n,nq,5),np.float32)
    counts=np.zeros(n,np.int32)
    ids=np.flatnonzero(usable)
    if len(ids)<17:return scores,signed,counts
    center=np.median(content[ids],axis=0)
    scale=np.maximum(1.4826*np.median(np.abs(content[ids]-center),axis=0),[16,.25,.25,.25,8,.25])
    normalized=(content-center)/scale/np.sqrt(content.shape[1])
    tree=cKDTree(normalized[ids]);k=min(128,len(ids))
    yy,xx=np.indices(shape);coords=np.column_stack([yy.ravel(),xx.ravel()])
    floor=np.maximum(.1*1.4826*np.median(np.abs(profiles[ids]-np.median(profiles[ids],axis=0)),axis=0),[.08,.08,.08,.06,.06])
    for lo in range(0,n,128):
        check(cancel);end=min(n,lo+128)
        distances,ix=tree.query(normalized[lo:end],k=k,workers=1)
        neighbours=ids[ix]
        eligible=(distances<=1.25)&(np.max(np.abs(coords[neighbours]-coords[lo:end,None]),axis=2)>max(2, min(shape)//5))
        eligible&=np.cumsum(eligible,axis=1)<=64
        count=eligible.sum(1);counts[lo:end]=np.where(usable[lo:end],count,0)
        supported=(count>=16)&usable[lo:end]
        if not supported.any():continue
        p=profiles[neighbours[supported]].copy()
        p[~eligible[supported]]=np.nan
        median=np.nanmedian(p,axis=1)
        mad=1.4826*np.nanmedian(np.abs(p-median[:,None]),axis=1)
        z=(profiles[lo:end][supported]-median)/np.maximum(mad,floor)
        indices=np.flatnonzero(supported)+lo;signed[indices]=z
        # Require a profile discrepancy, not one unstable descriptor alone.
        largest=np.sort(np.abs(z),axis=2)[:,:,-2:]
        scores[indices]=np.sqrt(np.mean(largest*largest,axis=2))
    return scores,signed,counts


def coherent_scores(signed, supported):
    """Single-descriptor evidence corroborated in quality, space and direction.

    Signed medians retain discrepancies persistent at >=2 quality probes and
    >=5 cells of a 3x3 neighbourhood. Unknown/outside cells contribute zero,
    never fabricated border copies. Cap by the centre's own evidence: this is
    not mask dilation or an independent-sample significance multiplier.
    """
    persistent = np.median(signed, axis=2)
    result = np.zeros(supported.shape, np.float32)
    for descriptor in range(persistent.shape[2]):
        own = np.where(supported, persistent[:, :, descriptor], 0)
        nearby = median_filter(own, size=3, mode='constant', cval=0)
        corroborated = np.where(own * nearby > 0,
                                np.minimum(np.abs(own), np.abs(nearby)), 0)
        np.maximum(result, corroborated, out=result)
    return result


class ElaBiomeEngine:
    def __init__(self,image,filename=None):
        if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
            raise ValueError('ELA biomes require an 8-bit BGR image.')
        self.image=image;self.filename=filename;self.cached=None;self.encodes=0;self.ghost_engine=None
        self.background_cached=None
        self.energy_cached=None
        self.energy_auto_cached=None
        from .cache_budget import ArrayCache
        self.energy_maps=ArrayCache(192)

    def prepare(self,block=DEFAULT_BLOCK,quality=0,cancel=lambda:False,*,ghost=False,all_grids=False,background=False,energy=False,energy_quantiles=(.1,.9),energy_auto=False,energy_auto_profile='sensitive'):
        request_start=perf_counter()
        if energy:
            from .ela_energy import prepare_energy,LOG_SCALE,ENERGY_FLOOR,WINDOW
            base=self.prepare(block,quality,cancel,ghost=ghost,all_grids=all_grids,background=background)
            if energy_auto_profile not in ('sensitive','conservative'):raise ValueError('Unknown automatic energy profile')
            automatic=None
            if energy_auto:
                from .ela_energy_auto import estimate,deviations,CONSERVATIVE_TAIL_FACTOR,CONSERVATIVE_SHADOW_GAIN,CONSERVATIVE_HIGHLIGHT_GAIN
                if self.energy_auto_cached is None or self.energy_auto_cached[0]!=base['key']:
                    reference=prepare_energy(self.image,base['energy_planes'],cancel)
                    automatic=estimate({**base,**reference},cancel)
                    proposal=prepare_energy(self.image,base['energy_planes'],cancel,automatic['quantiles'])
                    automatic={**automatic,'thresholds':deviations(proposal,cancel)}
                    check(cancel);self.energy_auto_cached=(base['key'],automatic)
                    self.energy_cached=((base['key'],tuple(automatic['quantiles'])),proposal)
                    self.energy_maps.put(self.energy_cached[0],proposal)
                automatic=self.energy_auto_cached[1]
                energy_quantiles=automatic['quantiles']
                if energy_auto_profile=='conservative':
                    energy_quantiles=[round(energy_quantiles[0]*CONSERVATIVE_TAIL_FACTOR,3),round(1-(1-energy_quantiles[1])*CONSERVATIVE_TAIL_FACTOR,3)]
            energy_key=(base['key'],tuple(energy_quantiles))
            cached_energy=self.energy_maps.get(energy_key)
            if cached_energy is not None:self.energy_cached=(energy_key,cached_energy)
            if self.energy_cached is None or self.energy_cached[0]!=energy_key:
                values=prepare_energy(self.image,base['energy_planes'],cancel,energy_quantiles)
                check(cancel);self.energy_cached=(energy_key,values)
            self.energy_maps.put(energy_key,self.energy_cached[1])
            if automatic is not None:
                thresholds=automatic['thresholds']
                if energy_auto_profile=='conservative':
                    thresholds=[round(min(20.,x*g),1) for x,g in zip(thresholds,(CONSERVATIVE_SHADOW_GAIN,CONSERVATIVE_HIGHLIGHT_GAIN))]
                automatic={**automatic,'thresholds':thresholds,
                           'quantiles':list(energy_quantiles),'profile':energy_auto_profile,
                           'calibration':dict(tail_factor=CONSERVATIVE_TAIL_FACTOR,shadow_gain=CONSERVATIVE_SHADOW_GAIN,highlight_gain=CONSERVATIVE_HIGHLIGHT_GAIN) if energy_auto_profile=='conservative' else None,
                           'method':'quantile_tail_knees_one_sided_otsu_v1'}
            return {**base,**self.energy_cached[1],
                'metadata':{**base['metadata'],'version':5,'seconds':perf_counter()-request_start,
                    'energy':dict(enabled=True,kind='descriptive_energy_contrast',probability=False,
                        reference='per_detected_panel_trimmed_mean',quantiles=list(energy_quantiles),window=WINDOW,
                        panels=self.energy_cached[1]['energy_summary'],automatic=automatic,
                        log_scale=LOG_SCALE,energy_floor=ENERGY_FLOOR,quality_confirmation='median',
                        segmentation='pixel_hysteresis_half_threshold',grouping='reference_panel_and_energy_class',minimum_strong_pixels=base['metadata']['block']**2)}}
        base=self._prepare_profiles(block,quality,cancel)
        result=self._with_ghost(base,all_grids,cancel,request_start) if ghost else base
        if not background:return result
        from .ela_background import score_background,TRIM,MAXIMUM_CONTENT_DISTANCE,LOG_FLOOR
        if self.background_cached is None or self.background_cached[0]!=base['key']:
            values=score_background(base['content'],base['background_profiles'],base['supported'],cancel)
            self.background_cached=(base['key'],values)
        values=self.background_cached[1]
        return {**result,**values,'pre_background_score':result['score'],
                'score':np.maximum(result['score'],values['background_score']),
                'metadata':{**result['metadata'],'version':4,'seconds':perf_counter()-request_start,
                    'score_combination':'maximum_existing_and_background',
                    'background':dict(enabled=True,trim_quantiles=list(TRIM),
                        descriptors=['trimmed_mean','quartile_25','quartile_75'],
                        residual='luminance_of_absolute_rgb_difference_before_display_clipping',
                        direction='deficit_only',maximum_content_distance=MAXIMUM_CONTENT_DISTANCE,
                        log_floor=LOG_FLOOR,minimum_peers=16,maximum_peers=64,
                        confirmation='median_descriptors_then_median_qualities',probability=False)}}

    def _with_ghost(self,base,all_grids,cancel,request_start):
        from .ela_ghosts import GhostBiomeEngine
        if self.ghost_engine is None:self.ghost_engine=GhostBiomeEngine(self.image)
        start=perf_counter();values=self.ghost_engine.compute(base,all_grids,cancel)
        return {**base,**values,'ela_score':base['score'],
                'score':np.maximum(base['score'],values['ghost_score']),
                'metadata':{**base['metadata'],'version':3,'seconds':perf_counter()-request_start,'ghost':dict(
                    enabled=True,qualities=list(range(30,101)),grids=64 if all_grids else 1,
                    normalization='per_16px_cell_across_qualities',comparison='distant_content_matched_curve_deficit',
                    quality_median=3,normalized_floor=.10,maximum_content_distance=.5,minimum_peers=16,
                    seconds=perf_counter()-start,quality_is='maximum_deficit_probe_not_original_quality',probability=False)}}

    def _prepare_profiles(self,block=DEFAULT_BLOCK,quality=0,cancel=lambda:False):
        check(cancel)
        q,origin,deviation=reference_quality(self.filename,quality)
        requested_block=int(block)
        block=max(16,int(block));block=((block+7)//8)*8
        h,w=self.image.shape[:2]
        while (h//block)*(w//block)>16384:block+=8
        key=(block,q,origin,deviation)
        if self.cached is not None and self.cached['key']==key:return self.cached
        rows,cols=h//block,w//block
        if rows*cols<25:raise ValueError('Image trop petite : choisir des cellules plus petites (au moins 25 cellules complètes).')
        start=perf_counter();profiles=[];background_profiles=[];energy_planes=[];levels=qualities(q);preview=None
        for level in levels:
            check(cancel);compressed=compress_jpg(self.image,level);self.encodes+=1
            from .ela_energy import describe_energy
            energy_planes.append(describe_energy(self.image,compressed))
            if level==q:
                preview=cv.convertScaleAbs(cv.absdiff(self.image,compressed),alpha=50)
                cv.LUT(preview,create_lut(25,25),dst=preview)
            content,profile,usable,middle=describe(self.image,compressed,block,cancel,background=True)
            profiles.append(profile)
            background_profiles.append(middle)
        del compressed
        values=np.stack(profiles,axis=1)
        scores,signed,counts=peer_scores(content,values,usable,(rows,cols),cancel)
        # Median demands persistence in at least 2 of the 3 normal quality probes.
        legacy_score=np.median(scores,axis=1).reshape(rows,cols)
        signed=signed.reshape(rows,cols,len(levels),5)
        supported=(counts>=16).reshape(rows,cols)
        coherent=coherent_scores(signed,supported)
        score=np.maximum(legacy_score,coherent)
        result=dict(key=key,score=score,legacy_score=legacy_score,coherent_score=coherent,quality_scores=scores.reshape(rows,cols,len(levels)),
                    signed_scores=signed,
                    profiles=values.reshape(rows,cols,len(levels),5),
                    background_profiles=np.stack(background_profiles,axis=1).reshape(rows,cols,len(levels),3),
                    energy_planes=np.stack(energy_planes),
                    content=content.reshape(rows,cols,6),peer_count=counts.reshape(rows,cols),
                    supported=supported,ela=preview,
                    metadata=dict(method='ela_biomes',version=2,opencv=cv.__version__,
                        image_pixels_sha256=hashlib.sha256(np.ascontiguousarray(self.image)).hexdigest(),
                        quality=q,quality_origin=origin,
                        table_deviation=deviation,qualities=levels,block=block,requested_block=requested_block,
                        image_shape=[h,w],valid_shape=[rows*block,cols*block],linear=True,scale=50,contrast=20,
                        descriptors=PROFILE_NAMES,seconds=perf_counter()-start,experimental=True,
                        score_kind='uncalibrated_robust_profile_distance',
                        coherence=dict(neighbourhood=3,minimum_signed_cells=5,
                            qualities='signed_median',centre_cap=True,unknown_value=0,
                            requires_legacy_biome=True),
                        score_combination='maximum_legacy_and_coherent'))
        check(cancel);self.cached=result;return result


def segment(base,threshold=DEFAULT_THRESHOLD,minimum=DEFAULT_MINIMUM_CELLS,*,energy_thresholds=None):
    """No forced partition: connected, supported anomalies only; zero is possible."""
    if not np.isfinite(threshold) or threshold <= 0 or minimum < 1:
        raise ValueError("Seuil et support doivent être positifs.")
    mask=(base['score']>=threshold)&base['supported']
    count,labels,stats,_=cv.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
    # Corroboration may complete a retained legacy biome, but cannot create an
    # extra unseeded region. This avoids promoting weak natural-content islands.
    seeded = None
    if 'legacy_score' in base:
        seed_mask=(base['legacy_score']>=threshold)&base['supported']
        _,seed_labels,seed_stats,_=cv.connectedComponentsWithStats(seed_mask.astype(np.uint8),connectivity=8)
        keep=np.flatnonzero(seed_stats[:,cv.CC_STAT_AREA]>=minimum)
        keep=keep[keep!=0]
        seeded=set(np.unique(labels[np.isin(seed_labels,keep)]).tolist())
        if 'ghost_score' in base:
            gm=(base['ghost_score']>=threshold)&base['ghost_supported']&base['supported']
            _,gl,gs,_=cv.connectedComponentsWithStats(gm.astype(np.uint8),connectivity=8)
            keep=np.flatnonzero(gs[:,cv.CC_STAT_AREA]>=minimum);keep=keep[keep!=0]
            seeded.update(np.unique(labels[np.isin(gl,keep)]).tolist())
        if 'background_score' in base:
            bm=(base['background_score']>=threshold)&base['background_supported']&base['supported']
            _,bl,bs,_=cv.connectedComponentsWithStats(bm.astype(np.uint8),connectivity=8)
            keep=np.flatnonzero(bs[:,cv.CC_STAT_AREA]>=minimum);keep=keep[keep!=0]
            seeded.update(np.unique(labels[np.isin(bl,keep)]).tolist())
    out=np.zeros_like(labels);regions=[];block=base['metadata']['block']
    for i in range(1,count):
        if stats[i,cv.CC_STAT_AREA]<minimum or (seeded is not None and i not in seeded):continue
        selected=labels==i;rid=len(regions)+1;out[selected]=rid
        x,y,w,h,area=map(int,stats[i]);profile=np.median(base['signed_scores'][selected],axis=(0,1))
        regions.append(dict(id=rid,cells=area,pixels=area*block*block,bbox=[x*block,y*block,w*block,h*block],
            score=float(np.median(base['score'][selected])),dominant_descriptor=PROFILE_NAMES[int(np.argmax(np.abs(profile)))],
            signed_profile=profile.tolist(),
            sources=(["ELA"] if np.any(base.get('ela_score',base['score'])[selected]>=threshold) else [])+
                    (["JPEG Ghosts"] if 'ghost_score' in base and np.any(base['ghost_score'][selected]>=threshold) else [])+
                    (["ELA background"] if 'background_score' in base and np.any(base['background_score'][selected]>=threshold) else [])))
    extra={}
    if 'energy_scope' in base:
        from .ela_energy import segment_energy
        pixel_labels,pixel_regions=segment_energy(base,threshold,minimum,len(regions),energy_thresholds)
        extra['energy_labels']=pixel_labels
        regions.extend(pixel_regions)
    return {**base,**extra,'labels':out,'metadata':{**base['metadata'],'threshold':float(threshold),'minimum_cells':int(minimum),'energy_thresholds':dict(zip(('low','high'),map(float,energy_thresholds if energy_thresholds is not None else (threshold,threshold)))),'regions':regions}}
