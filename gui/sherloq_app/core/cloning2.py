"""Spatially bounded copy/move matching and paired-endpoint biome grouping.

New analysis contract, independent of the historical CloningEngine. No image
resampling. Candidate geometry is filtered BEFORE descriptor comparisons.
"""
import colorsys
import cv2 as cv
import numpy as np
from scipy.spatial import cKDTree
from .interactive import ArrayCache
from .cloning import pack_keypoints,check

POPCOUNT=np.unpackbits(np.arange(256,dtype=np.uint8)[:,None],axis=1).sum(axis=1)
ALGORITHMS=('SIFT','RootSIFT','AKAZE','BRISK','ORB','PatchMatch Zernike','PatchMatch SIFT','XFeat','XFeat + LighterGlue','ALIKED','ALIKED rotation','ALIKED + LightGlue','ALIKED rotation + LightGlue','SIFT + LightGlue')
COMBINED='PatchMatch Zernike + PatchMatch SIFT'
EXTENDED='Extended: '+COMBINED
SYMMETRIC=COMBINED+' + Mirror'
EXTENDED_SYMMETRIC='Extended: '+SYMMETRIC
ALGORITHMS=(*ALGORITHMS,COMBINED,'SIFT + G2NN + RANSAC',EXTENDED,SYMMETRIC,EXTENDED_SYMMETRIC)
MAX_PAIRS=100_000


def polygon_mask(shape,regions,excluded=()):
    if not regions and not excluded:return None
    mask=np.zeros(shape,np.uint8) if regions else np.full(shape,255,np.uint8)
    for poly in regions:
        if len(poly)>=3:cv.fillPoly(mask,[np.rint(poly).astype(np.int32)],255)
    for poly in excluded:
        if len(poly)>=3:cv.fillPoly(mask,[np.rint(poly).astype(np.int32)],0)
    return mask


def memberships(points,shape,regions):
    if not regions:return np.ones((len(points),1),bool)
    out=np.zeros((len(points),len(regions)),bool)
    xy=np.rint(points[:,:2]).astype(int);xy[:,0]=np.clip(xy[:,0],0,shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,shape[0]-1)
    # At most one full mask at a time, independent of the number of zones.
    for i,poly in enumerate(regions):
        m=polygon_mask(shape,(poly,));out[:,i]=m[xy[:,1],xy[:,0]]>0
    return out


def spatial_matches(points,desc,members,radius,minimum,threshold,binary,compare=False,cancel=lambda:False,progress=lambda *x:None,radii=None,gap=(0.,0.),axes=None):
    if compare and members.shape[1]!=2:raise ValueError('Compare requires exactly two zones.')
    from .auto_zones import compact_points
    n=len(points);coords=compact_points(points[:,:2],axes);tree=cKDTree(coords);parts=[];count=0;evaluated=0
    gap=np.asarray(gap)
    query_radius=radius+float(np.linalg.norm(gap))
    for i in range(n):
        check(cancel)
        js=np.asarray(tree.query_ball_point(coords[i],query_radius),np.int64)
        js=js[js>i]
        if not len(js):continue
        delta=coords[js]-coords[i]
        if compare and np.any(gap):
            direction=np.where(members[i,0]&members[js,1],1.,-1.)
            delta=delta-direction[:,None]*gap
        distances=np.linalg.norm(delta,axis=1)
        keep=(distances>=minimum)&(distances<=radius)
        if compare:keep &= (members[i,0]&members[js,1])|(members[i,1]&members[js,0])
        else:
            common=members[js]&members[i]
            if radii is not None:common&=distances[:,None]<=np.asarray(radii)
            keep &= np.any(common,axis=1)
        js=js[keep];distances=distances[keep];evaluated+=len(js)
        for start in range(0,len(js),512):
            check(cancel);batch=js[start:start+512]
            if binary:similarity=POPCOUNT[np.bitwise_xor(desc[batch],desc[i])].sum(axis=1)/(desc.shape[1]*8)
            else:similarity=np.linalg.norm(desc[batch]-desc[i],axis=1)
            good=similarity<=threshold
            if not good.any():continue
            targets=batch[good];s=similarity[good];d=distances[start:start+512][good]
            count+=len(targets)
            if count>MAX_PAIRS:raise ValueError('More than 100,000 correspondences. Tighten similarity, spacing or zones.')
            parts.append(np.column_stack((np.full(len(targets),i),targets,s,d)))
        if i%32==0:progress(15+int(60*(i+1)/max(n,1)),'Comparing nearby descriptors')
    pairs=np.concatenate(parts) if parts else np.empty((0,4),np.float64)
    return pairs,evaluated


def biomes(points,pairs,tolerance,cancel=lambda:False):
    """Exact connected components of paired endpoints, without enumerating edges.

    Flood only unvisited links; rebuild the spatial index as that set shrinks.
    This preserves the old distance test and sorted output, including reversed
    pairs, while avoiding quadratic Python unions in very dense components.
    """
    from collections import deque
    n=len(pairs)
    if not n:return ()
    xy=points[:,:2];ids=pairs[:,:2].astype(int);a=xy[ids[:,0]];b=xy[ids[:,1]]
    oriented=np.column_stack((a,b));remaining=np.ones(n,bool);left=n
    active=np.arange(n);both=np.vstack((oriented,np.column_stack((b,a))))
    tree=cKDTree(both);groups=[];steps=0
    for seed in range(n):
        if not remaining[seed]:continue
        queue=deque([seed]);remaining[seed]=False;left-=1;group=[seed]
        while queue and left:
            i=queue.popleft();steps+=1
            if steps%64==0:check(cancel)
            if left<len(active)//2:
                active=np.flatnonzero(remaining);both=np.vstack((oriented[active],np.column_stack((b[active],a[active]))));tree=cKDTree(both)
            candidates=np.asarray(tree.query_ball_point(oriented[i],tolerance*np.sqrt(2)),int)
            candidates=candidates[remaining[active[candidates%len(active)]]]
            if not len(candidates):continue
            d=both[candidates]-oriented[i]
            good=(np.linalg.norm(d[:,:2],axis=1)<=tolerance)&(np.linalg.norm(d[:,2:],axis=1)<=tolerance)
            found=np.unique(active[candidates[good]%len(active)])
            remaining[found]=False;left-=len(found);queue.extend(found.tolist());group.extend(found.tolist())
        groups.append(np.array(sorted(group),np.int64));check(cancel)
    return tuple(sorted(groups,key=lambda g:tuple(np.mean(np.minimum(a[g],b[g]),axis=0))+(-len(g),)))


def regional_biomes(points,pairs,members,tolerance,compare=False,cancel=lambda:False):
    """Keep independent Search zones separate even when their endpoints touch.

    A correspondence shared by overlapping zones belongs to the first common
    zone, avoiding duplicate evidence. Compare groups across its two zones.
    """
    if compare or members.shape[1]==1:
        return biomes(points,pairs,tolerance,cancel)
    if not len(pairs):return ()
    ids=pairs[:,:2].astype(int)
    common=members[ids[:,0]] & members[ids[:,1]]
    if not common.any(axis=1).all():raise ValueError('Cross-zone pair in Search.')
    owners=common.argmax(axis=1);groups=[]
    for zone in np.unique(owners):
        check(cancel);rows=np.flatnonzero(owners==zone)
        groups.extend(rows[g] for g in biomes(points,pairs[rows],tolerance,cancel))
    return tuple(groups)


def biome_sides(points,pairs,group):
    """Orient each undirected pair against a common reference before hulls."""
    ids=pairs[group,:2].astype(int);a=points[ids[:,0],:2].copy();b=points[ids[:,1],:2].copy()
    ref=a[0];other=b[0]
    swap=np.linalg.norm(a-ref,axis=1)+np.linalg.norm(b-other,axis=1)>np.linalg.norm(a-other,axis=1)+np.linalg.norm(b-ref,axis=1)
    a[swap],b[swap]=b[swap].copy(),a[swap].copy()
    return a,b


def palette(groups,pairs,threshold):
    colors=np.zeros((len(pairs),3),np.uint8);bases=[]
    for i,g in enumerate(groups):
        hue=(.76+i*.618033988749895)%1;band=min(1/12,.3/max(1,len(groups)))
        bases.append(tuple(round(c*255) for c in colorsys.hsv_to_rgb(hue,.75,.95)[::-1]))
        offsets=np.linspace(-band/2,band/2,len(g)) if len(g)>1 else [0.]
        for j,offset in zip(g,offsets):
            value=.4+.6*min(1,float(pairs[j,2])/max(threshold,1e-9))
            colors[j]=[round(c*255) for c in colorsys.hsv_to_rgb((hue+offset)%1,.8,value)[::-1]]
    return colors,tuple(bases)


def supported_selection(result,group,low,high,minimum,maximum_overlap=None):
    pairs=result['pairs'];selected=group[(pairs[group,3]>=low)&(pairs[group,3]<=high)]
    if len(selected)<minimum:return np.empty(0,np.int64)
    a,b=biome_sides(result['points'],pairs,selected)
    # Repeated orientations/scales at one keypoint centre must not manufacture
    # several independent spatial witnesses for the same biome.
    if min(len(np.unique(np.rint(a),axis=0)),len(np.unique(np.rint(b),axis=0)))<minimum:return np.empty(0,np.int64)
    if result.get('self_match_filter'):maximum_overlap=result['self_match_filter']['maximum_overlap']
    policy=result.get('mirror_policy')
    if maximum_overlap is None and policy and len(group) and group[0]>=policy['base_pair_count']:
        maximum_overlap=policy['maximum_overlap']
    if maximum_overlap is not None:
        from .copy_overlap import overlap
        if overlap(a,b)>=maximum_overlap:return np.empty(0,np.int64)
    return selected


def render(image,result,style,cancel=lambda:False):
    low,high,min_support,chosen,circles,lines,points,areas=style[:8]
    hidden=set(style[8]) if len(style)>8 else set()
    out=image.copy();matches=result['pairs'];keypoints=result['points'];colors=result['colors'];visible=[];legend=[]
    for index,group in enumerate(result['groups']):
        check(cancel)
        selected=supported_selection(result,group,low,high,min_support)
        if not len(selected):continue
        legend.append((index,len(selected),len(group)))
        if index in hidden or (chosen and index not in chosen):continue
        visible.append((index,len(selected)))
        color=result['bases'][index]
        if areas:
            sides=biome_sides(keypoints,matches,selected)
            for coords in sides:
                if len(np.unique(coords,axis=0))<3:continue
                hull=cv.convexHull(np.rint(coords).astype(np.int32))
                x,y,w,h=cv.boundingRect(hull);roi=out[y:y+h,x:x+w];overlay=roi.copy()
                local=hull-np.array([[[x,y]]]);cv.fillConvexPoly(overlay,local,color,cv.LINE_AA);cv.addWeighted(overlay,.28,roi,.72,0,dst=roi)
                cv.polylines(out,[hull],True,color,2,cv.LINE_AA)
        for z,j in enumerate(selected):
            if z%256==0:check(cancel)
            a,b=matches[j,:2].astype(int);pa,pb=tuple(np.rint(keypoints[a,:2]).astype(int)),tuple(np.rint(keypoints[b,:2]).astype(int));rgb=tuple(map(int,colors[j]))
            if lines:cv.line(out,pa,pb,rgb,1,cv.LINE_AA)
            if circles:
                cv.circle(out,pa,max(2,round(keypoints[a,2]/2)),rgb,1,cv.LINE_AA);cv.circle(out,pb,max(2,round(keypoints[b,2]/2)),rgb,1,cv.LINE_AA)
            if points:cv.circle(out,pa,2,rgb,-1,cv.LINE_AA);cv.circle(out,pb,2,rgb,-1,cv.LINE_AA)
    return out,visible,legend


class Cloning2Engine:
    def __init__(self,image):
        self.image=image;self.gray=None;self.features=ArrayCache(256);self.descriptors=ArrayCache(64);self.matches=ArrayCache(128);self.results=ArrayCache(128);self.grouped=ArrayCache(64)
        self.counts=dict(detections=0,matchings=0,groupings=0);self.dense=None

    def analyze(self,params,regions=(),compare=False,cancel=lambda:False,progress=lambda *x:None):
        algorithm,limit,radius,minimum,threshold,tolerance=params[:6]
        excluded=params[16] if len(params)>16 else ()
        geometry=tuple(params[6:9]) if len(params)>6 else ('None',3.,6)
        if len(geometry)!=3:raise ValueError('Invalid geometry settings.')
        if algorithm not in ALGORITHMS or not 100<=limit<=20000 or not 0<radius or not 0<=minimum<=radius or not 0<threshold<=1 or tolerance<=0:raise ValueError('Invalid analysis settings.')
        if compare and len(regions)!=2:raise ValueError('Compare requires exactly two zones.')
        from .auto_zones import distance_policy
        radius,radii,gap=distance_policy(params,regions,compare)
        from .auto_zones import compact_axes
        guides=params[17] if len(params)>17 and params[15] and not compare else ()
        axes=compact_axes(self.image.shape,guides)
        check(cancel)
        if algorithm in (SYMMETRIC,EXTENDED_SYMMETRIC):
            key=('symmetric',params,regions,compare);cached=self.results.get(key)
            if cached is not None:return cached
            from .copy_mirror import analyze
            result=analyze(self,params,regions,compare,cancel,progress)
            check(cancel);self.results.put(key,result);return result
        if algorithm==EXTENDED:return self.analyze_extended(params,regions,compare,cancel,progress)
        if algorithm==COMBINED:return self.analyze_combined(params,regions,compare,cancel,progress)
        if algorithm.startswith('PatchMatch '):return self.analyze_dense(params,regions,compare,cancel,progress)
        progress(0,'Detecting features')
        family='SIFT-G2NN' if algorithm=='SIFT + G2NN + RANSAC' else 'SIFT' if algorithm=='RootSIFT' else 'XFeat' if algorithm.startswith('XFeat') else algorithm
        if algorithm.startswith('ALIKED'):family='aliked-n16rot' if 'rotation' in algorithm else 'aliked-n16'
        if algorithm=='SIFT + LightGlue':family='sift-lg'
        learned=family in ('XFeat','aliked-n16','aliked-n16rot','sift-lg')
        cpu=bool(params[13]) if len(params)>13 else False
        # LighterGlue's measured ~1.81e-4 MPS score difference was explicitly
        # accepted by the user; retain the CPU reference as an option.
        backend='cpu'
        if learned:
            from .learned_copy import device_for
            backend=device_for(cpu)
        per_zone=family=='SIFT-G2NN' and len(params)>19 and bool(params[19])
        key=(family,limit,regions,cpu if learned else None,excluded,per_zone);features=self.features.get(key)
        if features is None:
            if family=='SIFT-G2NN':
                from .sift_g2nn import extract,_extract_global
                packed,desc,members,total=(extract if per_zone else _extract_global)(self.image,limit,regions,excluded,cancel)
            elif family=='XFeat':
                from .learned_copy import extract
                packed,desc,members,total=extract(self.image,limit,regions,cpu,cancel,excluded=excluded)
            elif learned:
                from .learned_copy import extract_lightglue
                packed,desc,members,total=extract_lightglue(self.image,limit,regions,cpu,'sift' if family=='sift-lg' else family,cancel,excluded=excluded)
            else:
                if self.gray is None:self.gray=cv.cvtColor(self.image,cv.COLOR_BGR2GRAY)
                detector={'SIFT':lambda:cv.SIFT_create(nfeatures=limit),'AKAZE':cv.AKAZE_create,'BRISK':cv.BRISK_create,'ORB':lambda:cv.ORB_create(nfeatures=limit)}[family]()
                kp,desc=detector.detectAndCompute(self.gray,polygon_mask(self.gray.shape,regions,excluded));packed=pack_keypoints(kp)
                if desc is None:desc=np.empty((0,detector.descriptorSize()),np.float32 if family=='SIFT' else np.uint8)
                total=len(packed)
                ids=np.argsort(-packed[:,4],kind='stable')[:limit];packed=packed[ids];desc=desc[ids]
                members=memberships(packed,self.gray.shape,regions)
            features=(packed,desc,members,total);check(cancel);self.features.put(key,features);self.counts['detections']+=1
        packed,desc,members,total=features
        if family=='SIFT-G2NN':
            from .sift_g2nn import matching_backend
            backend=matching_backend(desc,members,cpu)
        if family=='SIFT':
            descriptor_key=(key,algorithm);normalized=self.descriptors.get(descriptor_key)
            if normalized is None:
                if algorithm=='RootSIFT':normalized=np.sqrt(desc/np.maximum(desc.sum(axis=1,keepdims=True),1e-12))
                else:normalized=desc/np.maximum(np.linalg.norm(desc,axis=1,keepdims=True),1e-12)
                self.descriptors.put(descriptor_key,normalized)
            desc=normalized
        matchkey=(key,algorithm,radius,minimum,threshold,compare,radii,gap,guides,backend);matched=self.matches.get(matchkey)
        if matched is None:
            if family=='SIFT-G2NN':
                from .sift_g2nn import match
                matched=match(packed,desc,members,radius,minimum,threshold,compare,cancel,progress,radii,gap,axes,backend=backend)
            elif 'Glue' in algorithm:
                from .learned_copy import match
                matched=match(packed,desc,members,radius,minimum,threshold,compare,cpu,self.image.shape,cancel,progress,kind='xfeat' if family=='XFeat' else 'sift' if family=='sift-lg' else 'aliked',radii=radii,gap=gap,axes=axes)
            else:
                matched=spatial_matches(packed,desc,members,radius,minimum,threshold,family in ('AKAZE','BRISK','ORB'),compare,cancel,progress,radii,gap,axes)
            check(cancel);self.matches.put(matchkey,matched);self.counts['matchings']+=1
        pairs,evaluated=matched;groupkey=(matchkey,tolerance);resultkey=(groupkey,geometry,params);result=self.results.get(resultkey)
        if result is None:
            progress(80,'Grouping paired regions');groups=self.grouped.get(groupkey)
            if groups is None:
                groups=regional_biomes(packed,pairs,members,tolerance,compare,cancel);self.grouped.put(groupkey,groups);self.counts['groupings']+=1
            from .copy_geometry import verify
            progress(90,'Checking geometric consistency');groups,models=verify(packed,pairs,groups,*geometry,cancel=cancel)
            if family=='SIFT-G2NN' and geometry[0]!='None':
                from .sift_g2nn import filter_models
                groups,models=filter_models(groups,models)
            from .copy_overlap import GUARDED,MAX_OVERLAP,reject_self
            rejected=();self_filter=None
            if algorithm in GUARDED:
                groups,models,rejected=reject_self(packed,pairs,groups,models,minimum,cancel)
                self_filter=dict(maximum_overlap=MAX_OVERLAP,metric='intersection_over_smaller_hull',minimum_model_displacement_px=minimum)
            colors,bases=palette(groups,pairs,threshold)
            result=dict(feature_policy='independent_roi_v2' if per_zone else 'global',points=packed,pairs=pairs,groups=groups,colors=colors,bases=bases,params=params,regions=regions,compare=compare,total_features=total,candidate_comparisons=evaluated,models=models,backend=backend,self_match_filter=self_filter,rejected_biomes=rejected,distance_policy=dict(radii=radii,comparison_radius=radius,gap=gap,compact_guides=params[17] if len(params)>17 and params[15] and not compare else ()))
            check(cancel);self.results.put(resultkey,result)
        progress(100,'Analysis retained');return result

    def analyze_dense(self,params,regions,compare,cancel,progress):
        from .dense_copy import DenseCopyEngine
        from .copy_geometry import verify
        algorithm,limit,radius,minimum,threshold,tolerance=params[:6]
        geometry=tuple(params[6:9]) if len(params)>6 else ('None',3.,6)
        options=tuple(params[9:13]) if len(params)>9 else (8,8,False,2.)
        if len(options)!=4:raise ValueError('Invalid dense settings.')
        key=('dense',params,regions,compare);result=self.results.get(key)
        if result is not None:return result
        if self.dense is None:self.dense=DenseCopyEngine(self.image)
        before=self.dense.counts.copy()
        from .auto_zones import distance_policy
        radius,radii,gap=distance_policy(params,regions,compare)
        from .metal_dense import available as metal_available,sift_enabled
        descriptor_backend='metal' if (algorithm=='PatchMatch Zernike' or (algorithm=='PatchMatch SIFT' and sift_enabled())) and len(params)>13 and not params[13] and metal_available() else 'cpu'
        data=self.dense.analyze(algorithm,limit,radius,minimum,threshold,regions,compare,options,cancel,progress,geometry=geometry,radii=radii,gap=gap,excluded=params[16] if len(params)>16 else (),guides=params[17] if len(params)>17 and params[15] and not compare else (),workers=params[18] if len(params)>18 else 1,backend=descriptor_backend,quarter_turn=len(params)>19 and params[19]=='quarter_turn',target_patch=params[20] if len(params)>20 else None)
        for name in before:self.counts[name]+=self.dense.counts[name]-before[name]
        points,pairs=data['points'],data['pairs'];groupkey=('dense',params,regions,compare,options,geometry[0]!='None',geometry[1:])
        groups=self.grouped.get(groupkey)
        if groups is None:
            progress(80,'Grouping dense correspondences');groups=regional_biomes(points,pairs,data['members'],tolerance,compare,cancel);self.grouped.put(groupkey,groups);self.counts['groupings']+=1
        groups,models=verify(points,pairs,groups,*geometry,cancel=cancel,reflection=bool(options[2]));colors,bases=palette(groups,pairs,threshold)
        result=dict(points=points,pairs=pairs,groups=groups,models=models,colors=colors,bases=bases,params=params,regions=regions,compare=compare,total_features=len(points),candidate_comparisons=data['candidate_comparisons'],dense_count=data['dense_count'],dense_consistent_count=data['dense_consistent_count'],dense_maps=data['dense_maps'],backend='hybrid' if descriptor_backend=='metal' else 'cpu',descriptor_backend=descriptor_backend,distance_policy=dict(radii=radii,comparison_radius=radius,gap=gap,compact_guides=params[17] if len(params)>17 and params[15] and not compare else ()))
        ids=pairs[:,:2].astype(int)
        result['_pair_regions']=(data['members'][ids[:,0]] & data['members'][ids[:,1]]).argmax(1) if not compare else np.zeros(len(pairs),int)
        result['_image_shape']=self.image.shape[:2]
        check(cancel);self.results.put(key,result);progress(100,'Dense field retained');return result

    def analyze_combined(self,params,regions,compare,cancel,progress):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event,Lock
        key=('combined',params,regions,compare);cached=self.results.get(key)
        if cached is not None:return cached
        if not hasattr(self,'combined_engines'):self.combined_engines=[Cloning2Engine(self.image),Cloning2Engine(self.image)]
        names=('PatchMatch Zernike','PatchMatch SIFT');abort=Event();lock=Lock();values=[0,0]
        stopped=lambda:cancel() or abort.is_set()
        def run(i):
            def update(n,message):
                with lock:values[i]=n;progress(sum(values)//2,names[i]+' · '+message)
            defaults=(names[i],6000,600.,5.,.3,50.,'None',3.,6,8,8,False,2.,False,False,False,(),())
            child_params=(names[i],*params[1:18],*defaults[len(params):18],4)
            before=self.combined_engines[i].counts.copy()
            result=self.combined_engines[i].analyze(child_params,regions,compare,stopped,update)
            return result,{k:self.combined_engines[i].counts[k]-before[k] for k in before}
        with ThreadPoolExecutor(max_workers=2,thread_name_prefix='cm2-algorithm') as pool:
            futures=[pool.submit(run,i) for i in range(2)]
            try:answers=[future.result() for future in futures]
            except BaseException:
                abort.set()
                for future in futures:future.cancel()
                raise
        a,b=[answer[0] for answer in answers]
        for _,counts in answers:
            for name,value in counts.items():self.counts[name]+=value
        pairs=b['pairs'].copy();pairs[:,:2]+=len(a['points']);pairs=np.concatenate([a['pairs'],pairs])
        groups=(*a['groups'],*(g+len(a['pairs']) for g in b['groups']))
        colors,bases=palette(groups,pairs,params[4])
        models=[]
        for name,r,offset in zip(names,(a,b),(0,len(a['points']))):
            for model in r['models']:
                updated=dict(model,algorithm=name)
                for index_key in ('source_point_indices','destination_point_indices'):
                    if index_key in updated:updated[index_key]=[i+offset for i in updated[index_key]]
                models.append(updated)
        result=dict(points=np.concatenate([a['points'],b['points']]),pairs=pairs,groups=groups,models=tuple(models),
            colors=colors,bases=bases,params=params,regions=regions,compare=compare,
            total_features=a['total_features']+b['total_features'],candidate_comparisons=a['candidate_comparisons']+b['candidate_comparisons'],
            dense_count=a['dense_count']+b['dense_count'],dense_consistent_count=a['dense_consistent_count']+b['dense_consistent_count'],
            dense_maps=tuple({**field,'algorithm':name} for name,r in zip(names,(a,b)) for field in r['dense_maps']),
            group_algorithms=tuple([names[0]]*len(a['groups'])+[names[1]]*len(b['groups'])),
            pair_algorithms=np.r_[np.zeros(len(a['pairs']),np.uint8),np.ones(len(b['pairs']),np.uint8)],
            source_algorithms=names,workers_per_algorithm=4,backend='hybrid' if any(r.get('descriptor_backend')=='metal' for r in (a,b)) else 'cpu',descriptor_backends={name:r.get('descriptor_backend','cpu') for name,r in zip(names,(a,b))},distance_policy=a['distance_policy'])
        check(cancel);self.results.put(key,result);progress(100,'Deux algorithmes denses conservés');return result

    def analyze_extended(self,params,regions,compare,cancel,progress):
        """Keep the existing duo and add quarter-turn SIFT at several supports.

        Scales are discrete, expressed as descriptor-bin sizes in the export.
        This is additional detection work, not an exact speed optimisation.
        """
        key=('extended',params,regions,compare);cached=self.results.get(key)
        if cached is not None:return cached
        defaults=(EXTENDED,6000,600.,5.,.3,50.,'None',3.,6,8,8,False,2.,False,False,False,(),())
        effective=(*params[:18],*defaults[len(params):18])
        normal=(COMBINED,*effective[1:])
        base=self.analyze_combined(normal,regions,compare,cancel,lambda n,s:progress(n//3,s))
        if not hasattr(self,'frame_engines'):self.frame_engines={}
        patch=effective[9]
        # Equal parity puts both descriptor grids on exactly the same pixel
        # centres, without interpolation or resampling of descriptor vectors.
        bins=tuple(sorted(set(max(3+((patch-3)%2),min(32-(32-patch)%2,patch+2*round((patch*factor-patch)/2)))
                              for factor in (.75,1.,1.25,1.5))))
        bins=tuple(size for size in bins if min(self.image.shape[:2])>3*max(patch,size))
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event,Lock
        from .copy_detail import detail_image,corroborated_groups,DETAIL_POLICY
        detail=None
        if effective[6]!='None':
            check(cancel)
            if not hasattr(self,'_detail_image'):self._detail_image=detail_image(self.image)
            detail=self._detail_image
        abort=Event();lock=Lock();values=[0]*len(bins)
        stopped=lambda:cancel() or abort.is_set()
        for target in bins:
            if target not in self.frame_engines:self.frame_engines[target]=Cloning2Engine(self.image)
        def run(frame_index):
            target=bins[frame_index];engine=self.frame_engines[target]
            settings=('PatchMatch SIFT',*normal[1:],4,'quarter_turn',target)
            before=engine.counts.copy()
            def update(n,message):
                with lock:
                    values[frame_index]=n;progress(33+sum(values)*66//(100*len(bins)),message)
            extra=engine.analyze_dense(settings,regions,compare,stopped,update)
            from .sift_frames import transformed_groups
            groups,models=transformed_groups(extra['groups'],extra['models'],patch,target,effective[11])
            if detail is not None:
                groups,models=corroborated_groups(detail,extra['points'],groups,models,stopped)
            return dict(extra,groups=groups,models=models),{k:engine.counts[k]-before[k] for k in before}
        # Two frames x four region slots: at most eight independent CPU jobs.
        # WorkspaceBudget additionally gates concurrent large descriptor fields.
        with ThreadPoolExecutor(max_workers=2,thread_name_prefix='cm2-sift-frame') as pool:
            futures=[pool.submit(run,i) for i in range(len(bins))]
            try:answers=[future.result() for future in futures]
            except BaseException:
                abort.set()
                for future in futures:future.cancel()
                raise
        result=dict(base,params=params,group_frames=tuple(['normal']*len(base['groups'])))
        for target,(extra,counts) in zip(bins,answers):
            check(cancel)
            for name,value in counts.items():self.counts[name]+=value
            point_offset=len(result['points']);pair_offset=len(result['pairs'])
            pairs=extra['pairs'].copy();pairs[:,:2]+=point_offset
            pairs=np.concatenate((result['pairs'],pairs));groups=(*result['groups'],*(g+pair_offset for g in extra['groups']))
            colors,bases=palette(groups,pairs,params[4]);frame=f'quarter_turn/bin={target}'
            models=[]
            for model in extra['models']:
                updated=dict(model)
                for name in ('source_point_indices','destination_point_indices'):
                    if name in updated:updated[name]=[i+point_offset for i in updated[name]]
                updated['descriptor_frame']=frame;models.append(updated)
            result=dict(result,points=np.concatenate((result['points'],extra['points'])),pairs=pairs,
                groups=groups,models=(*result['models'],*models),colors=colors,bases=bases,
                total_features=result['total_features']+extra['total_features'],
                candidate_comparisons=result['candidate_comparisons']+extra['candidate_comparisons'],
                dense_count=result['dense_count']+extra['dense_count'],dense_consistent_count=result['dense_consistent_count']+extra['dense_consistent_count'],
                dense_maps=(*result['dense_maps'],*({**field,'algorithm':'PatchMatch SIFT','descriptor_frame':frame,'source_bin':patch,'target_bin':target} for field in extra['dense_maps'])),
                group_algorithms=(*result['group_algorithms'],*(['PatchMatch SIFT']*len(extra['groups']))),
                pair_algorithms=np.r_[result['pair_algorithms'],np.ones(len(extra['pairs']),np.uint8)],
                group_frames=(*result['group_frames'],*([frame]*len(extra['groups']))))
        result.update(extension='sift_frames_v2',extension_bins=bins,orientation_energy_ratio=.1,frame_workers=2,
                      extension_detail=dict(enabled=detail is not None,**DETAIL_POLICY),
                      extension_geometry=dict(angle_tolerance_degrees=15,scale_relative_tolerance=.15,maximum_anisotropy=1.15))
        check(cancel);self.results.put(key,result);progress(100,'Extended dense fields retained');return result
