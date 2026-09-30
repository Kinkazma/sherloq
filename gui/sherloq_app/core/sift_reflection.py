"""Additional reflected keypoint SIFT pass in original image coordinates.

Re-extract reflected pixels; do not pretend ordinary rotation-invariant SIFT
is reflection-invariant. Native normal-pass evidence is retained unchanged.
"""
import numpy as np
from .cloning import check


def analyze(engine, params, regions, compare, cancel, progress):
    from .sift_g2nn import extract, match, matching_backend, filter_models
    from .cloning2 import regional_biomes
    from .copy_geometry import verify
    from .auto_zones import distance_policy, compact_axes
    limit=params[1];excluded=params[16];width=engine.image.shape[1]
    def reflected(polygons):
        return tuple(tuple((float(width-1-x),float(y)) for x,y in p) for p in polygons)
    # Reuse exactly the normal per-ROI extraction, including normalization/masks.
    key=('SIFT-G2NN',limit,regions,None,excluded,True)
    base=engine.features.get(key)
    if base is None:
        base=extract(engine.image,limit,regions,excluded,cancel)
        engine.features.put(key,base);engine.counts['detections']+=1
    mirror_key=('reflected-sift-roi-v1',limit,regions,excluded)
    mirror=engine.features.get(mirror_key)
    if mirror is None:
        p,d,m,n=extract(engine.image[:,::-1],limit,reflected(regions),reflected(excluded),cancel)
        p[:,0]=width-1-p[:,0]
        # Angles are descriptive only; descriptors remain in reflected frame.
        p[:,3]=(180-p[:,3])%360
        mirror=(p,d,m,n);engine.features.put(mirror_key,mirror);engine.counts['detections']+=1
    points=np.concatenate((base[0],mirror[0]));desc=np.concatenate((base[1],mirror[1]))
    members=np.concatenate((base[2],mirror[2]));variants=np.arange(len(points))>=len(base[0])
    radius,radii,gap=distance_policy(params,regions,compare)
    axes=compact_axes(engine.image.shape,params[17] if params[15] and not compare else ())
    backend=matching_backend(desc,members,params[13])
    key=('reflected-sift-match-v1',params,regions,compare)
    cached=engine.matches.get(key)
    if cached is None:
        cached=match(points,desc,members,radius,params[3],params[4],compare,cancel,progress,
                     radii,gap,axes,backend=backend,variants=variants)
        check(cancel);engine.matches.put(key,cached);engine.counts['matchings']+=1
    pairs,evaluated=cached
    groups=regional_biomes(points,pairs,members,params[5],compare,cancel)
    # Even when geometric checks are disabled for the normal pass, reflected
    # detections require support for a negative-determinant transform.
    geometry=params[6] if params[6]!='None' else 'Affine'
    groups,models=verify(points,pairs,groups,geometry,params[7],max(4,params[8]),cancel,reflection=True)
    groups,models=filter_models(groups,models)
    keep=[i for i,m in enumerate(models) if np.linalg.det(np.asarray(m['matrix'])[:2,:2])<0]
    ids=pairs[:,:2].astype(int)
    owners=(members[ids[:,0]] & members[ids[:,1]]).argmax(1).astype(np.int32) if not compare else np.full(len(pairs),-1,np.int32)
    return dict(points=points,pairs=pairs,pair_search_regions=owners,groups=tuple(groups[i] for i in keep),
                models=tuple(dict(models[i],variant='reflection') for i in keep),
                total_features=mirror[3],candidate_comparisons=evaluated,backend=backend)


def append(base, extra):
    """Append arrays without changing a normal pair, group or transformation."""
    offset=len(base['points']);rows=len(base['pairs']);pairs=extra['pairs'].copy();pairs[:,:2]+=offset
    models=[]
    for model in extra['models']:
        model=dict(model)
        for key in ('source_point_indices','destination_point_indices'):
            model[key]=[i+offset for i in model[key]]
        models.append(model)
    return dict(base,points=np.concatenate((base['points'],extra['points'])),
                pairs=np.concatenate((base['pairs'],pairs)),
                pair_search_regions=np.concatenate((base['pair_search_regions'],extra['pair_search_regions'])),
                groups=(*base['groups'],*(g+rows for g in extra['groups'])),
                models=(*base['models'],*models),
                total_features=base['total_features']+extra['total_features'],
                candidate_comparisons=base['candidate_comparisons']+extra['candidate_comparisons'],
                reflection_backend=extra['backend'])
