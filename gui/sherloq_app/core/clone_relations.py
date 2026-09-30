"""Keep search origin, anatomical panel relation and display area distinct."""
import hashlib
import json
import cv2 as cv
import numpy as np


def polygon_key(polygon):
    return hashlib.sha256(json.dumps(sorted(map(tuple,np.round(np.asarray(polygon,np.float64),3).tolist()))).encode()).hexdigest()[:16]


def panels(regions, envelope=None):
    """Remove only the encompassing search rectangle, never all search regions."""
    regions=[np.asarray(p,np.float32) for p in regions]
    if len(regions)<=1:return list(enumerate(regions))
    if envelope is not None:
        key=polygon_key(np.asarray(envelope))
        return [(i,p) for i,p in enumerate(regions) if polygon_key(p)!=key]
    # Core results have their active regions but not the UI envelope marker.
    largest=max(range(len(regions)),key=lambda i:cv.contourArea(regions[i]))
    p=regions[largest]
    if all(all(cv.pointPolygonTest(p,tuple(map(float,q)),False)>=0 for q in other)
           for i,other in enumerate(regions) if i!=largest):
        return [(i,p) for i,p in enumerate(regions) if i!=largest]
    return list(enumerate(regions))


def memberships(xy, regions):
    result=np.zeros((len(xy),len(regions)),bool)
    for column,(_,p) in enumerate(regions):
        lo=p.min(0);hi=p.max(0)
        candidates=np.flatnonzero(((xy>=lo)&(xy<=hi)).all(1))
        if not len(candidates):continue
        rectangle=len(p)==4 and len(np.unique(p[:,0]))==2 and len(np.unique(p[:,1]))==2
        if rectangle:result[candidates,column]=True
        else:result[candidates,column]=[cv.pointPolygonTest(p,tuple(map(float,q)),False)>=0 for q in xy[candidates]]
    return result


def pair_relations(result):
    """Relation per pair, before hull construction can mix unrelated panels."""
    regions=sorted(panels(result.get('regions',())),key=lambda entry:cv.contourArea(entry[1]))
    points=result['points'];ids=result['pairs'][:,:2].astype(int)
    if not regions:return np.full((len(ids),2),-1,np.int32)
    member=memberships(points[:,:2],regions)
    labels=np.where(member.any(1),member.argmax(1),-1)
    relation=labels[ids].copy()
    common=member[ids[:,0]]&member[ids[:,1]]
    shared=common.any(1)
    relation[shared]=np.repeat(common[shared].argmax(1)[:,None],2,axis=1)
    actual=np.array([i for i,_ in regions],np.int32)
    known=relation>=0;relation[known]=actual[relation[known]]
    return np.sort(relation,axis=1)


def annotate(entries, regions, envelope, shape):
    """Attach view classification and search origin, without changing raw evidence.

    An envelope is not a physical panel. A relation is intra-panel only when
    both endpoint hulls fit the same smallest panel. Ambiguous hulls remain
    unassigned rather than silently being labelled inter-panel.
    """
    from .automatic_clones import SOURCES, AI_SOURCES
    zones=sorted(panels(regions,envelope),key=lambda entry:cv.contourArea(entry[1]))
    if not zones:
        h,w=shape[:2];zones=[(0,np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],np.float32))]
    output=[]
    for entry in entries:
        if entry['source'] not in SOURCES or 'pixel_mask' in entry:output.append(entry);continue
        polygons=[np.asarray(p,np.float32) for p in entry['polygons']]
        owners=[]
        for p in polygons:
            eligible=[i for i,z in zones if all(cv.pointPolygonTest(z,tuple(map(float,q)),True)>=-.75 for q in p)]
            owners.append(eligible)
        common=set(owners[0]) if owners else set()
        for ids in owners[1:]:common.intersection_update(ids)
        if common:
            zone=next(i for i,_ in zones if i in common);assigned=[zone]*len(polygons);kind='within'
        elif len(owners)==2 and all(owners):
            assigned=[ids[0] for ids in owners];kind='between'
        else:
            assigned=[ids[0] if ids else None for ids in owners];kind='unassigned'
        origin=entry.get('provenance',{}).get('search_region')
        context=entry.get('search_context') or ('roi:'+polygon_key(np.asarray(origin)) if origin else 'unspecified')
        search_index=next((i for i,p in enumerate(regions) if origin is not None and polygon_key(np.asarray(p))==polygon_key(np.asarray(origin))),None)
        search_label=('Zone '+str(search_index+1) if search_index is not None else
                      'Enclosing zone' if entry['source'] in AI_SOURCES else
                      'Whole image' if context=='whole-image' else 'Unspecified search')
        from_envelope=(len(regions)>1 and envelope is not None and origin is not None and
                       polygon_key(np.asarray(origin))==polygon_key(np.asarray(envelope)))
        # Opacity is presentation metadata ONLY: it is never used in counts.
        reference={i:max(1.,cv.contourArea(z)) for i,z in zones}
        fill_alpha=[.25 if entry['source'] in AI_SOURCES else
                    .25/(1+2*np.sqrt(min(1.,cv.contourArea(p)/reference.get(i,float(shape[0]*shape[1])))))
                    for p,i in zip(polygons,assigned)]
        output.append(dict(entry,search_label=search_label,heat_eligible=True,
                           from_enclosing_search=from_envelope,biome_fill_alpha=fill_alpha,relation=kind,endpoint_zones=assigned,
                           search_context=context))
    return tuple(output)
