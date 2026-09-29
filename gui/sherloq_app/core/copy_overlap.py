"""Reject local self-correspondence biomes for the learned copy/move variants.

Overlap is intersection / smaller hull area (containment counts as overlap),
using the same rounded original-image coordinates as the visible polygons.
Raw candidates remain available for audit; accepted groups/models stay aligned.
"""
import cv2 as cv
import numpy as np
from .cloning import check

GUARDED=('XFeat','XFeat + LighterGlue','ALIKED','ALIKED rotation',
         'ALIKED + LightGlue','ALIKED rotation + LightGlue')
MAX_OVERLAP=.8

def overlap(a,b):
    hulls=[cv.convexHull(np.rint(p).astype(np.float32)) for p in (a,b)]
    areas=[cv.contourArea(h) for h in hulls]
    if min(areas)<=0:return 0.
    intersection=cv.intersectConvexConvex(*hulls,handleNested=True)[0]
    return float(np.clip(intersection/min(areas),0,1))

def reject_self(points,pairs,groups,models,minimum,cancel=lambda:False):
    from .cloning2 import biome_sides
    from .copy_geometry import project
    accepted=[];kept_models=[];rejected=[]
    for i,group in enumerate(groups):
        check(cancel);a,b=biome_sides(points,pairs,group);ratio=overlap(a,b)
        record=dict(candidate_biome=i,overlap_fraction=ratio,pair_indices=group.tolist())
        reason=None
        if ratio>=MAX_OVERLAP:reason='overlapping_hulls'
        if models:
            model=models[i];source=points[model['source_point_indices'],:2]
            displacement=float(np.median(np.linalg.norm(project(source,np.asarray(model['matrix']))-source,axis=1)))
            record['model_median_displacement_px']=displacement
            # RANSAC's residual tolerance must not turn a near-identity transform
            # into a valid copy when its movement is below the search minimum.
            if displacement+1e-6<minimum:reason=reason or 'near_identity_model'
        if reason:
            record['reason']=reason;rejected.append(record)
        else:
            accepted.append(group)
            if models:kept_models.append(models[i])
    return tuple(accepted),tuple(kept_models),tuple(rejected)
