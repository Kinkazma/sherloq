"""Additional dense SIFT frame, invariant to quarter turns of its 4x4x8 grid.

This is a new descriptor variant, not a numerical optimisation of normal SIFT.
It supplements the normal pass; arbitrary rotations/scale invariance are not
claimed. No image pixels are resized or rotated by this operation.
"""
import numpy as np
from .cloning import check

def transformed_groups(groups,models,source_bin,target_bin,mirror=False):
    """Corroborate the transformation tested by this supplementary pass.

    Normal translation evidence already comes from the unchanged base duo.
    A descriptor-scale hypothesis must not turn a near-identity match into
    evidence for a resized copy. Models can be oriented in either direction.
    Explicitly disabling geometry retains raw groups for manual inspection.
    """
    if not models:return groups,models
    selected=[];accepted=[];ratio=target_bin/source_bin
    for group,model in zip(groups,models):
        linear=np.asarray(model['matrix'],np.float64)[:2,:2]
        u,s,v=np.linalg.svd(linear)
        if s[-1]<=0 or s[0]/s[-1]>1.15:continue
        scale=float(np.sqrt(s.prod()));rotation=u@v
        reflected=np.linalg.det(rotation)<0
        if reflected!=mirror:continue
        if reflected:rotation=rotation@np.diag([-1.,1.])
        angle=float(np.degrees(np.arctan2(rotation[1,0],rotation[0,0])))%360
        turn=int(round(angle/90))%4
        if abs((angle-90*turn+180)%360-180)>15:continue
        if min(abs(np.log(scale/ratio)),abs(np.log(scale*ratio)))>np.log(1.15):continue
        if target_bin==source_bin and turn==0 and not mirror:continue
        selected.append(group);accepted.append(model)
    return tuple(selected),tuple(accepted)

def orientation_diversity(descriptors,cancel=lambda:False):
    """Reject an ambiguous straight-edge frame in the supplementary pass.

    The doubled-angle histogram treats opposite gradients as the same edge
    direction. Its minor/major directional-energy ratio must be at least .1.
    This criterion does not suppress any result of the normal SIFT pass.
    """
    source=descriptors.reshape(-1,4,4,8);result=np.empty(len(source),bool)
    cosine=np.array([1,0,-1,0,1,0,-1,0],np.float32)
    sine=np.array([0,1,0,-1,0,1,0,-1],np.float32)
    for start in range(0,len(source),8192):
        check(cancel);hist=source[start:start+8192].sum((1,2));total=hist.sum(1)
        anisotropy=np.sqrt((hist*cosine).sum(1)**2+(hist*sine).sum(1)**2)
        result[start:start+len(hist)]=(total>0)&((total-anisotropy)>=.1*(total+anisotropy))
    return result.reshape(descriptors.shape[:2])


def quarter_turn_frame(descriptors,cancel=lambda:False):
    if descriptors.ndim!=3 or descriptors.shape[-1]!=128:
        raise ValueError('Quarter-turn SIFT requires 128-dimensional dense descriptors.')
    source=descriptors.reshape(-1,4,4,8)
    result=np.empty_like(source)
    # Bound gather temporaries independently of image dimensions.
    for start in range(0,len(source),8192):
        check(cancel);block=source[start:start+8192]
        histogram=block.sum((1,2)).reshape(-1,4,2).sum(2)
        orientation=histogram.argmax(1)
        target=result[start:start+len(block)]
        for turn in range(4):
            ids=np.flatnonzero(orientation==turn)
            target[ids]=np.roll(np.rot90(block[ids],turn,axes=(1,2)),-2*turn,axis=3)
    check(cancel)
    return result.reshape(descriptors.shape)
