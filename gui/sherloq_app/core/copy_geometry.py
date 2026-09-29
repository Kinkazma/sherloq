"""Optional, repeatable multi-model geometric verification of copy/move links.

Each biome is fitted independently, so Search's ROI separation is preserved.
The result is geometric consistency, never a verdict about authenticity.
"""
import cv2 as cv
import numpy as np
from .cloning import check

MODELS=('None','Similarity','Affine','Homography')


def project(points,matrix):
    homogeneous=np.column_stack((points,np.ones(len(points)))) @ matrix.T
    denominator=homogeneous[:,2]
    out=np.full((len(points),2),np.inf)
    good=np.abs(denominator)>1e-12
    out[good]=homogeneous[good,:2]/denominator[good,None]
    return out


def fit(a,b,model,threshold,reflection=False):
    # A direct similarity has positive determinant. Reflect source coordinates
    # explicitly for a requested mirror, then compose back to image coordinates.
    if reflection and model=='Similarity':
        reflected=a.copy();reflected[:,0]*=-1
        matrix=fit(reflected,b,model,threshold)
        return None if matrix is None else matrix@np.diag([-1.,1.,1.])
    # OpenCV's per-worker RNG is reset for reproducibility of each hypothesis.
    cv.setRNGSeed(1701)
    if model=='Homography':
        matrix,_=cv.findHomography(a,b,cv.USAC_MAGSAC,threshold,maxIters=5000,confidence=.995)
    else:
        estimate=cv.estimateAffinePartial2D if model=='Similarity' else cv.estimateAffine2D
        matrix,_=estimate(a,b,method=cv.RANSAC,ransacReprojThreshold=threshold,maxIters=5000,confidence=.995,refineIters=10)
        if matrix is not None:matrix=np.vstack((matrix,[0.,0.,1.]))
    if matrix is None or not np.isfinite(matrix).all() or np.linalg.cond(matrix)>1e12:return None
    return matrix


def verify(points,pairs,groups,model='Similarity',threshold=3.,minimum=6,cancel=lambda:False,reflection=False):
    """Peel multiple transformations, retaining original pair row indices.

    Inliers meet the pixel tolerance in BOTH mapping directions. Repeated
    orientations at a single pixel cannot supply independent support.
    """
    from .cloning2 import biome_sides
    if model not in MODELS or not np.isfinite(threshold) or threshold<=0 or minimum<4:
        raise ValueError('Invalid geometric verification settings.')
    if model=='None':return groups,()
    verified=[];models=[];attempts=0
    for parent,group in enumerate(groups):
        check(cancel)
        if len(group)<minimum:continue
        a,b=biome_sides(points,pairs,group)
        remaining=np.arange(len(group))
        while len(remaining)>=minimum:
            check(cancel);attempts+=1
            if attempts>2000:raise ValueError('Too many geometric hypotheses. Tighten regions or descriptor tolerance.')
            # Avoid fitting an orientation-rich centre many times as evidence.
            _,unique=np.unique(np.rint(np.column_stack((a[remaining],b[remaining]))),axis=0,return_index=True)
            sample=remaining[np.sort(unique)]
            if len(sample)<minimum:break
            matrix=fit(a[sample],b[sample],model,threshold,reflection)
            if matrix is None:break
            forward=np.linalg.norm(project(a[remaining],matrix)-b[remaining],axis=1)
            backward=np.linalg.norm(project(b[remaining],np.linalg.inv(matrix))-a[remaining],axis=1)
            errors=np.maximum(forward,backward);mask=errors<=threshold
            selected=remaining[mask]
            if len(selected)<minimum:break
            unique_a=len(np.unique(np.rint(a[selected]),axis=0));unique_b=len(np.unique(np.rint(b[selected]),axis=0))
            if min(unique_a,unique_b)<minimum:break
            verified.append(group[selected])
            ids=pairs[group[selected],:2].astype(int).copy()
            flipped=np.any(points[ids[:,0],:2]!=a[selected],axis=1)
            ids[flipped]=ids[flipped,::-1]
            models.append(dict(source_point_indices=ids[:,0].tolist(),destination_point_indices=ids[:,1].tolist(),parent_biome=parent,model=model,matrix=matrix.tolist(),inliers=len(selected),
                               distinct_centres=[unique_a,unique_b],median_error_px=float(np.median(errors[mask])),
                               maximum_error_px=float(errors[mask].max())))
            remaining=remaining[~mask]
    return tuple(verified),tuple(models)
