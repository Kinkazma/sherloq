"""Copy/move stages with ordered exact Hamming matches and retained analysis."""
import cv2 as cv
import numpy as np
from .interactive import ArrayCache
from .jpeg_curve import Cancelled

MATCH_BUDGET = 256*1024*1024
CLUSTER_INDEX_BUDGET = 128*1024*1024


def check(cancel):
    if cancel():
        raise Cancelled()


def pack_keypoints(points):
    return np.asarray([(p.pt[0],p.pt[1],p.size,p.angle,p.response,p.octave,p.class_id)
                       for p in points],np.float64).reshape(-1,7)


def ordered_matches(desc, radius, cancel=lambda:False, progress=lambda *a:None):
    if len(desc)==0:
        return np.empty((0,3),np.float64)
    matcher=cv.BFMatcher_create(cv.NORM_HAMMING,True)
    parts=[];count=0
    # Bound native radiusMatch's temporary distance matrix and Python objects.
    for start in range(0,len(desc),64):
        check(cancel)
        rows=matcher.radiusMatch(desc[start:start+64],desc,radius)
        triples=[(m.queryIdx+start,m.trainIdx,m.distance) for row in rows for m in row
                 if m.queryIdx+start!=m.trainIdx]
        count+=len(triples)
        if count*24>MATCH_BUDGET:
            raise ValueError('Matching exceeds the 256 MiB result budget. Reduce Matching or Response.')
        parts.append(np.asarray(triples,np.float64).reshape(-1,3))
        progress(min(100,(start+64)*100//len(desc)),'Matching descriptors')
    check(cancel)
    return np.concatenate(parts) if parts else np.empty((0,3),np.float64)


def cluster_matches(points,matches,min_dist,minimum,cancel=lambda:False,progress=lambda *a:None):
    if not len(matches):return matches,()
    indices=matches[:,:2].astype(np.intp)
    coords=points[:,:2]
    displacements=np.linalg.norm(coords[indices[:,0]]-coords[indices[:,1]],axis=1)
    keep=displacements>min_dist
    # Fix the old index drift: distances and matches undergo the same filter.
    filtered=matches[keep];ds=displacements[keep];indices=indices[keep]
    qa=coords[indices[:,0]];ta=coords[indices[:,1]]
    groups=[];stored=0
    for i in range(len(filtered)):
        check(cancel)
        # Keep original order and strict spatial bounds. Work row by row rather
        # than allocating an MxM matrix or rebuilding points inside Python loops.
        candidates=np.flatnonzero(np.abs(ds[i+1:]-ds[i])<=min_dist)+i+1
        if len(candidates):
            reverse=(indices[candidates,0]==indices[i,1])&(indices[candidates,1]==indices[i,0])
            candidates=candidates[~reverse]
        if len(candidates):
            deltas=[qa[i]-qa[candidates],ta[i]-ta[candidates],qa[i]-ta[candidates],ta[i]-qa[candidates]]
            norms=[np.linalg.norm(d,axis=1) for d in deltas]
            # Match the original 1-D norm even at a rounding-sensitive boundary.
            for delta,norm in zip(deltas,norms):
                boundary=np.flatnonzero(np.abs(norm-min_dist)<=max(1.,min_dist)*1e-12)
                for j in boundary:norm[j]=np.linalg.norm(delta[j])
            aa,bb,ab,ba=norms
            good=((aa>0)&(aa<min_dist)&(bb>0)&(bb<min_dist))|((ab>0)&(ab<min_dist)&(ba>0)&(ba<min_dist))
            candidates=candidates[good]
        group=[i];seen={(int(indices[i,0]),int(indices[i,1]))}
        for j in candidates:
            query,train=map(int,indices[j])
            if (train,query) not in seen:
                group.append(int(j));seen.add((query,train))
        if len(group)>=minimum:
            stored+=len(group)
            if stored*8>CLUSTER_INDEX_BUDGET:
                raise ValueError('Clusters exceed the 128 MiB index budget. Reduce Matching or Response.')
            groups.append(np.asarray(group,np.int64))
        if i%64==0:progress((i+1)*100//max(1,len(filtered)),'Clustering matches')
    return filtered,tuple(groups)


def render(image,points,matches,groups,radius,show_points,hide_lines,cancel=lambda:False):
    output=image.copy();hsv=np.zeros((1,1,3));angles=[]
    if show_points:
        for i,p in enumerate(points):
            if i%512==0:check(cancel)
            cv.circle(output,(int(p[0]),int(p[1])),2,(250,227,72))
    for group in groups:
        check(cancel)
        for position,index in enumerate(group):
            if position%128==0:check(cancel)
            query,train,distance=matches[index];a=points[int(query)];b=points[int(train)]
            pa,pb=tuple(map(int,a[:2])),tuple(map(int,b[:2]));sa,sb=int(np.round(a[2])),int(np.round(b[2]))
            angle=np.arctan2(pb[1]-pa[1],pb[0]-pa[0])
            if angle<0:angle+=np.pi
            angles.append(angle)
            hsv[0,0]=(angle/np.pi*180,255,distance/radius*255)
            rgb=tuple(int(x) for x in cv.cvtColor(hsv.astype(np.uint8),cv.COLOR_HSV2BGR)[0,0])
            cv.circle(output,pa,sa,rgb,1,cv.LINE_AA);cv.circle(output,pb,sb,rgb,1,cv.LINE_AA)
            if not hide_lines:cv.line(output,pa,pb,rgb,1,cv.LINE_AA)
    return output,np.asarray(angles,np.float32).reshape(-1,1)


def region_count(angles):
    if not len(angles):return 0
    if np.std(angles)<.1:return 1
    # Fix k > number of samples; seed only this worker's OpenCV RNG. This
    # heuristic count was stochastic even when merely toggling drawing options.
    cv.setRNGSeed(0)
    criteria=(cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER,10,1.)
    compact=[cv.kmeans(angles,k,None,criteria,10,cv.KMEANS_PP_CENTERS)[0]
             for k in range(1,min(10,len(angles))+1)]
    compact=cv.normalize(np.asarray(compact),None,0,1,cv.NORM_MINMAX)
    return int(np.argmax(compact<.005)+1)


class CloningEngine:
    def __init__(self,image):
        self.image=image
        self.gray=None
        self.detected=ArrayCache(128)
        self.selected=ArrayCache(64)
        self.matched=ArrayCache(256)
        self.clustered=ArrayCache(128)
        self.renders=ArrayCache(128)
        self.counts={}

    def detect(self,algorithm,mask_id,mask,cancel):
        key=algorithm,mask_id
        value=self.detected.get(key)
        if value is None:
            check(cancel)
            if self.gray is None:self.gray=cv.cvtColor(self.image,cv.COLOR_BGR2GRAY)
            detector=(cv.BRISK_create,cv.ORB_create,cv.AKAZE_create)[algorithm]()
            points,desc=detector.detectAndCompute(self.gray,mask)
            packed=pack_keypoints(points)
            if desc is None:desc=np.empty((0,detector.descriptorSize()),np.uint8)
            check(cancel)
            value=self.detected.put(key,(packed,desc))
        return value

    def analyze(self,params,mask_id=0,mask=None,cancel=lambda:False,progress=lambda *a:None):
        algorithm,response,matching,distance,minimum,show,hide=params
        check(cancel)
        if min(self.image.shape[:2]) < 7:
            raise ValueError('Copy/move detection needs at least 7 × 7 pixels.')
        if not (algorithm in (0,1,2) and 0 <= response <= 100 and
                1 <= matching <= 100 and 1 <= distance <= 100 and 1 <= minimum <= 20):
            raise ValueError('Invalid copy/move detection settings.')
        basekey=(algorithm,mask_id)
        progress(0,'Detecting keypoints')
        points,desc=self.detect(algorithm,mask_id,mask,cancel)
        total=len(points)
        selectkey=basekey+(response,)
        selected=self.selected.get(selectkey)
        if selected is None:
            if len(points):
                strongest=(cv.normalize(points[:,4],None,0,100,cv.NORM_MINMAX)>=100-response).ravel()
                points,desc=points[strongest],desc[strongest]
            if len(points)>30000:raise ValueError(f'Too many filtered keypoints ({len(points)}). Reduce Response.')
            selected=self.selected.put(selectkey,(points,desc))
        points,desc=selected
        matchkey=selectkey+(matching,)
        matches=self.matched.get(matchkey)
        radius=matching/100*255
        if matches is None:
            matches=ordered_matches(desc,radius,cancel,progress)
            self.matched.put(matchkey,matches)
        geometrykey=matchkey+(distance,)
        result=self.clustered.get(geometrykey)
        if result is None:
            min_dist=distance/100*np.min(self.image.shape[:2])/2
            result=cluster_matches(points,matches,min_dist,1,cancel,progress)
            check(cancel)
            self.clustered.put(geometrykey,result)
        filtered,all_groups=result
        # Minimum group size affects selection only, never pair geometry.
        groups=tuple(group for group in all_groups if len(group)>=minimum)
        clusterkey=geometrykey+(minimum,)
        renderkey=clusterkey+(show,hide)
        output=self.renders.get(renderkey)
        if output is None:
            progress(95,'Drawing matches')
            self.renders.reserve(self.image.nbytes)
            output,angles=render(self.image,points,filtered,groups,radius,show,hide,cancel)
            check(cancel)
            count=self.counts.get(clusterkey)
            if count is None:count=region_count(angles)
            self.renders.put(renderkey,output)
            # Counts are tiny, but keep a finite number of arbitrary settings.
            self.counts[clusterkey]=count
            if len(self.counts)>128:self.counts.pop(next(iter(self.counts)))
        count=self.counts.get(clusterkey)
        if count is None:
            _,angles=render(self.image,points,filtered,groups,radius,False,True)
            count=region_count(angles);self.counts[clusterkey]=count
        check(cancel)
        progress(100,'Copy/move analysis ready')
        return output,dict(total=total,filtered=len(points),matches=len(filtered),clusters=len(groups),regions=count)
