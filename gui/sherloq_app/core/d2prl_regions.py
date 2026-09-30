"""Exact D2PRL segmentation layer for automatic analyses, without hull inference."""
import hashlib
import cv2 as cv
import numpy as np


def regions(result, minimum=500):
    if result is None:return ()
    from .d2prl import refilter
    from .automatic_clones import D2PRL_SOURCE
    current=result['metadata'].get('min_component',500)
    filtered=result if minimum==current else refilter((result,minimum))
    mask=np.ascontiguousarray(filtered['mask'],dtype=np.uint8)
    n,labels,stats,_=cv.connectedComponentsWithStats(mask,connectivity=8)
    entries=[]
    for i in range(1,n):
        x,y,w,h,count=map(int,stats[i]);pixels=(labels[y:y+h,x:x+w]==i).astype(np.uint8)
        key=hashlib.sha256(pixels.tobytes()+str((x,y,w,h)).encode()).hexdigest()[:24]
        contours,_=cv.findContours(pixels,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        entries.append(dict(id='d2prl-'+key,source=D2PRL_SOURCE,label=D2PRL_SOURCE,
            count=count,count_kind='pixels',color=(230,170,30),pixel_mask=pixels,origin=(x,y),
            polygons=[(p.reshape(-1,2)+(x,y)).tolist() for p in contours],
            # This is the union of the selected AI searches, not independent
            # corroborating methods. Overlapping passes contribute only once.
            search_context='d2prl-selected-zones',
            provenance=dict(min_component=minimum,native_grid=[448,448],
                            evidence='segmentation',boxes=result['metadata'].get('boxes',()))))
    return tuple(entries)


class RegionCache:
    def __init__(self):self.result=None;self.minimum=None;self.value=()
    def __call__(self,result,minimum):
        if result is not self.result or minimum!=self.minimum:
            self.value=regions(result,minimum);self.result=result;self.minimum=minimum
        return self.value


def paint(image,entry,fill=.25,outline=None,opacity=1.):
    """Fill selected pixels and their inner perimeter, keeping holes intact."""
    x,y=entry['origin'];mask=entry['pixel_mask'];h,w=mask.shape
    roi=image[y:y+h,x:x+w];keep=mask.astype(bool)
    if fill:
        mixed=cv.addWeighted(roi,1-fill,np.full_like(roi,entry['color']),fill,0)
        roi[keep]=mixed[keep]
    contours,_=cv.findContours(mask,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    boundary=np.zeros_like(mask);cv.drawContours(boundary,contours,-1,1,1)
    border=(boundary&mask).astype(bool)
    color=np.full_like(roi,entry['color'] if outline is None else outline)
    if opacity<1:color=cv.addWeighted(roi,1-opacity,color,opacity,0)
    roi[border]=color[border]
