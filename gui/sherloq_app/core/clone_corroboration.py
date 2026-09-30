"""Local preview: exact clone counts, preserving search contexts.

ELA is deliberately absent. These counts are not calibrated probabilities or
independent statistical trials. Raw detections remain available per source.
"""
import cv2 as cv
import numpy as np

# BGR, fixed absolute scale (never renormalized when a method is hidden).
COLORS=np.array([[0,0,0],[255,100,30],[210,210,0],[0,220,255],[20,30,255]],np.uint8)


def _iou(a,b):
    a=np.asarray(a,np.float32);b=np.asarray(b,np.float32)
    if np.any(a.max(0)<b.min(0)) or np.any(b.max(0)<a.min(0)):return 0.
    intersection,_=cv.intersectConvexConvex(a,b)
    union=cv.contourArea(a)+cv.contourArea(b)-intersection
    return float(intersection/union) if union>0 else 0.


def unique_envelopes(entries, threshold=.9):
    """Deduplicate only almost identical PAIRS, not nested or single shared ends.

    No union hull is formed: it could invent a bridge across unobserved areas.
    Members and source provenance survive in the representative display entry.
    """
    from .automatic_clones import SOURCES, AI_SOURCES
    result=[]
    for entry in entries:
        match=None
        if 'pixel_mask' not in entry and len(entry['polygons'])==2:
            a,b=entry['polygons']
            for old in result:
                # AI remains an independent display layer even if a classical
                # detector outlines the same pair of panels.
                if 'pixel_mask' in old:continue
                if (entry['source'] in AI_SOURCES or old['source'] in AI_SOURCES) and entry['source']!=old['source']:continue
                if len(old['polygons'])!=2:continue
                c,d=old['polygons']
                if ((_iou(a,c)>=threshold and _iou(b,d)>=threshold) or
                    (_iou(a,d)>=threshold and _iou(b,c)>=threshold)):
                    match=old;break
        if match is None:
            result.append(dict(entry,member_ids=[entry['id']],corroborating_sources=[entry['source']]))
        else:
            match['member_ids'].append(entry['id'])
            if entry['source'] not in match['corroborating_sources']:
                match['corroborating_sources'].append(entry['source'])
    return tuple(result)


def _fill_strip(mask, entry, top):
    """Rasterize exact segmentation masks; never replace them by convex hulls."""
    if 'pixel_mask' in entry:
        x,y=entry['origin'];pixels=entry['pixel_mask'];h,w=pixels.shape
        y0,y1=max(top,y),min(top+len(mask),y+h)
        x0,x1=max(0,x),min(mask.shape[1],x+w)
        if y0<y1 and x0<x1:
            np.maximum(mask[y0-top:y1-top,x0:x1],pixels[y0-y:y1-y,x0-x:x1-x],
                       out=mask[y0-top:y1-top,x0:x1])
        return
    for polygon in entry['polygons']:
        p=np.rint(polygon).astype(np.int32)
        if p[:,1].max()<top or p[:,1].min()>=top+len(mask):continue
        cv.fillConvexPoly(mask,p-(0,top),1)


def _counts(shape,entries,excluded,by_context):
    from .automatic_clones import SOURCES
    h,w=shape[:2];result=np.zeros((h,w),np.uint32 if by_context else np.uint8);groups={}
    for e in entries:
        if e['source'] not in SOURCES:continue
        key=(e['source'],e.get('search_context','unspecified')) if by_context else e['source']
        groups.setdefault(key,[]).append(e)
    for top in range(0,h,256):
        strip=result[top:top+256];mask=np.zeros(strip.shape,np.uint8)
        for group in groups.values():
            mask.fill(0)
            for entry in group:_fill_strip(mask,entry,top)
            strip+=mask
        target=strip.view(np.int32) if by_context else strip
        for p in excluded:cv.fillPoly(target,[np.rint(np.asarray(p)-(0,top)).astype(np.int32)],0)
    return result


def votes(shape, entries, excluded=()):
    """Distinct methods, with exact masks and bounded temporary workspace."""
    return _counts(shape,entries,excluded,False)


def context_counts(shape, entries, excluded=()):
    """Integer count of distinct (method, search region) covering a pixel."""
    return _counts(shape,entries,excluded,True)


COUNT_COLORS=np.array([[0,0,0],[255,100,30],[210,210,0],[60,210,60],
                       [0,220,255],[0,128,255],[20,30,255]],np.uint8)


def render_heat(image, entries, excluded, mode, opacity, count=None):
    count=context_counts(image.shape,entries,excluded) if count is None else count
    out=image.copy() if mode=='overlay' else np.zeros_like(image)
    for top in range(0,len(image),256):
        values=count[top:top+256]
        color=COUNT_COLORS[np.minimum(values,6)];keep=values>0
        roi=out[top:top+256]
        if mode=='overlay':color=cv.addWeighted(roi,1-opacity,color,opacity,0)
        roi[keep]=color[keep]
    # Thin boundaries expose nested biomes without changing a single map value.
    if mode=='overlay':
        from .automatic_clones import SOURCES, AI_SOURCES
        ordered=sorted((e for e in entries if e['source'] in SOURCES),
                       key=lambda e:sum(cv.contourArea(np.asarray(p,np.float32)) for p in e['polygons']),reverse=True)
        if opacity>0:
            for e in ordered:
                if 'pixel_mask' in e:
                    from .d2prl_regions import paint
                    paint(out,e,fill=0.,outline=(245,245,245),opacity=opacity)
                    continue
                for p in e['polygons']:
                    p=np.rint(p).astype(np.int32);x,y,w,h=cv.boundingRect(p)
                    x0,y0=max(0,x),max(0,y);x1,y1=min(out.shape[1],x+w),min(out.shape[0],y+h)
                    if x1<=x0 or y1<=y0:continue
                    roi=out[y0:y1,x0:x1];line=roi.copy()
                    cv.polylines(line,[p-(x0,y0)],True,(245,245,245),1,cv.LINE_AA)
                    cv.addWeighted(line,opacity,roi,1-opacity,0,dst=roi)
            # Respect excluded geometry after the optional contours as well.
            for top in range(0,len(out),256):
                strip=out[top:top+256];mask=np.zeros(strip.shape[:2],np.uint8)
                for p in excluded:cv.fillPoly(mask,[np.rint(np.asarray(p)-(0,top)).astype(np.int32)],1)
                strip[mask.astype(bool)]=image[top:top+256][mask.astype(bool)]
    return out


class CachedRenderer:
    """LatestJob serializes calls; opacity changes reuse the same count map."""
    def __init__(self, render, complete=False):
        self.render=render;self.complete=complete
        self.key=None;self.entries=();self.counts=None;self.builds=0

    def __call__(self, request):
        from .automatic_clones import SOURCES, AI_SOURCES
        position=5 if self.complete else 3
        if len(request)<=position or request[position]=='biomes':return self.render(request)
        image,entries,excluded=request[:3]
        clones=tuple(e for e in entries if e['source'] in SOURCES)
        key=(image.shape,tuple(id(e) for e in clones),
             tuple(tuple(map(tuple,p)) for p in excluded))
        if key!=self.key:
            count=context_counts(image.shape,clones,excluded)
            self.entries=clones;self.counts=count;self.key=key;self.builds+=1
        return self.render((*request[:position+2],self.counts))
