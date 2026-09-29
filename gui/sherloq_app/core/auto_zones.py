"""Find rectangular panels separated by flat coloured gutters, without a panel count prior."""
import cv2 as cv
import numpy as np
from .cloning import check


def detect_panels(image,cancel=lambda:False):
    h,w=image.shape[:2]
    # Candidate colours from flat pixels; no assumption of white or black.
    step=max(1,int(np.ceil(max(h,w)/1800)))
    small=image[::step,::step]
    flat=np.max(np.abs(small.astype(np.int16)-np.roll(small,1,0).astype(np.int16)),axis=2)<=2
    flat&=np.max(np.abs(small.astype(np.int16)-np.roll(small,1,1).astype(np.int16)),axis=2)<=2
    samples=small[flat]
    if not len(samples):return ()
    quant=samples//8;codes=quant[:,0].astype(int)*1024+quant[:,1].astype(int)*32+quant[:,2]
    ids,counts=np.unique(codes,return_counts=True);order=np.argsort(-counts,kind='stable')[:8]
    proposals=[]
    for code in ids[order]:
        check(cancel);color=np.median(samples[codes==code],axis=0).astype(np.int16)
        mask=(np.max(np.abs(image.astype(np.int16)-color),axis=2)>14).astype(np.uint8)
        mask=cv.morphologyEx(mask,cv.MORPH_CLOSE,np.ones((3,3),np.uint8))
        n,labels,stats,_=cv.connectedComponentsWithStats(mask,connectivity=8)
        for x,y,bw,bh,area in stats[1:]:
            if bw<max(24,w*.012) or bh<max(16,h*.012) or bw*bh<h*w*.0005:continue
            if bw*bh>h*w*.95 or area/(bw*bh)<.82:continue
            # Each side must largely meet a flat gutter or the outer image edge.
            borders=[]
            for strip in (image[max(0,y-2):y,x:x+bw],image[y+bh:min(h,y+bh+2),x:x+bw],
                          image[y:y+bh,max(0,x-2):x],image[y:y+bh,x+bw:min(w,x+bw+2)]):
                borders.append(1. if not strip.size else np.mean(np.max(np.abs(strip.astype(np.int16)-color),axis=2)<=14))
            if min(borders)<.65:continue
            proposals.append((int(x),int(y),int(bw),int(bh),float(area/(bw*bh))))
    # Keep largest non-overlapping panels; repeated colour candidates merge.
    selected=[]
    for p in sorted(proposals,key=lambda p:(-p[2]*p[3],p[1],p[0])):
        x,y,bw,bh,_=p
        overlap=False
        for a,b,c,d,_ in selected:
            inter=max(0,min(x+bw,a+c)-max(x,a))*max(0,min(y+bh,b+d)-max(y,b))
            if inter/min(bw*bh,c*d)>.2:overlap=True;break
        if not overlap:selected.append(p)
    rows=[]
    for panel in sorted(selected,key=lambda p:(p[1],p[0])):
        if not rows or panel[1]-rows[-1][0][1]>max(4,h*.02):rows.append([])
        rows[-1].append(panel)
    selected=[p for row in rows for p in sorted(row,key=lambda p:p[0])]
    return tuple(tuple((float(a),float(b)) for a,b in ((x,y),(x+bw-1,y),(x+bw-1,y+bh-1),(x,y+bh-1))) for x,y,bw,bh,_ in selected)


def enclosing(regions):
    if not regions:return None
    xy=np.concatenate(regions);lo=xy.min(0);hi=xy.max(0)
    return tuple(map(tuple,[(float(lo[0]),float(lo[1])),(float(hi[0]),float(lo[1])),(float(hi[0]),float(hi[1])),(float(lo[0]),float(hi[1]))]))


def diagonal(region):
    xy=np.asarray(region);return float(np.linalg.norm(xy.max(0)-xy.min(0)))


def gap_vector(regions):
    """Translate zone 2 towards zone 1 only across the empty bbox intervals."""
    if len(regions)!=2:return (0.,0.)
    a,b=map(np.asarray,regions);alo,ahi=a.min(0),a.max(0);blo,bhi=b.min(0),b.max(0)
    # One pixel separates neighbouring integer pixel centres after closing a gutter.
    gap=np.where(blo>ahi,np.maximum(0,blo-ahi-1),np.where(alo>bhi,-np.maximum(0,alo-bhi-1),0))
    return tuple(map(float,gap))


def distance_policy(params,regions,compare):
    radius=float(params[2]);auto=len(params)>14 and params[14];compact=len(params)>15 and params[15]
    gap=gap_vector(regions) if compare and compact else (0.,0.)
    radii=tuple(max(.01,diagonal(r)) if auto else radius for r in regions) or (radius,)
    if compare and auto:
        a=np.asarray(regions[0]);b=np.asarray(regions[1])-gap
        radius=max(.01,diagonal(np.concatenate([a,b])))
    elif auto and regions:radius=max(radii)
    return radius,radii,gap


def compact_axes(shape,regions):
    """Remove only axis intervals outside every panel bbox; keep original pixels."""
    if not regions:return None
    output=[]
    for axis,length in ((0,shape[1]),(1,shape[0])):
        occupied=np.zeros(length,bool)
        for region in regions:
            values=np.asarray(region)[:,axis]
            lo=max(0,int(np.floor(values.min())));hi=min(length,int(np.ceil(values.max()))+1)
            occupied[lo:hi]=True
        # Consecutive occupied pixel centres become adjacent across a gutter.
        output.append((np.cumsum(occupied)-1).astype(np.float32))
    return tuple(output)


def compact_points(points,axes):
    if axes is None:return points
    return np.column_stack([np.interp(points[:,i],np.arange(len(axes[i])),axes[i]) for i in (0,1)])
