"""Extrema with halos; density blocks anchored to global image coordinates."""
import cv2 as cv
import numpy as np
from .memory_resources import TemporaryArrays,MiB,tiles
from .bounded_ops import map_local,normalize_u8
from .minmax import MinMaxEngine,_std_table,_COLORS


def density(mask,radius,store):
    h,w=mask.shape;block=2*radius+1
    result=store.array(mask.shape,np.uint8,zero=True)
    if min(h,w)<=radius:return result
    values=store.array(((h+block-1)//block,(w+block-1)//block),np.float32,zero=True)
    # Each batch contains complete density cells, including original short
    # right/bottom cells. Never restart the cell grid at an arbitrary tile.
    for dst,_,_ in tiles(mask.shape,(block*32,block*32)):
        roi=mask[dst];rh,rw=roi.shape
        ys=np.arange(0,rh,block);xs=np.arange(0,rw,block)
        counts=np.add.reduceat(np.add.reduceat(roi,ys,axis=0,dtype=np.uint32),xs,axis=1,dtype=np.uint32)
        heights=np.minimum(block,rh-ys);widths=np.minimum(block,rw-xs)
        areas=heights[:,None]*widths[None,:];valid=(heights[:,None]>radius)&(widths[None,:]>radius)
        v=np.zeros(counts.shape,np.float32)
        for size in np.unique(areas[valid]):
            selected=valid&(areas==size);v[selected]=_std_table(int(size))[counts[selected]]
        gy,gx=dst[0].start//block,dst[1].start//block
        values[gy:gy+len(ys),gx:gx+len(xs)]=v;store.checkpoint()
    small=normalize_u8(values,store,maximum=127,direct=True)
    for dst,_,_ in tiles(mask.shape,(block*32,block*32)):
        y,x=dst[0].start,dst[1].start;rh=dst[0].stop-y;rw=dst[1].stop-x
        v=small[y//block:(y+rh+block-1)//block,x//block:(x+rw+block-1)//block]
        result[dst]=np.repeat(np.repeat(v,block,axis=0),block,axis=1)[:rh,:rw];store.checkpoint()
    return result


def compute(image,params):
    channel,minimum,maximum,radius=params;store=TemporaryArrays(32*MiB)
    low=store.array(image.shape[:2],bool);high=store.array(image.shape[:2],bool)
    map_local(image,lambda roi:MinMaxEngine(roi).extrema(channel),(low,high),halo=1,store=store)
    result=store.array(image.shape,np.uint8,zero=True)
    if radius:
        for mask,color in ((low,minimum),(high,maximum)):
            if color==4:continue
            plane=density(mask,radius+3,store)
            for dst,_,_ in tiles(image.shape,(512,512)):
                if color<=2:result[dst][:,:,2-color]+=plane[dst]
                else:result[dst]+=plane[dst][:,:,None]
                store.checkpoint()
            del plane
        result=normalize_u8(result,store)
    else:
        palette=np.zeros((256,1,3),np.uint8);palette[1,0]=_COLORS[minimum];palette[2,0]=_COLORS[maximum]
        for dst,_,_ in tiles(image.shape,(512,512)):
            code=high[dst].astype(np.uint8);code*=2;code+=low[dst]
            result[dst]=cv.applyColorMap(code,palette);store.checkpoint()
    store.checkpoint(force=True)
    return result,low,high
