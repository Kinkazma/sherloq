"""Luminance Gradient adapter: exact local Sobel, global normalization."""
import cv2 as cv
import numpy as np
from .memory_resources import TemporaryArrays,MiB
from .bounded_ops import map_local,normalize_u8,extrema,equalization_lut
from .utility import create_lut


def channels(image,store=None):
    store=store or TemporaryArrays(64*MiB);shape=image.shape[:2]
    dx=store.array(shape,np.int16);dy=store.array(shape,np.int16)
    map_local(image,lambda a:cv.spatialGradient(cv.cvtColor(a,cv.COLOR_BGR2GRAY)),(dx,dy),halo=1,store=store)
    mx=max(abs(x) for x in extrema(dx,store=store));my=max(abs(x) for x in extrema(dy,store=store))
    absolute=store.array(shape,np.float32)
    output=[store.array(shape,np.uint8) for _ in range(4)]
    for start in range(0,len(image),128):
        x=dx[start:start+128].astype(np.float32);y=dy[start:start+128].astype(np.float32)
        absolute[start:start+128]=np.abs(x)+np.abs(y)
        for dst,values,maximum,invert in zip(output,(x,y,x,y),(mx,my,mx,my),(False,False,True,True)):
            if maximum==0:dst[start:start+128]=127
            else:dst[start:start+128]=(((-values if invert else values)/maximum*127)+127).astype(np.uint8)
        store.checkpoint()
    return (*output,normalize_u8(absolute,store))


def compute(image,params):
    store=TemporaryArrays(64*MiB)
    percent,mode,invert,equalize=params;intensity=0 if equalize else int(percent/100*127)
    red,green,rr,rg,absolute=channels(image,store)
    if invert:red,green=rr,rg
    if mode==3:
        squared=store.array(red.shape,np.float64)
        for start in range(0,len(red),128):
            block=np.square(red[start:start+128],dtype=np.float64)
            block+=np.square(green[start:start+128],dtype=np.float64);np.sqrt(block,out=block)
            squared[start:start+128]=block;store.checkpoint()
        blue=normalize_u8(squared,store)
    elif mode==2:blue=absolute
    elif mode not in (0,1):raise ValueError('Unknown blue channel mode')
    result=store.array(image.shape,np.uint8)
    histograms=np.zeros((3,256),np.int64)
    for start in range(0,len(image),128):
        shape=red[start:start+128].shape
        b=blue[start:start+128] if mode in (2,3) else np.full(shape,255 if mode==1 else 0,np.uint8)
        block=cv.merge([b,green[start:start+128],red[start:start+128]])
        if equalize:
            for channel in range(3):histograms[channel]+=np.bincount(block[:,:,channel].ravel(),minlength=256)
        elif intensity>0:block=cv.LUT(block,create_lut(intensity,intensity))
        result[start:start+128]=block;store.checkpoint()
    if equalize:
        # Use OpenCV itself to create its exact global uint8 equalization LUT.
        # Counts can be huge, so implement its float32 scale / round-to-even.
        luts=[equalization_lut(histogram) for histogram in histograms]
        for start in range(0,len(image),128):
            block=result[start:start+128]
            result[start:start+128]=cv.merge([cv.LUT(block[:,:,c],luts[c]) for c in range(3)])
            store.checkpoint()
    store.checkpoint(force=True)
    return result
