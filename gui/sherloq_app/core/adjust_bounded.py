"""Adjustment stages with local halos and unchanged global histogram/CLAHE."""
import cv2 as cv
import numpy as np
from .memory_resources import TemporaryArrays,MiB,tiles
from .bounded_ops import map_local,equalize_u8


def reflect_indices(first,last,length):
    if length==1:return np.zeros(last-first,np.intp)
    indices=np.arange(first,last)%(2*(length-1))
    return np.minimum(indices,2*(length-1)-indices)


def clahe(values,level,store):
    h,w=values.shape
    # Match CLAHE's global 8x8 lattice, including its extra complete tile in
    # one dimension when only the other dimension needs padding.
    ph,pw=(h,w) if h%8==w%8==0 else (h+8-h%8,w+8-w%8)
    source=values
    if (ph,pw)!=(h,w):
        source=store.array((ph,pw),np.uint8)
        for dst,_,_ in tiles(source.shape,(512,512)):
            yy=reflect_indices(dst[0].start,dst[0].stop,h);xx=reflect_indices(dst[1].start,dst[1].stop,w)
            source[dst]=values[np.ix_(yy,xx)];store.checkpoint()
    out=store.array(source.shape,np.uint8)
    cv.createCLAHE((2,5,10,20)[level-2]).apply(source,out)
    store.checkpoint(force=True)
    return out[:h,:w]


def compute(image,params):
    from .adjust import AdjustEngine
    store=TemporaryArrays(32*MiB)
    bright,saturation,hue,gamma,shadows,highlights,sweep,width,sharp,threshold,equalize,invert=params
    local=(bright,saturation,hue,gamma,shadows,highlights,sweep,width,0,255,0,False)
    result=store.array(image.shape,np.uint8)
    source=image
    if sharp//4:
        def sharpen(roi):
            radius=sharp//4
            return cv.addWeighted(roi,1.5,cv.GaussianBlur(roi,(2*radius+1,2*radius+1),0),-.5,0)
        map_local(image,sharpen,(result,),halo=sharp//4,store=store)
        source=result
    # Keep point conversions on globally aligned 512-pixel columns. OpenCV's
    # SIMD and scalar HSV tails differ; including the sharpening halo in this
    # stage would shift those tails and change a few thresholded pixels.
    map_local(source,lambda roi:AdjustEngine(roi)._compute(local),(result,),store=store)
    if equalize:
        hsv=store.array(image.shape,np.uint8);v=store.array(image.shape[:2],np.uint8)
        map_local(result,lambda roi:cv.cvtColor(roi,cv.COLOR_BGR2HSV),(hsv,),store=store)
        map_local(hsv[:,:,2],lambda roi:roi,(v,),store=store)
        v=equalize_u8(v,store) if equalize==1 else clahe(v,equalize,store)
        for dst,_,_ in tiles(image.shape,(512,512)):
            block=np.array(hsv[dst]);block[:,:,2]=v[dst]
            result[dst]=cv.cvtColor(block,cv.COLOR_HSV2BGR);store.checkpoint()
        del hsv,v
    if threshold==0:
        gray=store.array(image.shape[:2],np.uint8)
        map_local(result,lambda roi:cv.cvtColor(roi,cv.COLOR_BGR2GRAY),(gray,),store=store)
        threshold,_=cv.threshold(gray,0,255,cv.THRESH_OTSU,gray)
        store.checkpoint(force=True);del gray
    if params[9]<255 or invert:
        def finish(roi):
            if params[9]<255:_,roi=cv.threshold(roi,threshold,255,cv.THRESH_BINARY)
            return cv.bitwise_not(roi) if invert else roi
        map_local(result,finish,(result,),store=store)
    store.checkpoint(force=True)
    return result
