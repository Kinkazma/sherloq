"""Reusable bounded image operations; global statistics are not per-tile.

Only explicitly finite-support operators may use map_local. Full FFTs, learned
attention and clone matching require their own global execution plan.
"""
import cv2 as cv
import numpy as np
from .memory_resources import tiles


def map_local(image,operator,outputs,*,halo=0,tile_shape=(512,512),store,cancel=lambda:False):
    from .cloning import check
    store.watch(image,outputs)
    for dst,src,crop in tiles(image.shape,tile_shape,halo):
        check(cancel);values=operator(np.ascontiguousarray(image[src]))
        if not isinstance(values,tuple):values=(values,)
        if len(values)!=len(outputs):raise ValueError('Local operator output count mismatch')
        for out,value in zip(outputs,values):out[dst]=value[crop]
        store.checkpoint()
    return outputs


def extrema(array,*,rows=256,store=None):
    if store is not None:store.watch(array)
    low=float('inf');high=-float('inf')
    for dst,_,_ in tiles(array.shape,(rows,512)):
        block=array[dst];low=min(low,float(block.min()));high=max(high,float(block.max()))
        if store is not None:store.checkpoint()
    return low,high


def normalize_u8(array,store,*,rows=128,maximum=255,direct=False):
    """Same cv.normalize conversion using global extrema in every block."""
    low,high=extrema(array,store=store);result=store.array(array.shape,np.uint8)
    for dst,_,_ in tiles(array.shape,(rows,512)):
        block=array[dst]
        # Sentinel extrema preserve OpenCV's exact convertTo scale, offset and
        # integer rounding even for a constant local block or nonzero minimum.
        values=np.empty(block.size+2,array.dtype);values[:block.size]=block.ravel();values[-2:]=(low,high)
        out=cv.normalize(values,None,0,maximum,cv.NORM_MINMAX,
                         dtype=cv.CV_8U if direct else -1).astype(np.uint8)
        result[dst]=out[:block.size].reshape(block.shape)
        store.checkpoint()
    return result


def equalization_lut(histogram):
    """OpenCV uint8 equalization using the histogram of the entire image."""
    first=int(np.flatnonzero(histogram)[0]);total=int(histogram.sum())
    lut=np.zeros(256,np.uint8)
    if histogram[first]==total:lut[:]=first
    else:
        scale=np.float32(255)/np.float32(total-int(histogram[first]));cumulative=0
        for i in range(first+1,256):
            cumulative+=int(histogram[i])
            lut[i]=np.uint8(np.clip(np.rint(np.float32(cumulative)*scale),0,255))
    return lut


def equalize_u8(array,store):
    store.watch(array)
    channels=1 if array.ndim==2 else array.shape[2]
    hist=np.zeros((channels,256),np.int64)
    for dst,_,_ in tiles(array.shape,(512,512)):
        block=array[dst].reshape(-1,channels)
        for c in range(channels):hist[c]+=np.bincount(block[:,c],minlength=256)
        store.checkpoint()
    luts=[equalization_lut(h) for h in hist];out=store.array(array.shape,np.uint8)
    for dst,_,_ in tiles(array.shape,(512,512)):
        block=array[dst]
        out[dst]=cv.LUT(block,luts[0]) if channels==1 else cv.merge([cv.LUT(block[:,:,c],luts[c]) for c in range(channels)])
        store.checkpoint()
    return out


def local_result(image,operator,*,halo=0,dtypes=(np.uint8,),store=None,cancel=lambda:False):
    """BGR results of a local operator, backed by the shared spill mechanism."""
    from .memory_resources import TemporaryArrays,MiB
    store=store or TemporaryArrays(64*MiB)
    outputs=tuple(store.array(image.shape,dtype) for dtype in dtypes)
    map_local(image,operator,outputs,halo=halo,store=store,cancel=cancel)
    store.checkpoint(force=True)
    return outputs
