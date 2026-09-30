"""Bounded contrast maps on the unchanged, globally anchored padded grid."""
import cv2 as cv
import numpy as np
from .contrast import histogram_error
from .cloning import check
from .memory_resources import TemporaryArrays,MiB,tiles
from .bounded_ops import map_local


def compute(engine,params,cancel=lambda:False,progress=lambda *a:None):
    block,mode=params
    if block not in (32,64,128,256):raise ValueError('Block size must be 32, 64, 128 or 256')
    if mode not in (0,1,2):raise ValueError('Unknown contrast indicator')
    check(cancel);cached=engine.renders.get((block,mode))
    if cached is not None:return cached
    store=TemporaryArrays(64*MiB);store.watch(engine.image)
    h,w=engine.image.shape[:2];ph,pw=h+block-h%block,w+block-w%block
    nr,nc=ph//block,pw//block;maps=engine.maps.get(block)
    if maps is None:
        if engine.partial is None or engine.partial[0]!=block:
            arrays=tuple(store.array((nr+1,nc+1),np.float32) for _ in range(3))
            for array in arrays:array[:]=0
            engine.partial=[block,0,arrays]
        _,first,arrays=engine.partial;store.watch(arrays)
        kx,ky=cv.getDerivKernels(1,1,1)
        for index in range(first,nr*nc):
            check(cancel);r,c=divmod(index,nc)
            y,x=r*block,c*block;y0,x0=max(0,y-1),max(0,x-1)
            y1,x1=min(ph,y+block+1),min(pw,x+block+1)
            color=np.zeros((y1-y0,x1-x0,3),np.uint8)
            sh,sw=max(0,min(h,y1)-y0),max(0,min(w,x1)-x0)
            color[:sh,:sw]=engine.image[y0:y0+sh,x0:x0+sw]
            crop=(slice(y-y0,y-y0+block),slice(x-x0,x-x0+block))
            gray=cv.cvtColor(color,cv.COLOR_BGR2GRAY)[crop]
            bd,gd,rd=[cv.sepFilter2D(ch,cv.CV_32F,kx,ky)[crop] for ch in cv.split(color)]
            tri=(np.abs(gd-rd)+np.abs(gd-bd)+np.abs(rd-bd))/3
            avg=(np.abs(bd)+np.abs(gd)+np.abs(rd))/3
            error=histogram_error(gray);avg_m=np.mean(avg)
            chsim=0 if avg_m==0 else np.mean(tri)/avg_m
            chsim=1 if chsim>.75 else chsim/.75
            arrays[0][r,c],arrays[1][r,c],arrays[2][r,c]=error,chsim,error*chsim
            engine.partial[1]=index+1
            if c==nc-1:progress((r+1)*100//nr,'Analyzing contrast blocks')
            store.checkpoint()
        check(cancel);engine.partial=None;maps=engine.maps.put(block,arrays)
    store.watch(maps)
    plane=store.array(maps[mode].shape,np.uint8)
    # convertScaleAbs then median use a one-cell halo, preserving the extra
    # historical zero row/column before filtering.
    map_local(maps[mode],lambda x:cv.medianBlur(cv.convertScaleAbs(x,None,255),3),
              (plane,),halo=1,store=store,cancel=cancel)
    output=store.array(engine.image.shape,np.uint8);step=max(1,512//block)
    for dst,_,_ in tiles((nr,nc),(step,step)):
        check(cancel);expanded=np.repeat(np.repeat(plane[dst],block,axis=0),block,axis=1)
        y,x=dst[0].start*block,dst[1].start*block
        eh,ew=min(h-y,expanded.shape[0]),min(w-x,expanded.shape[1])
        if eh>0 and ew>0:output[y:y+eh,x:x+ew]=expanded[:eh,:ew,None]
        store.checkpoint()
    store.checkpoint(force=True)
    return engine.renders.put((block,mode),output)
