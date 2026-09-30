"""Bounded regional histograms with the same full-image illuminant reference.

Cell origins, short boundary cells, integer histogram sums and floating point
moment order match the RAM engine. Storage pages never define analysis regions.
"""
import cv2 as cv
import numpy as np
from .illuminant import estimate
from .cloning import check
from .memory_resources import TemporaryArrays,MiB,tiles


def compute(engine,params,cancel=lambda:False,progress=lambda *a:None):
    settings,mode=params
    block,method,linear,exclude=settings
    if block not in (32,64,128,256) or method not in (0,1,2):
        raise ValueError('Invalid illuminant settings.')
    if mode not in (0,1,2):raise ValueError('Unknown illuminant view.')
    check(cancel)
    store=TemporaryArrays(64*MiB);store.watch(engine.image)
    h,w=engine.image.shape[:2];shape=((h+block-1)//block,(w+block-1)//block)
    result=engine.estimates.get(tuple(settings))
    if result is None:
        histogram=engine.histograms.get((block,exclude))
        if histogram is None:
            # Same resumable partial format as the RAM engine.
            if engine.partial is None or engine.partial[0]!=(block,exclude):
                engine.partial=[(block,exclude),0,store.array((*shape,3,256),np.uint32),
                                store.array(shape,np.uint32)]
            _,start,hist,areas=engine.partial;store.watch(hist,areas)
            for index in range(start,shape[0]*shape[1]):
                check(cancel);r,c=divmod(index,shape[1])
                roi=engine.image[r*block:min(h,(r+1)*block),c*block:min(w,(c+1)*block)]
                pixels=roi.reshape(-1,3);areas[r,c]=len(pixels)
                if exclude:pixels=pixels[(pixels.min(axis=1)>8)&(pixels.max(axis=1)<250)]
                for channel in range(3):hist[r,c,channel]=np.bincount(pixels[:,2-channel],minlength=256)
                engine.partial[1]=index+1
                if c==shape[1]-1:progress((r+1)*100//shape[0],'Estimating local illumination')
                store.checkpoint()
            check(cancel);engine.partial=None
            engine.histograms.put((block,exclude),(hist,areas))
        else:hist,areas=histogram;store.watch(hist,areas)
        rgb=store.array((*shape,3),np.float64);count=store.array(shape,np.uint64)
        valid=store.array(shape,np.bool_);angle=store.array(shape,np.float64)
        global_hist=np.zeros((3,256),np.uint64)
        for dst,_,_ in tiles(shape,(8,16)):
            check(cancel);piece=np.asarray(hist[dst]);colors,counts=estimate(piece,method,linear)
            rgb[dst]=colors;count[dst]=counts
            global_hist+=piece.sum(axis=(0,1),dtype=np.uint64)
            store.checkpoint()
        global_rgb,_=estimate(global_hist,method,linear)
        for dst,_,_ in tiles(shape,(32,64)):
            check(cancel);colors=rgb[dst].copy()
            keep=(count[dst]>=np.minimum(16,areas[dst]))&(np.max(colors,axis=-1)>0)
            colors[~keep]=0;rgb[dst]=colors;valid[dst]=keep
            sine=np.sqrt(np.sum(np.cross(colors,global_rgb)**2,axis=-1))
            cosine=np.sum(colors*global_rgb,axis=-1)
            angles=np.degrees(np.arctan2(sine,cosine));angles[~keep]=0;angle[dst]=angles
            store.checkpoint()
        result=(rgb,count,areas,valid,global_rgb,angle)
        engine.estimates.put(tuple(settings),result)
    store.watch(result)
    key=(*settings,mode);output=engine.renders.get(key)
    if output is None:
        rgb,count,areas,valid,global_rgb,angle=result
        output=store.array(engine.image.shape,np.uint8)
        # Expand only a few globally anchored cells at a time. Do not stretch
        # the last row/column when the image dimensions are not block multiples.
        step=max(1,512//block)
        for dst,_,_ in tiles(shape,(step,step)):
            check(cancel);colors=np.asarray(rgb[dst])
            if mode==0:
                maximum=np.max(colors,axis=-1,keepdims=True)
                normalized=np.divide(colors,maximum,out=np.zeros_like(colors),where=maximum>0)
                if linear:normalized=np.where(normalized<=.0031308,12.92*normalized,1.055*normalized**(1/2.4)-.055)
                pixels=np.rint(np.clip(normalized,0,1)*255).astype(np.uint8)[:,:,::-1].copy()
            elif mode==1:
                pixels=cv.applyColorMap(np.rint(np.clip(angle[dst]/90,0,1)*255).astype(np.uint8),cv.COLORMAP_VIRIDIS)
            else:pixels=cv.cvtColor(np.rint(count[dst]/areas[dst]*255).astype(np.uint8),cv.COLOR_GRAY2BGR)
            if mode!=2:pixels[~valid[dst]]=(64,64,64)
            y,x=dst[0].start*block,dst[1].start*block
            expanded=np.repeat(np.repeat(pixels,block,axis=0),block,axis=1)
            eh,ew=min(h-y,len(expanded)),min(w-x,expanded.shape[1])
            output[y:y+eh,x:x+ew]=expanded[:eh,:ew];store.checkpoint()
        engine.renders.put(key,output)
    check(cancel);store.checkpoint(force=True)
    return output,result
