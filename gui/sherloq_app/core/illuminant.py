"""Local Minkowski illuminant colour estimates, not light directions.

New analysis contract: regional uint8 histograms, optional inverse-sRGB,
float64 moments in increasing-bin order, unit RGB illuminant vectors. No
claim that object colour can be separated unambiguously from illumination.
"""
import csv
import os
import tempfile
from pathlib import Path
import cv2 as cv
import numpy as np
from .interactive import ArrayCache
from .jpeg_curve import Cancelled


def transfer_table(linear):
    x=np.arange(256,dtype=np.float64)/255
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4) if linear else x


def estimate(histogram,method,linear):
    table=transfer_table(linear)
    count=histogram.sum(axis=-1,dtype=np.uint64)[...,0]
    if method==2:
        bins=np.max(np.where(histogram>0,np.arange(256),0),axis=-1)
        moment=table[bins]
    else:
        p=1 if method==0 else 6
        # Explicit last-axis sum avoids BLAS-dependent dot product dispatch.
        moment=np.sum(histogram*(table**p),axis=-1)/np.maximum(count[...,None],1)
        moment=moment**(1/p)
    norm=np.sqrt(np.sum(moment**2,axis=-1))
    return np.divide(moment,norm[...,None],out=np.zeros_like(moment),where=norm[...,None]>0),count


class IlluminantEngine:
    def __init__(self,image):
        if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:raise ValueError('Expected an 8-bit BGR image.')
        self.image=image
        self.histograms=ArrayCache(192)
        self.estimates=ArrayCache(32)
        self.renders=ArrayCache(192)
        self.partial=None

    def histogram(self,block,exclude,cancel=lambda:False,progress=lambda *a:None):
        key=block,exclude
        cached=self.histograms.get(key)
        if cached is not None:return cached
        h,w=self.image.shape[:2];nr,nc=(h+block-1)//block,(w+block-1)//block
        if self.partial is None or self.partial[0]!=key:
            self.histograms.reserve(nr*nc*3*256*4)
            self.partial=[key,0,np.zeros((nr,nc,3,256),np.uint32),np.zeros((nr,nc),np.uint32)]
        _,start,hist,areas=self.partial
        for index in range(start,nr*nc):
            if cancel():raise Cancelled()
            r,c=divmod(index,nc);roi=self.image[r*block:min(h,(r+1)*block),c*block:min(w,(c+1)*block)]
            pixels=roi.reshape(-1,3);areas[r,c]=len(pixels)
            if exclude:pixels=pixels[(pixels.min(axis=1)>8)&(pixels.max(axis=1)<250)]
            for channel in range(3):hist[r,c,channel]=np.bincount(pixels[:,2-channel],minlength=256)
            self.partial[1]=index+1
            if c==nc-1:progress((r+1)*100//nr,'Estimating local illumination')
        if cancel():raise Cancelled()
        self.partial=None
        return self.histograms.put(key,(hist,areas))

    def analyze(self,block,method,linear,exclude,cancel=lambda:False,progress=lambda *a:None):
        if block not in (32,64,128,256) or method not in (0,1,2):raise ValueError('Invalid illuminant settings.')
        if cancel():raise Cancelled()
        key=block,method,linear,exclude
        cached=self.estimates.get(key)
        if cached is not None:return cached
        hist,areas=self.histogram(block,exclude,cancel,progress)
        rgb,count=estimate(hist,method,linear)
        global_rgb,_=estimate(hist.sum(axis=(0,1),dtype=np.uint64),method,linear)
        valid=(count>=np.minimum(16,areas))&(np.max(rgb,axis=-1)>0)
        rgb[~valid]=0
        # atan2 is stable near zero, unlike acos of an almost-unit dot product.
        sine=np.sqrt(np.sum(np.cross(rgb,global_rgb)**2,axis=-1))
        cosine=np.sum(rgb*global_rgb,axis=-1)
        angle=np.degrees(np.arctan2(sine,cosine));angle[~valid]=0
        value=(rgb,count,areas,valid,global_rgb,angle)
        if cancel():raise Cancelled()
        return self.estimates.put(key,value)

    def render(self,settings,mode,result):
        key=(*settings,mode);cached=self.renders.get(key)
        if cached is not None:return cached
        rgb,count,areas,valid,global_rgb,angle=result
        if mode==0:
            normalized=np.divide(rgb,np.max(rgb,axis=-1,keepdims=True),out=np.zeros_like(rgb),where=np.max(rgb,axis=-1,keepdims=True)>0)
            if settings[2]:normalized=np.where(normalized<=.0031308,12.92*normalized,1.055*normalized**(1/2.4)-.055)
            pixels=np.rint(np.clip(normalized,0,1)*255).astype(np.uint8)[:,:,::-1].copy()
        elif mode==1:
            pixels=cv.applyColorMap(np.rint(np.clip(angle/90,0,1)*255).astype(np.uint8),cv.COLORMAP_VIRIDIS)
        elif mode==2:
            values=np.rint(count/areas*255).astype(np.uint8);pixels=cv.cvtColor(values,cv.COLOR_GRAY2BGR)
        else:raise ValueError('Unknown illuminant view.')
        if mode!=2:pixels[~valid]=(64,64,64)
        h,w=self.image.shape[:2];block=settings[0]
        self.renders.reserve(self.image.nbytes)
        # Repeat then crop, never stretch a short boundary cell to full width.
        output=np.repeat(np.repeat(pixels,block,axis=0),block,axis=1)[:h,:w].copy()
        return self.renders.put(key,output)

    def compute(self,params,cancel=lambda:False,progress=lambda *a:None):
        settings,mode=params
        result=self.analyze(*settings,cancel,progress)
        output=self.render(settings,mode,result)
        if cancel():raise Cancelled()
        return output,result


def export_csv(path,settings,result,image_shape,cancel=lambda:False):
    rgb,count,areas,valid,global_rgb,angles=result;block,method,linear,exclude=settings
    destination=Path(path);temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',newline='',dir=destination.parent,prefix='.illuminant-',suffix='.csv',delete=False) as stream:
            temporary=Path(stream.name);writer=csv.writer(stream)
            writer.writerow(['x','y','width','height','valid_pixels','total_pixels','valid_estimate','R_unit','G_unit','B_unit','angle_to_global_degrees','global_R_unit','global_G_unit','global_B_unit','method','linearize_srgb','exclude_dark_clipped'])
            for r,c in np.ndindex(valid.shape):
                if cancel():raise Cancelled()
                writer.writerow([c*block,r*block,min(block,image_shape[1]-c*block),min(block,image_shape[0]-r*block),int(count[r,c]),int(areas[r,c]),int(valid[r,c]),*rgb[r,c],angles[r,c],*global_rgb,['Gray World','Shades of Gray p=6','White Patch'][method],int(linear),int(exclude)])
        if cancel():raise Cancelled()
        os.replace(temporary,destination)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
