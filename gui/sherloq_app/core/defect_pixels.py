"""Conservative isolated-pixel candidates on decoded uint8 images.

A candidate channel lies outside all neighbours by at least a user threshold,
while that neighbourhood's range is small. This does not diagnose a sensor.
"""
import csv,os,tempfile
from collections import OrderedDict
from pathlib import Path
import cv2 as cv
import numpy as np
from .interactive import ArrayCache
from .jpeg_curve import Cancelled


class DefectEngine:
    def __init__(self,image,backend='auto'):
        self.image=image;self.backend=backend
        self.bases=ArrayCache(384);self.masks=ArrayCache(128);self.renders=ArrayCache(192)
        self.last_backend='CPU'
        self.counts=OrderedDict()
        self.pixel_masks=ArrayCache(64)

    def prepare(self,radius,cpu,cancel=lambda:False):
        gpu=not cpu and radius==2 and (self.backend=='gpu' or self.image.shape[0]*self.image.shape[1]>=8_000_000) and cv.ocl.haveOpenCL() and cv.ocl.useOpenCL()
        key=radius,bool(gpu);value=self.bases.get(key)
        if value is not None:
            self.last_backend='GPU' if gpu else 'CPU';return value
        self.bases.reserve(self.image.nbytes*3)
        kernel=np.ones((radius*2+1,)*2,np.uint8);kernel[radius,radius]=0
        def compute(use_gpu):
            source=cv.UMat(self.image) if use_gpu else self.image
            low=cv.erode(source,kernel)
            if cancel():raise Cancelled()
            high=cv.dilate(source,kernel)
            if cancel():raise Cancelled()
            median=cv.medianBlur(source,radius*2+1)
            if use_gpu:low,high,median=low.get(),high.get(),median.get()
            return low,high,median
        try:value=compute(gpu)
        except cv.error:
            if not gpu:raise
            gpu=False;key=radius,False;value=compute(False)
        if cancel():raise Cancelled()
        self.last_backend='GPU' if gpu else 'CPU'
        return self.bases.put(key,value)

    def analyze(self,settings,cancel=lambda:False,progress=lambda *a:None):
        radius,threshold,spread,kind,cpu=settings
        if radius not in (1,2) or not 1<=threshold<=255 or not 0<=spread<=255 or kind not in (0,1,2):raise ValueError('Invalid pixel-defect settings.')
        if cancel():raise Cancelled()
        found=self.masks.get(settings)
        if found is not None:return found
        low,high,median=self.prepare(radius,cpu,cancel)
        h,w=self.image.shape[:2];self.masks.reserve(self.image.nbytes)
        flags=np.zeros_like(self.image)
        # Bounded stripes keep masks/intermediates small and allow cancellation.
        for first in range(radius,max(radius,h-radius),256):
            if cancel():raise Cancelled()
            last=min(first+256,h-radius);ys=slice(first,last);xs=slice(radius,max(radius,w-radius))
            source=self.image[ys,xs];lo=low[ys,xs];hi=high[ys,xs]
            if source.size:
                smooth=cv.subtract(hi,lo)<=spread
                if kind!=2:flags[ys,xs][smooth&(cv.subtract(source,hi)>=threshold)]=1
                if kind!=1:flags[ys,xs][smooth&(cv.subtract(lo,source)>=threshold)]=2
            progress(last*100//max(1,h),'Finding isolated pixel candidates')
        if cancel():raise Cancelled()
        return self.masks.put(settings,flags)

    def pixels(self,settings,flags):
        cached=self.pixel_masks.get(settings)
        if cached is not None:return cached
        self.pixel_masks.reserve(flags.shape[0]*flags.shape[1])
        return self.pixel_masks.put(settings,flags[:,:,0]|flags[:,:,1]|flags[:,:,2])

    def render(self,settings,mode,flags,cancel=lambda:False):
        key=(*settings,mode);cached=self.renders.get(key)
        if cached is not None:return cached
        self.renders.reserve(self.image.nbytes)
        output=self.image.copy() if mode in (0,2) else np.zeros_like(self.image)
        if mode==2:
            median=self.prepare(settings[0],settings[4],cancel)[2]
            for first in range(0,len(output),256):
                if cancel():raise Cancelled()
                region=slice(first,first+256);mask=flags[region]!=0;output[region][mask]=median[region][mask]
        elif mode in (0,1):
            pixels=self.pixels(settings,flags)
            colours=np.array([[0,0,0],[0,0,255],[255,0,0],[255,0,255]],np.uint8)
            for first in range(0,len(output),256):
                if cancel():raise Cancelled()
                region=slice(first,first+256);code=pixels[region];selected=code!=0
                output[region][selected]=colours[code[selected]]
        else:raise ValueError('Unknown defect-pixel view.')
        return self.renders.put(key,output)

    def compute(self,params,cancel=lambda:False,progress=lambda *a:None):
        settings,mode=params;flags=self.analyze(settings,cancel,progress);output=self.render(settings,mode,flags,cancel)
        count=self.counts.get(settings)
        if count is None:
            count=0
            for i in range(0,len(flags),256):
                if cancel():raise Cancelled()
                count+=np.count_nonzero(self.pixels(settings,flags)[i:i+256])
            self.counts[settings]=int(count)
            if len(self.counts)>128:self.counts.popitem(last=False)
        if cancel():raise Cancelled()
        median=self.prepare(settings[0],settings[4],cancel)[2]
        return output,flags,int(count),median


def export_csv(path,settings,image,flags,median,cancel=lambda:False):
    radius,threshold,spread,kind,cpu=settings;destination=Path(path);temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',newline='',dir=destination.parent,prefix='.defect-pixels-',suffix='.csv',delete=False) as stream:
            temporary=Path(stream.name);writer=csv.writer(stream)
            writer.writerow(['x','y','channel','candidate','original_value','replacement_value','radius','minimum_deviation','maximum_neighbour_range'])
            for y in range(image.shape[0]):
                if cancel():raise Cancelled()
                columns,channels=np.nonzero(flags[y])
                for x,c in zip(columns,channels):writer.writerow([int(x),y,'BGR'[c],'hot' if flags[y,x,c]==1 else 'dead',int(image[y,x,c]),int(median[y,x,c]),radius,threshold,spread])
        if cancel():raise Cancelled()
        os.replace(temporary,destination)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
