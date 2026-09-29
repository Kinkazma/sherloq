"""Exact retained stereogram offset search and lazy optical-flow views."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import cv2 as cv
import numpy as np
from .jpeg_curve import Cancelled
from .interactive import ArrayCache
from .utility import norm_img,norm_mat,gray_to_bgr


class StereoEngine:
    def __init__(self,image):
        self.image=image;self.small=None;self.difference=None;self.done=None
        self.searched=False;self.offset=None;self.flow=None;self.views=ArrayCache(256)
        self.workers=4

    def search(self,cancel=lambda:False,progress=lambda *a:None):
        if cancel():raise Cancelled()
        if self.searched:return self.offset
        h,w=self.image.shape[:2]
        # The original needs at least two tested offsets and a nonempty resize.
        if h<2 or w//3<=11:self.searched=True;return None
        if self.small is None:
            self.small=cv.resize(cv.cvtColor(self.image,cv.COLOR_BGR2GRAY),None,None,1,.5)
            self.difference=np.zeros(w//3-10,np.float32);self.done=np.zeros(len(self.difference),bool)
        def one(index):
            if cancel():raise Cancelled()
            offset=int(index)+10
            value=cv.mean(cv.absdiff(self.small[:,offset:],self.small[:,:-offset]))[0]
            return int(index),value
        missing=np.flatnonzero(~self.done)
        workers=self.workers if self.image.shape[0]*self.image.shape[1]>=250_000 else 1
        with ThreadPoolExecutor(workers,thread_name_prefix='stereo-offset') as pool:
            for start in range(0,len(missing),32):
                if cancel():raise Cancelled()
                futures=[pool.submit(one,index) for index in missing[start:start+32]]
                try:
                    for future in as_completed(futures):
                        index,value=future.result();self.difference[index]=value;self.done[index]=True
                finally:
                    if cancel():
                        for future in futures:future.cancel()
                progress(int(np.count_nonzero(self.done)*100/len(self.done)),'Finding stereogram offset')
        if cancel():raise Cancelled()
        _,maximum,_,argmax=cv.minMaxLoc(np.ediff1d(self.difference))
        self.offset=argmax[1]+10 if maximum>=2 else None
        self.searched=True
        return self.offset

    def pattern(self,cancel=lambda:False,progress=lambda *a:None):
        cached=self.views.get(0)
        if cached is not None:return cached
        offset=self.search(cancel,progress)
        if offset is None:return None
        if cancel():raise Cancelled()
        self.views.reserve(self.image.shape[0]*(self.image.shape[1]-offset)*3)
        return self.views.put(0,norm_img(cv.absdiff(self.image[:,offset:],self.image[:,:-offset])))

    def optical_flow(self,cancel=lambda:False,progress=lambda *a:None):
        if self.flow is None:
            if cancel():raise Cancelled()
            progress(0,'Estimating relative disparity')
            left=cv.cvtColor(self.image[:,self.offset:],cv.COLOR_BGR2GRAY)
            right=cv.cvtColor(self.image[:,:-self.offset],cv.COLOR_BGR2GRAY)
            # Same CPU function/parameters as the original, no scale reduction.
            self.flow=cv.calcOpticalFlowFarneback(left,right,None,.5,5,15,5,5,1.2,cv.OPTFLOW_FARNEBACK_GAUSSIAN)[:,:,0].copy()
            if cancel():raise Cancelled()
        return self.flow

    def compute(self,mode,cancel=lambda:False,progress=lambda *a:None):
        if mode not in (0,1,2,3):raise ValueError('Unknown stereogram view.')
        if cancel():raise Cancelled()
        cached=self.views.get(mode)
        if cached is not None:return cached
        pattern=self.pattern(cancel,progress)
        if pattern is None:return None
        if mode==0:return pattern
        self.views.reserve(pattern.nbytes)
        if mode==1:
            gray=cv.cvtColor(pattern,cv.COLOR_BGR2GRAY);threshold,_=cv.threshold(gray,0,255,cv.THRESH_TRIANGLE)
            output=cv.medianBlur(gray_to_bgr(cv.threshold(gray,threshold,255,cv.THRESH_BINARY)[1]),3)
        else:
            flow=self.optical_flow(cancel,progress)
            if mode==2:output=gray_to_bgr(norm_mat(flow))
            else:
                # Broadcasting avoids allocating three copies of normalized
                # flow. The elementwise multiply/normalization stays float32.
                normal=cv.normalize(flow,None,0,1,cv.NORM_MINMAX)
                shaded=pattern.astype(np.float32)*normal[:,:,None]
                output=cv.normalize(shaded,None,0,255,cv.NORM_MINMAX).astype(np.uint8)
        if cancel():raise Cancelled()
        return self.views.put(mode,output)
