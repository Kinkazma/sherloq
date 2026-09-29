"""Local Laplacian with exact, global channel normalization."""
import cv2 as cv
import numpy as np
from .bounded_ops import map_local,normalize_u8
from .memory_resources import TemporaryArrays,MiB
from .utility import create_lut,bgr_to_gray3


def compute(image,params):
    radius,contrast,gray=params;store=TemporaryArrays(32*MiB)
    result=store.array(image.shape,np.uint8)
    for channel in range(3):
        derivative=store.array(image.shape[:2],np.float64)
        def one(roi):
            d=cv.Laplacian(roi,cv.CV_64F,ksize=2*radius+1)
            np.abs(d,out=d);return d
        map_local(image[:,:,channel],one,(derivative,),halo=radius,store=store)
        normalized=normalize_u8(derivative,store,direct=True)
        map_local(normalized,lambda roi:roi,(result[:,:,channel],),store=store)
        del derivative,normalized
    lut=create_lut(0,int(contrast/100*255))
    def display(roi):
        out=cv.LUT(roi,lut)
        return bgr_to_gray3(out) if gray else out
    map_local(result,display,(result,),store=store)
    store.checkpoint(force=True)
    return result
