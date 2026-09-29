"""Finite-window denoising with whole-image residual equalization."""
import cv2 as cv
import numpy as np
from .memory_resources import TemporaryArrays,MiB
from .bounded_ops import map_local,equalize_u8
from .utility import create_lut


def compute(engine,params):
    mode,radius,sigma,gray,denoised,levels=params
    store=TemporaryArrays(32*MiB);image=engine.image
    shape=image.shape[:2] if gray else image.shape
    residual=store.array(shape,np.uint8)
    # Default NLM search radius 10 plus template radius 3. Colored NLM also
    # computes a 3x3 local mean when weighting luminance patches.
    halo=14 if mode==4 else radius
    def one(roi):
        original=cv.cvtColor(roi,cv.COLOR_BGR2GRAY) if gray else roi
        filtered=engine.denoise(original,mode,radius,sigma)
        return filtered if denoised else cv.absdiff(original,filtered)
    map_local(image,one,(residual,),halo=halo,store=store)
    if not denoised:
        if levels==0:residual=equalize_u8(residual,store)
        else:
            lut=create_lut(0,255-levels)
            map_local(residual,lambda roi:cv.LUT(roi,lut),(residual,),store=store)
    if gray:
        result=store.array(image.shape,np.uint8)
        map_local(residual,lambda roi:cv.cvtColor(roi,cv.COLOR_GRAY2BGR),(result,),store=store)
    else:result=residual
    store.checkpoint(force=True)
    return result
