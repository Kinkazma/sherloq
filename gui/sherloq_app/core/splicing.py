"""Pure preparation/display and exact quality choice for Noiseprint analysis."""
import numpy as np
import cv2 as cv
from .jpeg_curve import RecompressionCurve
from .utility import norm_mat


def estimate_model(image, progress=lambda *a: None):
    qualities = tuple(range(1,101))
    curve = RecompressionCurve(image,qualities).compute(
        progress=lambda value,text:progress(value//5,'Estimating JPEG quality'))
    curve = cv.normalize(curve,None,0,1,cv.NORM_MINMAX).ravel()
    # argmin is an index; model names encode the actual 1-based JPEG quality.
    return qualities[int(np.argmin(curve))]


def noise_display(noise):
    if noise.ndim!=2 or not np.isfinite(noise).all():
        raise ValueError('Noiseprint returned an invalid noise array.')
    interior = noise[34:-34,34:-34] if min(noise.shape)>68 else noise
    low,high,_,_ = cv.minMaxLoc(interior)
    return norm_mat(noise.clip(low,high),to_bgr=True)
