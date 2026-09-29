"""Lazy binary planes, preserving the historical RGB norm modulo 256."""
import cv2 as cv
import numpy as np
from .interactive import ArrayCache


_SQUARES = np.arange(256, dtype=np.uint32)**2
_NORM = cv.sqrt(np.arange(3*255**2+1, dtype=np.float64)).ravel().astype(np.uint8)


class PlanesEngine:
    def __init__(self, image):
        self.image = image
        self.channels = ArrayCache(128)
        self.planes = ArrayCache(128)
        self.results = ArrayCache(128)

    def channel(self, channel):
        if channel in (1, 2, 3):
            return self.image[:, :, 3-channel]
        result = self.channels.get(channel)
        if result is None:
            if channel == 0:
                result = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
            elif channel == 4:
                squared = _SQUARES[self.image[:, :, 0]]
                squared += _SQUARES[self.image[:, :, 1]]
                squared += _SQUARES[self.image[:, :, 2]]
                result = _NORM[squared]
            else:
                raise ValueError('Unknown bit-plane channel')
            self.channels.put(channel, result)
        return result

    def compute(self, params):
        channel, bit, filtering = params
        result = self.results.get(params)
        if result is None:
            key = channel, bit
            plane = self.planes.get(key)
            if plane is None:
                # Fixed bit meaning: 0->black and 1->white, including constants.
                # NumPy also preserves a 1x1 shape (OpenCV's scalar overload
                # can interpret that particular array as a four-value scalar).
                plane = np.right_shift(self.channel(channel), bit)
                plane &= 1
                plane *= 255
                self.planes.put(key, plane)
            if filtering == 1:
                plane = cv.medianBlur(plane, 3)
            elif filtering == 2:
                plane = cv.GaussianBlur(plane, (3, 3), 0)
            result = cv.cvtColor(plane, cv.COLOR_GRAY2BGR)
            self.results.put(params, result)
        return result
