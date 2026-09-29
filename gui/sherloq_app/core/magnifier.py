"""Exact ROI operations; one engine per immutable loaded image, serial worker."""
import cv2 as cv
from .interactive import ArrayCache
from .utility import auto_lut, equalize_img


class MagnifierEngine:
    def __init__(self, image, megabytes=128):
        self.image = image
        self.regions = ArrayCache(megabytes)

    def bounds(self, rect):
        x, y, width, height = rect
        rows, cols = self.image.shape[:2]
        # QRect right/bottom are inclusive; Python slice stops are exclusive.
        return max(0, min(cols, x)), max(0, min(rows, y)), max(0, min(cols, x + width)), max(0, min(rows, y + height))

    def compute(self, params):
        bounds, mode, percent, channel = params
        cached = self.regions.get(params)
        if cached is not None:
            return bounds, cached
        x1, y1, x2, y2 = bounds
        roi = self.image[y1:y2, x1:x2]
        if roi.size == 0:
            return bounds, None
        if mode == 'equalize':
            result = equalize_img(roi)
        elif channel:
            result = cv.merge([cv.LUT(c, auto_lut(c, percent / 200)) for c in cv.split(roi)])
        else:
            result = cv.LUT(roi, auto_lut(cv.cvtColor(roi, cv.COLOR_BGR2GRAY), percent / 200))
        result.flags.writeable = False
        return bounds, self.regions.put(params, result)
