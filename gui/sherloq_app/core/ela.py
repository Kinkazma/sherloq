"""Retained JPEG recompressions and exact float32 ELA bases."""
import cv2 as cv
import numpy as np
from .jpeg import compress_jpg
from .interactive import ArrayCache
from .utility import create_lut, desaturate


class ElaEngine:
    def __init__(self, image):
        self.image = image
        self.compressed = ArrayCache(128)
        self.bases = ArrayCache(256)
        self.linear = ArrayCache(128)
        self.results = ArrayCache(128)

    def recompressed(self, quality):
        result = self.compressed.get(quality)
        if result is None:
            self.compressed.reserve(self.image.nbytes)
            result = self.compressed.put(quality, compress_jpg(self.image, quality))
        return result

    def base(self, quality, linear):
        cache = self.linear if linear else self.bases
        result = cache.get(quality)
        if result is None:
            cache.reserve(self.image.size * (1 if linear else 4))
            compressed = self.recompressed(quality)
            if linear:
                # Linear ELA is still an absolute difference. cv.subtract on
                # uint8 silently discarded negative compression errors.
                result = cv.absdiff(compressed, self.image)
            else:
                result = self.image.astype(np.float32) / 255
                temporary = compressed.astype(np.float32) / 255
                cv.absdiff(result, temporary, dst=result)
                del temporary
                cv.sqrt(result, dst=result)
                result *= 255
            cache.put(quality, result)
        return result

    def compute(self, params):
        quality, scale, contrast_percent, linear, grayscale = params
        # At 100%, historical endpoints crossed (128 > 127), inverting tone.
        contrast = min(127, int(contrast_percent / 100 * 128))
        key = quality, scale, contrast, linear, grayscale
        result = self.results.get(key)
        if result is None:
            self.results.reserve(self.image.nbytes)
            result = cv.convertScaleAbs(self.base(quality, linear), None,
                                        scale if linear else scale / 20)
            cv.LUT(result, create_lut(contrast, contrast), dst=result)
            if grayscale:
                result = desaturate(result)
            self.results.put(key, result)
        return result
