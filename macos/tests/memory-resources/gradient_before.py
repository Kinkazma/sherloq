"""Luminance-gradient channels retained independently of presentation settings."""
import cv2 as cv
import numpy as np
from .interactive import ArrayCache
from .utility import norm_mat, create_lut, equalize_img


class GradientEngine:
    def __init__(self, image):
        self.image = image
        self.channels = ArrayCache(128)
        self.bases = ArrayCache(128)
        self.results = ArrayCache(128)

    def prepare(self):
        found = self.channels.get('channels')
        if found is not None:
            return found
        dx, dy = cv.spatialGradient(cv.cvtColor(self.image, cv.COLOR_BGR2GRAY))
        dx, dy = dx.astype(np.float32), dy.astype(np.float32)
        ax, ay = np.abs(dx), np.abs(dy)
        mx, my = np.max(ax), np.max(ay)
        blue = norm_mat(ax + ay)
        del ax, ay
        def channel(values, maximum, invert):
            if maximum == 0:
                # A zero derivative has neutral direction, not NaN cast to 0.
                return np.full(values.shape, 127, np.uint8)
            if invert:
                values = -values
            return ((values / maximum * 127) + 127).astype(np.uint8)
        red, green = channel(dx, mx, False), channel(dy, my, False)
        reverse_red, reverse_green = channel(dx, mx, True), channel(dy, my, True)
        return self.channels.put('channels', (red, green, reverse_red, reverse_green, blue))

    def compute(self, params):
        percent, mode, invert, equalize = params
        intensity = 0 if equalize else int(percent / 100 * 127)
        key = intensity, mode, invert, equalize
        result = self.results.get(key)
        if result is not None:
            return result
        base = self.bases.get((mode, invert))
        if base is None:
            red, green, reverse_red, reverse_green, absolute = self.prepare()
            if invert:
                red, green = reverse_red, reverse_green
            if mode == 0:
                blue = np.zeros_like(red)
            elif mode == 1:
                blue = np.full_like(red, 255)
            elif mode == 2:
                blue = absolute
            elif mode == 3:
                # Same two-term float64 sum and sqrt as linalg.norm, without
                # its interleaved 2-channel float64 copy and squared copy.
                squared = np.square(red, dtype=np.float64)
                squared += np.square(green, dtype=np.float64)
                np.sqrt(squared, out=squared)
                blue = norm_mat(squared)
            else:
                raise ValueError('Unknown blue channel mode.')
            base = self.bases.put((mode, invert), cv.merge([blue, green, red]))
        if equalize:
            result = equalize_img(base)
        elif intensity > 0:
            result = cv.LUT(base, create_lut(intensity, intensity))
        else:
            result = base
        return self.results.put(key, result)
