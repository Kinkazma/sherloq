"""Original adjustment order, exact LUT composition and bounded stage reuse."""
from functools import lru_cache
import cv2 as cv
import numpy as np
from .interactive import ArrayCache
from .utility import create_lut


@lru_cache(maxsize=128)
def adjustment_lut(gamma_tenths, shadows, highlights, sweep, width):
    inverse = 1 / (gamma_tenths / 10)
    lut = np.array([((i / 255) ** inverse) * 255 for i in np.arange(256)]).astype(np.uint8)
    if shadows:
        lut = create_lut(int(shadows / 100 * 255), 0)[lut]
    if highlights:
        lut = create_lut(0, int(highlights / 100 * 255))[lut]
    if width < 255:
        radius = width // 2
        lut = create_lut(max(sweep - radius, 0), 255 - min(sweep + radius, 255))[lut]
    lut.flags.writeable = False
    return lut


class AdjustEngine:
    def __init__(self, image, megabytes=256):
        self.image = image
        self.cache = ArrayCache(megabytes)

    def compute(self, params):
        from .memory_resources import MEMORY,MiB
        pixels=self.image.shape[0]*self.image.shape[1]
        # CLAHE/Otsu keep the original global OpenCV kernels. Their mapped
        # source/destination passes have no per-tile Python checkpoint.
        bounded=64*MiB+(pixels*3 if params[10]>=2 or params[9]==0 else 0)
        def adapted():
            cached=self.cache.get(('bounded',params))
            if cached is not None:return cached
            from .adjust_bounded import compute
            return self.cache.put(('bounded',params),compute(self.image,params))
        return MEMORY.execute(pixels*48,bounded,lambda:self._compute(params),adapted)

    def _compute(self, params):
        bright, saturation, hue, gamma, shadows, highlights, sweep, width, sharp, threshold, equalize, invert = params
        sharp //= 4
        if width == 255:
            sweep = 0
        key = (sharp, bright, saturation, hue, gamma, shadows, highlights, sweep, width, equalize, threshold, invert)
        final = self.cache.get(('result', key))
        if final is not None:
            return final
        result = self.image
        completed = 0
        # Seek the furthest retained prefix first. Otherwise an evicted early
        # stage could evict the useful late stage while rebuilding the chain.
        for name, prefix, level in (
            ('equalize', key[:10], 4), ('lut', key[:9], 3),
            ('hsv', key[:4], 2), ('sharpen', key[:1], 1),
        ):
            cached = self.cache.get((name, prefix))
            if cached is not None:
                result, completed = cached, level
                break

        def stage(name, prefix, compute):
            stage_key = (name, prefix)
            value = self.cache.get(stage_key)
            return value if value is not None else self.cache.put(stage_key, compute())

        if sharp and completed < 1:
            def sharpen():
                gaussian = cv.GaussianBlur(result, (2 * sharp + 1, 2 * sharp + 1), 0)
                return cv.addWeighted(result, 1.5, gaussian, -0.5, 0)
            result = stage('sharpen', key[:1], sharpen)
        if (bright or saturation or hue) and completed < 2:
            def hsv():
                h, s, v = cv.split(cv.cvtColor(result, cv.COLOR_BGR2HSV))
                if hue:
                    # uint8 H + slider is in [0,359]; int16 matches the former
                    # float64 arithmetic exactly, including historical H=180.
                    h = h.astype(np.int16) + hue
                    h[h > 180] -= 180
                    h = h.astype(np.uint8)
                if saturation:
                    s = cv.add(s, saturation)
                if bright:
                    v = cv.add(v, bright)
                return cv.cvtColor(cv.merge([h, s, v]), cv.COLOR_HSV2BGR)
            result = stage('hsv', key[:4], hsv)
        lut = adjustment_lut(gamma, shadows, highlights, sweep, width)
        if completed < 3 and not np.array_equal(lut, np.arange(256, dtype=np.uint8)):
            result = stage('lut', key[:9], lambda: cv.LUT(result, lut))
        if equalize and completed < 4:
            def equalization():
                h, s, v = cv.split(cv.cvtColor(result, cv.COLOR_BGR2HSV))
                v = cv.equalizeHist(v) if equalize == 1 else cv.createCLAHE((2, 5, 10, 20)[equalize-2]).apply(v)
                return cv.cvtColor(cv.merge([h, s, v]), cv.COLOR_HSV2BGR)
            result = stage('equalize', key[:10], equalization)
        if threshold < 255:
            if threshold == 0:
                gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
                threshold, _ = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)
            _, result = cv.threshold(result, threshold, 255, cv.THRESH_BINARY)
        if invert:
            result = cv.bitwise_not(result)
        # Cache entries may share an array; accounting is deliberately
        # conservative and never underestimates the configured array budget.
        return self.cache.put(('result', key), result)
