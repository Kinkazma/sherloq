"""Retain denoising separately from residual contrast and display choices."""
import cv2 as cv
from .interactive import ArrayCache
from .utility import create_lut, equalize_img


class NoiseEngine:
    def __init__(self, image, backend='auto'):
        self.image = image
        self.backend = backend
        self.last_backend = 'cpu'
        self.gray = ArrayCache(64)
        self.filtered = ArrayCache(256)
        self.residuals = ArrayCache(128)
        self.results = ArrayCache(128)

    @staticmethod
    def parameters(params):
        mode, radius, sigma, gray, denoised, levels = params
        return mode, radius, sigma if mode == 3 else 0, gray, denoised, 0 if denoised else levels

    def source(self, gray):
        if not gray:
            return self.image
        original = self.gray.get('gray')
        if original is None:
            original = self.gray.put('gray', cv.cvtColor(self.image, cv.COLOR_BGR2GRAY))
        return original

    def denoise(self, original, mode, radius, sigma):
        kernel = radius*2+1
        self.last_backend = 'cpu'
        if mode == 0:
            # Median is an integer order statistic, not a floating reduction.
            # Only the measured 5x5 OpenCL path is faster with transfers.
            if (radius == 2 and self.backend != 'cpu'
                    and (self.backend == 'gpu' or original.shape[0]*original.shape[1] >= 1_000_000)
                    and cv.ocl.haveOpenCL() and cv.ocl.useOpenCL()):
                try:
                    result = cv.medianBlur(cv.UMat(original), kernel).get()
                    self.last_backend = 'opencl'
                    return result
                except cv.error:
                    pass  # CPU remains the reference and the fallback.
            return cv.medianBlur(original, kernel)
        if mode == 1:
            return cv.GaussianBlur(original, (kernel, kernel), 0)
        if mode == 2:
            return cv.blur(original, (kernel, kernel))
        if mode == 3:
            return cv.bilateralFilter(original, kernel, sigma, sigma)
        if mode == 4:
            # Historical radius control actually sets NLM strength h, not a
            # window radius. Preserve h=2*radius+1 and OpenCV default windows.
            if original.ndim == 2:
                return cv.fastNlMeansDenoising(original, None, kernel)
            return cv.fastNlMeansDenoisingColored(original, None, kernel, kernel)
        raise ValueError('Unknown denoising method')

    def compute(self, params):
        params=self.parameters(params)
        cached=self.results.get(params)
        if cached is not None:return cached
        from .memory_resources import MEMORY,MiB
        def bounded():
            from .noise_bounded import compute
            result=compute(self,params)
            return self.results.put(params,result)
        return MEMORY.execute(self.image.shape[0]*self.image.shape[1]*32,64*MiB,
                              lambda:self._compute(params),bounded)

    def _compute(self, params):
        params = self.parameters(params)
        result = self.results.get(params)
        if result is not None:
            return result
        mode, radius, sigma, gray, denoised, levels = params
        key = mode, radius, sigma, gray
        original = self.source(gray)
        filtered = self.filtered.get(key)
        if filtered is None:
            filtered = self.filtered.put(key, self.denoise(original, mode, radius, sigma))
        if denoised:
            result = filtered
        else:
            noise = self.residuals.get(key)
            if noise is None:
                noise = self.residuals.put(key, cv.absdiff(original, filtered))
            if levels == 0:
                result = cv.equalizeHist(noise) if gray else equalize_img(noise)
            else:
                result = cv.LUT(noise, create_lut(0, 255-levels))
        if gray:
            result = cv.cvtColor(result, cv.COLOR_GRAY2BGR)
        return self.results.put(params, result)
