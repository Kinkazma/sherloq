"""Retained db8 diagonal detail and vectorized block median noise estimates."""
import cv2 as cv
import numpy as np
import pywt
from .interactive import ArrayCache


class WaveletBlockingEngine:
    def __init__(self, filename, image):
        self.filename = str(filename)
        self.image = image
        self.gray = ArrayCache(128)
        self.details = ArrayCache(256)
        self.maps = ArrayCache(128)
        self.displays = ArrayCache(128)
        self.source_mode = 'file grayscale'

    def detail(self):
        detail = self.details.get('db8')
        if detail is None:
            gray = self.gray.get('gray')
            if gray is None:
                gray = cv.imread(self.filename, cv.IMREAD_GRAYSCALE)
                if gray is None:
                    # Formats decoded by the shared loader (e.g. RAW) can be
                    # unavailable to imread. Use the already loaded image.
                    gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
                    self.source_mode = 'loaded image grayscale'
                self.gray.put('gray', gray)
            low, high = pywt.dwt(gray.astype(np.float64), 'db8', axis=0)
            del low
            low, detail = pywt.dwt(high, 'db8', axis=1)
            self.details.put('db8', detail)
        return detail

    def compute(self, blocksize):
        from .memory_resources import MEMORY,MiB
        pixels=self.image.shape[0]*self.image.shape[1]
        def bounded():
            from .wavelet_blocking_bounded import compute
            return compute(self,blocksize)
        return MEMORY.execute(pixels*64,pixels*4+64*MiB,lambda:self._compute(blocksize),bounded)

    def _compute(self, blocksize):
        detail = self.detail()
        if blocksize < 1 or blocksize > min(detail.shape):
            raise ValueError(f'Block size must be between 1 and {min(detail.shape)}')
        noise = self.maps.get(blocksize)
        if noise is None:
            rows, cols = (size//blocksize for size in detail.shape)
            cropped = detail[:rows*blocksize, :cols*blocksize]
            blocks = cropped.reshape(rows, blocksize, cols, blocksize).transpose(0, 2, 1, 3)
            # Flatten in the same order as each original block, without a
            # Python loop. Median may partition this disposable abs buffer.
            absolute = np.array(blocks, order='C', copy=True).reshape(rows, cols, blocksize*blocksize)
            np.abs(absolute, out=absolute)
            noise = np.median(absolute, axis=2, overwrite_input=True) / .6745
            self.maps.put(blocksize, noise)
        display = self.displays.get(blocksize)
        if display is None:
            normalized = cv.normalize(noise, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
            resized = cv.resize(normalized, (self.image.shape[1], self.image.shape[0]), interpolation=cv.INTER_NEAREST)
            display = self.displays.put(blocksize, cv.cvtColor(resized, cv.COLOR_GRAY2BGR))
        return display, noise
