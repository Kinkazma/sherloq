"""Bounded caches and pure computations for interactive analysis tools.

Each engine belongs to one tool and is called serially by LatestJob. Arrays in
caches are never modified by consumers. No Qt GUI objects enter worker threads.
"""
from collections import OrderedDict
import cv2 as cv
import numpy as np
import pywt
from .utility import norm_mat, create_lut, bgr_to_gray3
from .wavelet_threshold import threshold_band
from .frequency_mask import circular_mask


from .cache_budget import ArrayCache


class FrequencyEngine:
    def __init__(self, image):
        self.image = image
        self.dft = None
        self.masks = ArrayCache(128)
        self.results = ArrayCache(256)
        self.displays = ArrayCache(128)

    def prepare(self):
        if self.dft is not None:
            return
        gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        rows, cols = gray.shape
        padded = cv.copyMakeBorder(gray, 0, cv.getOptimalDFTSize(rows)-rows,
                                   0, cv.getOptimalDFTSize(cols)-cols, cv.BORDER_CONSTANT)
        self.dft = np.fft.fftshift(cv.dft(padded.astype(np.float32), flags=cv.DFT_COMPLEX_OUTPUT), axes=(0, 1))
        # Preserve the old magnitude's operand order/rounding. Historical
        # fftshift also swapped the real/imaginary axis; only its phase was
        # wrong. The corrected phase uses real then imaginary components.
        magnitude, _ = cv.cartToPolar(self.dft[:, :, 1], self.dft[:, :, 0])
        _, phase = cv.cartToPolar(self.dft[:, :, 0], self.dft[:, :, 1])
        self.magnitude0 = cv.normalize(cv.log(magnitude), None, 0, 255, cv.NORM_MINMAX)
        self.phase0 = cv.normalize(phase, None, 0, 255, cv.NORM_MINMAX)

    def compute(self, params):
        split, smooth, threshold, filtering = params
        self.prepare()
        key = (split, smooth, threshold)
        result = self.results.get(key)
        if result is None:
            rows, cols = self.dft.shape[:2]
            mask = self.masks.get((split, smooth))
            if mask is None:
                mask = circular_mask((rows, cols), split, smooth)
                self.masks.put((split, smooth), mask)
            if threshold:
                mask = mask.copy()
                mask[self.magnitude0 < int(threshold/100*255)] = 0
                zeros = (mask.size-np.count_nonzero(mask))/mask.size*100
            else:
                zeros = 0
            r, c = self.image.shape[:2]
            low = cv.idft(np.fft.ifftshift(self.dft*mask[:, :, None], axes=(0, 1)), flags=cv.DFT_SCALE)
            low = norm_mat(cv.magnitude(low[:, :, 0], low[:, :, 1])[:r, :c], to_bgr=True)
            high = cv.idft(np.fft.ifftshift(self.dft*(1-mask[:, :, None]), axes=(0, 1)), flags=cv.DFT_SCALE)
            high = norm_mat(cv.magnitude(high[:, :, 0], high[:, :, 1]), to_bgr=True)[:r, :c].copy()
            magnitude = (self.magnitude0*mask).astype(np.uint8)
            phase = (self.phase0*mask).astype(np.uint8)
            result = self.results.put(key, (low, high, magnitude, phase, zeros))
        low, high, magnitude, phase, zeros = result
        display = self.displays.get((key, filtering))
        if display is None:
            if filtering:
                kernel = 2*filtering+1
                magnitude = cv.GaussianBlur(magnitude, (kernel, kernel), 0)
                phase = cv.GaussianBlur(phase, (kernel, kernel), 0)
            display = self.displays.put((key, filtering),
                                       (cv.cvtColor(magnitude, cv.COLOR_GRAY2BGR),
                                        cv.cvtColor(phase, cv.COLOR_GRAY2BGR)))
        return low, high, *display, zeros


class WaveletEngine:
    def __init__(self, image):
        self.image = image
        self.transforms = ArrayCache(256)
        self.prefixes = ArrayCache(128)
        self.results = ArrayCache(128)

    def compute(self, params):
        wavelet, threshold, level, mode = params
        if threshold == 0 or level == 0:
            threshold, level, mode = 0, 0, 'soft'
        key = wavelet, threshold, level, mode
        result = self.results.get(key)
        if result is not None:
            return result
        transform = self.transforms.get(wavelet)
        if transform is None:
            coeffs = pywt.wavedec2(self.image[:, :, 0], wavelet)
            maxima = [tuple(np.max(np.abs(p)) for p in octave) for octave in coeffs[1:]]
            transform = self.transforms.put(wavelet, (coeffs, maxima))
        coeffs, maxima = transform
        # Copy the list, not all coefficient arrays. threshold() allocates only
        # the detail bands being changed; cached coefficients remain intact.
        changed = list(coeffs)
        if threshold:
            level = min(level, len(coeffs)-1)
            if level:
                prefix = self.prefixes.get((wavelet, level))
                if prefix is None:
                    prefix = self.prefixes.put((wavelet, level), pywt.waverec2(coeffs[:-level], wavelet))
                changed = [prefix] + list(coeffs[-level:])
                for i in range(1, level+1):
                    changed[-i] = tuple(threshold_band(p, threshold/100*m, mode)
                                        if m else p for p, m in zip(coeffs[-i], maxima[-i]))
        # Symmetric boundary extension can reconstruct one extra row/column.
        # Preserve every original-coordinate value but exclude that padding.
        rows, cols = self.image.shape[:2]
        reconstructed = pywt.waverec2(changed, wavelet)[:rows, :cols]
        result = cv.cvtColor(reconstructed.astype(np.uint8), cv.COLOR_GRAY2BGR)
        return self.results.put(key, result)


class EchoEngine:
    def __init__(self, image, backend="auto"):
        self.image = image
        self.backend = backend
        self.last_backend = "cpu"
        self.derivatives = ArrayCache(128)
        self.results = ArrayCache(128)

    def compute(self, params):
        radius, contrast, gray = params
        result = self.results.get(params)
        if result is not None:
            return result
        derivative = self.derivatives.get(radius)
        if derivative is None:
            planes = []
            # For uint8 and radii <= 4, integer kernels have absolute partial
            # sum bounds <= 4,700,160, below float32's exact integer limit.
            # Convert back to float64 before historical normalization. Larger
            # kernels are NOT exact in float32 and always keep the CPU path.
            use_gpu = (self.backend != "cpu" and self.image.dtype == np.uint8
                       and radius <= 4 and cv.ocl.useOpenCL()
                       and (self.backend == "gpu" or self.image.shape[0]*self.image.shape[1] >= 1_048_576))
            self.last_backend = "opencl" if use_gpu else "cpu"
            for channel in cv.split(self.image):
                if use_gpu:
                    try:
                        d = cv.Laplacian(cv.UMat(channel), cv.CV_32F, ksize=2*radius+1).get().astype(np.float64)
                    except cv.error:
                        use_gpu = False
                        self.last_backend = "cpu-fallback"
                        d = cv.Laplacian(channel, cv.CV_64F, None, 2*radius+1)
                else:
                    d = cv.Laplacian(channel, cv.CV_64F, None, 2*radius+1)
                np.abs(d, out=d)
                planes.append(cv.normalize(d, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1))
                del d
            derivative = self.derivatives.put(radius, cv.merge(planes))
        result = cv.LUT(derivative, create_lut(0, int(contrast/100*255)))
        if gray:
            result = bgr_to_gray3(result)
        return self.results.put(params, result)


class PlotEngine:
    def __init__(self, image):
        self.image = image
        self.colors = ArrayCache(1024)
        self.positions = ArrayCache(512)
        self.rgba = ArrayCache(384)

    def compute(self, scale):
        colors = self.colors.get(scale)
        if colors is None:
            image = self.image
            for _ in range(scale):
                image = cv.pyrDown(image)
            rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.float32)
            rgb /= 255
            hsv = cv.cvtColor(rgb, cv.COLOR_RGB2HSV)
            hsv[:, :, 0] /= 360
            colors = self.colors.put(scale, np.concatenate((rgb.reshape(-1, 3), hsv.reshape(-1, 3)), axis=1))
        return colors


    def prepare_plot(self, params):
        scale, x, y, z, colored, alpha, kind = params
        data = self.compute(scale)
        position_key = scale, x, y, z if kind == '3d' else None, kind
        positions = self.positions.get(position_key)
        if positions is None:
            if kind == '3d':
                positions = np.ascontiguousarray(data[:, [x, y, z]])
            elif kind == '2d':
                positions = np.empty((len(data), 3), np.float32)
                positions[:, :2] = data[:, [x, y]]
                positions[:, 2] = 0
            else:
                positions = np.ascontiguousarray(data[:, [x, y]])
            self.positions.put(position_key, positions)
        color_key = scale, colored, alpha
        colors = self.rgba.get(color_key) if colored else None
        if colored and colors is None:
            colors = np.empty((len(data), 4), np.float32)
            colors[:, :3] = data[:, :3]
            colors[:, 3] = alpha
            self.rgba.put(color_key, colors)
        if not colored:
            colors = (.12, .47, .71, alpha) if kind == '3d' else (31/255, 119/255, 180/255, alpha)
        return dict(params=params, data=data, positions=positions, colors=colors,
                    position_key=position_key, color_key=color_key)
