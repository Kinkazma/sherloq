"""Resampling maps with reference float64 arithmetic and bounded stage caches."""
import numpy as np
import cv2 as cv
from .interactive import ArrayCache
from .jpeg_curve import Cancelled


def check(cancel):
    if cancel():
        raise Cancelled()


def normalize_gray(image):
    shifted = image - image.min()
    maximum = shifted.max()
    return shifted / maximum if maximum else np.zeros(image.shape, np.float64)


def matrices(image, size):
    border = size // 2
    h, w = image.shape
    if min(h, w) < size:
        raise ValueError(f'The selected region must be at least {size} × {size} pixels.')
    rows = (h - 2 * border) * (w - 2 * border)
    # Preserve row-major neighbor order, including omission of the center.
    F = np.empty((rows, size * size - 1), np.float64)
    column = 0
    for y in range(size):
        for x in range(size):
            if (y, x) != (border, border):
                F[:, column] = image[y:y+h-2*border, x:x+w-2*border].ravel()
                column += 1
    f = image[border:-border, border:-border].ravel().copy()
    return F, f


def radial_window(shape, highpass=False):
    rows, cols = shape
    cy, cx = rows // 2, cols // 2
    radius = np.sqrt(cy ** 2 + cx ** 2)
    if radius == 0:
        return np.zeros(shape) if highpass else np.ones(shape)
    y, x = np.ogrid[:rows, :cols]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2) / radius * np.sqrt(2)
    result = np.zeros(shape)
    if highpass:
        mask = r <= np.sqrt(2)
        result[mask] = .5 - .5 * np.cos(np.pi * r[mask] / np.sqrt(2))
    else:
        result[r < 3/4] = 1
        mask = (r >= 3/4) & (r <= np.sqrt(2))
        result[mask] = .5 + .5 * np.cos(np.pi * (r[mask] - 3/4) / (np.sqrt(2) - 3/4))
    return result


class ResamplingEngine:
    def __init__(self, gray):
        self.gray = gray
        self.composites = ArrayCache(256)
        self.features = ArrayCache(256)
        self.maps = ArrayCache(512)
        self.windows = ArrayCache(128)
        self.spectra = ArrayCache(1024)
        self.magnitudes = ArrayCache(512)
        self.renders = ArrayCache(512)
        self.partial = None
        self.counts = dict(probability=0, fft=0, magnitude=0)

    def probability(self, rect, size, cancel=lambda:False, progress=lambda *_:None):
        key = (tuple(rect), size)
        cached = self.maps.get(key)
        if cached is not None:
            return cached
        x0, y0, x1, y1 = rect
        image = self.gray[y0:y1, x0:x1]
        if min(image.shape) < size:
            raise ValueError(f'The selected region must be at least {size} × {size} pixels.')
        if not np.isfinite(image).all() or image.max() == image.min():
            raise ValueError('The selected region has no usable intensity variation.')
        rows = (image.shape[0] - size + 1) * (image.shape[1] - size + 1)
        estimated = rows * ((size * size - 1) * 4 + 16) * 8
        if estimated > 12 * 1024**3:
            raise ValueError('This region exceeds the 12 GiB working-memory budget. Select a smaller region.')
        check(cancel)
        features = self.features.get(key)
        if features is None:
            self.features.reserve(rows * size * size * 8)
            features = matrices(image, size)
            self.features.put(key, features)
        F, f = features
        if self.partial is not None and self.partial[0] == key:
            _, a, s, iteration = self.partial
        else:
            a = np.random.RandomState(0).rand(size * size - 1)
            a = a / a.sum()
            s, iteration = .005, 0
        self.counts['probability'] += 1
        while iteration < 100:
            check(cancel)
            total = a[0] * F[:, 0]
            for column in range(1, len(a)):
                total += a[column] * F[:, column]
            r = f - total
            # Scalar np.float64 ** 2 uses pow, unlike array ** 2's multiply
            # shortcut. float_power preserves its last-bit rounding.
            squared = np.float_power(r, 2)
            if not np.isfinite(s) or s <= 0:
                raise ValueError('The region is degenerate for this interpolation model.')
            g = np.exp(-squared / s)
            w = g / (g + .1)
            weight_sum = w.sum()
            if weight_sum == 0:
                raise ValueError('The region has no valid interpolation weights.')
            # Preserve the original sequential accumulation, not a pairwise sum.
            s2 = np.cumsum(w * squared)[-1]
            s = s2 / weight_sum
            check(cancel)
            try:
                a2 = np.linalg.inv(F.T * w * w @ F) @ F.T * w * w @ f
            except np.linalg.LinAlgError as exc:
                raise ValueError('The region is singular for this model. Select a more varied region.') from exc
            if not np.isfinite(a2).all():
                raise ValueError('The interpolation model returned non-finite coefficients.')
            progress(iteration + 1, 100)
            if np.linalg.norm(a - a2) < .01:
                break
            a = a2
            iteration += 1
            self.partial = (key, a, s, iteration)
        result = w.reshape(image.shape[0] - size + 1, image.shape[1] - size + 1)
        self.partial = None
        self.maps.put(key, result)
        return result

    def fourier(self, image, identity, params, cancel=lambda:False):
        window, upsample, center, highpass, gamma, rescale = params
        key = (identity, tuple(params))
        cached = self.renders.get(key)
        if cached is not None:
            return cached
        if min(image.shape) < 2:
            raise ValueError('A Fourier region must be at least 2 × 2 pixels.')
        check(cancel)
        transform_key = (identity, window, upsample)
        fourier = self.spectra.get(transform_key)
        if fourier is None:
            h, w = image.shape
            half = min(h, w) // 2
            square = image[h//2-half:h//2+half, w//2-half:w//2+half]
            window_key = (square.shape, window)
            weights = self.windows.get(window_key)
            if weights is None:
                weights = (np.hanning(square.shape[0])[:, None] * np.hanning(square.shape[1])
                           if window == 'hanning' else radial_window(square.shape))
                self.windows.put(window_key, weights)
            source = square * weights
            if upsample:
                source = cv.pyrUp(source)
            check(cancel)
            fourier = np.fft.fftshift(np.fft.fft2(source))
            self.counts['fft'] += 1
            self.spectra.put(transform_key, fourier)
        magnitude_key = (transform_key, center, highpass)
        magnitude = self.magnitudes.get(magnitude_key)
        if magnitude is None:
            check(cancel)
            if center:
                h, w = fourier.shape
                half = w // 4
                if half == 0:
                    raise ValueError('The Fourier region is too small to take its center.')
                spectrum = fourier[h//2-half:h//2+half, w//2-half:w//2+half]
            else:
                spectrum = fourier
            if highpass == 'simple':
                h, w = spectrum.shape
                y, x = np.ogrid[:h, :w]
                radius = max(1, int(.1 * (min(h, w) / 2)))
                mask = np.sqrt((x-w//2)**2 + (y-h//2)**2) <= radius
                filtered = spectrum.copy()
                filtered[mask] = 0
            else:
                filtered = spectrum * radial_window(spectrum.shape, True)
            magnitude = np.abs(filtered)
            self.magnitudes.put(magnitude_key, magnitude)
            self.counts['magnitude'] += 1
        low, high = magnitude.min(), magnitude.max()
        scaled = (magnitude - low) / (high - low) if high > low else np.zeros(magnitude.shape)
        result = np.power(scaled, gamma)
        if rescale:
            result = result * high
        self.renders.put(key, result)
        return result
