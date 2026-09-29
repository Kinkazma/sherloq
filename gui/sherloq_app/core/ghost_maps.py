"""Stream full-resolution JPEG errors into retained 16x16 block means."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import RLock
import math
import numpy as np
import cv2 as cv
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from .interactive import ArrayCache
from .jpeg_curve import Cancelled

_RENDER_LOCK = RLock()


def _check(cancel):
    if cancel():
        raise Cancelled()


def block_error(image, quality):
    # Inputs are decoded uint8: the historical float64 imencode cast them back
    # to the same bytes. Avoid that round-trip and its full-image copies.
    ok, encoded = cv.imencode('.jpg', image, [cv.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError('JPEG encoding failed')
    decoded = cv.imdecode(encoded, cv.IMREAD_COLOR)
    difference = cv.absdiff(image, decoded)
    del encoded, decoded
    total = cv.transform(cv.multiply(difference, difference, dtype=cv.CV_32F),
                         np.ones((1, 3), np.float32))
    del difference
    # The exact sum is <=3*255², hence this accumulation has precisely the
    # same value as the old float64 channel additions. All these integers are
    # exactly representable in float32 (<2**24). Divide in float64 before mean.
    rows, cols = image.shape[0] // 16, image.shape[1] // 16
    result = np.empty((rows, cols), np.float64)
    # Limit float64 temporaries while preserving each block's C-order reduction.
    for first in range(0, rows, 32):
        count = min(32, rows-first)
        values = total[first*16:(first+count)*16, :cols*16].astype(np.float64)
        values /= 3
        blocks = values.reshape(count, 16, cols, 16).transpose(0, 2, 1, 3).copy().reshape(count, cols, 256)
        result[first:first+count] = blocks.mean(axis=2)
    return result


class GhostEngine:
    def __init__(self, image, workers=4):
        self.image, self.workers = image, workers
        self.shifted = ArrayCache(64)
        self.blocks = ArrayCache(128)
        self.normalized = ArrayCache(64)
        self.renders = ArrayCache(128)

    def shifted_image(self, x, y):
        if x == y == 0:
            return self.image
        value = self.shifted.get((x, y))
        if value is None:
            self.shifted.reserve(self.image.nbytes)
            value = self.shifted.put((x, y), np.roll(self.image, (y, x), axis=(0, 1)))
        return value

    @staticmethod
    def validate(params):
        low, high, step, x, y, gray, original = params
        if not 0 <= low <= high <= 100 or not 1 <= step <= 20:
            raise ValueError('Use 0 ≤ lower quality ≤ upper quality ≤ 100 and a positive step')
        if not 0 <= x <= 7 or not 0 <= y <= 7:
            raise ValueError('Offsets must be between 0 and 7')
        return tuple(range(low, high+1, step))

    def maps(self, params, cancel=lambda: False, progress=lambda *args: None):
        qualities = self.validate(params)
        _check(cancel)
        if min(self.image.shape[:2]) < 16:
            raise ValueError('JPEG Ghost Maps needs at least 16 × 16 pixels')
        low, high, step, x, y, _, _ = params
        key = low, high, step, x, y
        cached = self.normalized.get(key)
        if cached is not None:
            return cached
        _check(cancel)
        planes, missing = {}, []
        for q in qualities:
            value = self.blocks.get((x, y, q))
            if value is None:
                missing.append(q)
            else:
                planes[q] = value
        if missing:
            shifted = self.shifted_image(x, y)
            with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='jpeg-ghost') as pool:
                def one(q):
                    _check(cancel)
                    return block_error(shifted, q)
                futures = {pool.submit(one, q): q for q in missing}
                try:
                    for future in as_completed(futures):
                        _check(cancel)
                        q = futures[future]
                        value = future.result()
                        planes[q] = self.blocks.put((x, y, q), value)
                        progress(int(90 * len(planes)/len(qualities)), f'JPEG quality {q}')
                finally:
                    if cancel():
                        for future in futures:
                            future.cancel()
        _check(cancel)
        cube = np.stack([planes[q] for q in qualities], axis=2)
        minimum = np.min(cube, axis=2)
        span = np.max(cube, axis=2) - minimum
        cube -= minimum[:, :, None]
        np.divide(cube, span[:, :, None], out=cube, where=span[:, :, None] != 0)
        cube[span == 0] = 0
        self.normalized.put(key, cube)
        return cube

    def render(self, params, maps):
        low, high, step, x, y, gray, include = params
        qualities = tuple(range(low, high+1, step))
        # Explicit Agg figure: no global pyplot state, GUI backend or temporary file.
        with _RENDER_LOCK:
            figure = Figure(figsize=(12, 8), dpi=200)
            canvas = FigureCanvasAgg(figure)
            side = math.ceil(math.sqrt(len(qualities) + int(include)))
            if include:
                axis = figure.add_subplot(side, side, 1)
                rgb = cv.cvtColor(self.image, cv.COLOR_BGR2RGB).astype(np.float32)/255
                axis.imshow(rgb)
                axis.set_title('Original Image')
                axis.axis('off')
                del rgb
            for i, q in enumerate(qualities):
                axis = figure.add_subplot(side, side, i+1+int(include))
                axis.imshow(maps[:, :, i], cmap='gray' if gray else None, vmin=0, vmax=1)
                axis.axis('off')
                axis.set_title(f'Quality {q}')
            figure.suptitle(f'Ghost plots for grid offset X = {x} and Y = {y}')
            canvas.draw()
            result = cv.cvtColor(np.asarray(canvas.buffer_rgba()), cv.COLOR_RGBA2BGR)
            figure.clear()
            return result

    def compute(self, params, cancel=lambda: False, progress=lambda *args: None):
        self.validate(params)
        _check(cancel)
        maps = self.maps(params, cancel, progress)
        plot = self.renders.get(params)
        if plot is None:
            progress(95, 'Rendering ghost maps')
            plot = self.render(params, maps)
            _check(cancel)
            self.renders.put(params, plot)
        _check(cancel)
        progress(100, 'JPEG Ghost Maps ready')
        return plot, maps
