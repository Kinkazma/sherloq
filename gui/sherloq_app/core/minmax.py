"""Local extrema excluding the center and retained block-density maps."""
from functools import lru_cache
import cv2 as cv
import numpy as np
from .interactive import ArrayCache
from .utility import norm_mat


_NEIGHBORS = np.ones((3, 3), np.uint8)
_NEIGHBORS[1, 1] = 0
_COLORS = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0],
                    [255, 255, 255], [0, 0, 0]], np.uint8)


@lru_cache(maxsize=128)
def _std_table(size):
    # Match np.std's float64 computation and its original float32 storage.
    table = np.empty(size+1, np.float32)
    block = np.zeros(size, bool)
    for count in range(size+1):
        table[count] = np.std(block)
        if count < size:
            block[count] = True
    return table


def block_density(mask, radius):
    rows, cols = mask.shape
    block = 2*radius+1
    if rows <= radius or cols <= radius:
        return np.zeros(mask.shape, np.uint8)
    row_starts = np.arange(0, rows, block)
    col_starts = np.arange(0, cols, block)
    counts = np.add.reduceat(np.add.reduceat(mask, row_starts, axis=0, dtype=np.uint32),
                             col_starts, axis=1, dtype=np.uint32)
    heights = np.minimum(block, rows-row_starts)
    widths = np.minimum(block, cols-col_starts)
    areas = heights[:, None]*widths[None, :]
    values = np.zeros(counts.shape, np.float32)
    valid = (heights[:, None] > radius) & (widths[None, :] > radius)
    for size in np.unique(areas[valid]):
        selected = valid & (areas == size)
        values[selected] = _std_table(int(size))[counts[selected]]
    expanded = np.repeat(np.repeat(values, heights, axis=0), widths, axis=1)
    return cv.normalize(expanded, None, 0, 127, cv.NORM_MINMAX, cv.CV_8UC1)


class MinMaxEngine:
    def __init__(self, image):
        self.image = image
        self.masks = ArrayCache(128)
        self.densities = ArrayCache(128)
        self.results = ArrayCache(128)

    def extrema(self, channel):
        result = self.masks.get(channel)
        if result is not None:
            return result
        rows, cols = self.image.shape[:2]
        if min(rows, cols) < 3:
            return self.masks.put(channel, (np.zeros((rows, cols), bool), np.zeros((rows, cols), bool)))
        if channel == 0:
            values = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        elif channel == 4:
            # uint8 squared norms are exact integers <=195075. sqrt preserves
            # their order, so neither float64 planes nor square roots are needed.
            values = np.sum(self.image.astype(np.uint32)**2, axis=2, dtype=np.uint32).astype(np.float32)
        else:
            values = self.image[:, :, 3-channel]
        low = values < cv.erode(values, _NEIGHBORS)
        high = values > cv.dilate(values, _NEIGHBORS)
        for mask in (low, high):
            mask[[0, -1], :] = False
            mask[:, [0, -1]] = False
        return self.masks.put(channel, (low, high))

    def compute(self, params):
        cached=self.results.get(params)
        if cached is not None:
            masks=self.masks.get(params[0])
            if masks is not None:return cached,*masks
        from .memory_resources import MEMORY,MiB
        def bounded():
            from .minmax_bounded import compute
            result,low,high=compute(self.image,params)
            self.results.put(params,result);self.masks.put(params[0],(low,high))
            return result,low,high
        return MEMORY.execute(self.image.shape[0]*self.image.shape[1]*32,64*MiB,
                              lambda:self._compute(params),bounded)

    def _compute(self, params):
        channel, minimum, maximum, radius = params
        low, high = self.extrema(channel)
        result = self.results.get(params)
        if result is None:
            if radius:
                result = np.zeros_like(self.image)
                for which, mask, color in ((0, low, minimum), (1, high, maximum)):
                    if color == 4:
                        continue
                    key = channel, which, radius
                    density = self.densities.get(key)
                    if density is None:
                        density = self.densities.put(key, block_density(mask, radius+3))
                    if color <= 2:
                        result[:, :, 2-color] += density
                    else:
                        result += density[:, :, None]
                result = norm_mat(result)
            else:
                code = high.astype(np.uint8)
                code *= 2
                code += low
                palette = np.zeros((256, 1, 3), np.uint8)
                palette[1, 0] = _COLORS[minimum]
                palette[2, 0] = _COLORS[maximum]
                result = cv.applyColorMap(code, palette)
            self.results.put(params, result)
        return result, low, high
