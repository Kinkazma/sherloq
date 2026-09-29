"""Exact spatial smoothing of a circular mask, sharing identical row work."""
from concurrent.futures import ThreadPoolExecutor
import cv2 as cv
import numpy as np


# Shared across all Frequency panels: at most four independent column tasks.
_COLUMN_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sherloq-frequency")

def circular_mask(shape, split, smooth):
    rows, cols = shape
    half = np.sqrt(rows ** 2 + cols ** 2) / 2
    radius = int(half * split / 100)
    kernel = 2 * int(half * smooth / 100) + 1
    mask = np.zeros(shape, np.float32)
    cv.circle(mask, (cols // 2, rows // 2), radius, 1, cv.FILLED)
    if kernel <= 63 or min(shape) == 1:
        blurred = cv.GaussianBlur(mask, (kernel, kernel), 0)
    else:
        # A filled circle has centered contiguous runs of ones. Their lengths
        # identify identical rows, including clipping at the image boundary.
        lengths = np.count_nonzero(mask, axis=1)
        _, representatives, inverse = np.unique(lengths, return_index=True, return_inverse=True)
        weights = cv.getGaussianKernel(kernel, 0, cv.CV_32F)
        identity = np.ones(1, np.float32)
        horizontal = cv.sepFilter2D(mask[representatives], cv.CV_32F, weights, identity)
        del mask
        expanded = np.ascontiguousarray(horizontal[inverse])
        del horizontal
        # Vertical convolutions do not depend on neighboring columns. Starts
        # and widths are multiples of 32 except the final tail, preserving
        # OpenCV's SIMD grouping and float32 accumulation order.
        if rows*cols >= 1_048_576:
            step = max(32, ((cols+127)//128)*32)
            blocks = [(start, min(cols, start+step)) for start in range(0, cols, step)]
            def vertical(block):
                start, end = block
                return cv.sepFilter2D(expanded[:, start:end], cv.CV_32F, identity, weights)
            blurred = np.concatenate(list(_COLUMN_POOL.map(vertical, blocks)), axis=1)
        else:
            blurred = cv.sepFilter2D(expanded, cv.CV_32F, identity, weights)
    blurred /= np.max(blurred)
    return blurred
