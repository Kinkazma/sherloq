"""Exact uint8 BGR histograms and distinct-colour count with bounded scratch RAM."""
import cv2 as cv
import numpy as np

CHUNK_PIXELS = 1 << 20


def exact_channel_histogram(image):
    """Preserve integer counts beyond the exact range of float32 calcHist."""
    values = image if image.ndim == 2 else image[:, :, 0]
    if values.size <= CHUNK_PIXELS:
        return cv.calcHist([values], [0], None, [256], [0, 256]).ravel().astype(np.int64)
    flat = values.reshape(-1)
    hist = np.zeros(256, dtype=np.int64)
    for start in range(0, len(flat), CHUNK_PIXELS):
        block = flat[start:start + CHUNK_PIXELS]
        hist += cv.calcHist([block], [0], None, [256], [0, 256]).ravel().astype(np.int64)
    return hist


def analyze_histogram(image):
    flat = image.reshape(-1, 3)
    hist = np.zeros((4, 256), dtype=np.int64)
    # A 24-bit RGB colour is an integer in [0, 2**24). Presence needs 16 MiB,
    # independent of the input resolution, without sorting millions of tuples.
    seen = np.zeros(1 << 24, dtype=bool) if len(flat) > CHUNK_PIXELS else None
    unique = 0
    for start in range(0, len(flat), CHUNK_PIXELS):
        pixels = flat[start:start + CHUNK_PIXELS]
        block = np.ascontiguousarray(pixels).reshape(1, -1, 3)
        for slot, channel in enumerate((2, 1, 0)):
            hist[slot] += cv.calcHist([block], [channel], None, [256], [0, 256]).ravel().astype(np.int64)
        gray = cv.cvtColor(block, cv.COLOR_BGR2GRAY)
        hist[3] += cv.calcHist([gray], [0], None, [256], [0, 256]).ravel().astype(np.int64)
        # Each float32 histogram is <=2**20 pixels: every count is exact.
        # Accumulation in int64 avoids float32's lost units above 2**24.
        codes = pixels[:, 0].astype(np.uint32)
        codes <<= 8
        codes |= pixels[:, 1]
        codes <<= 8
        codes |= pixels[:, 2]
        if seen is None:
            unique = len(np.unique(codes))
        else:
            seen[codes] = True
    if seen is not None:
        unique = int(np.count_nonzero(seen))
    hist.flags.writeable = False
    return hist, unique, np.round(unique / len(flat) * 100, 2)
