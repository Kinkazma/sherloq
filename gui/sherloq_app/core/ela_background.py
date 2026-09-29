"""One-sided mid-residual deficit against distant, comparable image content.

This is an exploratory robust distance, not an authenticity probability.
White residual tails cannot dominate the trimmed mean. Dark tails are excluded
from that mean too; the quartiles retain broad distribution shifts.
"""
import numpy as np
from scipy.spatial import cKDTree

TRIM = (.1, .9)
MAXIMUM_CONTENT_DISTANCE = .4
LOG_FLOOR = .1


def mid_profile(levels, valid):
    """Log trimmed mean and quartiles of raw, unsaturated absolute RGB luma."""
    cells = len(levels)
    values = levels.reshape(cells, -1)
    valid = valid.reshape(cells, -1)
    count = valid.sum(1)
    ordered = np.sort(np.where(valid, values, np.inf), axis=1)
    start = np.floor(count * TRIM[0]).astype(int)
    end = np.ceil(count * TRIM[1]).astype(int)
    ranks = np.arange(values.shape[1])[None, :]
    middle = (ranks >= start[:, None]) & (ranks < end[:, None])
    mean = np.where(middle, ordered, 0).sum(1) / np.maximum(end - start, 1)
    indices = np.floor(np.maximum(count - 1, 0)[:, None] * np.array([.25, .75])).astype(int)
    quartiles = np.take_along_axis(ordered, indices, axis=1)
    result = np.column_stack((mean, quartiles))
    result[count == 0] = 0
    return np.log1p(result).astype(np.float32)


def score_background(content, profiles, supported, cancel=lambda: False):
    from .ela_biomes import check
    shape = supported.shape
    n = supported.size
    c = content.reshape(n, 6)
    p = profiles.reshape(n, profiles.shape[2], 3)
    z = np.zeros_like(p)
    reference = np.zeros_like(p)
    counts = np.zeros(n, np.int32)
    ids = np.flatnonzero(supported)
    if len(ids) >= 17:
        center = np.median(c[ids], axis=0)
        scale = np.maximum(1.4826 * np.median(np.abs(c[ids] - center), axis=0),
                           [16, .25, .25, .25, 8, .25])
        normal = (c - center) / scale / np.sqrt(6)
        tree = cKDTree(normal[ids])
        coords = np.column_stack([a.ravel() for a in np.indices(shape)])
        for lo in range(0, n, 128):
            check(cancel)
            hi = min(n, lo + 128)
            distances, indices = tree.query(normal[lo:hi], k=min(128, len(ids)), workers=1)
            peers = ids[indices]
            eligible = ((distances <= MAXIMUM_CONTENT_DISTANCE) &
                        (np.max(np.abs(coords[peers] - coords[lo:hi, None]), axis=2)
                         > max(2, min(shape) // 5)))
            eligible &= np.cumsum(eligible, axis=1) <= 64
            count = eligible.sum(1)
            good = (count >= 16) & supported.ravel()[lo:hi]
            counts[lo:hi] = np.where(good, count, 0)
            if not good.any():
                continue
            values = p[peers[good]].copy()
            values[~eligible[good]] = np.nan
            median = np.nanmedian(values, axis=1)
            mad = 1.4826 * np.nanmedian(np.abs(values - median[:, None]), axis=1)
            rows = np.flatnonzero(good) + lo
            reference[rows] = median
            z[rows] = (median - p[lo:hi][good]) / np.maximum(mad, LOG_FLOOR)
    # Positive = deficit. Confirmation by >=2 descriptors and >=2 qualities.
    quality_score = np.median(z, axis=2)
    score = np.maximum(0, np.median(quality_score, axis=1))
    check(cancel)
    return dict(background_score=score.reshape(shape),
                background_quality_scores=quality_score.reshape(*shape, -1),
                background_signed_scores=z.reshape(*shape, *p.shape[1:]),
                background_reference=reference.reshape(*shape, *p.shape[1:]),
                background_peer_count=counts.reshape(shape),
                background_supported=counts.reshape(shape) >= 16)
