"""Descriptive ELA energy disparities, distinct from peer-based anomaly scores.

Compare low/high residual levels against the central 10–90% reference of each
panel. Neither class attributes an editing operation or estimates fraud odds.
"""
import colorsys
import cv2 as cv
import numpy as np

LOG_SCALE = .2
ENERGY_FLOOR = .25
WINDOW = 7


def describe_energy(image, compressed):
    difference = np.abs(image.astype(np.float32) - compressed.astype(np.float32))
    return cv.blur(cv.cvtColor(difference, cv.COLOR_BGR2GRAY), (WINDOW, WINDOW))


def prepare_energy(image, planes, cancel=lambda: False, quantiles=(.1,.9)):
    from .ela_biomes import check, Cancelled
    from .auto_zones import detect_panels
    from .jpeg_curve import Cancelled as ZoneCancelled
    if len(quantiles)!=2 or not (0<=quantiles[0]<=.5 and .5<=quantiles[1]<=1):
        raise ValueError('Energy histogram bounds must lie in [0, .5] and [.5, 1].')
    check(cancel)
    h, w = image.shape[:2]
    try:
        regions = detect_panels(image, cancel)
    except ZoneCancelled:
        raise Cancelled() from None
    if not regions:
        regions = (((0., 0.), (w-1., 0.), (w-1., h-1.), (0., h-1.)),)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    scope = np.zeros((h, w), np.int32)
    # Saturated originals and gutters are not residual-energy evidence.
    valid = (gray > 3) & (gray < 252)
    low = np.zeros_like(planes)
    high = np.zeros_like(planes)
    summaries = []
    for index, polygon in enumerate(regions, 1):
        check(cancel)
        x0, y0 = np.min(polygon, axis=0).astype(int)
        x1, y1 = np.max(polygon, axis=0).astype(int)
        sl = np.s_[y0:y1+1, x0:x1+1]
        allowed = valid[sl]
        if np.count_nonzero(allowed) < 64:
            continue
        scope[sl] = np.where(allowed, index, 0)
        probes = []
        for q, plane in enumerate(planes):
            check(cancel)
            energy = plane[sl]
            values = energy[allowed]
            q10, q90 = np.quantile(values, quantiles)
            middle = values[(values >= q10) & (values <= q90)]
            # At 50/50 the interpolated median may lie between two samples.
            # Keep the chosen bounds fixed, using that central value directly.
            if not middle.size:middle=np.asarray([(q10+q90)/2])
            center = float(middle.mean())
            score = np.log((center + ENERGY_FLOOR) / (energy + ENERGY_FLOOR)) / LOG_SCALE
            low[q][sl] = np.where(allowed, np.maximum(score, 0), 0)
            high[q][sl] = np.where(allowed, np.maximum(-score, 0), 0)
            dark = values[values <= q10]
            bright = values[values >= q90]
            probes.append(dict(q10=float(q10), q90=float(q90), central_mean=center,
                               central_variance=float(middle.var()), low_mean=float(dark.mean()),
                               low_variance=float(dark.var()), high_mean=float(bright.mean()),
                               high_variance=float(bright.var())))
        summaries.append(dict(id=index, bbox=[int(x0), int(y0), int(x1+1), int(y1+1)], probes=probes))
    check(cancel)
    return dict(energy_low_score=np.median(low, axis=0),
                energy_high_score=np.median(high, axis=0), energy_scope=scope,
                energy_summary=summaries)


def segment_energy(base, threshold, minimum, offset=0, energy_thresholds=None):
    """Grow within measured weak support; never fill a panel or its convex hull."""
    thresholds = (threshold, threshold) if energy_thresholds is None else tuple(energy_thresholds)
    if len(thresholds)!=2 or not all(np.isfinite(t) and t>=0 for t in thresholds):
        raise ValueError('Energy thresholds must be two nonnegative finite values.')
    shape = base['energy_scope'].shape
    labels = np.zeros(shape, np.int32)
    regions = []
    block = base['metadata']['block']
    allowed = base.get('energy_allowed')
    for region in base['energy_summary']:
        x0, y0, x1, y1 = region['bbox']
        sl = np.s_[y0:y1, x0:x1]
        valid = base['energy_scope'][sl] == region['id']
        if allowed is not None:
            valid = valid & allowed[sl]
        for kind, key in (('low', 'energy_low_score'), ('high', 'energy_high_score')):
            threshold = thresholds[0 if kind=='low' else 1]
            score = base[key][sl]
            if threshold==0:
                weak=strong=(score>0)&valid
            else:
                weak = (score >= threshold / 2) & valid
                strong = (score >= threshold) & valid
            n, components, stats, _ = cv.connectedComponentsWithStats(weak.astype(np.uint8), connectivity=8)
            strong_counts = np.bincount(components[strong], minlength=n)
            kept=[i for i in range(1,n) if stats[i,cv.CC_STAT_AREA]>=minimum*block*block
                  and strong_counts[i]>=block*block]
            if not kept:
                continue
            # One class per reference panel, even if its measured territory has
            # disconnected pieces. Never paint the gaps between those pieces.
            selected=np.isin(components,kept)
            x=int(stats[kept,cv.CC_STAT_LEFT].min());y=int(stats[kept,cv.CC_STAT_TOP].min())
            right=int((stats[kept,cv.CC_STAT_LEFT]+stats[kept,cv.CC_STAT_WIDTH]).max())
            bottom=int((stats[kept,cv.CC_STAT_TOP]+stats[kept,cv.CC_STAT_HEIGHT]).max())
            area=int(selected.sum());rid=offset+len(regions)+1
            labels[sl][selected]=rid
            regions.append(dict(id=rid,kind=kind,energy_region=region['id'],components=len(kept),
                cells=int(np.ceil(area/(block*block))),pixels=area,bbox=[x0+x,y0+y,right-x,bottom-y],
                score=float(np.median(score[selected])),strong_pixels=int(strong_counts[kept].sum()),
                dominant_descriptor='energy_'+kind,
                sources=['Low ELA energy' if kind=='low' else 'High ELA energy']))
    return labels, regions


def energy_color(region):
    """Stable panel/class identity, independent of mask shape and label ordering."""
    identity=2*(int(region['energy_region'])-1)+(region['kind']=='high')
    rgb=colorsys.hsv_to_rgb((.73+identity*.61803398875)%1,.7,.95)
    return tuple(round(x*255) for x in rgb[::-1])
