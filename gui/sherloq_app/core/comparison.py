"""Comparison arithmetic, pair-local caches and cancellable helper commands."""
import math
from contextlib import nullcontext
import tempfile
import subprocess
import warnings
from pathlib import Path
from threading import Event
import cv2 as cv
import numpy as np
import sewar
from .utility import norm_mat, equalize_img, desaturate, butter_exe, ssimul_exe
from .interactive import ArrayCache

def rmse(x, y):
    return np.sqrt(np.mean(np.square(x - y)))

def mb(x, y):
    mx = np.mean(x)
    my = np.mean(y)
    return (mx - my) / mx

def pfe(x, y):
    return np.linalg.norm(x - y) / np.linalg.norm(x) * 100

def ssim(x, y):
    c1 = 6.5025
    c2 = 58.5225
    k = (11, 11)
    s = 1.5
    x2 = x ** 2
    y2 = y ** 2
    xy = x * y
    mu_x = cv.GaussianBlur(x, k, s)
    mu_y = cv.GaussianBlur(y, k, s)
    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y
    del mu_x, mu_y
    s_x2 = cv.GaussianBlur(x2, k, s) - mu_x2
    del x2
    s_y2 = cv.GaussianBlur(y2, k, s) - mu_y2
    del y2
    s_xy = cv.GaussianBlur(xy, k, s) - mu_xy
    del xy
    t1 = 2 * mu_xy + c1
    del mu_xy
    t2 = 2 * s_xy + c2
    del s_xy
    t3 = t1 * t2
    t1 = mu_x2 + mu_y2 + c1
    del mu_x2, mu_y2
    t2 = s_x2 + s_y2 + c2
    del s_x2, s_y2
    t1 *= t2
    del t2
    ssim_map = cv.divide(t3, t1)
    del t3, t1
    ssim = cv.mean(ssim_map)[0]
    return (ssim, 255 - norm_mat(ssim_map, to_bgr=True))

def corr(x, y):
    return np.corrcoef(x, y)[0, 1]

def psnr(x, y):
    # PSNR = 10 log10(MAX**2 / MSE), not 20 log10 of this power ratio.
    k = np.mean(np.square(x - y))
    if k == 0:
        return float('inf')
    return 10 * math.log10(255 ** 2 / k)


class ComparisonCancelled(Exception):
    pass


def helper_inputs(folder, x, y):
    paths = [str(Path(folder) / name) for name in ('evidence.png', 'reference.png', 'map.ppm')]
    if not cv.imwrite(paths[0], x) or not cv.imwrite(paths[1], y):
        raise OSError('Unable to prepare images for comparison helper.')
    return paths


def helper(program, paths, heatmap=False, cancelled=None):
    if not program:
        raise RuntimeError('Comparison helper is unavailable on this platform.')
    arguments = [program, *paths[:2]] + ([paths[2]] if heatmap else [])
    with subprocess.Popen(arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        import time
        deadline = time.monotonic() + 120
        while True:
            if cancelled is not None and cancelled.is_set():
                process.kill();process.communicate()
                raise ComparisonCancelled()
            if time.monotonic() > deadline:
                process.kill();process.communicate()
                raise RuntimeError('Comparison helper timed out.')
            try:
                output, errors = process.communicate(timeout=.1)
                break
            except subprocess.TimeoutExpired:
                continue
        if process.returncode:
            raise RuntimeError(errors.decode(errors='replace')[-4096:] or 'Comparison helper failed.')
        value = float(output)
        if not np.isfinite(value) or value < 0:
            raise ValueError('Comparison helper returned an invalid score.')
        if heatmap:
            image = cv.imread(paths[2], cv.IMREAD_COLOR)
            if image is None:
                raise ValueError('Comparison helper did not produce a readable map.')
            return value, image
        return value


def butter(x, y):
    with tempfile.TemporaryDirectory(prefix='sherloq-comparison-') as folder:
        return helper(butter_exe(), helper_inputs(folder, x, y), heatmap=True)


def ssimul(x, y):
    with tempfile.TemporaryDirectory(prefix='sherloq-comparison-') as folder:
        return helper(ssimul_exe(), helper_inputs(folder, x, y))


METRICS = ('rmse', 'sam', 'ergas', 'mb', 'pfe', 'psnr', 'ssim', 'msssim',
           'rase', 'scc', 'uqi', 'vifp', 'ssimul', 'butter',
           'hist_0', 'hist_1', 'hist_4', 'hist_2', 'hist_3', 'hist_5')


class HistogramPair:
    """Keep OpenCV's ordinary results; correct counts beyond float32 integers."""
    def __init__(self, evidence, reference):
        self.histograms = [cv.calcHist([a], [0, 1, 2], None, [256]*3, [0, 256]*3)
                           for a in (evidence, reference)]
        self.exact = any(h.max() >= 2**24 for h in self.histograms)
        if self.exact:
            self.histograms = None
            counts = []
            for image in (evidence, reference):
                pixels = image.reshape(-1, 3)
                codes = pixels[:, 0].astype(np.uint32)
                codes <<= 8
                codes |= pixels[:, 1]
                codes <<= 8
                codes |= pixels[:, 2]
                counts.append(np.bincount(codes, minlength=256**3))
            indices = np.flatnonzero((counts[0] != 0) | (counts[1] != 0))
            self.a, self.b = (h[indices].astype(np.float64) for h in counts)

    def compare(self, method):
        if not self.exact:
            return cv.compareHist(*self.histograms, method)
        a, b = self.a, self.b
        sa, sb = a.sum(), b.sum()
        if method == cv.HISTCMP_CORREL:
            # Empty bins still count in the mean of the full 256^3 histogram.
            n = 256**3
            denominator = (np.dot(a, a) - sa*sa/n) * (np.dot(b, b) - sb*sb/n)
            return float((np.dot(a, b) - sa*sb/n) / np.sqrt(denominator)) if denominator > 0 else 1.0
        if method == cv.HISTCMP_CHISQR:
            keep = a > 0
            return float(np.sum((a[keep]-b[keep])**2 / a[keep]))
        if method == cv.HISTCMP_CHISQR_ALT:
            return float(2*np.sum((a-b)**2 / (a+b)))
        if method == cv.HISTCMP_INTERSECT:
            return float(np.minimum(a, b).sum())
        if method == cv.HISTCMP_BHATTACHARYYA:
            return float(np.sqrt(max(1-np.sqrt(a*b).sum()/np.sqrt(sa*sb), 0)))
        if method == cv.HISTCMP_KL_DIV:
            keep = a > 0
            return float(np.sum(a[keep]*np.log(a[keep]/np.where(b[keep] > 0, b[keep], 1e-10))))
        raise ValueError('Unknown histogram comparison method.')


class ComparisonEngine:
    def __init__(self, evidence, reference):
        if evidence.shape != reference.shape:
            raise ValueError('Evidence and reference must have the same size.')
        self.evidence, self.reference = evidence, reference
        self.cancelled = Event()
        self.progress = (0, '')
        self.checkpoint = None
        self.gray_pair = None
        self.metric_runner = None
        self.helper_folder = None
        self.values = {}
        self.maps = {}
        self.display_cache = ArrayCache(128)

    def display(self, mode, equalized, gray):
        # This method is serialized by a separate LatestJob. Metrics only
        # publishes complete new maps; cached display arrays are never edited.
        key = mode, equalized, gray
        found = self.display_cache.get(key)
        if found is not None:
            return found
        if mode == 'normal':
            result = self.reference
        elif mode == 'difference':
            if mode not in self.maps:
                self.maps[mode] = norm_mat(cv.absdiff(self.evidence, self.reference))
            result = self.maps[mode]
        else:
            result = self.maps[mode]
        if equalized:
            result = equalize_img(result)
        if gray:
            result = desaturate(result)
        return self.display_cache.put(key, result)

    def compute(self):
        errors = {}
        if len(self.values) == len(METRICS):
            return {'values': dict(self.values), 'errors': errors, 'cancelled': False}
        try:
            self._check_cancelled()
            img1, img2 = self.gray_pair if self.gray_pair is not None else (
                cv.cvtColor(self.evidence, cv.COLOR_BGR2GRAY),
                cv.cvtColor(self.reference, cv.COLOR_BGR2GRAY),
            )
            x, y = img1.astype(np.float64), img2.astype(np.float64)
            histograms = None
            with (tempfile.TemporaryDirectory(prefix='sherloq-comparison-') if self.helper_folder is None
                  else nullcontext(self.helper_folder)) as folder:
                paths = None
                for index, name in enumerate(METRICS):
                    self._check_cancelled()
                    self.progress = index, name
                    if self.checkpoint is not None:
                        self.checkpoint(self.values, self.maps, errors, self.progress)
                    if index == 7:
                        x = y = None  # Later metrics perform their own conversion.
                    if name in self.values:
                        continue
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore', RuntimeWarning)
                            if name in ('rmse', 'mb', 'pfe', 'psnr'):
                                value = globals()[name](x, y)
                            elif name == 'ssim':
                                value, self.maps[name] = ssim(x, y)
                            elif name in ('ssimul', 'butter'):
                                if paths is None:
                                    paths = helper_inputs(folder, img1, img2)
                                result = helper(butter_exe() if name == 'butter' else ssimul_exe(), paths, name == 'butter', self.cancelled)
                                if name == 'butter':
                                    value, self.maps[name] = result
                                else:
                                    value = result
                            elif name.startswith('hist_'):
                                if histograms is None:
                                    histograms = HistogramPair(self.evidence, self.reference)
                                value = histograms.compare(int(name[5:]))
                            else:
                                value = (self.metric_runner(name) if self.metric_runner is not None and name in ('msssim', 'vifp')
                                         else getattr(sewar, name)(img1, img2))
                                if name == 'msssim':
                                    value = value.real
                        if not np.isfinite(value) and not (name == 'psnr' and value == np.inf):
                            raise ValueError('Metric is undefined for this image pair.')
                        self.values[name] = value
                    except (ComparisonCancelled, MemoryError):
                        raise
                    except Exception as exc:
                        errors[name] = str(exc) or type(exc).__name__
            self.progress = len(METRICS), ''
            return {'values': dict(self.values), 'errors': errors, 'cancelled': False}
        except ComparisonCancelled:
            return {'values': dict(self.values), 'errors': errors, 'cancelled': True}

    def _check_cancelled(self):
        if self.cancelled.is_set():
            raise ComparisonCancelled()
