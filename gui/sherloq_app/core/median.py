"""Retained median-filter features and exact CPU model scores.

The feature definitions (including PSNR=-1 for equal images and signed MD)
are the trained model's input contract, not generic metric conventions.
"""
from concurrent.futures import ThreadPoolExecutor
import cv2 as cv
import numpy as np
from .interactive import ArrayCache
from .jpeg_curve import Cancelled
from .utility import pad_image


def _statistics(image):
    values = image.astype(np.float64)
    square = values ** 2
    mean = cv.GaussianBlur(values, (11, 11), 1.5)
    variance = cv.GaussianBlur(square, (11, 11), 1.5) - mean ** 2
    return values, np.sum(square), np.sum(values), mean, variance


def _metrics(left, right):
    x, x2, xs, mx, vx = left
    y, y2, _, my, vy = right
    e = x-y
    m = np.zeros(8)
    m[0] = np.mean(np.square(e))
    m[1] = 20*np.log10(255/np.sqrt(m[0])) if m[0] > 0 else -1
    m[2] = np.sum(x*y)/x2 if x2 > 0 else -1
    m[3] = np.mean(e)
    m[4] = x2/y2 if y2 > 0 else -1
    m[5] = np.max(e)
    m[6] = np.sum(np.abs(e))/xs if xs > 0 else -1
    mu_xy = mx*my
    covariance = cv.GaussianBlur(x*y,(11,11),1.5)-mu_xy
    numerator = (2*mu_xy+(0.01*255)**2)*(2*covariance+(0.03*255)**2)
    denominator = mx**2+my**2+(0.01*255)**2
    denominator *= vx+vy+(0.03*255)**2
    m[7] = cv.mean(cv.divide(numerator,denominator))[0]
    return m


def get_features(image, windows, levels):
    features = np.zeros(windows*levels*8)
    original = _statistics(image)
    index = 0
    for window in range(windows):
        previous, stats = image, original
        for _ in range(levels):
            filtered = cv.medianBlur(previous,2*(window+1)+1)
            next_stats = _statistics(filtered)
            features[index:index+8] = _metrics(stats,next_stats)
            previous, stats = filtered,next_stats
            index += 8
    return features


class MedianEngine:
    block = 64

    def __init__(self, image, model_file):
        self.image, self.model_file = image, model_file
        self.model = self.gray = self.features = self.prob = self.var = None
        self.completed = 0
        self.workers = 2
        self.filtered = {}
        self.renders = ArrayCache(256)

    def _prepare(self):
        if self.model is None:
            import xgboost as xgb
            model = xgb.Booster({'nthread':4})
            model.load_model(self.model_file)
            model.set_param({'nthread':4})
            columns = model.num_features()
            # Preserve historical effective ordering for every accepted model:
            # the old caller passes (levels, windows) to get_features.
            formats = {8:(1,1),24:(3,1),96:(3,4),128:(4,4)}
            if columns not in formats:
                raise ValueError(f'Unknown median model format: {columns} features.')
            self.windows,self.levels = formats[columns]
            self.model = model
        if self.gray is None:
            self.gray = pad_image(cv.cvtColor(self.image,cv.COLOR_BGR2GRAY),self.block)
            nr,nc = (s//self.block for s in self.gray.shape)
            # Extra padded block and zero row/column are historical display
            # boundary conditions; preserve them before median/linear resize.
            self.prob = np.zeros((nr+1,nc+1))
            self.var = np.zeros_like(self.prob)
            self.features = np.zeros((nr*nc,self.model.num_features()))

    def analyze(self, cancel=lambda:False, progress=lambda *a:None):
        if cancel():raise Cancelled()
        self._prepare()
        nc = self.gray.shape[1]//self.block
        total = len(self.features)
        if self.completed == total:return self.prob,self.var
        def one(index):
            if cancel():raise Cancelled()
            r,c = divmod(index,nc)
            roi = self.gray[r*64:(r+1)*64,c*64:(c+1)*64]
            return get_features(roi,self.windows,self.levels),np.var(roi)
        # A bounded batch limits cancellation latency and temporary memory.
        with ThreadPoolExecutor(self.workers,thread_name_prefix='median-features') as pool:
            while self.completed<total:
                if cancel():raise Cancelled()
                start,end = self.completed,min(total,self.completed+32)
                indices = range(start,end)
                values = list(map(one,indices)) if self.workers==1 else list(pool.map(one,indices))
                if cancel():raise Cancelled()
                features = np.array([v[0] for v in values])
                predictions = self.model.inplace_predict(features)
                if not np.isfinite(predictions).all():raise ValueError('Non-finite median model scores.')
                for offset,(feature,variance) in enumerate(values):
                    index = start+offset;r,c = divmod(index,nc)
                    self.features[index] = feature
                    self.var[r,c] = variance;self.prob[r,c] = predictions[offset]
                self.completed = end
                progress(end*100//total,'Analyzing median-filter traces')
        if cancel():raise Cancelled()
        return self.prob,self.var

    def render(self, params):
        variance,threshold,show_score,speckle = params
        key = (variance,None if show_score else threshold,show_score,speckle)
        cached = self.renders.get(key)
        if cached is not None:return cached
        if speckle not in self.filtered:
            prob = self.prob.astype(np.float32)
            self.filtered[speckle] = cv.medianBlur(prob,3) if speckle else prob
        prob = self.filtered[speckle]
        mask = self.var<variance
        if show_score:
            output = np.repeat(prob[:,:,None],3,axis=2);output[mask]=0
        else:
            output = np.zeros((*prob.shape,3))
            blue,green,red = cv.split(output)
            blue[mask]=1;green[prob<threshold]=1;green[mask]=0
            red[prob>=threshold]=1;red[mask]=0
            output = cv.merge((blue,green,red))
        output = cv.convertScaleAbs(output,None,255)
        h,w = self.image.shape[:2]
        self.renders.reserve(self.image.nbytes)
        output = cv.resize(output,None,None,64,64,cv.INTER_LINEAR)[:h,:w].copy()
        mean = cv.mean(prob,1-mask.astype(np.uint8))[0]
        return self.renders.put(key,(output,mean))

    def compute(self, params, cancel=lambda:False, progress=lambda *a:None):
        self.analyze(cancel,progress)
        result = self.render(params)
        if cancel():raise Cancelled()
        return result
