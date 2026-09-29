"""Exact lazy PCA views; float64 model and historical normalization retained."""
import cv2 as cv
import numpy as np
from .interactive import ArrayCache
from .utility import norm_mat, equalize_img


class PcaEngine:
    def __init__(self, image):
        self.image = image
        self.model = None
        self.centered = ArrayCache(512)
        self.projections = ArrayCache(128)
        self.bases = ArrayCache(256)
        self.results = ArrayCache(128)

    def prepare(self):
        centered = self.centered.get('data')
        if centered is None:
            centered = self.image.reshape(-1, 3).astype(np.float64)
            if self.model is None:
                # PCA needs three samples to return all three color axes.
                # Repeating a 1/2-pixel sample preserves its mean/covariance.
                samples = np.tile(centered, (3, 1)) if len(centered) < 3 else centered
                self.model = cv.PCACompute2(samples, np.array([]))
                del samples
            centered -= self.model[0]
            self.centered.put('data', centered)
        return centered

    @staticmethod
    def cross_plane(centered, vector, channel):
        j, k = ((1, 2), (2, 0), (0, 1))[channel]
        return centered[:, j]*vector[k] - centered[:, k]*vector[j]

    def base(self, component, mode):
        key = component, mode
        result = self.bases.get(key)
        if result is not None:
            return result
        centered = self.prepare()
        mu, vectors, values = self.model
        vector = vectors[component]
        shape = self.image.shape[:2]
        if mode == 'distance':
            distance = None
            for channel in range(3):
                plane = self.cross_plane(centered, vector, channel)
                plane *= plane
                if distance is None:
                    distance = plane
                else:
                    distance += plane
            np.sqrt(distance, out=distance)
            distance /= np.linalg.norm(vector)
            result = norm_mat(distance.reshape(shape))
        elif mode == 'project':
            planes = self.projections.get('planes')
            if planes is None:
                # Keep all three columns: a single-column GEMM changes rounding.
                raw = cv.PCAProject(centered, np.zeros_like(mu), vectors).reshape(self.image.shape)
                planes = tuple(norm_mat(raw[:, :, i]) for i in range(3))
                self.projections.put('planes', planes)
            result = planes[component]
        elif mode == 'crossprod':
            result = cv.merge([norm_mat(self.cross_plane(centered, vector, i).reshape(shape))
                               for i in range(3)])
        else:
            raise ValueError('Unknown PCA view')
        return self.bases.put(key, result)

    def compute(self, params):
        result = self.results.get(params)
        if result is None:
            component, mode, invert, equalize = params
            result = self.base(component, mode)
            if invert:
                result = cv.bitwise_not(result)
            if equalize:
                result = equalize_img(result) if result.ndim == 3 else cv.equalizeHist(result)
            if result.ndim == 2:
                result = cv.cvtColor(result, cv.COLOR_GRAY2BGR)
            self.results.put(params, result)
        return result, self.model
