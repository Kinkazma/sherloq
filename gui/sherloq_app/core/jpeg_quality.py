"""Exact historical JPEG recompression curve, bounded parallel work and header tables."""
from functools import lru_cache
import numpy as np
import cv2 as cv
from PIL import Image
from .jpeg import compress_jpg, get_tables
from .jpeg_curve import RecompressionCurve, Cancelled
from ..paths import model_path


def read_quantization(filename):
    # Reject non-JPEG by signature, including RAWs already decoded by the app.
    # Pillow then reads JPEG markers without decoding pixels or copying the file.
    with open(filename, 'rb') as stream:
        if stream.read(2) != b"\xff\xd8":
            return None
        stream.seek(0)
        with Image.open(stream) as image:
            tables = {key: np.asarray(value, dtype=np.int64).reshape(8, 8)
                      for key, value in image.quantization.items()}
            if not tables or any(np.any(t <= 0) for t in tables.values()):
                raise ValueError('Missing or invalid JPEG quantization tables')
            return tables, tuple(layer[3] for layer in image.layer)


@lru_cache(maxsize=1)
def standard_tables():
    base = get_tables(50)
    # libjpeg jpeg_quality_scaling uses INTEGER division below quality 50.
    # Historical 5000 / q float scaling falsely labelled genuine tables nonstandard.
    return np.stack([np.clip((base * (5000 // max(q, 1) if q < 50 else 200 - 2*q) + 50) // 100,
                             1, 255) for q in range(101)])


def table_estimate(tables, components):
    standards = standard_tables()
    if len(components) not in (1, 3) or any(k not in tables for k in components):
        return None
    luma = tables[components[0]]
    lu_diff = np.mean(np.abs(luma[None] - standards[:, :, :, 0]), axis=(1, 2))
    if len(components) == 1:
        distance = lu_diff
    else:
        cb, cr = (tables[k] for k in components[1:])
        ch_diff = np.mean(np.abs(cb[None] - standards[:, :, :, 1]), axis=(1, 2))
        if np.array_equal(cb, cr):
            distance = (lu_diff + 2 * ch_diff) / 3
        else:
            cr_diff = np.mean(np.abs(cr[None] - standards[:, :, :, 1]), axis=(1, 2))
            distance = (lu_diff + ch_diff + cr_diff) / 3
    closest = int(np.argmin(distance))
    deviation = float(distance[closest])
    quality = closest if deviation == 0 else int(np.round(closest - deviation))
    return int(np.clip(quality, 1, 100)), deviation, distance


@lru_cache(maxsize=1)
def quality_booster():
    # The bundled historical sklearn pickle predates the installed sklearn/XGB
    # wrapper API. Its embedded tree model remains readable by XGBoost itself.
    from joblib import load
    model = load(model_path('jpeg_qf.mdl'))
    booster = model.get_booster()
    if booster.num_features() != 100:
        raise ValueError('JPEG quality model must have 100 inputs')
    return booster


def predict_quality(curve):
    from xgboost import DMatrix
    # Bundled model has 140 rounds, no best_iteration/early-stop attributes.
    # All trees are used, as historical ntree_limit=None requested.
    return float(quality_booster().predict(DMatrix(curve.reshape(1, -1)))[0])


class QualityEngine:
    def __init__(self, filename, image, workers=4):
        self.filename, self.image = filename, image
        self.workers = workers
        self.curve = RecompressionCurve(image, workers=workers)
        self.raw = self.curve.raw
        self.result = None

    def compute(self, cancel=lambda: False, progress=lambda *args: None):
        if cancel():
            raise Cancelled()
        if self.result is not None:
            return self.result
        metadata_error = None
        try:
            quantization = read_quantization(self.filename)
        except (OSError, ValueError, AttributeError) as exc:
            quantization = None
            metadata_error = str(exc)
        self.curve.workers = self.workers
        values = self.curve.compute(cancel, progress, compress_jpg)
        curve = cv.normalize(values, None, 0, 1, cv.NORM_MINMAX).flatten()
        minimum = int(np.argmin(curve[:-5])) + 1
        if minimum == 95:
            minimum = 100
        result = {'curve': curve, 'minimum': minimum, 'quantization': quantization,
                  'metadata_error': metadata_error, 'estimate': None,
                  'prediction': None, 'model_error': None}
        if quantization is not None:
            result['estimate'] = table_estimate(*quantization)
        elif metadata_error is None:
            try:
                result['prediction'] = predict_quality(curve)
            except Exception as exc:
                result['model_error'] = str(exc)
        if cancel():
            raise Cancelled()
        self.result = result
        return result
