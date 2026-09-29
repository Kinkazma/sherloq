"""Serial reductions inside independently parallel JPEG recompressions."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2 as cv
import numpy as np
from .jpeg import compress_jpg


class Cancelled(Exception):
    pass


class RecompressionCurve:
    def __init__(self, image, qualities=tuple(range(1, 101)), workers=4):
        self.image, self.qualities, self.workers = image, tuple(qualities), workers
        self.raw = {}

    def compute(self, cancel=lambda: False, progress=lambda *args: None,
                encoder=compress_jpg):
        if cancel():
            raise Cancelled()
        missing = [q for q in self.qualities if q not in self.raw]
        if not missing:
            return np.array([self.raw[q] for q in self.qualities])
        gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY) if self.image.ndim > 2 else self.image
        def one(quality):
            if cancel():
                raise Cancelled()
            return cv.mean(cv.absdiff(encoder(gray, quality, False), gray))[0]
        workers = self.workers if gray.size >= 100_000 else 1
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='jpeg-curve') as pool:
            futures = {pool.submit(one, q): q for q in missing}
            try:
                for future in as_completed(futures):
                    if cancel():
                        raise Cancelled()
                    self.raw[futures[future]] = future.result()
                    progress(len(self.raw), 'Computing JPEG recompression curve')
            finally:
                if cancel():
                    for future in futures:
                        future.cancel()
        if cancel():
            raise Cancelled()
        return np.array([self.raw[q] for q in self.qualities])
