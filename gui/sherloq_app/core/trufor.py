"""Retained TruFor displays and atomic exports, without GUI dependencies."""
import os
import tempfile
from pathlib import Path
import cv2 as cv
import numpy as np
from .interactive import ArrayCache
from .jpeg_curve import Cancelled


class TruForRenderer:
    def __init__(self, result):
        self.result = result
        self.displays = ArrayCache(256)
        self.percentiles = None

    def render(self, choice):
        cached = self.displays.get(choice)
        if cached is not None:
            return cached
        if choice == 0:
            from matplotlib import colormaps
            cmap = colormaps['RdBu_r']
            # Exactly the original colormap bins, with RGB quantized once.
            lut = (cmap(np.arange(cmap.N))[:, :3] * 255).astype(np.uint8)
            values = np.clip(self.result['map'], 0, 1)
            index = (values * cmap.N).astype(np.int32)
            np.minimum(index, cmap.N - 1, out=index)
            output = cv.cvtColor(lut[index], cv.COLOR_RGB2BGR)
        elif choice == 1:
            output = cv.cvtColor(np.uint8(np.clip(self.result['conf'], 0, 1) * 255), cv.COLOR_GRAY2BGR)
        elif choice == 2:
            values = self.result['np++']
            if self.percentiles is None:
                self.percentiles = np.percentile(values, [1, 99])
            low, high = self.percentiles
            normalized = np.clip((values - low) / max(float(high - low), 1e-8), 0, 1)
            output = cv.cvtColor(np.uint8(normalized * 255), cv.COLOR_GRAY2BGR)
        else:
            raise ValueError('Unknown TruFor map.')
        return self.displays.put(choice, output)


def export_npz(result, destination, cancel=lambda: False):
    destination = Path(destination)
    if cancel():
        raise Cancelled()
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.'+destination.name+'-', suffix='.npz', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **result)
        if cancel():
            raise Cancelled()
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return str(destination)
