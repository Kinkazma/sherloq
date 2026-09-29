"""Real float64 thresholding with the same operation order as PyWavelets."""
import numpy as np
import pywt


def threshold_band(data, value, mode):
    if mode not in ('soft', 'garrote'):
        return pywt.threshold(data, value, mode)
    work = np.abs(data)
    if mode == 'garrote':
        np.square(work, out=work)
        value = value ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        np.divide(value, work, out=work)
        np.subtract(1, work, out=work)
        work.clip(min=0, max=None, out=work)
        np.multiply(data, work, out=work)
    return work
