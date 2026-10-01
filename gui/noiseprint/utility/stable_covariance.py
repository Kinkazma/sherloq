"""Scale-aware numerical regularization for Composite's statistical model.

This is part of the analysis, not a hardware calibration. Float64's sqrt(eps)
limits covariance conditioning to about 6.7e7, reserving roughly half the
significant digits for roundoff amplification. The rule has no image-specific
constant and does not remove features, observations or EM restarts.
"""
import numpy as np

STATISTICS_POLICY = "covariance-floor-v1"
RELATIVE_VARIANCE_FLOOR = float(np.sqrt(np.finfo(np.float64).eps))


def floor_covariance(covariance, reference_scale=0.0, relative_floor=RELATIVE_VARIANCE_FLOOR):
    """Lift only unreliable eigendirections; preserve an adequate matrix exactly.

    reference_scale is the largest variance of the full fitting sample, before
    EM weights. It keeps an EM component collapsing to a point from removing
    its own numerical floor. It is not estimated from a separate pilot run.
    """
    # Most images need no correction: do not compute eigenvectors in that case.
    values = np.linalg.eigvalsh(covariance)
    scale = max(float(values[-1]), float(reference_scale))
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(values).all():
        raise ValueError("Insufficient finite variation for the statistical model.")
    floor = relative_floor * scale
    if values[0] >= floor:
        return covariance, False
    values, vectors = np.linalg.eigh(covariance)
    # Add only the missing variance instead of reconstructing the entire matrix:
    # the well-resolved part retains its original floating-point values.
    correction = (vectors * np.maximum(floor - values, 0.0)) @ vectors.T
    result = covariance + correction
    return (result + result.T) * 0.5, True
