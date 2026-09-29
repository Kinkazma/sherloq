# LightGlue port

Upstream: cvg/LightGlue, Apache 2.0, revision
`eb42fee2d71449efb0aa5c10549752b5d75384d8`.
Archives and weights have SHA-256 manifests in third_party/research and
models/external of the local distribution.

lightglue.py unchanged; application constructs features=None to disable hub
loading, then loads local parameters strictly in core/learned_copy.py.
Package __init__.py avoids eager imports of unused extractors. ALIKED loads
local weights only; ROI before feature limit, empty detection guard added.
SIFT permits unlimited detection before local point selection, uses an OpenCV
ROI mask, and handles empty descriptors. The spatially constrained learned
matcher is an application adaptation, not the unmodified upstream matcher.
