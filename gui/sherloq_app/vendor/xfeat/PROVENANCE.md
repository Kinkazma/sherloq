# XFeat port

Upstream: verlab/accelerated_features, Apache 2.0, revision
`e92685f57f8318b18725c5c8c0bd28c7fe188d9a`.
Archive URL and SHA-256: `/third_party/research/manifest.json` in the local distribution.
Weights: `/models/external/manifest.json`.

Changes in modules/xfeat.py: relative imports; explicit CPU/MPS device, strict
local weights_only loading, minimum dimensions, ROI mask before point limit,
empty NMS handling. Official floor-to-32 resize retained. MPS lacks bicubic
sampling here: interpolator.py transfers just that interpolation to CPU.
Unused upstream lighterglue.py retained for provenance; application does not
call its hub loader. Integration and constrained matcher: core/learned_copy.py.
