# CAT-Net v2

mjkwon2021/CAT-Net (author Myung-Joon Kwon), pinned revision
331b8059c3f55efec1d9075de79dd153413f2061.
The pinned README announces Apache2 for CAT-Net code and CC-BY4.0 for its weights
(2026-08-05 update). HRNet's MIT notice is retained in LICENSE-HRNet.
config.py (upstream default.py) and CAT_full.yaml are unchanged. network_CAT.py adds two optional branches delegating memory-bounded DCT/head calculations to core/catnet.py.

Adapter core/catnet.py directly constructs CAT_Net and strictly loads the full
checkpoint; partial ImageNet pretraining is not needed or silently substituted.
The restricted weights loader additionally permits NumPy numeric scalar types
because the published checkpoint includes an evaluation metric.
SHA-256/source URLs: third_party/research/manifest.json and
models/external/manifest.json in the local distribution.
