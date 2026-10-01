# Composite covariance stability checks

These synthetic fixtures and development tests accompany
[covariance-floor-v1](../../docs/COMPOSITE-STABILITY.md).
All six NPZ fixtures are unchanged from the frozen native delivery.
Files named `supplied-*.json` are its recorded evidence; `public-*.json` are
independent checks of the merged public sources.

Copy this directory to `tests/f32-stability/` inside a restored native installation,
then run with that installation's bundled Python:

```sh
venv/bin/python tests/f32-stability/regression.py
QT_QPA_PLATFORM=offscreen venv/bin/python tests/f32-stability/delivery.py
```

The regression checks BLAS thread counts, serial/parallel EM, nearly singular and
well-conditioned controls, a known residual anomaly, rank deficiency, constant
features and scale equivariance. It writes fresh `results.json` and stable fixtures
in the test copy. The optional `--large` flag requires the earlier local F32 input
and residual fixtures; these larger historical fixtures are not included here.

The delivery test uses real subprocesses and the Qt panel with supplied synthetic
residual/SPAM data. It checks cache invalidation, CPU/MPS statistics policy, reuse,
backend changes and an exact PNG export. It does not rerun the network. The normal
distribution hook still ensures the selected model is present; complete installs
already include it, and light installs can download it on demand.

To reproduce the separate capacity qualification, run:

```sh
venv/bin/python tests/f32-stability/large96.py
```

This macOS Apple Silicon test generates a noisy 96 MP input with a copied region
and runs the real MPS network, global statistics, rendering and PNG export. The
supplied run peaked at about 19.78 GB worker RSS; temporary arrays also need disk
space. This is a development test, never an installation preflight or user-facing
calibration. Its recorded result does not establish general forensic accuracy.
