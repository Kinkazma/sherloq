# Composite / Noiseprint: covariance-floor-v1

I corrected a numerical instability in Composite Splicing's PCA/EM statistical
model. On two nearly singular synthetic cases, changing only the BLAS thread
count previously changed the rendered map by up to 28 and 188 levels out of 255.
The Noiseprint residual, data, seed and analysis parameters were unchanged.

The default policy is now **covariance-floor-v1**, on both CPU and GPU. The CPU
button still selects TensorFlow for Noiseprint extraction; it does not restore
the unstable statistical policy. Scripts can explicitly select
`numerical_policy="legacy"` for historical reproduction. Network weights are
unchanged and this correction requires no new model download.

## Numerical rule

`gui/noiseprint/utility/stable_covariance.py` defines
`r = sqrt(eps(float64)) = 1.4901161193847656e-8`.

- Before PCA whitening, selected eigenvalues are bounded below by
  `r * lambda_max(PCA)`. All 32 components and their directions are retained.
  Completely constant or non-finite variation produces a recoverable error.
- EM retains the historical one-ULP addition, then floors eigenvalues at
  `r * max(lambda_max(Sigma), initial_max_variance)`. The reference variance comes
  from the entire fitting sample before EM weighting, so a collapsing component
  cannot remove its own floor.
- An adequate covariance matrix is returned unchanged. Otherwise only missing
  variance is added in the affected eigendirections, followed by symmetrization.
  Eigenvectors are computed only when a correction is needed.
- The rule applies at initialization and every maximization, before likelihoods,
  replicate selection and final distances.

This limits conditioning to approximately 6.7e7. It is a documented numerical
choice, not a claimed statistical optimum. It adds no hardware calibration,
training or image-specific tuning. Global PCA, SPAM windows, ten initializations
with seed 0, at most 100 iterations and the outlier term 42 are retained.
`gaussianMixture.gm` keeps its historical default for other callers;
`EMgu_img` explicitly enables stabilization, including `noiseprint_blind_post`.

Results record `statistics_policy`, `covariance_regularizations` for the selected
model, and `pca_regularized_components`. Original PCA eigenvalues remain exported.

## Cached results and installation

The worker writes `map-statistics.json` atomically after the raw and rendered maps.
The UI verifies that marker before reuse. Old maps are recomputed, while completed
Noiseprint residuals and SPAM data are retained. GPU remains the default backend.

Apply the [cumulative native update](../README.md#apply-the-cumulative-rc1-updates)
and restart SHERLOQ to load the corrected code. Reopen previously computed panels
to replace maps retained in memory. The update includes the new covariance module,
both statistical callers, the worker and its UI cache check; model-download hooks
remain intact.

## Supplied qualification and independent checks

The frozen native delivery includes synthetic fixtures, scripts and recorded
results. Files prefixed `supplied-` retain those measurements. Files prefixed
`public-` report independent reruns from the merged public sources; they do not
replace the supplied measurements. The independent checks rerun the small
numerical cases and actual worker/Qt cache/export path, using retained synthetic
residuals for the latter. They do not rerun the unchanged network or the 96 MP job.

| Case | Recorded result |
| --- | --- |
| Two nearly singular cases, BLAS 1/2/4/8 threads | Maximum normalized map difference below 6e-9; at most one uint8 level. |
| Four concurrent EM replicates | Raw map exactly equal to serial execution at the same BLAS setting. |
| Well-conditioned control | Raw map exact against the archived two-thread reference; rendered map exact at all four thread counts. |
| Known residual anomaly | AUC 1 before and after; raw maps exact. This tests post-processing locally. |
| Rank-five PCA input | 27 whitened components floored; normalized differences below 1e-5. |
| Supplied 0.39 / 3.15 / 20.16 MP F32 cases | Raw maps bit-exact; archived MPS residuals unchanged. |
| Actual CPU/MPS worker and Qt panel | Stale maps rejected, expensive stages reused, PNG export exact, backend changes invalidate the relevant context. |
| Supplied installed-application 3.15 MP run | Real MPS network, map, rendering and export checked. |
| Supplied full 96 MP run | Real MPS network, global PCA/EM and ten fits, full-resolution rendering and exact PNG export/readback. |

The corrected singular-case maps **intentionally differ from the old two-thread
reference**, by up to 18/255 and 211/255. This is not bit-for-bit reproduction of
the old defect. Raw distances are unbounded: absolute inter-thread differences
can still reach a few units for distances around 5e8, while relative/normalized
differences stay below 6e-9. The final one-level difference reflects uint8
quantization.

The recorded 96 MP input is 8,000 × 12,000 noisy pixels with a copied region,
generated with seed 20261001. The worker actually executes the model-101 network;
no cached residual, resized input or local replacement for global statistics is
injected. SPAM is 988 × 1,488 × 512, with 719,643 valid cells. Recorded time was
115.49 seconds for calculation and 118.11 seconds including PNG export/readback;
peak worker RSS was 19,783,778,304 bytes (19.78 GB / 18.43 GiB). Other engines were
active on the machine, so this is not an isolated performance benchmark or a
prediction for every 96 MP image. Large temporary data are generated by the test
and removed after successful validation; they are not distributed.

Stabilization does not establish forensic accuracy on arbitrary images or turn
scores into fraud probabilities. Some singular cases still classify many features
as outliers; the correction does not replace Noiseprint's statistical model.

[Source and merge hashes](COMPOSITE-STABILITY-SOURCE.json) ·
[Test instructions and evidence](../tests/f32-stability/README.md).
