# D2PRL — native integration

I integrated [the original D2PRL inference](https://github.com/byc33/D2PRL/tree/a4314b614ea3186b4fac98e9e96939e37f275fc5)
into Clone Detectors. The GPU is selected by default; CPU remains an explicit
choice. The panel provides union, source and target views, raw-map exports and
cancellation. Each selected region is processed independently. This segmentation
model does not perform a correspondence search between two selected regions.

## Protocol and implementation

The adapter preserves RGB values in [0,1], PIL-to-tensor conversion followed by
bilinear antialiased tensor resizing to **448 × 448**, **40 PatchMatch iterations**,
evaluation mode and seed 22. It loads the local checkpoint on CPU with
`weights_only=True` and strict key matching before transferring to the device.
The four MAT filters and checkpoint are checked against the recorded SHA256s.

The three raw maps are union probability, target residual and source residual,
not class logits. Role thresholds are **> 0**, with removal of objects smaller
than 500 model-grid pixels and the original 50 × 50 majority filter. Categorical
masks are restored with nearest-neighbour interpolation; the continuous map uses
bilinear interpolation. Raw maps remain available before display interpolation.

On ARM CPU, interpolation uses float32 on the already float16-quantized values,
then converts back to float16. This avoids crashes observed in the PyTorch 2.8
half-precision CPU grid sampler. On MPS, evaluations group eight candidates with
the same float16 operations and retain constant grids on the device. There is no
performance calibration before an analysis.

The [runtime manifest](../integration/clone_detectors/runtime-manifest.json)
records adaptations of the original code. The
[D2PRLu fork](https://github.com/nPr0nn/D2PRLu/tree/70b804a16575fe160695ec6c4740e6f10337d3b1)
is an additional reference; its training preprocessing is not substituted for
the original inference used here.

## Recorded measurements

These development measurements used an M1 Max and the authors' example. They
are individual runs, not medians or comparisons against the original CUDA code.

| Path | Recorded time | Comparison |
| --- | ---: | --- |
| Initial MPS adaptation | 401.42 s, excluding model loading | Initial reference |
| Optimized MPS, public engine | 25.76 s, including model loading | All three raw maps identical to the initial MPS reference |
| Portable CPU | 241.85 s | Completed without the half-precision sampler crash |
| CPU, public engine | 230.33 s, including model loading | All three raw maps identical to the portable CPU reference |

CPU and GPU are **not numerically interchangeable**. With identical random
streams, their union probabilities had mean absolute difference 0.00709 and
maximum difference 0.999997; 0.6906% of mask pixels differed (mask IoU 0.9720).
I kept GPU as the default after reviewing these differences, with CPU available
for reproducible work on that backend. This choice is specific to D2PRL and does
not relax the numerical requirements of other engines.

The MPS union-mask IoU against the example's annotation was 0.8925. One example
is not an independent scientific validation. A 448-pixel model grid can lose
small details in a large image. The panel retains its 64 MP input limit; this
integration is not a full-resolution or universally bounded-memory detector.

Recorded validation includes twelve postprocessing cases, twelve evaluator
comparisons with bit-identical grouped/loop outputs, independent regions, and an
isolated Qt/worker test covering source/target views, export and cancellation.
[Original measurement records and scripts](../tests/d2prl/) and
[checks rerun from this public export](NATIVE-SEPT30-PUBLIC-CHECKS.json) are kept
separately. The latter record states which checks were repeated.

## Two additional image tests

Two whole-image MPS tests using the unchanged 448-grid/40-iteration protocol
finished in 22.76 and 19.50 seconds. Both final masks were empty. The largest
components were only 77 and 58 pixels on the model grid, below the original
500-pixel filter. This is not a display failure, and an empty result does not
establish that an image is unedited. No threshold was changed to manufacture a
positive result. [Measurements](../tests/d2prl/user-images/results.json) and
[component diagnostics](../tests/d2prl/user-images/diagnostics.json) are supplied
without the images or private paths.

## Installation and external files

Apply the [cumulative RC1 update](ADAPTIVE-MEMORY.md#install-the-cumulative-rc1-update)
to the restored installation, or use the [source layout](../README.md#source-development).
The adapter uses PyTorch, torchvision, NumPy, SciPy, OpenCV, Pillow and
scikit-image from the existing native environment; no new Python package is
required beyond those dependencies.

Place the separately obtained final model at this installation-relative path:

```text
models/external/clone_detectors/01_d2prl/d2prl.pth
```

Expected size: **540,533,481 bytes**. SHA256:
`2749c7436169ce689deaeb197ce5dae3d1a4533999833928168ec0b0d703df36`.
OSN is for training and is not required for this inference.

The following filters from the pinned original repository must be present under
`third_party/research/clone_detectors/01_d2prl/`:

- `ZM_polar_k13.mat`
- `VV_mvf7.mat`
- `VV_mvf9.mat`
- `VV_mvf11.mat`

Their hashes are retained in the merged
[file inventory](../integration/clone_detectors/files.sha256.json). The restored
RC1 includes those filters; a fresh source export needs the external assets.
The updater includes the adapter and merged inventories, but **does not contain
or download the checkpoint**. The frozen six-part RC1 download predates this
integration and does not include the D2PRL checkpoint.

Code notices and checkpoint distribution rights are separate. The Apache-2.0
notice of the reference D2PRLu fork does not relicense the original repository
or its checkpoints. This publication contains no private download links,
correspondence or checkpoint files. See [component attribution](ATTRIBUTION.md).

To repeat the checks from the staged layout:

```sh
python tests/d2prl/contracts.py
python tests/d2prl/evaluator.py
# With the external model, filters and upstream example installed:
python tests/d2prl/public_inference.py mps
python tests/d2prl/public_inference.py cpu
QT_QPA_PLATFORM=offscreen python tests/d2prl/ui.py
```

The evaluator uses both CPU and MPS and therefore requires an MPS-capable Mac.
The [source manifest](D2PRL-SOURCE-MANIFEST.json) identifies the delivered files.
