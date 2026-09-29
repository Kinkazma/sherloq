# Public change index

Scope: exported native source relative to the upstream revision in README.
The patch is an exact source transfer, not a claim that every feature has passed
scientific or end-user validation. Private changelogs and test images are omitted.
For an exact inherited/added comparison, see [the 50-entry tool inventory](TOOLS.md).

## 29 September 2026 — ELA manual input takes precedence

I fixed a race where an automatic estimate already in progress could overwrite
all four ELA controls after manual input at a slider limit. Both the ELA biomes
panel and Complete Automatic Analysis now take manual control before Qt processes
the input and reject the obsolete response. The four values and their geometry
passed 64 Cocoa cases; the numerical formulas are unchanged.
[Reproduction, validation and cumulative RC1 update](ELA-SLIDERS.md).

## 29 September 2026 — whole-image analysis override

I added **Run on whole image** to Complete Automatic Analysis and Automatic
Clone Search. Subimage detection remains the default; the new button clears
regions and exclusions, cancels the old work and restarts on the entire image.
An early click is preserved instead of being overwritten by delayed startup
detection. Users can return to **Detect subimages** afterwards.

The French interface includes **Exécuter sur toute l’image**. Tests cover both
panels and obsolete asynchronous results, using real Qt jobs and ELA with
explicit clone fixtures. The scientific algorithms are unchanged.
[Usage, RC1 updater and reproducible regression check](WHOLE-IMAGE.md).

## Tools made usable under existing menu entries

I implemented **Illuminant Map** and **Dead/Hot Pixels**, whose menu labels existed in the baseline but had no working dispatch or implementation. Illuminant Map estimates local illuminant colour with Gray World, Shades of Gray and White Patch; Dead/Hot Pixels detects isolated candidates and offers overlays, correction previews and coordinate exports. These are new implementations under inherited labels, not just interface adjustments.

I restored the disabled **Multiple Compression** launch path, added cancellation/resume and CSV/PNG exports to its recompression curve, and integrated an experimental aligned double-JPEG analysis tab. **TruFor** was already an upstream method; my work connects its model, CPU/Metal inference, exports, worker process and large-image memory handling to the macOS application.

## Correctness and stability fixes

| Area | Problem addressed | Result |
| --- | --- | --- |
| Linear ELA | Unsigned subtraction discarded negative compression errors | Absolute differences include errors in both directions |
| ELA contrast | Endpoints crossed at 100% | Monotone bounded contrast mapping |
| Reference Comparison | PSNR used a doubled factor | Correct power-ratio formula; individual metric errors handled separately |
| Histograms | Large float32 counts lost units; point/empty ranges and fullness were misleading | Blockwise exact counts, corrected range endpoints and percentages |
| Frequency Split | FFT shifting included the real/imaginary component axis | Explicit spatial axes preserve component order and correct phase |
| Wavelets | Odd-sized reconstruction retained an extra row/column; zero bands needed guards | Original dimensions retained, zero-band handling explicit |
| Colour conversion | Pure black produced incorrect CMYK values | Black maps to the expected zero-CMY/full-K representation |
| Bit planes | An all-one plane normalized to black | Constant active plane displayed as white |
| PCA and neighbourhood tools | Tiny images could fail in decomposition or neighbourhood operations | Explicit small-image handling |
| Copy-move | Uniform/no-feature images could fail without descriptors | Explicit no-descriptor handling |
| JPEG ghost normalization | Constant ranges caused division by zero | Defined masked normalization |
| Noiseprint | Removed NumPy aliases and changed SciPy interfaces broke compatibility | Compatible type and eigenvalue calls |
| File opening and UI lifetime | Cancelled openings, drag/drop, Finder events and closing workers needed handling | Current image preserved on failed/cancelled opening; worker/window cleanup improved |

## Performance and interaction

I separated expensive computation from display-only changes, retained intermediate results in bounded caches, added background jobs and improved cancellation and stale-result handling. Histogram counting avoids sorting millions of RGB triplets; comparison work uses separate processes; transform and reconstruction work is reused where applicable. Selected native CPU/Metal paths have fallbacks; prototypes that changed results or lost their gains to transfers were not adopted merely for being GPU-based.

[Recorded measurements](PERFORMANCE.md) include histogram readiness of 14.858 → 0.493 seconds, Frequency Split readiness of 10.805 → 3.164 seconds, and PCA peak process memory of 5.29 → 3.01 GB in specific local 20 MP profiles. The accompanying records state what was timed and the cases that became slower. These are historical development measurements, not a universal release-performance claim.

## Added analysis and everyday use

- Twelve additional menu entries: C2PA Validation, Noisesniffer, ZERO JPEG Grids, Copy-Move Forgery 2, Adaptive CFA, Clone Detectors, Automatic Clone Search, Complete Automatic Analysis, CAT-Net v2, SAFIRE, FOCAL and AdaIFL.
- Research-method integrations retain their original authors and licenses; my contribution includes the surrounding workers, parameters, displays and exports.
- Copy-move workflows add region selection, dense descriptors, constrained learned matching, mirror/overlap handling and grouped overlays.
- Automatic analysis combines clone-search branches with ELA/JPEG-ghost views; sources and result groups can be shown separately.
- Interface work adds language switching, persistent favorites, layered ELA views, profiles, region controls and zoom/gesture handling.
- macOS support includes native launch/build sources, ARM64 dependencies, bundled environments, restoration and locally signed application builds.

## Code map

| Area | Changes visible in source | Principal paths under source/gui/sherloq_app |
| --- | --- | --- |
| Correctness | absolute linear ELA differences; contrast endpoint bound; explicit decoding/display ownership; one-based JPEG quality selection for splicing | core/ela.py, core/image_io.py, core/utility.py, core/splicing.py |
| Responsiveness | retained computations, bounded caches, background jobs, cancellation and services | core/interactive.py, core/cache_budget.py, ui/jobs.py, ui/heavy_jobs.py, ui/*_service.py |
| Inspection | comparison, digest, histogram, magnifier and display controls | core/comparison*.py, core/digest.py, tools/inspection |
| JPEG / noise | ELA exploration, ghost maps, double JPEG, wavelet/noise methods, ZERO and Noisesniffer adapters | core/ela*.py, core/double_jpeg.py, core/zero.py, core/noisesniffer.py |
| Copy-move | ROI workflows, Copy-Move 2, dense descriptors, constrained learned matching, mirror and overlap analysis | core/cloning*.py, core/dense*.py, core/copy_*.py, core/learned_copy.py |
| Additional analysis | illuminant maps, defect candidates, provenance/C2PA and optional research adapters | tools/various, core/illuminant.py, core/defect_pixels.py, core/c2pa.py |
| Interface | localization, layered views, profiles, region selection and gestures | ui/locales, ui/localization.py, ui/ela_profiles.py, ui/region_viewer.py, ui/gestures.py |
| macOS kernels | CPU native descriptors and optional Metal kernels; ZERO reference/parallel source | vendor/patchmatch, vendor/zero; packaging/build_*.py |

See ATTRIBUTION.md for original notices and the confirmed handover permissions.
The native patch contains shared dependencies: splitting by filename alone would
not produce independently runnable PRs. See PR-PLAN.md.
