# Public change index

Scope: exported native source relative to the upstream revision in README.
The patch is an exact source transfer, not a claim that every feature has passed
scientific or end-user validation. Private changelogs and test images are omitted.
For an exact inherited/added comparison, see [the 50-entry tool inventory](TOOLS.md).

## 30 September 2026 — automatic analyses, mirrored SIFT and D2PRL regions

I added the full mirrored SIFT panels/text profile and D2PRL to both automatic
workflows. D2PRL analyses every enabled subimage plus the enabled enclosing image;
its native-grid size filter updates cached results without another inference.

Corroboration now counts distinct method × search-region contexts, with no size
weighting and at most one D2PRL vote per pixel. ELA is excluded from these maps.
Relation filters apply only to classical biomes; AI layers remain independent.
Tabs retain their own view/filter choices, display changes keep the zoom, and
progress reports real group states and regional passes. Near-identical classical
pairs are deduplicated visually while exact masks and raw results are retained.

The validated ELA slider behaviour is preserved. The known 1,000 px compact-layout
limit is documented. Download hooks, portable OCR and model files are preserved.
[Behaviour, validation and limits](AUTOMATIC-ANALYSIS-V2.md).

## 30 September 2026 — SIFT panels/text, model acquisition and AI category

I added SIFT + G2NN + RANSAC + Panels + Text to Copy-Move Forgery 2 and as a separate
layer in Complete Automatic Analysis. Surviving RANSAC links are regrouped to split
disconnected envelopes without losing links or imposing a size cap. Portable
Tesseract and English data are included with the cumulative source update.
[Method and checks](SIFT-PANELS-TEXT.md).

I added cancellable, resumable first-use model downloads with size/hash verification,
local reuse and dependency deduplication. [Supported jobs and offline preparation](MODEL-DOWNLOADS.md).

I renamed Clone Detectors to **AI Clone Detection** and moved it to **AI Solutions**,
after AdaIFL. Its internal identity, saved favorites and green marker are preserved.
The French label is **Détection de clones par IA**. The menu still has 50 entries.

## 30 September 2026 — Forgeryscope Auto in both automatic analyses

I connected the validated Auto pipeline to Automatic Clone Search and Complete
Automatic Analysis on the enabled enclosing zone, retaining subimage exclusions.
The region legend distinguishes microscopy, blots and lanes, with geometric or
similarity evidence. A display-only branch selector preserves hidden regions and
does not rerun models; exports retain this display state and the original maps.
PatchMatch keeps its extended mirror-and-scale profile and Complete Analysis
keeps ELA/Ghosts. The updater now validates 69 source/build/inventory files.
[Behaviour and validation](FORGERYSCOPE-AUTO.md#automatic-clone-search-and-complete-automatic-analysis).

## 30 September 2026 — standalone Forgeryscope Auto

I added a green Forgeryscope Auto choice in Clone Detectors, following the public
upstream pipeline for microscopy, western blots and lanes. It merges blot
candidate pairs by maximum score, preserves the public lane fallback ordering
and whole-panel rule, and exposes branch maps and distinct geometric/similarity
evidence in views and NPZ exports. Search classifies panels within each active
zone; Auto does not infer classes for manual Compare rectangles.
The seven existing weights are reused. The cumulative updater includes the new
module and checkpoint class inventory, with merged French translations.
[Usage, pipeline details and validation](FORGERYSCOPE-AUTO.md).

## 30 September 2026 — interactive D2PRL region filtering

I added a 0–5,000 region-size slider and synchronized numeric field, measured on
the 448 × 448 model grid. The default 500 preserves the original postprocessing;
0 disables only small-component removal. Cached raw maps are refiltered without
another inference. Masks, source/target views, overlays and exports follow the
latest setting; export waits for filtering. The probability-map view stays raw.
The inference checkpoint is also available as a separate GitHub Release asset,
with a size/SHA256-verifying installer; it is not added to Git history or the
frozen RC1 ZIP. [Model provenance and terms](D2PRL-MODEL-NOTICE.md).
D2PRL overlays now respect the retained mask. Rapid changes and invalidation
are covered by Qt tests. [Usage, measurements and checks](D2PRL-INTERACTIVE-FILTER.md).

## 30 September 2026 — D2PRL and four more bounded-memory engines

I integrated the original D2PRL inference into Clone Detectors, with GPU as the
default, an explicit CPU choice, independent regions and union/source/target
views and exports. The optimized MPS path preserved the initial MPS port's raw
maps on the recorded author example (401.42 to 25.76 seconds); CPU/GPU differences
remain documented. The original 448-pixel grid and 40 iterations are retained.
Weights are external and are not added to the frozen RC1 archive.
[Protocol, dependencies and measured limits](D2PRL-INTEGRATION.md).

I added exact bounded alternatives for Wavelet Threshold, Wavelet Blocking,
Illuminant Map and Contrast Enhancement, bringing the validated engine coverage
to thirteen. All four matched their previous reference on noisy 96 MP inputs.
Wavelet transforms keep their global axes. Optional resident pages use spare
capacity without a preliminary benchmark. The mapped Wavelet Blocking layout
improved from 185 to 20 seconds in its recorded test; RAM remained faster.
[Memory scope and measurements](ADAPTIVE-MEMORY.md).

The cumulative updater validates 65 source/build/inventory files, preserves
previous backups, adds the D2PRL package transactionally, and merges its inventory
entries while retaining the other detectors. I also restored the baseline hash
inventory and three runtime license files omitted from the earlier source export.
No checkpoints are downloaded by the updater.

## 29 September 2026 — adaptive memory, validated native paths

I added exact bounded alternatives for nine engines, shared resource admission
and cache pressure handling, compact/mapped Zernike and SIFT, and tiled display
with full-resolution source/export data. The normal RAM path stays preferred;
mapped execution has a measured cost. The rebuilt bridge passed 44 integrated
combinations. Recorded capacity checks include nine exact engines at 96 MP,
CMYK at one billion pixels, and all-pixel SIFT at 96 MP for one iteration only.
These do not establish complete-tool coverage or a completed eleven-pass profile.

A separate coherence fix restores mathematically integer filter sums, removing
false rejections of perfect translations at zero tolerance. Dense-link
deduplication remains exact. Worker input copies and result-map validation are
also reduced. The cumulative RC1 updater now validates 45 source/build files and
rebuilds the two native libraries before replacing installed files.
[Scope, measured costs, proofs, limitations and update](ADAPTIVE-MEMORY.md).

## 29 September 2026 — PatchMatch large-image memory

I reduced temporary allocations in the Metal SIFT extended + mirror path and
adapted its reservations, retaining the shared memory budget. On the 3510 × 4387
case, paired reservations fall from about 45 GiB to at most 17.41 GiB. The CPU
path is unchanged. Controlled descriptor/field comparisons remain bit-exact;
all four full-resolution mirrored descriptor preparations completed. The full
profile run was deliberately cancelled after several completed stages, so it
is not claimed as an end-to-end validation on that image.
[Changes, measured limits, checks and cumulative RC1 update](PATCHMATCH-MEMORY.md).

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
