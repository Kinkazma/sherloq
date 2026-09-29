# Public change index

Scope: exported native source relative to the upstream revision in README.
The patch is an exact source transfer, not a claim that every feature has passed
scientific or end-user validation. Private changelogs and test images are omitted.
For an exact inherited/added comparison, see [the 50-entry tool inventory](TOOLS.md).

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
