# Forgeryscope Auto in Clone Detectors

I added **Forgeryscope Auto** as a green choice in **Clone Detectors**, alongside
the four specialized Forgeryscope profiles. Select it and use **Search** to
classify panels and run the microscopy, western-blot and lane branches.
GPU is selected by default; CPU remains available explicitly.

This integration follows the [public Forgeryscope README at its pinned revision](https://github.com/vlad3996/forgeryscope/tree/63101e28a12daea6f8c9f6b1cd1ff3b8ffd2d7c8).
It uses the seven existing public `models-v1` weights and shared local keypoint
and LightGlue weights. It does not represent the complete competition submission.
The upstream code attribution and model terms remain applicable; the reference
code is MIT, while YOLO weights have their own terms.

## Search and interpretation

Each active zone is processed independently. Use an enclosing zone to compare
panels within it. Panels touching excluded areas are removed. **Compare** is
disabled for Auto because two manually selected rectangles do not supply their
panel classes. The specialized profiles retain their comparison controls.

1. Detect microscopy and blot panels once per active zone.
2. Combine blot duplication candidates (threshold 0.84) and overlap candidates
   (0.85), retaining the maximum score for each pair. Microscopy uses 0.58.
3. Exclude intersecting panels, then run SIFT/LightGlue or ALIKED/LightGlue
   matching. Microscopy requires at least 8 inliers and mean match score 0.73.
4. Retain blot decisions according to the public pipeline, including its
   fallback without geometric confirmation. Merge masks using the upstream
   clique routine.
5. Search lanes at threshold 0.65 only when blots exist and there was no blot
   candidate pair **before intersection exclusions**. The public rule covers
   the whole panel when more than 50% of its lanes are matched.

The combined mask includes the public pipeline's accepted blot and lane
decisions; it is not a mask of exclusively geometrically confirmed matches.
The separate `candidates` and `geometric` arrays preserve that distinction.

## Views and export

Alongside the combined overlay, map, mask and suggestions, the panel provides
**Microscopy**, **Western blots**, **Lanes** and **Geometric matches** views.
Changing views reuses the computed results. NPZ export includes the branch
arrays, panels, scores, thresholds, decision provenance, merged groups and lane
pairs. The existing D2PRL controls remain available when that backend is selected.

## Automatic Clone Search and Complete Automatic Analysis

Both combined analyses use Forgeryscope Auto on the enabled enclosing zone.
Disabled subimages remain excluded; disabling the enclosing zone disables the
Forgeryscope branch. PatchMatch retains its extended mirror-and-scale profile,
and Complete Automatic Analysis retains its ELA/Ghosts branch.

The combined region legend identifies microscopy, blots and lanes, and labels
geometric evidence separately from similarity evidence. Lane regions follow
the upstream whole-panel expansion while retaining the original lane rectangles
in metadata. These regions represent pairs; original masks and merged clique
groups remain available in exports.

A branch selector filters only the Forgeryscope display, without recomputing
models or hiding the other engines. Hidden regions, identifiers and colours
remain stable. The existing display defaults remain 10 pixels minimum and an
80% overlap exclusion; these are display filters, not detector thresholds.
Complete Analysis keeps its independent source checkboxes and ELA controls.
Exports record the selected branch and hidden-region state.

## Installation

Apply the [cumulative RC1 update](../README.md#apply-the-cumulative-rc1-updates)
to a restored complete installation. It includes `core/forgeryscope_auto.py`,
the adapters, panel, merged French translations and the checkpoint class
inventory required by the local YOLO loader. That inventory was missing from
the earlier distribution. No new weights are required for Auto on an installation
with the existing Forgeryscope and shared matching weights. The updater does
not download checkpoints or run a preliminary calibration.

Model loading remains local: YOLO, keypoints and geometry use CPU; embeddings
and LightGlue use MPS when GPU is selected. This is not a claim of numerical
identity with a CUDA implementation.

## Validation and provenance

The [standalone source manifest](FORGERYSCOPE-AUTO-SOURCE-MANIFEST.json) and
[combined-analysis source manifest](FORGERYSCOPE-AUTOMATIC-SOURCE-MANIFEST.json)
identify the two delivered steps and merged translations. The
[public integration checks](FORGERYSCOPE-AUTO-PUBLIC-CHECKS.json) distinguish
tests rerun on the exported source from the native delivery's recorded trial.

- Controlled contracts cover maximum-score pair fusion, cliques, blot fallback,
  lane fallback and its ordering, whole-panel lane coverage, exclusions,
  microscopy rejection, independent zones and branch propagation.
- The real Qt panel and resident worker pass on a uniform synthetic image:
  no panels, empty output, branch views and NPZ export.
- A delivered native trial on a 3226 × 936 image found 16 panels and 23 candidate
  pairs, with 955,182 pixels in the final mask. The mask exactly matched a
  separate transcription of the public README. That trial's retained output
  was entirely microscopy; it does not establish western-blot detection quality.
  Model/device adapters were shared, so the comparison validates orchestration,
  not independent model accuracy. Its 91.41-second duration includes loading
  and concurrent activity and is not a speed guarantee.
- D2PRL postprocessing and cached-filter UI regression checks pass. The
  cumulative updater passes transactional failure checks and migration from
  prior public revisions, including safe repeated application.

The [combined-analysis checks](FORGERYSCOPE-AUTOMATIC-PUBLIC-CHECKS.json) cover
both panels, branch routing, evidence labels, lane expansion, display-only
filtering, hidden-state persistence, exclusions and exports. A real combined
run on a synthetic 256 × 128 image exercised PatchMatch, Forgeryscope Auto and
ELA/Ghosts together and checked exact exported arrays. The test checks wiring;
Forgeryscope retained no matches on that input, so it does not demonstrate detection
quality on scientific figures. Mirror matching and all four SIFT scales were
verified. The cumulative updater was exercised from eight prior public revisions.

The native trial image and its maps are not included. Test programs and recorded
results are under [forgeryscope-auto](../tests/forgeryscope-auto); their layout
expects `tests/` and `source/gui/` under an isolated installation root with the
installed clone-detector environment and resources. Run contracts with that
environment; run the UI test with `QT_QPA_PLATFORM=offscreen` in the GUI environment.
