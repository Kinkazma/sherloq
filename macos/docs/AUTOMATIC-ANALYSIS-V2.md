# Automatic analyses: mirrored SIFT, D2PRL and corroboration

I integrated the full **SIFT + G2NN + RANSAC + Panels + Text** pipeline into both
Automatic Clone Search and Complete Automatic Analysis. Its tab is shortened to
**SIFT + G2NN + RANSAC**, while the tooltip, internal identifier and exports retain
the full profile name. This is the panels/text pipeline, not the earlier classic
SIFT profile. Its real reflected-feature pass is enabled by default. PatchMatch
retains its extended profile with both mirrors and scales.

Both workflows combine Forgeryscope Auto, PatchMatch Zernike, PatchMatch SIFT,
the panels/text SIFT profile and D2PRL. Complete Automatic Analysis also includes
ELA/JPEG-ghost views. There are four computational job groups without ELA and five
with ELA; the number of tabs is not the number of jobs.

## Regions, views and corroboration

Classical methods search each active region and the active enclosing region.
Forgeryscope Auto retains its global scope. D2PRL now runs a pass for every active
subimage and the active enclosing image, preserving exclusions. Turning off the
enclosing pass does not remove D2PRL's enabled local passes.

The combined tab initially shows corroboration on the image, **Within zones**,
with 70% opacity. An isolated method initially opens in **Biomes / All relations**.
Each tab then remembers its own display choices, including separate relation
filters for biomes and corroboration. Map-only and overlaid maps share a relation
selection. These display changes reuse cached results.

Within / Between / All relations filters affect classical biomes only. Forgeryscope
and D2PRL remain visible and counted unless their source is unchecked or another
method is isolated. The enclosing search is retained. Unassigned classical
relations remain available under All relations.

Corroboration counts distinct **method × search-region contexts** at each pixel,
using exact uint32 integers without size weighting. Multiple biomes from the same
context contribute once; different regional searches preserve their provenance.
D2PRL contributes at most one vote per pixel from the union of its passes. ELA never
votes. The colours show 1 through 6+, while NPZ export preserves uncapped counts.
These counts are neither probabilities nor evidence of statistical independence
between the methods.

In a corroboration view, ELA's background and biomes are temporarily hidden.
Returning to Biomes restores the earlier ELA preferences. Large classical biomes
remain present, with less opaque fill for readability; smaller biomes are drawn
later and take priority when selected under the pointer. Near-identical classical
pairs are drawn once when both endpoints overlap by at least 90% IoU, without
creating a new convex hull. AI masks remain separate. D2PRL retains holes and
disconnected areas rather than replacing them with a convex envelope.

## Interactive D2PRL, zoom and progress

The 0–5,000 D2PRL size filter defaults to **500 pixels** and is available in the
combined and D2PRL tabs, beside Export. Filtering occurs independently on each
native 448 × 448 model grid, followed by reprojection and union. Raw cached outputs
are reused: changing the filter updates biomes/corroboration without running the
network or the other detectors again. GPU remains the default; CPU is selectable.

Updates with the same image dimensions preserve the zoom and viewport. Progress
shows completed groups, real method states/percentages and the most recent active
engine message. D2PRL progress includes all requested regions. Waiting, running,
finalizing, finished, failed and cancelled are distinguished in the existing status
lines without changing their height.

## Validation and limits

The supplied native delivery passed 64 Qt offscreen ELA input/race cases, D2PRL UI
and detailed progress checks. The ELA profiles and equations are unchanged by this
integration. Independent public checks and source/merge hashes are linked below.
UI fixtures that substitute detectors are distinct from real inference checks.
The real three-region D2PRL development test substituted other detectors; it is not
a full multi-panel board or scientific accuracy validation.

The historical compact-layout assertion at **1,000 px width fails** with the added
controls. That width is not validated. The ELA input regression evidence is Qt
offscreen, not a new physical trackpad/macOS gesture test.

[Source and merge inventory](AUTOMATIC-V2-SOURCE-MANIFEST.json) ·
[Independent checks](AUTOMATIC-V2-PUBLIC-CHECKS.json) ·
[Install the cumulative update](../README.md#apply-the-cumulative-rc1-updates).
