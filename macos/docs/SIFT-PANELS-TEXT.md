# SIFT + G2NN + RANSAC + Panels + Text

I added this green profile to **Copy-Move Forgery 2**. When no manual zones are
selected, it locates panels and an enclosing zone. Tesseract locates likely labels
on flat backgrounds; those regions are excluded before SIFT extraction. Each zone
is extracted independently, using the existing G2NN matching and RANSAC geometric
verification. Manual regions and exclusions remain available; pairs whose two
regions overlap by at least 80% are rejected.

The profile is a SHERLOQ integration. It does not reproduce the luc_pub competition
ensemble or add a graph classifier. Text exclusion is conservative: labels on
textured backgrounds may remain.

## Disconnected groups after RANSAC

I regroup the surviving geometric correspondences at both endpoints using the same
proximity setting. This separates disconnected pieces whose earlier connecting
matches were rejected. There is no new size cap, and retained links are not deleted.
Connected groups keep their geometry and colour. Separated pieces retain a family
of shades and the parent's transform; a small piece is not a new independent RANSAC
estimate. Exports include parent/group provenance and geometric errors.

In the supplied development-board test, 26 groups became 41 parts across nine
split parents. The large red pair retained all 313 links and its geometry/colour.
Regrouping alone took 0.134 seconds in that test. These are recorded observations,
not general speed or detection-quality guarantees. The image and raw private
reference are not part of the test fixtures. Envelopes illustrate match extent;
they are not precise segmentation masks or proof of manipulation.

## Complete Automatic Analysis

The complete workflow runs this profile as an independent layer, with a tab,
visibility checkbox, cancellation, active regions and exclusions. Display changes
reuse cached results. ELA settings do not restart SIFT. NPZ export retains SIFT
results, parameters, text exclusions and group provenance.

Automatic Clone Search alone keeps its previous sources. Extended PatchMatch
retains both mirror and scale matching. Forgeryscope Auto and ELA/Ghosts keep their
existing roles.

## Installation and validation

Apply the [cumulative RC1 update](../README.md#apply-the-cumulative-rc1-updates).
The update includes portable ARM64 Tesseract 5.5.3, its dependent libraries,
English language data and component notices. `SHERLOQ_TESSERACT` can explicitly
select another executable; otherwise the bundled executable is preferred to PATH.
Missing OCR dependencies raise an error rather than silently dropping text exclusion.

Synthetic checks cover real OCR/SIFT, isolated search versus comparison, labels,
exclusions, cancellation, cache reuse, UI display and NPZ export. Group contracts
include a 693-link connected group preserved intact and separated clusters with no
lost links. Complete-workflow UI fixtures explicitly stub other clone jobs; a
separate 128 × 256 check runs all engines for real and verifies the exported arrays.
[Public integration results](SIFT-MODELS-PUBLIC-CHECKS.json).
