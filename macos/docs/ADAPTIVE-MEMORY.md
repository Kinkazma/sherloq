# Adaptive memory: validated native paths

I added a shared memory coordinator, exact bounded-memory alternatives for nine
engines, compact and disk-backed dense descriptors, and tiled display of large
images. The normal RAM path remains the first choice when it fits. These changes
preserve the analysis resolution, selected iterations and global search domain;
display tiles are separate from analysis inputs and full-resolution exports.

This is a **partial set of validated paths**, not a claim that every SHERLOQ
function can process arbitrarily large images. Resource reservations apply within
a process and are not a strict application-wide RSS ceiling. Temporary disk
space, codecs and native index sizes still impose real limits.

## Nine engines with exact bounded alternatives

The public entry points for Gradient, Pixel Statistics, Bit Planes, Dead/Hot
Pixels, Space Conversion, Signal Separation, Echo, Min/Max and Global Adjustments
select between their RAM implementation and an exact bounded alternative.
Finite-support operations retain the necessary border overlap; global statistics,
equalisation, CLAHE and Otsu operations preserve their global calculations.

Recorded noisy 8000 × 12000 inputs (96 MP) matched the previous engine outputs
bit-for-bit. These measurements deliberately forced disk-backed operation even
though the images fitted in RAM on the test machine:

| Engine | Bounded peak RSS, bytes | Bounded / RAM time, seconds |
| --- | ---: | ---: |
| Luminance Gradient | 617,644,032 | 18.07 / 1.01 |
| Pixel Statistics | 476,938,240 | 1.17 / 0.36 |
| Bit Plane Values | 492,011,520 | 1.64 / 0.72 |
| Dead/Hot Pixels | 708,345,856 | 8.17 / 4.39 |
| Space Conversion | 493,076,480 | 1.54 / 0.97 |
| Signal Separation | 545,849,344 | 5.28 / 0.24 |
| Echo Edge Filter | 564,772,864 | 36.46 / 1.24 |
| Min/Max Deviation | 529,563,648 | 17.96 / 2.81 |
| Global Adjustments | 616,316,928 | 77.30 / 1.52 |

These are capacity measurements, **not speed gains**. Their source image was in
RAM. Disk-backed execution can be substantially slower and is not selected just
because an image has 96 MP. Variant and boundary tests include 72 Gradient,
54 Stats/Planes/Defects, 71 Color/Noise/Echo, 135 Min/Max and 90 Adjustments cases.
The public numerical checks were rerun from the exported repository before
publication. [Recorded cases and scripts](../tests/memory-resources/).

CMYK conversion also completed on one billion noisy pixels in 66.64 seconds at
245,153,792 bytes peak RSS. Every output pixel was checked against the earlier
pointwise calculation in blocks. Source and output were mapped; a full-size RAM
reference was not allocated. This result applies specifically to that conversion.
[CMYK record](../tests/memory-resources/gigapixel-results.json).

## Dense Zernike and SIFT

Dense searches can use ordinary RAM, exact compact descriptors in RAM, or mapped
storage. The compact SIFT representation reconstructs the original descriptor
components; search candidates and connected components remain global across
blocks. The CPU and Metal preparation paths are covered by the integrated checks.

- The freshly rebuilt public native bridge passed **44 integrated combinations**:
  normal/mirror, quarter turns, support sizes, full-image and region searches,
  cross-region comparisons, exclusions, CPU/Metal and compact/mapped storage.
- At 4,404,201 pixels and one iteration, a forced 512 MiB budget selected mapped
  execution. Results and maps matched RAM; the recorded times were **47.57 seconds
  mapped versus 12.15 seconds in RAM**.
- A separate 96 MP run searched **all pixels for one complete SIFT iteration**:
  2,179,905,460 comparisons, 1,492.74 seconds, 7,362,494,464 bytes peak RSS.
  This is not a validation of the complete eleven-pass profile at that size or
  a scientific accuracy benchmark. Completion of that full profile on the
  separately tested user photograph is also not established by this release.

[Integrated checks](../tests/dense-streaming/integration-production-results.json),
[forced mapped run](../tests/dense-streaming/automatic-fallback-results.json),
[96 MP all-pixel record](../tests/dense-streaming/huge-full-results.json).

I also fixed a separate coherence error: mathematically integer filter sums could
contain floating-point residues, incorrectly rejecting a perfect translation at
zero tolerance. Restoring those integer sums removes that defect in both paths.
This is an intentional numerical bug fix, distinct from the exact storage changes.
[Regression record](../tests/dense-streaming/coherence-rounding-results.json).

Dense-link deduplication remains exact. A recorded one-million-link benchmark
went from 0.709 to 0.038 seconds (18.7× in that test), benefiting both RAM and
mapped paths. It is not an overall application speed multiplier.

## Shared resources, display and worker inputs

Inactive caches and display tiles share an LRU budget. Memory reservations are
rechecked at admission, and the nine adapted entry points retry their bounded
path once after a reported NumPy/OpenCV allocation failure. Eighteen injected
failures preserve the expected results. This cannot recover a process killed by
the operating system.

Mapped storage remains alive while a viewer or worker holds a view. The tests
cover cancellation, errors, concurrent admission and release of resident pages,
including read-only mappings. Large images are displayed by visible tiles, with
exact pixels at 100% and source coordinates retained. A visible window from a
40000 × 25000 source was verified without allocating a full-size display texture.

Worker hashes and result validation are progressive. SAFIRE/FOCAL input resizing
precedes colour permutation, avoiding an unnecessary complete RGB copy; their
inference kernels are unchanged. The result-map and source-staging checks include
an actual 513 MiB file. These input/display improvements do not establish bounded
memory inside every model or codec.

## Install the cumulative RC1 update

The frozen six-part RC1 archive is unchanged. Restore it first if necessary,
close SHERLOQ, and run from the current repository with Python 3.11 or newer:

```sh
python3 macos/apply_rc1_updates.py "/path/to/SHERLOQ-installation"
```

This now validates **45 source/build files** and rebuilds the native PatchMatch
library plus its dense companion in a temporary directory, before replacing any
installed file. Apple's command-line developer tools are required. A compiler
failure leaves the installation unchanged. The older whole-image, ELA-slider
and PatchMatch-memory corrections are included.

The updater accepts original RC1 and the previously published source versions;
unknown local source edits are refused. Source files and existing native libraries
are backed up under `.updates/rc1-adaptive-memory-v1-20260929/files/`. Empty
`.absent` markers record newly added files. A replacement failure rolls back both
source and native files. Earlier update backups remain intact. A verified build
receipt avoids recompilation when the same update is run again.

No model download or app-bundle rebuild is needed. Reopen the application after
the update. Each native library file is replaced atomically.

## Reproduce the public checks

Use [the exported source layout](../README.md#source-development) and the
application's dependencies on a compatible Apple Silicon Mac:

```sh
python packaging/build_patchmatch.py
python tests/memory-resources/contracts.py
python tests/memory-resources/gradient_contract.py
python tests/memory-resources/local_contracts.py
python tests/memory-resources/more_local_contracts.py
python tests/memory-resources/minmax_contract.py
python tests/memory-resources/adjust_contract.py
python tests/memory-resources/retry_contract.py
QT_QPA_PLATFORM=offscreen python tests/memory-resources/viewer_contract.py
python tests/memory-resources/image_buffers_contract.py
python tests/memory-resources/worker_buffers_contract.py
python tests/dense-streaming/integration.py --production
python tests/dense-streaming/coherence_rounding.py
python tests/memory-resources/dense_links.py
```

[Public-export validation](ADAPTIVE-MEMORY-PUBLIC-CHECKS.json) records this
verification separately from the original large-input measurements. The
[source manifest](ADAPTIVE-MEMORY-SOURCE-MANIFEST.json) identifies the transferred
snapshot files. The optional `huge_*` and `gigapixel.py` scripts are capacity tests
with substantially greater runtime and temporary-storage needs.

From the repository, `python3 macos/tests/rc1-updater/check.py` exercises source
and native replacement failures on temporary fixtures. The
[installation checks](../tests/rc1-updater/adaptive-installation-results.json)
cover four previous source versions and a real isolated native build.
