# PatchMatch: large-image memory on Apple Silicon

This note records the earlier correction published in commit `7992b90`. The later
[adaptive-memory changes](ADAPTIVE-MEMORY.md) extend the CPU/Metal paths and supersede
its installation procedure; the measurements below describe that earlier state.

I fixed an early memory-budget refusal in the extended + mirror PatchMatch
profile. A 3510 × 4387 image was rejected before SIFT matching because the Metal
path was charged for copies used by the older CPU implementation. A pair of
support sizes requested about 45 GiB against a shared budget of 21.33 GiB on
the 64 GiB test Mac.

The Metal SIFT path now normalises descriptors in blocks, mirrors and compacts
fresh output buffers in place, and builds only the source and target fields
actually used by a pair. Quarter-turn canonicalisation can reuse these fresh
buffers; its public default still returns a separate copy for other callers.
The image resolution, descriptors and selected profile passes are retained.

For that image, the reservation is about 10.13 GiB for a normal field and up to
17.41 GiB for a pair. The shared budget remains one third of physical RAM,
capped at 24 GiB: 21.33 GiB on the test machine. Large SIFT pairs are therefore
serialised by the existing budget. This is a working-memory reservation, not a
limit on total process memory, retained caches or other applications.

The CPU path and its existing guards are unchanged; it can still refuse this
image size. Smaller-memory machines may also need smaller regions. No native
library rebuild is required.

## What was validated

- 30 descriptor comparisons and six PatchMatch fields are bit-for-bit equal
  to the previous implementation on controlled inputs, including mirrored
  supports, sizes 6/8/10/12 and compaction. In-place canonicalisation is exact.
- Budget checks verify that large pairs run one at a time and that exceptions
  and cancellation release their reservations.
- The native extended + mirror regression suite passed, including retained
  normal/mirror outputs, a controlled CPU/Metal comparison, caching and region
  isolation. Its recorded result is included; its separate image corpus is not.
- All four mirrored descriptor preparations at 3510 × 4387 completed, including
  normalisation, compaction, canonicalisation and orientation checks. Peak
  process RSS was 15.63 GiB. **This check does not perform matching.**
- During a separate full-resolution profile run, the normal duo, canonical
  SIFT at size 8 and the 8/6 pair completed. The run was then deliberately
  cancelled during the next pair. **The complete profile on that image has not
  been validated to completion.** No final detection result or total runtime
  is claimed for it.

The controlled numerical and budget checks were also rerun from an independent
export of the public source before publication. Recorded files and scripts are
in [tests/dense-large-memory](../tests/dense-large-memory/). Private images and
their run logs are not included.

## Update an installed RC1

The current cumulative updater also installs the later adaptive-memory paths and
rebuilds the native bridge. Follow [the current update instructions](ADAPTIVE-MEMORY.md#install-the-cumulative-rc1-update).

## Reproduce the checks

After [exporting the source](../README.md#source-development), use the
application's Python environment and existing native library:

```sh
python tests/dense-large-memory/check.py
python tests/dense-large-memory/budget.py
python tests/dense-large-memory/full_descriptors.py "/path/to/input-image.png"
```

These Metal checks require a compatible Apple Silicon environment. The supplied
large-image budget assertions describe the 64 GiB test machine. The last command
uses the full resolution of the input and can require substantial memory; it
only tests preparation. `real_image.py` is a separate optional full-profile
runner and only writes a success result if the entire run completes.

The updater's file handling can be checked directly from the repository with
`python3 macos/tests/rc1-updater/check.py`; it operates only on temporary fixtures.
