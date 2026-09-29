# IPOL dense descriptors — SHERLOQ ARM64 adaptation

Source: https://www.ipol.im/pub/art/2018/213/ (forgeryDetection_1.tgz).
Archive SHA-256: 1bb512ac36b540a437a0064b7f1b78b15a39a5fc205fef2db2d0d50ac1e34bd4.
Original author: Thibaud Ehret. GPL-3.0-or-later (GPLv3.txt); bundled VLFeat uses its retained BSD-2-Clause COPYING.

Original README retained as UPSTREAM.txt. Local changes: guarded x86 CPUID, removed unused FFTW include, virtual feature-manager destructor, contiguous feature export, no redundant unflipped second field, deterministic independent-pixel Zernike parallelism using macOS libdispatch. bridge.cpp adds a constrained, seeded PatchMatch variant for ROIs/radius, with cancellation and explicit unmatched pixels. The Python adapter normalizes descriptors and applies full-field local affine consistency before display sampling. These are adaptations, not the original IPOL final segmentation pipeline.

Build: python packaging/build_patchmatch.py from the kit root. Requires Apple command-line developer tools; no CUDA or FFTW dependency. No fast-math.
