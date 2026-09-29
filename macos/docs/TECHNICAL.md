# Technical handover

The application remains Python/Qt with shared pure computation helpers and worker
services. Exports retain the modified modules and new helpers as one dependency
closure. Research adapters and their present third-party implementations are retained.
Weights and installed runtimes are supplied in the companion installation ZIP.
The contributor confirms having obtained the necessary permissions by email.

The reference is a frozen list of input SHA-256 values, not the mutable local
working tree. `SOURCE-INPUTS.json` names every selected source, lock and build
script. An original absolute build command in the PatchMatch README was replaced
with a relative command. No algorithm was edited for packaging.

The dependency version list is the existing macOS environment inventory. It is
not a wheel-hash lock, does not lock native system libraries, and does not establish
availability on other platforms. Fresh dependency installation is a separate
validation gate. Build scripts reconstruct the native libraries from retained
sources on macOS with command-line developer tools. Code signing and compiler
versions can change binary bytes; bit-reproducible binaries are not claimed.
The source ZIP itself has deterministic ordering, modes, timestamps and contents.

Synthetic tests check ELA against a direct OpenCV formula, including negative
compression errors; exact repeatability and retained cache behavior; image
memory ownership; and a Qt main-window startup in an isolated settings directory.
These tests do not measure false positives or scientific efficacy. They do not
validate all 50 tool entries, a physical GPU, excluded weights or every native backend.
Timing values from a small synthetic test are not end-to-end speedups. No legacy
performance claim is carried into this candidate without a redistributable test.
