# macOS Apple Silicon contribution — 29 September 2026

This is an independent, AI-assisted contribution to SHERLOQ by Guido Bartoli.
The original GPLv3 license and author credits are preserved. Third-party
components retain their individual licenses. This is a release candidate, not
an official upstream release. See [attribution](docs/ATTRIBUTION.md),
[changes](docs/CHANGELOG.md) and [technical notes](docs/TECHNICAL.md).

## Complete installation

The [GitHub Release](https://github.com/Kinkazma/sherloq/releases/tag/native-macos-arm64-2026.09.29-rc1)
contains the complete 10,894,627,579-byte installation ZIP in six parts smaller
than 2 GiB each. It includes the application, Python environments, native
dependencies, research components and all model weights present in the frozen
installation. The files are hosted on GitHub; no Mega account is needed.

Download this repository, then run with Python 3.11 or newer:

```sh
python3 macos/download_installation.py ~/Downloads/SHERLOQ-download
```

The script verifies every part, reuses completed parts after interruption,
assembles the ZIP and verifies its SHA-256. It does not execute the downloaded
software. Allow approximately 22 GB for downloads and assembly, plus 15 GB for
the extracted installation. Parts can be removed after successful assembly.

Extract `SHERLOQ-installation-macos-arm64.zip`. Inside its
`SHERLOQ-installation` directory, run:

```sh
python3 restore_installation.py
venv/bin/python app_start.py
```

Restoration requires macOS ARM64, Python 3.11+ and Apple's command-line developer
tools. It verifies the installation manifest, relocates bundled environments,
locally signs changed native libraries and builds `build/SHERLOQ.app`.
The application bundle can subsequently be opened in Finder.

## Source development

The fork keeps upstream's `gui/` tree so changes remain reviewable against the
original project. Native integration expects a containing installation directory
with `source/gui/`, `integration/`, `native/`, and `packaging/` as siblings.
Export a working layout to a new directory outside this repository:

```sh
python3 macos/stage_source.py ../SHERLOQ-native-source
```

The command copies sources without modifying their content. It does not supply
environments or models. The complete Release supplies those. Build scripts and
synthetic tests run from the exported layout, using a compatible environment:

```sh
cd ../SHERLOQ-native-source
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-macos-arm64.lock
.venv/bin/python tests/public/synthetic.py
.venv/bin/python tests/public/startup.py
```

The requirements file is an installed-version inventory, not a wheel-hash lock.
A clean network installation has not been validated. Do not apply the old
source-kit patch to this fork: the changes are already incorporated.

## Validation and limits

The frozen full ZIP was actually extracted and restored on the same Apple
Silicon Mac. Both included Python environments imported NumPy/OpenCV/PyTorch;
synthetic ELA, cache and image-memory tests passed; the Qt entry point reached
its main window offscreen. Native PatchMatch and ZERO libraries were rebuilt
from the exported sources. All 185 present checkpoint files matched the source
installation. Reports are in `docs/`.

This is not a test on a second Mac or a validation of every analysis/model.
Missing optional checkpoints are listed in `MISSING-WEIGHTS.md` inside the
installation. The browser port is separate and is not included here.
Private photographs, settings, logs, correspondence and private Git history
are excluded. Analysis outputs are investigative signals, not authenticity proofs.
