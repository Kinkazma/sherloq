# macOS Apple Silicon contribution — 29 September 2026

I maintain this independent fork of SHERLOQ by Guido Bartoli, with AI-assisted development.
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

### Prerequisites

Use an Apple Silicon Mac (ARM64), Python 3.11 or newer, and Apple's command-line developer tools. Check the active Python with `python3 --version` and the architecture with `uname -m` (expected: `arm64`). If your Python 3.11+ executable is named `python3.11` or `python3.12`, use that name in place of `python3` in the download and restoration commands.

Check the developer tools with `xcode-select -p`. If they are absent, run `xcode-select --install` and complete Apple's installer before restoring. The bundled launcher is built locally; this is not an Apple-notarized application distribution.

### Download and restore

Obtain this repository with GitHub's **Code → Download ZIP**, then extract it and open Terminal in its folder. Alternatively:

```sh
git clone https://github.com/Kinkazma/sherloq.git
cd sherloq
```

Run with Python 3.11 or newer:

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
```

Restoration requires macOS ARM64, Python 3.11+ and Apple's command-line developer
tools. It verifies the installation manifest, relocates bundled environments,
locally signs changed native libraries and builds `build/SHERLOQ.app`.
The application bundle can subsequently be opened in Finder. Keep it with the complete installation: `build/SHERLOQ.app` refers to the containing directory and is not a standalone app that can be moved on its own. Choose the final installation location before restoring. If you need to relocate it later, extract a fresh copy of the verified ZIP at the new location and restore that copy; the original manifest describes the files before relocation.

### Apply the cumulative RC1 updates

The frozen RC1 archive predates **Run on whole image**, the ELA slider correction and the adaptive-memory changes. After restoration, close
SHERLOQ and return to the **current repository folder** in Terminal. Run:

```sh
python3 macos/apply_rc1_updates.py "/path/to/SHERLOQ-installation"
```

Replace the quoted path with your restored installation folder. The updater
validates 65 source/build/inventory files, rebuilds the native PatchMatch libraries, and backs up replaced files; it refuses unknown local changes and can be
rerun safely. Then open `build/SHERLOQ.app` in the installation folder, or run
`venv/bin/python app_start.py` there. Existing restored installations use the
same update command without repeating restoration. [Whole-image behaviour](docs/WHOLE-IMAGE.md) · [ELA controls](docs/ELA-SLIDERS.md) · [Adaptive-memory scope and update details](docs/ADAPTIVE-MEMORY.md).

### Manual download and assembly

If you prefer browser downloads, save **all six** `.part01` through `.part06` files and `SHA256SUMS` from the Release into the same folder. In Terminal, enter that folder and run:

```sh
shasum -a 256 -c SHA256SUMS
cat SHERLOQ-installation-macos-arm64.zip.part0{1,2,3,4,5,6} > SHERLOQ-installation-macos-arm64.zip
shasum -a 256 SHERLOQ-installation-macos-arm64.zip
```

Do not continue if any part fails verification. The final ZIP must have SHA-256:

```text
f1cbdaedf5f287630eb0fdb6c06a161643e9b16614c65663e963afd33a5f427d
```

Extract the ZIP and follow the restoration commands above. The numbered files are raw parts of one ZIP; do not try to extract them individually. GitHub's automatic **Source code (zip)** and **Source code (tar.gz)** downloads provide sources, not the installed environments or model bundle.

If a downloaded part is interrupted or incorrect, rerun the downloader: completed valid parts are kept, and invalid parts are downloaded again. After successful assembly and verification, you may remove the six part files to recover space; keep the verified ZIP if you want an untouched restoration copy.

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
.venv/bin/python packaging/build_patchmatch.py
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
installation.
Private photographs, settings, logs, correspondence and private Git history
are excluded. Analysis outputs are investigative signals, not authenticity proofs.
