# SHERLOQ — macOS Apple Silicon community fork

**SHERLOQ is the work of [Guido Bartoli and the original contributors](https://github.com/GuidoBartoli/sherloq).** Start with the [original project](https://github.com/GuidoBartoli/sherloq) for its introduction, history, research references and upstream development.

This independent fork is maintained by [Kinkazma](https://github.com/Kinkazma). It brings together macOS Apple Silicon installation work, fixes and interface improvements, additional analysis workflows, and integrations of existing research methods. Development was assisted by AI. The original project and research authors retain their credits; adding a method to this interface does not make its underlying algorithm our invention.

[Download the complete macOS installation](https://github.com/Kinkazma/sherloq/releases/tag/native-macos-arm64-2026.09.29-rc1) · [Installation guide](macos/README.md) · [Full tool inventory](macos/docs/TOOLS.md) · [Changes](macos/docs/CHANGELOG.md) · [Credits and licenses](macos/docs/ATTRIBUTION.md)

## What comes from the original project

The comparison baseline is [upstream revision `3fe95fc`](https://github.com/GuidoBartoli/sherloq/tree/3fe95fcb56037e47e2eefbc2d3785804a31a74f5). It already provides **38 tool entries**, including image inspection, metadata, colour and noise analysis, ELA, JPEG ghosts, copy-move detection, resampling, PRNU and **TruFor**. These are retained here, with adaptations in several areas.

For the original descriptions and scientific references, please use [Guido's README](https://github.com/GuidoBartoli/sherloq#readme). The original [source history](https://github.com/GuidoBartoli/sherloq/commits/master/) and [GPLv3 license](LICENSE) are preserved in this fork.

## What this fork adds

The current source defines **50 distinct tool entries: 38 inherited and 12 added**. Favorites are shortcuts and are not counted twice. This is an interface inventory, not a count of independently validated detectors.

| Added entry or group | Contribution in this fork |
| --- | --- |
| C2PA Validation | Interface for checking signed provenance, asset integrity and local trust |
| Noisesniffer | Integration of the existing research implementation for noise inconsistencies |
| ZERO JPEG Grids | Integration of the existing ZERO method for JPEG-grid analysis |
| Copy-Move Forgery 2 | Additional matching workflow with regions, dense descriptors, geometric controls and grouped results |
| Adaptive CFA | Integration of an existing colour-filter-array analysis method |
| Clone Detectors | Interface to additional research backends; availability depends on the included backend and weights |
| Automatic Clone Search | Combined clone-search workflow, including microscopy and PatchMatch branches |
| Complete Automatic Analysis | Combined workflow with clone-search results and ELA/JPEG-ghost views |
| CAT-Net v2, SAFIRE, FOCAL, AdaIFL | Four additional research-method integrations, with original component credits and licenses |

See the [complete inventory](macos/docs/TOOLS.md) for all inherited and added entries and their current categories. Dependencies and model availability vary; optional missing checkpoints are recorded in the installation's `MISSING-WEIGHTS.md`.

### Changes to existing tools and the interface

The contribution also includes ELA correctness and caching work; additional JPEG-analysis behavior; comparison and inspection controls; background processing and cancellation; region and layer views; localization and favorites; and native macOS build and launcher support. **TruFor is adapted from the original project, not a newly introduced method.** Illuminant Map and Dead/Hot Pixels also had entries upstream; implementation work on them is listed as an improvement to existing tools.

The [change index](macos/docs/CHANGELOG.md) and [file-level index](macos/docs/NATIVE-CHANGE-INDEX.md) describe the code changes. Colored outlines in historical test captures mark additions or extensions; they are not a scientific confidence rating or a complete authorship map.

## Screenshots

The gallery below preserves screenshots from the original SHERLOQ project, with attribution. It illustrates the inherited interface; the [tool inventory](macos/docs/TOOLS.md) describes the additional tools in this fork.

<details>
<summary>Original SHERLOQ screenshots — credited to the upstream project</summary>

These inherited captures illustrate the original interface, not this fork's complete feature set. They are retained from [Guido Bartoli's project](https://github.com/GuidoBartoli/sherloq#screenshots).

| Area | Original capture |
| --- | --- |
| General | [Original image, hex editor, digest and similarity search](screenshots/0_general.png) |
| Metadata | [EXIF and header structure](screenshots/1_metadata.png) |
| Inspection | [Magnifier, histogram and comparison](screenshots/2_inspection.png) |
| Detail | [Gradient, echo, wavelet and frequency tools](screenshots/3_detail.png) |
| Colors | [Plots, conversion, PCA and statistics](screenshots/4_colors.png) |
| Noise | [Signal, min/max, bit planes and wavelet tools](screenshots/5_noise.png) |
| JPEG | [Quality estimation and ELA](screenshots/6_jpeg.png) |
| Tampering | [Contrast, copy-move, splicing and median filtering](screenshots/7_tampering.png) |

</details>

## Install the macOS candidate

The supported packaged target is **macOS on Apple Silicon (ARM64)**. The complete installation is approximately **10.9 GB**, hosted in this repository's Release as six parts. It includes the application, installed Python environments, native dependencies and all 185 model/checkpoint files present in the frozen snapshot. It does not require a Mega account.

1. Download this repository with **Code → Download ZIP**, or clone it. Open Terminal in the extracted repository folder.
2. With **Python 3.11 or newer**, run:

   ```sh
   python3 macos/download_installation.py ~/Downloads/SHERLOQ-download
   ```

3. The script downloads, verifies and assembles `SHERLOQ-installation-macos-arm64.zip`. Extract that ZIP into the location where you intend to keep the installation.
4. Open Terminal in the extracted `SHERLOQ-installation` folder and run:

   ```sh
   python3 restore_installation.py
   venv/bin/python app_start.py
   ```

Restoration needs **Apple's command-line developer tools**. It verifies the manifest, relocates the supplied environments and native libraries, signs modified binaries locally, and builds `build/SHERLOQ.app`. This app uses the surrounding installation directory; it is not a self-contained app to move on its own.

Allow roughly **37 GB** for downloaded parts, the assembled ZIP and the extracted installation. Completed parts are reused when the downloader is rerun. See the [installation guide](macos/README.md) for prerequisites, manual assembly, relocation and source-development instructions. GitHub's automatic **Source code** archives contain the repository; the six separate Release files provide the complete installed environment.

## Validation and contribution

The release candidate was extracted, checked and restored on the same Apple Silicon Mac. Synthetic ELA, caching, QImage ownership and offscreen startup tests passed; both supplied Python environments imported NumPy/OpenCV/PyTorch, and native PatchMatch/ZERO libraries were rebuilt from retained sources. These checks do not establish scientific accuracy for every method or validation on another machine. See [validation details](macos/README.md#validation-and-limits).

The current packaged release is a native desktop candidate. Browser work is separate. For SHERLOQ's scientific background and upstream roadmap, return to [the original project](https://github.com/GuidoBartoli/sherloq). For this fork's code, start with `gui/` and `macos/`, or the [proposed contribution groups](macos/docs/PR-PLAN.md).

SHERLOQ retains its [GPLv3 license](LICENSE). Third-party implementations, models and images retain their applicable notices and rights; see [attribution](macos/docs/ATTRIBUTION.md). This is an independent contribution, not an official release or endorsement by Guido Bartoli.
