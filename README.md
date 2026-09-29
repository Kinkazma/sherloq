# SHERLOQ — macOS Apple Silicon community fork

**SHERLOQ is the work of [Guido Bartoli and the original contributors](https://github.com/GuidoBartoli/sherloq).** Start with the [original project](https://github.com/GuidoBartoli/sherloq) for its introduction, history, research references and upstream development.

I started this fork to get SHERLOQ running on my Apple Silicon Mac. With help from AI tools, I then worked through installation problems, bugs, slow or blocking operations, and tools that were listed but not usable. I have also added analysis workflows and integrated additional research methods.

Here I share the version I use, its source code and its complete macOS installation. My contribution is the adaptation, implementation and integration work described below. SHERLOQ remains Guido's project, and the research methods retain their original authors and licenses. For the project's introduction, history and scientific references, I refer you to [Guido's repository](https://github.com/GuidoBartoli/sherloq#readme).

[Download the complete macOS installation](https://github.com/Kinkazma/sherloq/releases/tag/native-macos-arm64-2026.09.29-rc1) · [Installation guide](macos/README.md) · [Full tool inventory](macos/docs/TOOLS.md) · [Changes](macos/docs/CHANGELOG.md) · [Credits and licenses](macos/docs/ATTRIBUTION.md)

## What I have changed

I have worked on three areas: making existing tools usable, improving correctness and responsiveness, and adding new analysis workflows. The comparison baseline is [upstream revision `3fe95fc`](https://github.com/GuidoBartoli/sherloq/tree/3fe95fcb56037e47e2eefbc2d3785804a31a74f5).

### Existing tools I implemented or brought back into use

| Tool | State in the baseline | What is available here |
| --- | --- | --- |
| Illuminant Map | Listed in the menu, with no working dispatch or implementation | Local illuminant-colour estimation using Gray World, Shades of Gray or White Patch; views, validity masks and CSV/PNG exports |
| Dead/Hot Pixels | Listed in the menu, with no working dispatch or implementation | Isolated-pixel candidate detection, overlays, masks, a correction preview and coordinate exports; the original image is retained |
| Multiple Compression | The launch branch was commented out | Restored recompression curves, cancellation/resume and CSV/PNG exports, plus a separate experimental aligned double-JPEG analysis tab |
| TruFor | Already part of SHERLOQ; required adaptation for my macOS environment | CPU/Metal selection, a separate inference process, progress/cancellation, map and score exports, and memory-management changes for large images |

These are substantial additions to usability even though they reuse existing menu entries. Illuminant Map estimates colour, not a 3D light direction; isolated-pixel candidates are not a hardware diagnosis; a recompression curve alone cannot establish the number of JPEG compressions.

### Bugs I fixed

- **ELA:** preserve both signs of compression differences in linear mode; correct the inverted contrast endpoint at 100%.
- **Reference Comparison:** correct a doubled PSNR calculation and improve handling of individual metric failures and cancellation.
- **Histogram:** correct large-image counting precision, single-level ranges, empty ranges and percentages exceeding 100%.
- **Frequency Split:** restrict the FFT shift to spatial axes so it no longer swaps real and imaginary components and corrupts the phase.
- **Image and colour edge cases:** correct odd-sized wavelet reconstruction, CMYK black, constant bit-plane display and tiny-image failures.
- **Opening and processing:** improve Finder/file-drop handling, preserve the current image after a cancelled or failed opening, handle images without copy-move descriptors, and update incompatible NumPy/SciPy calls in Noiseprint.

### Responsiveness and measured improvements

I separated computation from display updates, added bounded caches and background jobs, and improved cancellation and reuse of completed work. This lets many display changes reuse an existing result instead of repeating an expensive analysis. I also added native kernels where the tests supported their use, while keeping CPU paths where GPU transfers or numerical differences made them unsuitable.

The following are **recorded development measurements from 27 September 2026 on my Apple Silicon Mac, using 20-megapixel test images**. They describe those measured operations, not a universal speed multiplier for SHERLOQ.

| Measured operation | Before | After |
| --- | ---: | ---: |
| Channel Histogram: panel ready, including drawing | 14.858 s | 0.493 s |
| Frequency Split: initial panel ready | 10.805 s | 3.164 s |
| PCA Projection: initial panel ready | 2.434 s | 1.147 s |
| PCA Projection: peak process memory during the measured session | 5.29 GB | 3.01 GB |

[Measurement records and interpretation](macos/docs/PERFORMANCE.md) explain the protocols and tradeoffs. Some first visits to deferred views became slower, and caches or GPU buffers can increase retained memory. I do not claim that every operation is faster or that a faster display makes a detector more accurate.

I have also added language switching, persistent favorites, layered ELA views and profiles, region selection, zoom/gesture improvements, and automatic workflows that bring several analysis outputs into one interface. The [detailed changelog](macos/docs/CHANGELOG.md) connects these changes to the source.

## Additional tools I integrated

The current source defines **50 distinct tool entries: 38 inherited labels and 12 added entries**. The three restored or newly implemented entries described above belong to those 38, so the menu count alone understates the implementation work. Favorites are shortcuts and are not counted twice. This is an interface inventory, not a count of independently validated detectors.

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

## Screenshots

The galleries below show results and tool views from my version of SHERLOQ, followed by the screenshots from [Guido Bartoli’s original repository](https://github.com/GuidoBartoli/sherloq#screenshots), credited separately.

<details>
<summary>My examples — Complete Automatic Analysis</summary>

### Microscopy figure

An exported result from Complete Automatic Analysis, with the analysis overlays displayed across the figure.

![Complete Automatic Analysis result on a microscopy figure](screenshots/fork/automatic-analysis-microscopy.png)

### Edited texture

For this example, I modified the texture shown below and ran Complete Automatic Analysis. The first image shows the analysis overlays on my edited version; the second is the original before my edits, for visual comparison.

**Analysis of my edited version**

![Complete Automatic Analysis result on my edited texture](screenshots/fork/automatic-analysis-texture.png)

**Original before my edits**

![Original texture before my edits](screenshots/fork/texture-original.png)

</details>

<details>
<summary>My examples — histogram, noise, frequencies and bit planes</summary>

These views use the actual SHERLOQ tools, with the source image shown where useful. The microscopy image comes from BBBC039 via RSIID; the texture is the original shown above. [Sources, settings and reproduction instructions](examples/README.md).

### Channel Histogram

Intensity distribution and pixel statistics for a fluorescence microscopy image.

![Channel Histogram with the microscopy source and intensity statistics](screenshots/fork/histogram-microscopy.png)

### Noise Separation

Median-filter residuals, with equalization enabled to make their spatial structure visible.

![Noise Separation with the microscopy source and equalized residual](screenshots/fork/noise-microscopy.png)

### Frequency Split

Low and high frequencies, Fourier magnitude and phase on the original texture.

![Frequency Split showing four complementary views of the texture](screenshots/fork/frequency-texture.png)

### Bit Planes Values

Bit 5 of the luminance channel, alongside the original texture.

![Bit Planes Values showing luminance bit 5 and the source texture](screenshots/fork/bit-planes-texture.png)

</details>

<details>
<summary>Original SHERLOQ screenshots — credited to the upstream project</summary>

These inherited captures illustrate the original interface, not this fork's complete feature set. They are retained from [Guido Bartoli's project](https://github.com/GuidoBartoli/sherloq#screenshots).

### General

![Original image, hex editor, digest and similarity search](screenshots/0_general.png)

### Metadata

![EXIF and header structure](screenshots/1_metadata.png)

### Inspection

![Magnifier, histogram and comparison](screenshots/2_inspection.png)

### Detail

![Gradient, echo, wavelet and frequency tools](screenshots/3_detail.png)

### Colors

![Plots, conversion, PCA and statistics](screenshots/4_colors.png)

### Noise

![Signal, min/max, bit planes and wavelet tools](screenshots/5_noise.png)

### JPEG

![Quality estimation and ELA](screenshots/6_jpeg.png)

### Tampering

![Contrast, copy-move, splicing and median filtering](screenshots/7_tampering.png)

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

For SHERLOQ’s scientific background, please visit [the original project](https://github.com/GuidoBartoli/sherloq). My source changes are in `gui/`, with installation support in `macos/`; the [contribution reference](macos/docs/PR-PLAN.md) maps the available material. I welcome review and corrections.

SHERLOQ retains its [GPLv3 license](LICENSE). Third-party implementations, models and images retain their applicable notices and rights; see [attribution](macos/docs/ATTRIBUTION.md). This is an independent contribution, not an official release or endorsement by Guido Bartoli.
