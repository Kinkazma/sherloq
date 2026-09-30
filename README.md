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

- **Automatic analysis:** keep subimage detection by default and add **Run on whole image** to restart on the complete input if the detected regions are unsuitable. [Usage and RC1 update](macos/docs/WHOLE-IMAGE.md).
- **PatchMatch extended + mirror:** reduce temporary Metal SIFT allocations and correct reservations that rejected large images before matching. The shared memory budget is retained. [Measured scope, validation limits and RC1 update](macos/docs/PATCHMATCH-MEMORY.md).
- **ELA slider control:** manual edits take precedence over automatic estimates already in progress, including input at a slider limit. [Correction and update](macos/docs/ELA-SLIDERS.md).
- **ELA:** preserve both signs of compression differences in linear mode; correct the inverted contrast endpoint at 100%.
- **Reference Comparison:** correct a doubled PSNR calculation and improve handling of individual metric failures and cancellation.
- **Histogram:** correct large-image counting precision, single-level ranges, empty ranges and percentages exceeding 100%.
- **Frequency Split:** restrict the FFT shift to spatial axes so it no longer swaps real and imaginary components and corrupts the phase.
- **Image and colour edge cases:** correct odd-sized wavelet reconstruction, CMYK black, constant bit-plane display and tiny-image failures.
- **Opening and processing:** improve Finder/file-drop handling, preserve the current image after a cancelled or failed opening, handle images without copy-move descriptors, and update incompatible NumPy/SciPy calls in Noiseprint.

### Responsiveness and measured improvements

I added adaptive memory paths for thirteen engines, compact/mapped dense Zernike and SIFT, and tiled display of large images. The RAM path remains preferred, analysis resolution is retained, and the validated scope is partial. Disk-backed execution can be slower. [Validated engines, measurements, numerical correction and RC1 update](macos/docs/ADAPTIVE-MEMORY.md).

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
| Clone Detectors | Additional research backends, including Forgeryscope Auto for microscopy, western blots and lanes, and D2PRL with GPU/CPU selection and union/source/target views; external weights are required |
| Automatic Clone Search | Combined clone-search workflow with Forgeryscope Auto microscopy/blot/lane branches and extended PatchMatch mirror/scale matching |
| Complete Automatic Analysis | Combined workflow with clone-search results and ELA/JPEG-ghost views |
| CAT-Net v2, SAFIRE, FOCAL, AdaIFL | Four additional research-method integrations, with original component credits and licenses |

See the [complete inventory](macos/docs/TOOLS.md) for all inherited and added entries and their current categories. Dependencies and model availability vary; optional missing checkpoints are recorded in the installation's `MISSING-WEIGHTS.md`.

D2PRL now runs natively with GPU selected by default and CPU available explicitly.
The [checkpoint and verified installer](macos/docs/D2PRL-INTEGRATION.md#installation-and-model-files)
are available through GitHub. Its [region-size slider](macos/docs/D2PRL-INTERACTIVE-FILTER.md) refilters cached
results without rerunning inference; the original 500-pixel minimum is the default.
On the recorded author example, the optimized MPS adaptation took 25.76 seconds
versus 401.42 seconds for the initial MPS port, with identical raw maps. CPU and
GPU can produce different results; this is not a general accuracy or speed claim.
[Protocol, measurements, limits and required external files](macos/docs/D2PRL-INTEGRATION.md).

I added [Forgeryscope Auto](macos/docs/FORGERYSCOPE-AUTO.md) as a green choice in
Clone Detectors. It follows the public upstream pipeline for microscopy, western
blots and lanes, with separate branch views and NPZ exports. Geometric matches
remain distinguishable from similarity-based candidates. It reuses the seven
existing Forgeryscope weights and shared matching dependencies.

Automatic Clone Search and Complete Automatic Analysis also use this Auto
pipeline on the enclosing zone. A branch selector filters its displayed results
without another inference. PatchMatch keeps its extended mirror/scale matching,
and Complete Automatic Analysis keeps its ELA/Ghosts controls.

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
<summary>My examples — clone searches and ELA on my spiral texture</summary>

I supplied the original spiral, my edited image and a separate reference showing the areas I changed. The reference is shown only for comparison; it was never supplied to the detectors.

| Original | Edited input | My edit reference |
| --- | --- | --- |
| ![Original texture](screenshots/fork/texture-original.png) | ![Edited texture without analysis annotations](examples/advanced/texture-edited.png) | ![My spiral edit reference](examples/spiral/spiral-reference.png) |

**Same coordinates, four views:** compare the original, edited input and my reference with the actual Copy-Move Forgery 2 output. The shared colours connect matching regions; they do not distinguish a copied destination from its source.

![Spiral edits and copy-move results at matching coordinates](examples/spiral/comparison-clones.png)

The softened areas are easier to compare with ELA: they appear darker than the surrounding detailed texture. This response describes recompression differences, not the editing operation itself.

![Spiral edits and ELA results at matching coordinates](examples/spiral/comparison-ela.png)

[Reference, detailed comparisons and reproduction](examples/spiral/README.md).

### Copy-Move Forgery 2

Matching areas share a colour. Compare the paired shapes on the left and near the top with the repeated details in the edited texture.

![Copy-Move Forgery 2 on my edited texture](screenshots/fork/advanced/cloning2-texture.png)

### Automatic Clone Search

The combined search brings together the matching regions found by its different methods. Here the displayed matches come from the PatchMatch branches; Forgeryscope returns no supported pair.

![Automatic Clone Search on my edited texture](screenshots/fork/advanced/automatic_clones-texture.png)

### ELA — compression differences

The softened areas at the lower left and on the right produce darker residuals than the surrounding detailed texture. ELA displays recompression differences; it does not identify the editing operation by itself.

![Classic ELA on my edited texture](screenshots/fork/advanced/ela-texture.png)

### ELA — layers over the image

The layered view places the unusual profiles back onto the texture. In particular, the broad areas at the lower left and on the right can be compared directly with the classic ELA view above.

![ELA layers on my edited texture](screenshots/fork/advanced/ela-texture-layers.png)

[Inputs, recorded settings and reproduction script](examples/advanced/README.md).

</details>

<details>
<summary>Street Photo — original, edits and reference</summary>

I supplied the original photograph, my edited version and a separate image showing the areas I changed. The reference is for visual comparison only; none of the detectors receives it. The analyses use the 4387 × 3510 input, with any internal model resizing recorded in the example documentation.

| Original | Edited input | My edit reference |
| --- | --- | --- |
| ![Original city photograph](examples/street/street-original-preview.jpg) | ![Edited city photograph](examples/street/street-edited-preview.png) | ![My reference of touched areas](examples/street/street-reference-preview.jpg) |

### Street Photo

**Same coordinates, four views:** original, edited input, my edit reference, and the actual CAT-Net output. Look at the removed clouds, signs and pedestrian.

![Aligned comparisons of clouds, signs and pedestrian](examples/street/comparison-details.png)

[Full-resolution inputs, all 21 tool views, settings and reproduction](examples/street/README.md).

</details>

<details>
<summary>Street Photo — CAT-Net, SAFIRE, ZERO and ELA</summary>

### CAT-Net v2

Several strong responses coincide with the edited sky patches, removed signs and removed pedestrian. There are also responses elsewhere. The PNG input is analysed with a quality-100 JPEG companion generated by the integration.

![CAT-Net on my edited city photograph](examples/street/catnet.png)

### SAFIRE

The green source-consistency group isolates the large retouched sky patch. The other colours are clusters, not authenticity labels.

![SAFIRE source-consistency groups](examples/street/safire.png)

### ZERO — zoom at the removed pedestrian

The red foreign-grid response falls within the pedestrian's former location. Blue missing-grid responses are widespread and do not trace the edits. This is a display zoom of a full-resolution analysis.

![ZERO at the removed pedestrian](examples/street/zero-pedestrian.png)

### ELA and layers

The central sky edit is visible among the legacy cell-layer responses. Natural scene structure produces other responses as well.

![Classic ELA on my edited city photograph](examples/street/ela.png)
![Layered ELA on my edited city photograph](examples/street/ela-layers.png)

</details>

<details>
<summary>Street Photo — other methods and what they actually show</summary>

### Noisesniffer

The central sky edit is marked, along with many other sky and building regions. The 107 regions do not cleanly delineate all retouches.

![Noisesniffer on the city photograph](examples/street/noisesniffer.png)

### Adaptive CFA

The distributed map does not isolate the reference edits in this example.

![Adaptive CFA on the city photograph](examples/street/adaptive_cfa.png)

### TruFor

The image score is 0.0560 in this run; the map responds around lights and edges without clearly recovering the edit reference.

![TruFor anomaly map](examples/street/trufor.png)

### FOCAL and AdaIFL

FOCAL mainly selects a region on the left. AdaIFL's overlay remains weak. Neither displayed result closely matches the supplied reference here.

![FOCAL selected region](examples/street/focal.png)
![AdaIFL result](examples/street/adaifl.png)

### Multiple Compression — original JPEG

This view analyses the original JPEG; the edited TIFF has no stored JPEG coefficients. The double-JPEG result is inconclusive, with 0 of 9 supporting frequencies.

![Double-JPEG analysis of the original photograph](examples/street/multiple-double-jpeg.png)

### Illuminant Map — original photograph

Local colour estimates make the blue sky and warm lighting easy to compare. Scene colours affect these estimates; this is not an edit mask.

![Illuminant Map on the original photograph](examples/street/illuminant.png)

### Dead / Hot Pixels — original photograph

The tool finds 36 isolated-pixel candidates. This magnified view shows one among tree details; it does not diagnose a sensor defect.

![Magnified isolated-pixel candidate](examples/street/defect_pixels-detail.png)

[All views, settings, source files and interpretation](examples/street/README.md).

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
   ```

5. The frozen RC1 archive predates the whole-image button, ELA slider correction and adaptive-memory changes. Close SHERLOQ and, from the current **repository** folder, apply the verified cumulative update:

   ```sh
   python3 macos/apply_rc1_updates.py "/path/to/SHERLOQ-installation"
   ```

   This cumulative update also rebuilds the native PatchMatch bridge using Apple’s command-line developer tools. Replace the quoted path with your restored installation folder, then open its `build/SHERLOQ.app`. [Update details](macos/docs/WHOLE-IMAGE.md).

Restoration needs **Apple's command-line developer tools**. It verifies the manifest, relocates the supplied environments and native libraries, signs modified binaries locally, and builds `build/SHERLOQ.app`. This app uses the surrounding installation directory; it is not a self-contained app to move on its own.

Allow roughly **37 GB** for downloaded parts, the assembled ZIP and the extracted installation. Completed parts are reused when the downloader is rerun. See the [installation guide](macos/README.md) for prerequisites, manual assembly, relocation and source-development instructions. GitHub's automatic **Source code** archives contain the repository; the six separate Release files provide the complete installed environment.

## Validation and contribution

The release candidate was extracted, checked and restored on the same Apple Silicon Mac. Synthetic ELA, caching, QImage ownership and offscreen startup tests passed; both supplied Python environments imported NumPy/OpenCV/PyTorch, and native PatchMatch/ZERO libraries were rebuilt from retained sources. These checks do not establish scientific accuracy for every method or validation on another machine. See [validation details](macos/README.md#validation-and-limits).

For SHERLOQ’s scientific background, please visit [the original project](https://github.com/GuidoBartoli/sherloq). My source changes are in `gui/`, with installation support in `macos/`; the [contribution reference](macos/docs/PR-PLAN.md) maps the available material. I welcome review and corrections.

SHERLOQ retains its [GPLv3 license](LICENSE). Third-party implementations, models and images retain their applicable notices and rights; see [attribution](macos/docs/ATTRIBUTION.md). This is an independent contribution, not an official release or endorsement by Guido Bartoli.
