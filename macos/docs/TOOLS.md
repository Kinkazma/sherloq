# Tool inventory: inherited, added and adapted

Compared with Guido Bartoli's upstream revision `3fe95fcb56037e47e2eefbc2d3785804a31a74f5`, the frozen fork defines **50 canonical tool entries: 38 inherited and 12 added**. Favorite shortcuts do not create additional tools. This inventory is derived from `gui/sherloq_app/ui/tools.py`, including the move of Complete Automatic Analysis into Inspection. It counts panels/workflows, not validated scientific methods; a panel may offer several backends or depend on optional components.

The upstream labels are retained to make the comparison explicit. “Inherited” means the entry existed upstream; it does not imply its implementation is unchanged. “Added” means an interface entry was added by this fork; research-method integrations remain credited to their original authors.

| Current category | Inherited entries | Added entries |
| --- | --- | --- |
| General | Original Image; File Digest; Hex Editor; Similarity Search | — |
| Metadata | Header Structure; EXIF Full Dump; Thumbnail Analysis; Geolocation data | C2PA Validation |
| Inspection | Enhancing Magnifier; Channel Histogram; Global Adjustments; Reference Comparison | Complete Automatic Analysis |
| Detail | Luminance Gradient; Echo Edge Filter; Wavelet Threshold; Frequency Split | — |
| Colors | RGB/HSV Plots; Space Conversion; PCA Projection; Pixel Statistics | — |
| Noise | Signal Separation; Min/Max Deviation; Bit Plane Values; Wavelet Blocking; PRNU Identification | Noisesniffer |
| JPEG | Quality Estimation; Error Level Analysis; Multiple Compression; JPEG Ghost Maps | ZERO JPEG Grids |
| Tampering | Contrast Enhancement; Copy-Move Forgery; Composite Splicing; Image Resampling | Copy-Move Forgery 2; Adaptive CFA; Clone Detectors; Automatic Clone Search |
| AI Solutions | TruFor | CAT-Net v2; SAFIRE; FOCAL; AdaIFL |
| Various | Median Filtering; Illuminant Map; Dead/Hot Pixels; Stereogram Decoder | — |

## Adaptations to existing entries

- **Error Level Analysis:** absolute linear differences, caching, layered exploration and extended display controls.
- **Multiple Compression:** additional double-JPEG analysis work under an existing entry.
- **TruFor:** environment, worker and interface integration; TruFor itself was already present upstream.
- **Illuminant Map and Dead/Hot Pixels:** implementation work under existing upstream labels.
- **Inspection and copy-move tools:** computation helpers, region/display controls and responsiveness changes; the separate Copy-Move Forgery 2 panel is counted among additions.
- **Shared interface:** background jobs and cancellation, bounded caching, language support, favorites and region/layer interaction.

The colored outlines in the interface group additions and extensions, so an outlined entry is not necessarily a newly added tool. Neither entry counts nor outline colors establish validation or authorship of the underlying algorithms. See [changes](CHANGELOG.md), [component attribution](ATTRIBUTION.md) and [installation/validation limits](../README.md).
