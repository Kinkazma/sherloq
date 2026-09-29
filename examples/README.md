# Examples from my SHERLOQ version

## Complete Automatic Analysis

I exported the two illustrated results from my application. The microscopy
figure shows the combined analysis overlays. The texture example shows the
analysis of a version I edited, with the original before my edits alongside
it for visual comparison. These supplied exports are displayed unchanged.

## Individual tool views

The four additional illustrations are actual SHERLOQ tool widgets rendered
from this repository, with a title and, where useful, a source-image viewer.
They demonstrate the tools' output and controls on these images.

| Tool | Input | Settings |
| --- | --- | --- |
| Channel Histogram | BBBC039 microscopy | Value channel, range 0–255, default linear display |
| Noise Separation | BBBC039 microscopy | Median, radius 1 px, equalized residual display |
| Frequency Split | Original texture | Separation 15%, smoothing 25%, threshold/filter off |
| Bit Planes Values | Original texture | Luminance, bit 5, filter disabled |

The microscopy source and the original texture are used directly for these
four demonstrations; the coloured automatic-analysis overlays are not used
as analysis inputs. Histogram, residual, frequency and bit-plane views are
exploration tools, not labels declaring an image altered.

To render these four panels again, use Python with the native installation's
dependencies and run `python examples/render_examples.py` from the repository
root. The script uses the repository's actual tool code and writes PNGs into
`screenshots/fork/`. Font rendering and timing labels can vary by machine.

## Microscopy source and credits

`bbbc039-00809.png` is the unchanged `srcDataset/00809.png` file from the
[RSIID source archive](https://zenodo.org/records/15095089), credited to
João P. Cardenuto and Anderson Rocha, *Benchmarking scientific image forgery
detectors* (2022). RSIID is distributed under CC BY 4.0.

The underlying image is from **BBBC039v1**, Caicedo et al. (2018), available
from the Broad Bioimage Benchmark Collection, Ljosa et al. (2012).
[BBBC039's source page](https://bbbc.broadinstitute.org/BBBC039) places that
image collection under CC0. RSIID identifies this derived source crop as CC0;
the original source metadata is retained in `bbbc039-source.json`.

This is a fluorescence image of cell nuclei, not an AI-generated image.
The source filename identifies a converted crop; it is not the original
full-field 16-bit TIFF. No additional change was made to the supplied crop.

These credits apply to the BBBC039 image and the two panels using it;
the separately supplied microscopy analysis export is a different figure.
