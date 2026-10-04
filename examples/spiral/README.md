# Spiral — original, edited input and my edit reference

I supplied the original spiral, my edited TIFF and a WebP reference showing the
areas I changed. The three images share the same 1254 × 1254 pixel coordinates.
`spiral-reference.png` preserves the decoded colour pixels of my reference WebP
without additional compression loss. It is not a SHERLOQ result and was never
supplied to a detector.

| Original | Edited input | My edit reference |
| --- | --- | --- |
| ![Original spiral](../../screenshots/fork/texture-original.png) | ![Edited spiral](../advanced/texture-edited.png) | ![My edit reference](spiral-reference.png) |

## Copy-Move Forgery 2

Each row uses exactly the same crop in all four columns. Compare the patch on
the upper left, the shapes on the upper right, and the repeated vertical detail
on the left side. The actual tool output links matching regions with a shared
colour; it does not determine which member is the copied destination. The
coloured outlines are indicative envelopes, not exact edit boundaries.

![Original, edited input, reference and copy-move output](comparison-clones.png)

The tool also reports matches outside my reference and does not outline every
referenced change. The reference remains visible separately so those differences
can be judged directly.

## ELA

The softened regions at the lower left and on the right have darker
recompression residuals than the surrounding detailed texture. ELA visualises
those differences; it does not identify a specific editing operation or provide
a complete edit mask.

![Original, edited input, reference and ELA output](comparison-ela.png)

## Files and reproduction

The [full copy-move output](copy-move-output.png) and [full ELA output](ela-output.png)
are actual full-resolution processed images exported from the native widgets.
The detector input is the [existing edited PNG](../advanced/texture-edited.png),
never the reference. Copy-Move Forgery 2 uses PatchMatch Zernike, 6,000 displayed
links and 8 iterations; ELA uses quality 75, scale 50% and contrast 20%.
[Recorded controls](analysis-records.json), [crop coordinates](comparison-crops.json)
and [file hashes](manifest.json) accompany the figures.

To reproduce the processed images, follow the environment instructions in
[the earlier examples](../advanced/README.md#reproduce-the-captures), then run from that
`examples/advanced` directory:

```sh
"$SHERLOQ_NATIVE_ROOT/venv/bin/python" render_extended.py cloning2 texture-edited.png texture
"$SHERLOQ_NATIVE_ROOT/venv/bin/python" render_extended.py ela texture-edited.png texture
cp cloning2-texture-view0.png ../spiral/copy-move-output.png
cp ela-texture-view1.png ../spiral/ela-output.png
"$SHERLOQ_NATIVE_ROOT/venv/bin/python" ../spiral/build_comparison.py
```

The comparison script only crops and arranges the saved outputs. It performs no
inference and does not add the edit reference to an analysis result. The existing
full tool captures, Automatic Clone Search and layered ELA views remain in the
[main gallery](../../README.md#screenshots).
