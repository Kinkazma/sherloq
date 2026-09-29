"""Rebuild the aligned comparison after render_street.py catnet has completed."""
from pathlib import Path
import json
import cv2 as cv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
names = ('street-original.jpg', 'street-edited.png', 'street-reference.jpg', 'catnet-view0.png')
images = [cv.imread(str(root / name)) for name in names]
if any(image is None for image in images):
    raise SystemExit('Missing input: run render_street.py catnet street-edited.png first.')
assert len({image.shape for image in images}) == 1
crops = json.loads((root / 'comparison-crops.json').read_text())
fig, axes = plt.subplots(len(crops), 4, figsize=(16, 13), layout='constrained')
fig.suptitle('Street Photo — same coordinates, four views', fontsize=21, weight='bold')
for row, crop in enumerate(crops):
    x, y, w, h = (crop[key] for key in ('x', 'y', 'width', 'height'))
    for col, (name, image) in enumerate(zip(('Original', 'Edited input', 'My edit reference', 'CAT-Net output'), images)):
        ax = axes[row, col]
        ax.imshow(cv.cvtColor(image[y:y+h, x:x+w], cv.COLOR_BGR2RGB))
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(name, fontsize=14)
        if col == 0:
            ax.set_ylabel(crop['label'], fontsize=14)
fig.savefig(root / 'comparison-details.png', dpi=130)
plt.close(fig)
