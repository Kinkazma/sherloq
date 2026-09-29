"""Build aligned panels from supplied inputs and actual SHERLOQ output pixels."""
from pathlib import Path
import json
import cv2 as cv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
original = cv.imread(str(root / '../../screenshots/fork/texture-original.png'))
edited = cv.imread(str(root / '../advanced/texture-edited.png'))
reference = cv.imread(str(root / 'spiral-reference.png'))
crops = json.loads((root / 'comparison-crops.json').read_text())
for kind, filename, title in (
    ('clones', 'copy-move-output.png', 'Copy-Move Forgery 2'),
    ('ela', 'ela-output.png', 'ELA output'),
):
    images = [original, edited, reference, cv.imread(str(root / filename))]
    assert all(image is not None and image.shape == (1254, 1254, 3) for image in images)
    fig, axes = plt.subplots(len(crops[kind]), 4, figsize=(16, 4.4 * len(crops[kind])), layout='constrained')
    for row, crop in enumerate(crops[kind]):
        x, y, w, h = (crop[key] for key in ('x', 'y', 'width', 'height'))
        for col, (name, image) in enumerate(zip(('Original', 'Edited input', 'My edit reference', title), images)):
            ax = axes[row, col]
            ax.imshow(cv.cvtColor(image[y:y+h, x:x+w], cv.COLOR_BGR2RGB))
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(name, fontsize=14)
            if col == 0:
                ax.set_ylabel(crop['label'], fontsize=14)
    fig.savefig(root / ('comparison-' + kind + '.png'), dpi=130)
    plt.close(fig)
