"""Isolated RAW decoder. The parent owns the private output directory."""
import json
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sherloq_app.core.image_io import decode_image
import numpy as np


def main():
    filename, output = sys.argv[1:]
    _, _, image, metadata = decode_image(filename)
    output = Path(output)
    np.save(output / 'image.npy', image, allow_pickle=False)
    (output / 'metadata.json').write_text(json.dumps(metadata))


if __name__ == '__main__':
    main()
