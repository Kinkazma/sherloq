#!/usr/bin/env python3
"""Optional separate Google Drive acquisition; requires gdown from tools-venv."""
from fetch_assets import DOWNLOADS, WEIGHTS, HERE, MANIFEST, digest, unpack_zip, link
import json

def main():
    path = DOWNLOADS/'04_mgcfdn_weights.zip'
    if not path.exists():
        import gdown
        if not gdown.download(id='1o9iCAOukI3iwMW64hSvfapwaNHCI5-Hd', output=str(path), resume=True):
            raise RuntimeError('Google Drive download failed')
    data = json.loads(MANIFEST.read_text())
    expected = next((a.get('sha256') for a in data['assets'] if a['id']=='04_mgcfdn_weights'),None)
    if expected and expected != digest(path):
        raise ValueError('Archive checksum mismatch')
    unpack_zip(path, WEIGHTS/'04_mgcfdn')
    link(HERE/'weights/04_mgcfdn', WEIGHTS/'04_mgcfdn')
    print(WEIGHTS/'04_mgcfdn')
if __name__ == '__main__':
    main()
