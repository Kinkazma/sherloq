#!/usr/bin/env python3
"""Offline verification of staged files and project symlinks; imports no upstream code."""
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'integration/clone_detectors'

def main():
    inventory = json.loads((HERE/'files.sha256.json').read_text())
    errors = []
    for item in inventory['files']:
        p = ROOT/item['path']
        if not p.is_file():
            errors.append('MISSING '+item['path']); continue
        with p.open('rb') as f:
            h = hashlib.file_digest(f, 'sha256').hexdigest()
        if h != item['sha256'] or p.stat().st_size != item['size']:
            errors.append('CHANGED '+item['path'])
    for item in inventory['symlinks']:
        p = ROOT/item['path']
        if not p.is_symlink() or str(p.readlink()) != item['target'] or not p.exists():
            errors.append('INVALID LINK '+item['path'])
    missing = [x['id'] for x in json.loads((HERE/'assets.json').read_text())['assets'] if x.get('status') == 'missing']
    print(json.dumps(dict(checked_files=len(inventory['files']), checked_symlinks=len(inventory['symlinks']), errors=errors, known_missing_assets=missing), indent=2))
    return bool(errors)
if __name__ == '__main__':
    sys.exit(main())
