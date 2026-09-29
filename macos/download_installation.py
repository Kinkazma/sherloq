"""Download and verify the complete macOS snapshot using only GitHub Releases.

Python 3.11+. Completed parts are reused on retry. No installation code runs here.
"""
import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def valid(path, item):
    return path.is_file() and path.stat().st_size == item['bytes'] and digest(path) == item['sha256']


def download(manifest, destination):
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / manifest['archive']['name']
    if valid(archive, manifest['archive']):
        return archive
    for item in manifest['parts']:
        target = destination / item['name']
        if valid(target, item):
            print('Already verified:', target.name, flush=True)
            continue
        temporary = target.with_suffix(target.suffix + '.download')
        print('Downloading:', target.name, flush=True)
        request = urllib.request.Request(manifest['base_url'] + item['name'], headers={'User-Agent': 'SHERLOQ-installation-downloader'})
        with urllib.request.urlopen(request, timeout=120) as source, temporary.open('wb') as output:
            shutil.copyfileobj(source, output, 4 * 1024 * 1024)
        if not valid(temporary, item):
            raise RuntimeError('Download checksum mismatch: ' + item['name'])
        temporary.replace(target)
    print('Assembling and verifying installation ZIP...', flush=True)
    temporary = archive.with_suffix('.assembling')
    with temporary.open('wb') as output:
        for item in manifest['parts']:
            with (destination / item['name']).open('rb') as source:
                shutil.copyfileobj(source, output, 4 * 1024 * 1024)
    if not valid(temporary, manifest['archive']):
        raise RuntimeError('Assembled installation checksum mismatch')
    temporary.replace(archive)
    archive.with_suffix('.zip.sha256').write_text(manifest['archive']['sha256'] + '  ' + archive.name + '\n')
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path, help='Download folder; allow 22 GB for parts and assembled ZIP')
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name('installation-assets.json').read_text())
    archive = download(manifest, args.destination.expanduser().resolve())
    print('Verified:', archive)
    print('Extract the ZIP, then run python3 restore_installation.py from SHERLOQ-installation.')


if __name__ == '__main__':
    main()
