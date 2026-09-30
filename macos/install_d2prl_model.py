"""Install the D2PRL inference checkpoint from GitHub Releases, with SHA256 checks.

Run after restoring RC1 and applying its cumulative source update. Python 3.11+.
A verified existing checkpoint is reused; unknown existing files are not replaced.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request


def valid(path, asset):
    if not path.is_file() or path.stat().st_size != asset['bytes']:
        return False
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() == asset['sha256']


def install(installation, asset, local_file=None):
    root = installation.expanduser().resolve(strict=True)
    if not (root / 'source/gui').is_dir():
        raise ValueError('Choose the restored or staged SHERLOQ installation folder')
    relative = Path(asset['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Invalid model destination')
    target = root / relative
    if target.resolve() != target.absolute() or not target.resolve().is_relative_to(root):
        raise ValueError('Refusing a symlink or outside destination')
    if target.exists():
        if valid(target, asset):
            return target
        raise ValueError('Existing checkpoint has a different size or hash; left untouched')
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.d2prl-', suffix='.download', dir=target.parent)
    temporary = Path(name)
    try:
        request = urllib.request.Request(asset['url'], headers={'User-Agent': 'SHERLOQ-D2PRL-installer'})
        with os.fdopen(descriptor, 'wb') as output:
            with (Path(local_file).open('rb') if local_file else urllib.request.urlopen(request, timeout=120)) as source:
                size = 0
                while chunk := source.read(4 * 1024 * 1024):
                    size += len(chunk)
                    if size > asset['bytes']:
                        raise ValueError('Checkpoint exceeds the expected size')
                    output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if not valid(temporary, asset):
            raise ValueError('Checkpoint size or SHA256 mismatch; no model installed')
        # Link the verified temporary file atomically without replacing a file
        # another installer might have created while the download was running.
        try:
            os.link(temporary, target)
        except FileExistsError:
            if not valid(target, asset):
                raise ValueError('Checkpoint appeared during download; left untouched')
        return target
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('installation', type=Path)
    parser.add_argument('--file', type=Path, help='Verify and install a previously downloaded d2prl.pth')
    args = parser.parse_args()
    asset = json.loads(Path(__file__).with_name('d2prl-model.json').read_text())
    try:
        path = install(args.installation, asset, args.file)
    except (OSError, ValueError) as error:
        parser.exit(1, f'D2PRL installation failed: {error}\n')
    print('Verified D2PRL checkpoint:', path)


if __name__ == '__main__':
    main()
