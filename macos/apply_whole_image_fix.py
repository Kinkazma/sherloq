"""Apply the three-file whole-image fix to a restored RC1 installation.

Close SHERLOQ first. Run this script from a current checkout of the fork.
Only the known RC1 files or already updated files are accepted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

SUPPORT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def confined(root, relative):
    path = root / relative
    if path.resolve() != path.absolute() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Refusing a symlink or outside path: {relative}")
    return path


def replace(path, data, mode):
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def apply(installation):
    root = installation.expanduser().resolve(strict=True)
    manifest = json.loads((SUPPORT / 'whole-image-update.json').read_text())
    plan = []
    # Validate every input and backup before modifying any installed file.
    for relative, hashes in manifest.items():
        source = confined(SUPPORT.parent, relative).read_bytes()
        if digest(source) != hashes['after']:
            raise ValueError(f"Repository file does not match this update: {relative}")
        target = confined(root, 'source/' + relative)
        before = target.read_bytes()
        if digest(before) not in (hashes['before'], hashes['after']):
            raise ValueError(f"Unrecognized installed file; left untouched: {relative}")
        if digest(before) == hashes['after']:
            continue
        backup = confined(root, '.updates/whole-image-20260929/source/' + relative)
        if backup.exists() and backup.read_bytes() != before:
            raise ValueError(f"Existing backup differs: {backup}")
        plan.append((target, source, before, backup, target.stat().st_mode & 0o777))
    for target, source, before, backup, mode in plan:
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            with backup.open('xb') as stream:
                stream.write(before)
    changed = []
    try:
        for target, source, before, backup, mode in plan:
            replace(target, source, mode)
            changed.append((target, before, mode))
    except Exception:
        for target, before, mode in reversed(changed):
            replace(target, before, mode)
        raise
    return len(changed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('installation', type=Path, help='Restored SHERLOQ-installation folder')
    args = parser.parse_args()
    try:
        count = apply(args.installation)
    except (OSError, ValueError) as error:
        parser.exit(1, f'Update refused: {error}\n')
    print(f'Updated {count} files. You can now reopen SHERLOQ.')
