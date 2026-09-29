"""Apply cumulative whole-image, ELA slider and PatchMatch memory fixes to RC1.

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
BACKUP_ROOT = '.updates/rc1-patchmatch-memory-20260929/source/'


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
        before = target.read_bytes() if target.exists() else None
        before_hash = digest(before) if before is not None else None
        accepted = hashes['before'] if isinstance(hashes['before'], list) else [hashes['before']]
        if before_hash not in (*accepted, hashes['after']):
            raise ValueError(f"Unrecognized installed file; left untouched: {relative}")
        if before_hash == hashes['after']:
            continue
        backup = confined(root, BACKUP_ROOT + relative)
        absent = confined(root, BACKUP_ROOT + relative + '.absent')
        # An empty .absent marker records a module newly introduced by this update.
        if before is None:
            if backup.exists() or (absent.exists() and absent.read_bytes() != b''):
                raise ValueError(f"Existing backup differs: {backup}")
            backup, backup_data = absent, b''
        else:
            if absent.exists():
                raise ValueError(f"Existing backup differs: {absent}")
            backup_data = before
        if backup.exists() and backup.read_bytes() != backup_data:
            raise ValueError(f"Existing backup differs: {backup}")
        if not target.parent.is_dir():
            raise ValueError(f"Missing installation directory: {target.parent}")
        mode = target.stat().st_mode & 0o777 if before is not None else 0o644
        plan.append((target, source, before, backup, backup_data, mode))
    for target, source, before, backup, backup_data, mode in plan:
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            with backup.open('xb') as stream:
                stream.write(backup_data)
    changed = []
    try:
        for target, source, before, backup, backup_data, mode in plan:
            replace(target, source, mode)
            changed.append((target, before, mode))
    except Exception:
        for target, before, mode in reversed(changed):
            if before is None:
                target.unlink()
            else:
                replace(target, before, mode)
        raise
    return len(changed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('installation', type=Path, help='Restored SHERLOQ-installation folder')
    args = parser.parse_args()
    try:
        count = apply(args.installation)
    except (OSError, ValueError) as error:
        parser.exit(1, f'Update refused: {error}\n')
    print(f'Updated {count} files. You can now reopen SHERLOQ.')


if __name__ == '__main__':
    main()
