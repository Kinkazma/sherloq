"""Apply cumulative RC1 fixes, including adaptive memory and its native bridge.

Close SHERLOQ first. Run this script from a current checkout of the fork.
Only known installed source versions are accepted. Native binaries are rebuilt
from the repository before any installed file is replaced.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

SUPPORT = Path(__file__).resolve().parent
BACKUP_ROOT = '.updates/rc1-d2prl-filter-20260930/'
RECEIPT = BACKUP_ROOT + 'native-build.json'
NATIVE_FILES = ('native/runtime/libsherloq_patchmatch.dylib',
                'native/runtime/libsherloq_dense_stream.dylib')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def confined(root, relative):
    path = root / relative
    if path.resolve() != path.absolute() or not path.resolve().is_relative_to(root):
        raise ValueError(f'Refusing a symlink or outside path: {relative}')
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


def build_native_payload():
    """Compile in an isolated directory, never in the installed application."""
    with tempfile.TemporaryDirectory(prefix='sherloq-native-update-') as directory:
        stage = Path(directory).resolve()
        vendor = 'gui/sherloq_app/vendor/patchmatch'
        shutil.copytree(SUPPORT.parent / vendor, stage / 'source' / vendor,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        (stage / 'packaging').mkdir()
        (stage / 'native/runtime').mkdir(parents=True)
        script = stage / 'packaging/build_patchmatch.py'
        shutil.copyfile(SUPPORT / 'packaging/build_patchmatch.py', script)
        try:
            subprocess.run([sys.executable, str(script)], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as error:
            raise ValueError('Native build failed; installation unchanged. '
                             'Check Apple command-line developer tools.\n' + error.stderr[-4000:]) from error
        return {name: (stage / name).read_bytes() for name in NATIVE_FILES}


def native_receipt_valid(root, manifest_hash):
    try:
        receipt = json.loads(confined(root, RECEIPT).read_text())
        return receipt['manifest_sha256'] == manifest_hash and all(
            digest(confined(root, name).read_bytes()) == receipt['files'][name]
            for name in NATIVE_FILES)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def planned_file(root, relative, source, before, default_mode=0o644):
    target = confined(root, relative)
    backup = confined(root, BACKUP_ROOT + 'files/' + relative)
    absent = confined(root, BACKUP_ROOT + 'files/' + relative + '.absent')
    if before is None:
        if backup.exists() or (absent.exists() and absent.read_bytes() != b''):
            raise ValueError(f'Existing backup differs: {backup}')
        backup, backup_data = absent, b''
    else:
        if absent.exists():
            raise ValueError(f'Existing backup differs: {absent}')
        backup_data = before
    if backup.exists() and backup.read_bytes() != backup_data:
        raise ValueError(f'Existing backup differs: {backup}')
    # New model packages may add directories. Validate their existing ancestors
    # now; create them only after every source and native build has passed.
    for parent in target.parents:
        if parent == root:
            break
        if parent.exists() and not parent.is_dir():
            raise ValueError(f'Installation parent is not a directory: {parent}')
    mode = target.stat().st_mode & 0o777 if before is not None else default_mode
    return target, source, before, backup, backup_data, mode


def apply(installation):
    root = installation.expanduser().resolve(strict=True)
    manifest = json.loads((SUPPORT / 'whole-image-update.json').read_text())
    manifest_hash = digest(json.dumps(manifest, sort_keys=True).encode())
    plan = []
    # Validate every source and installed preimage before compiling or changing files.
    for relative, hashes in manifest.items():
        source = confined(SUPPORT.parent, relative).read_bytes()
        if digest(source) != hashes['after']:
            raise ValueError(f'Repository file does not match this update: {relative}')
        destination = hashes.get('target', 'source/' + relative)
        target = confined(root, destination)
        before = target.read_bytes() if target.exists() else None
        before_hash = digest(before) if before is not None else None
        accepted = hashes['before'] if isinstance(hashes['before'], list) else [hashes['before']]
        if before_hash not in (*accepted, hashes['after']):
            raise ValueError(f'Unrecognized installed file; left untouched: {destination}')
        if before_hash != hashes['after']:
            plan.append(planned_file(root, destination, source, before))
    receipt_data = None
    if any(entry.get('native_build') for entry in manifest.values()):
        # Check this path even if an invalid receipt needs to be regenerated.
        confined(root, RECEIPT)
        if not native_receipt_valid(root, manifest_hash):
            for name in NATIVE_FILES:
                target = confined(root, name)
                if not target.parent.is_dir():
                    raise ValueError(f'Missing installation directory: {target.parent}')
            payload = build_native_payload()
            if set(payload) != set(NATIVE_FILES) or not all(payload.values()):
                raise ValueError('Incomplete native build; installation unchanged')
            for name, source in payload.items():
                target = confined(root, name)
                before = target.read_bytes() if target.exists() else None
                if before != source:
                    plan.append(planned_file(root, name, source, before, 0o755))
            receipt_data = (json.dumps({'manifest_sha256': manifest_hash,
                            'files': {name: digest(data) for name, data in payload.items()}},
                            indent=2) + '\n').encode()
    for target, source, before, backup, backup_data, mode in plan:
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            with backup.open('xb') as stream:
                stream.write(backup_data)
    changed = []
    created_directories = []
    try:
        for target, source, before, backup, backup_data, mode in plan:
            missing = []
            parent = target.parent
            while not parent.exists():
                missing.append(parent)
                parent = parent.parent
            for parent in reversed(missing):
                parent.mkdir()
                created_directories.append(parent)
            replace(target, source, mode)
            changed.append((target, before, mode))
        if receipt_data is not None:
            receipt = confined(root, RECEIPT)
            receipt.parent.mkdir(parents=True, exist_ok=True)
            replace(receipt, receipt_data, 0o644)
    except Exception:
        for target, before, mode in reversed(changed):
            if before is None:
                target.unlink()
            else:
                replace(target, before, mode)
        for parent in reversed(created_directories):
            parent.rmdir()
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
