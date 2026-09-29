"""Exercise cumulative updates without touching an installed application."""
import importlib.util
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

support = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('rc1_updater', support / 'apply_whole_image_fix.py')
updater = importlib.util.module_from_spec(spec)
spec.loader.exec_module(updater)


def fixture(root):
    repository = root / 'repository'
    installation = root / 'installation'
    (repository / 'macos').mkdir(parents=True)
    (installation / 'source/gui').mkdir(parents=True)
    manifest = {}
    for name, before in [('first.py', b'old-first'), ('added.py', None), ('last.py', b'old-last')]:
        relative = 'gui/' + name
        source = repository / relative
        source.parent.mkdir(exist_ok=True)
        source.write_bytes(b'new-' + name.encode())
        if before is not None:
            (installation / 'source' / relative).write_bytes(before)
        manifest[relative] = {
            'before': updater.digest(before) if before is not None else None,
            'after': updater.digest(source.read_bytes()),
        }
    (repository / 'macos/whole-image-update.json').write_text(json.dumps(manifest))
    return repository, installation


def contents(installation):
    return {str(p.relative_to(installation)): p.read_bytes()
            for folder in ('source', 'native', 'packaging')
            for p in (installation / folder).rglob('*') if p.is_file()}


cases = []
for case in ('success', 'unknown-existing', 'unknown-new', 'missing-existing',
             'bad-source', 'conflicting-backup', 'symlink', 'rollback'):
    with tempfile.TemporaryDirectory() as directory:
        repository, installation = fixture(Path(directory).resolve())
        if case == 'unknown-existing':
            (installation / 'source/gui/last.py').write_bytes(b'local edits')
        elif case == 'unknown-new':
            (installation / 'source/gui/added.py').write_bytes(b'local module')
        elif case == 'missing-existing':
            (installation / 'source/gui/last.py').unlink()
        elif case == 'bad-source':
            (repository / 'gui/last.py').write_bytes(b'wrong delivery')
        elif case == 'conflicting-backup':
            backup = installation / updater.BACKUP_ROOT / 'files/source/gui/first.py'
            backup.parent.mkdir(parents=True)
            backup.write_bytes(b'different backup')
        elif case == 'symlink':
            target = installation / 'source/gui/last.py'
            target.unlink()
            target.symlink_to(repository / 'gui/last.py')
        before = contents(installation)
        original_replace = updater.replace

        def failing_replace(path, data, mode):
            if path.name == 'last.py' and data == b'new-last.py':
                raise OSError('injected replacement failure')
            original_replace(path, data, mode)

        with patch.object(updater, 'SUPPORT', repository / 'macos'):
            if case == 'success':
                assert updater.apply(installation) == 3
                assert updater.apply(installation) == 0
                for name in ('first.py', 'added.py', 'last.py'):
                    assert (installation / 'source/gui' / name).read_bytes() == b'new-' + name.encode()
                backup = installation / updater.BACKUP_ROOT / 'files/source/gui'
                assert (backup / 'first.py').read_bytes() == b'old-first'
                assert (backup / 'last.py').read_bytes() == b'old-last'
                assert (backup / 'added.py.absent').read_bytes() == b''
            else:
                with patch.object(updater, 'replace', failing_replace if case == 'rollback' else original_replace):
                    try:
                        updater.apply(installation)
                    except (ValueError, OSError):
                        pass
                    else:
                        raise AssertionError('Expected refusal: ' + case)
                assert contents(installation) == before, case
        cases.append(case)
for case in ('native-success', 'native-build-failure', 'native-incomplete',
             'native-rollback', 'receipt-failure'):
    with tempfile.TemporaryDirectory() as directory:
        repository, installation = fixture(Path(directory).resolve())
        manifest_path = repository / 'macos/whole-image-update.json'
        manifest = json.loads(manifest_path.read_text())
        # A packaging target uses the same validated preimage mechanism.
        (installation / 'packaging').mkdir()
        (installation / 'source/gui/first.py').rename(installation / 'packaging/first.py')
        manifest['gui/first.py'].update(target='packaging/first.py', native_build=True)
        manifest_path.write_text(json.dumps(manifest))
        (installation / 'native/runtime').mkdir(parents=True)
        (installation / updater.NATIVE_FILES[0]).write_bytes(b'old native library')
        payload = {name: ('compiled fixture ' + name).encode() for name in updater.NATIVE_FILES}
        before = contents(installation)
        original_replace = updater.replace

        def fail_native(path, data, mode):
            if case == 'native-rollback' and path.name == 'libsherloq_dense_stream.dylib':
                raise OSError('injected native replacement failure')
            if case == 'receipt-failure' and path.name == 'native-build.json':
                raise OSError('injected receipt failure')
            original_replace(path, data, mode)

        def compile_fixture():
            if case == 'native-build-failure':
                raise ValueError('injected compiler failure')
            if case == 'native-incomplete':
                return {updater.NATIVE_FILES[0]: b'incomplete'}
            return payload

        with patch.object(updater, 'SUPPORT', repository / 'macos'), \
             patch.object(updater, 'build_native_payload', side_effect=compile_fixture) as build, \
             patch.object(updater, 'replace', fail_native):
            if case == 'native-success':
                assert updater.apply(installation) == 5
                assert updater.apply(installation) == 0
                assert build.call_count == 1
                for name, data in payload.items():
                    assert (installation / name).read_bytes() == data
                assert (installation / 'packaging/first.py').read_bytes() == b'new-first.py'
            else:
                try:
                    updater.apply(installation)
                except (ValueError, OSError):
                    pass
                else:
                    raise AssertionError('Expected refusal: ' + case)
                assert contents(installation) == before, case
                assert not (installation / updater.RECEIPT).exists()
        cases.append(case)
print(json.dumps({'passed': True, 'cases': cases}))
