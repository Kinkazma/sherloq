"""Versioned, bounded-memory model acquisition; no framework imports or inference.

Files are verified before becoming visible at their final path. Interrupted
compressed payloads remain resumable. OS locks coordinate independent workers.
"""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import hashlib
import json
import os
import re
import shutil
import time
import urllib.request
import zlib

ROOT = Path(__file__).resolve().parents[4]
CHUNK = 1024 * 1024
_verified = {}


class Cancelled(Exception):
    pass


def check(cancel):
    if cancel():
        raise Cancelled('Model download cancelled; downloaded bytes are retained.')


def safe_path(root, relative):
    p = Path(relative)
    if p.is_absolute() or not p.parts or '..' in p.parts:
        raise ValueError('Invalid model path')
    target = root / p
    if target.resolve() != target.absolute() or not target.resolve().is_relative_to(root):
        raise ValueError('Model path contains a symlink or escapes the installation')
    return target


def digest(path, cancel=lambda: False):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(CHUNK):
            check(cancel)
            value.update(block)
    return value.hexdigest()


def valid(path, asset, cancel=lambda: False):
    if not path.is_file() or path.is_symlink():
        return False
    st = path.stat()
    if st.st_size != asset['bytes']:
        return False
    stamp = (str(path), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns,
             st.st_ctime_ns, asset['sha256'])
    if stamp not in _verified:
        if digest(path, cancel) != asset['sha256']:
            return False
        _verified[stamp] = True
    return True


@contextmanager
def locked(path, cancel):
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        while True:
            check(cancel)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(.1)
        yield
    finally:
        os.close(fd)


class Store:
    def __init__(self, root=ROOT, catalog=None, opener=urllib.request.urlopen):
        self.root = Path(root).expanduser().resolve(strict=True)
        self.catalog = catalog or json.loads(Path(__file__).with_name('model_catalog.json').read_text())
        self.opener = opener
        if self.catalog.get('schema') != 1:
            raise ValueError('Unsupported model catalogue')
        self.assets = self.catalog['assets']
        for key, asset in self.assets.items():
            safe_path(self.root, asset['path'])
            if not isinstance(asset['bytes'], int) or asset['bytes'] < 0 or not re.fullmatch(r'[a-f0-9]{64}', asset['sha256']):
                raise ValueError('Invalid model size or hash: ' + key)
            if asset['encoding'] not in ('identity', 'deflate'):
                raise ValueError('Unsupported model encoding')
            for part in asset['parts']:
                if not part['url'].startswith('https://') or part['bytes'] <= 0 or part.get('offset', 0) < 0:
                    raise ValueError('Invalid model source')

    def resolve(self, feature):
        selected, visiting = [], set()
        def visit(name):
            if name in visiting:
                raise ValueError('Cyclic model dependency')
            visiting.add(name)
            spec = self.catalog['features'][name]
            for dep in spec.get('requires', []):
                visit(dep)
            for key in spec.get('assets', []):
                if key not in selected:
                    if key not in self.assets:
                        raise ValueError('Unknown model asset')
                    selected.append(key)
            visiting.remove(name)
        visit(feature)
        return selected

    def ensure(self, feature, cancel=lambda: False, progress=lambda *args: None):
        ids = self.resolve(feature)
        total = sum(sum(p['bytes'] for p in self.assets[key]['parts']) for key in ids)
        completed = 0
        for key in ids:
            asset = self.assets[key]
            count = sum(p['bytes'] for p in asset['parts'])
            self.install(asset, cancel, lambda done, text: progress(completed + done, total, text))
            completed += count
        progress(total, total, 'Models ready')

    def install(self, asset, cancel, progress):
        check(cancel)
        target = safe_path(self.root, asset['path'])
        progress(0, 'Checking ' + target.name)
        if target.exists():
            if valid(target, asset, cancel):
                return target
            raise ValueError('Existing model has a different size or hash; left untouched: ' + asset['path'])
        cache = safe_path(self.root, '.model-downloads/' + asset['sha256'])
        cache.mkdir(parents=True, exist_ok=True)
        with locked(cache / 'lock', cancel):
            if target.exists():
                if valid(target, asset, cancel):
                    return target
                raise ValueError('Conflicting model appeared during download')
            remaining = sum(max(0, p['bytes'] - ((cache / f'{i}.partial').stat().st_size
                if (cache / f'{i}.partial').exists() else 0))
                for i, p in enumerate(asset['parts']))
            if shutil.disk_usage(cache).free < remaining + asset['bytes'] + 16 * CHUNK:
                raise OSError('Not enough free disk space to download and verify ' + target.name)
            files, completed = [], 0
            for index, part in enumerate(asset['parts']):
                temporary = safe_path(self.root, str((cache / f'{index}.partial').relative_to(self.root)))
                self.download(part, temporary, cancel,
                              lambda n: progress(completed + n, 'Downloading ' + target.name))
                files.append(temporary)
                completed += part['bytes']
            check(cancel)
            progress(completed, 'Verifying ' + target.name)
            decoded = safe_path(self.root, str((cache / 'decoded.partial').relative_to(self.root)))
            decoder = zlib.decompressobj(-15) if asset['encoding'] == 'deflate' else None
            size = 0
            hasher = hashlib.sha256()
            def write(output, block):
                nonlocal size
                size += len(block)
                if size > asset['bytes']:
                    raise ValueError('Model exceeds expected decoded size')
                output.write(block)
                hasher.update(block)
            try:
                with decoded.open('wb') as output:
                    for file in files:
                        with file.open('rb') as stream:
                            while block := stream.read(CHUNK):
                                check(cancel)
                                if decoder:
                                    while block:
                                        write(output, decoder.decompress(block, CHUNK))
                                        block = decoder.unconsumed_tail
                                else:
                                    write(output, block)
                    if decoder and (not decoder.eof or decoder.unused_data):
                        raise ValueError('Incomplete compressed model')
                    output.flush()
                    os.fsync(output.fileno())
                if size != asset['bytes'] or hasher.hexdigest() != asset['sha256']:
                    for file in files:
                        file.unlink(missing_ok=True)
                    raise ValueError('Model SHA256 mismatch; nothing installed. Retry the download.')
                check(cancel)
                target = safe_path(self.root, asset['path'])
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(decoded, target)
                except FileExistsError:
                    if not valid(target, asset, cancel):
                        raise ValueError('Conflicting model appeared; left untouched')
                for file in files:
                    file.unlink(missing_ok=True)
                return target
            except (zlib.error, ValueError):
                for file in files:
                    file.unlink(missing_ok=True)
                raise
            finally:
                decoded.unlink(missing_ok=True)

    def download(self, part, path, cancel, progress):
        expected = part['bytes']
        saved = path.stat().st_size if path.exists() else 0
        if saved > expected:
            path.unlink()
            saved = 0
        progress(saved)
        if saved == expected:
            return
        start = part.get('offset', 0) + saved
        end = part.get('offset', 0) + expected - 1
        request = urllib.request.Request(part['url'], headers={
            'User-Agent': 'SHERLOQ-model-store/1', 'Accept-Encoding': 'identity',
            'Range': f'bytes={start}-{end}'})
        check(cancel)
        with self.opener(request, timeout=15) as response:
            if response.status == 206:
                value = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                if value is None or (int(value[1]), int(value[2])) != (start, end):
                    raise ValueError('Server returned an unexpected model byte range')
            elif response.status == 200 and not part.get('range', False):
                # Servers may ignore Range for a standalone asset. Restart it.
                saved = 0
            else:
                raise ValueError('This model source did not honor the requested byte range')
            with path.open('ab' if saved else 'wb') as output:
                while saved < expected:
                    check(cancel)
                    block = response.read(min(CHUNK, expected - saved))
                    if not block:
                        raise OSError('Model download interrupted; retry to resume')
                    output.write(block)
                    output.flush()
                    saved += len(block)
                    progress(saved)
                os.fsync(output.fileno())


def feature_for(method, variant=None):
    if method == 'clone_detectors':
        return 'forgeryscope' if variant.startswith('Forgeryscope') else 'clone:' + variant
    if method == 'adaptive_cfa':
        return 'cfa:' + variant
    if method == 'noiseprint':
        return 'noiseprint:' + str(variant)
    if method == 'copy_move':
        if variant.startswith('XFeat'):
            return 'xfeat-lighter' if 'Glue' in variant else 'xfeat'
        if variant.startswith('ALIKED'):
            return 'aliked' + ('-rotation' if 'rotation' in variant else '') + ('-glue' if 'Glue' in variant else '')
        if variant == 'SIFT + LightGlue':
            return 'sift-glue'
        return 'none'
    return method
