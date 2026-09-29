"""Historical float64 Wiener/mean/NCC, with atomic database snapshots.

A database is an explicit training snapshot, never an automatically refreshed
folder cache. Input order and every scientific arithmetic operation are retained.
"""
import os
import hashlib
import json
import tempfile
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
import h5py
from scipy.signal import wiener
from .interactive import ArrayCache

WIENER_SIZE = 3
NCC_THRESHOLD = 0.005
SCHEMA = "sherloq-wiener-ncc-1"

def parse_camera_label(filename: str) -> str:
    stem = Path(filename).stem
    parts = stem.split("_")
    return "_".join(parts[:-1]) if len(parts) >= 3 else stem


def scan_dataset(path: str) -> dict:
    """Only cameras with at least 2 images are included in the result,
    since a single image is not enough to build a reliable PRNU fingerprint.
    """

    camera_imgs = defaultdict(list)
    ext = {".jpg", ".jpeg", ".JPG", ".JPEG"}
    for dirpath, _, files in os.walk(path):
        for f in files:
            if Path(f).suffix.lower() in {".jpg", ".jpeg"}:
                label = parse_camera_label(f)
                camera_imgs[label].append(os.path.join(dirpath, f))
    return {k: v for k, v in camera_imgs.items() if len(v) >= 2}


def extract_residual(image: np.ndarray) -> np.ndarray:
    if image.ndim != 2 or min(image.shape) < 3:
        raise ValueError("PRNU requires at least 3 × 3 pixels")
    residual= image - wiener(image, mysize=WIENER_SIZE)
    residual = np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)

    # zero-pads at the border internally, which biases the local mean/variance estimate for edge pixels. Crop that border out
    pad = (WIENER_SIZE - 1) // 2
    if pad > 0:
        residual = residual[pad:-pad, pad:-pad]
    return residual

def load_image_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a, b = a[:h, :w], b[:h, :w]
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    return float(np.sum(a * b) / denom) if denom > 1e-10 else 0.0


class Cancelled(Exception):
    pass


def check_cancel(cancel):
    if cancel():
        raise Cancelled()


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def image_identity(path):
    s = os.stat(path)
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def camera_images(folder, single_camera=None):
    if single_camera:
        if '/' in single_camera or single_camera in {'.', '..'}:
            raise ValueError('Camera label must not contain a slash or be . or ..')
        paths = [os.path.join(root, name) for root, _, names in os.walk(folder)
                 for name in names if Path(name).suffix.lower() in {'.jpg', '.jpeg'}]
        return {single_camera: paths} if len(paths) >= 2 else {}
    return scan_dataset(folder)


def validate_database(handle):
    schema = handle.attrs.get('schema')
    if schema is not None and (schema != SCHEMA or not handle.attrs.get('complete', False)):
        raise ValueError('Unsupported or incomplete fingerprint database')
    cameras = list(handle.keys())
    if not cameras:
        raise ValueError('Fingerprint database is empty')
    for cam in cameras:
        group = handle[cam]
        if not isinstance(group, h5py.Group) or 'fingerprint' not in group:
            raise ValueError(f'Invalid fingerprint group: {cam}')
        dataset = group['fingerprint']
        if dataset.ndim != 2 or min(dataset.shape) == 0 or dataset.dtype != np.float64:
            raise ValueError(f'Invalid float64 fingerprint: {cam}')
    return cameras


def build_database(path, folder, single_camera=None, progress=lambda *a: None,
                   cancel=lambda: False, excluded_path=None):
    """All-or-nothing publication; never removes/replaces a pre-existing database."""
    cameras = camera_images(folder, single_camera)
    if not cameras:
        raise ValueError('At least two JPEGs per camera are required. Use the single-camera label for ordinary camera filenames.')
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    query_hash = file_digest(excluded_path) if excluded_path and Path(excluded_path).is_file() else None
    fd, temporary = tempfile.mkstemp(prefix='.prnu-', suffix='.h5', dir=target.parent)
    os.close(fd)
    try:
        with h5py.File(temporary, 'w') as handle:
            handle.attrs['schema'] = SCHEMA
            handle.attrs['complete'] = False
            for cam_idx, (cam, paths) in enumerate(cameras.items()):
                mean_fp = None
                used, skipped = [], []
                for idx, source in enumerate(paths):
                    check_cancel(cancel)
                    progress(5 + int(60 * (cam_idx + (idx+1)/len(paths))/len(cameras)),
                             f'{cam}: image {idx+1}/{len(paths)}')
                    try:
                        before = image_identity(source)
                        digest = file_digest(source)
                        if digest == query_hash:
                            skipped.append({'name': Path(source).name, 'reason': 'query excluded from training'})
                            continue
                        residual = extract_residual(load_image_gray(source))
                        if before != image_identity(source):
                            raise IOError('Source changed during extraction')
                    except (OSError, ValueError, cv2.error) as exc:
                        skipped.append({'name': Path(source).name, 'reason': str(exc)})
                        continue
                    used.append({'name': Path(source).name, 'sha256': digest})
                    if mean_fp is None:
                        mean_fp = residual.copy()
                    else:
                        h = min(residual.shape[0], mean_fp.shape[0])
                        w = min(residual.shape[1], mean_fp.shape[1])
                        mean_fp = mean_fp[:h, :w]
                        mean_fp += (residual[:h, :w] - mean_fp) / len(used)
                check_cancel(cancel)
                if len(used) < 2:
                    raise ValueError(f'{cam}: fewer than two readable training images after query exclusion')
                group = handle.create_group(cam)
                group.create_dataset('fingerprint', data=mean_fp, dtype=np.float64,
                                     compression='gzip', compression_opts=4)
                group.attrs['n_images'] = len(paths)
                group.attrs['n_used'] = len(used)
                group.attrs['shape'] = mean_fp.shape
                group.attrs['training_manifest'] = json.dumps(used)
                group.attrs['skipped_images'] = json.dumps(skipped)
                del mean_fp, residual
            handle.attrs['complete'] = True
            handle.flush()
        check_cancel(cancel)
        # Hard link publishes without overwriting a concurrently created destination.
        os.link(temporary, target)
    finally:
        os.unlink(temporary)


class PrnuEngine:
    def __init__(self, image, filename=None):
        self.image, self.filename = image, filename
        self.residuals = ArrayCache(192)
        self.centered = ArrayCache(192)
        self._scores_key = None
        self._scores = None

    def residual(self):
        value = self.residuals.get('query')
        if value is None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
            value = self.residuals.put('query', extract_residual(gray).copy())
        return value

    def identify(self, path, folder=None, single_camera=None,
                 progress=lambda *a: None, cancel=lambda: False):
        check_cancel(cancel)
        if not os.path.exists(path):
            if not folder:
                raise ValueError('No training folder selected')
            build_database(path, folder, single_camera, progress, cancel, self.filename)
        identity = image_identity(path)
        key = (os.path.abspath(path), identity)
        if key == self._scores_key:
            return list(self._scores)
        results = []
        with h5py.File(path, 'r') as handle:
            cameras = validate_database(handle)
            query_hash = file_digest(self.filename) if self.filename and Path(self.filename).is_file() else None
            for cam in cameras:
                manifest = json.loads(handle[cam].attrs.get('training_manifest', '[]'))
                if query_hash and any(x['sha256'] == query_hash for x in manifest):
                    raise ValueError('The query belongs to the training set. Use a held-out image.')
            check_cancel(cancel)
            progress(68, 'Extracting query residual')
            residual = self.residual()
            for idx, cam in enumerate(cameras):
                check_cancel(cancel)
                progress(70 + int(28 * (idx+1)/len(cameras)), f'Matching: {cam}')
                fp = handle[cam]['fingerprint'][:]
                if not np.isfinite(fp).all():
                    raise ValueError(f'Non-finite fingerprint: {cam}')
                shape = min(residual.shape[0], fp.shape[0]), min(residual.shape[1], fp.shape[1])
                h, w = shape
                prepared = self.centered.get(shape)
                if prepared is None:
                    a = residual[:h, :w]
                    a = a - a.mean()
                    prepared = self.centered.put(shape, (a, (a**2).sum()))
                a, a2 = prepared
                b = fp[:h, :w]
                b = b - b.mean()
                denom = np.sqrt(a2 * (b**2).sum())
                score = float(np.sum(a * b) / denom) if denom > 1e-10 else 0.0
                results.append((cam, score))
                del fp, b
        check_cancel(cancel)
        if identity != image_identity(path):
            raise IOError('Fingerprint database changed during identification; retry')
        results.sort(key=lambda x: x[1], reverse=True)
        self._scores_key, self._scores = key, results
        progress(100, 'Identification complete')
        return list(results)
