"""Local, cancellable Tesseract text localization for clone preprocessing.

Only boxes/confidence leave this module, never transcribed text. Tiling keeps
native resolution and bounds OCR memory; no training or startup calibration.
"""
import csv
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import cv2 as cv
import numpy as np

from .cloning import check


def executable():
    root = Path(__file__).resolve().parents[4]
    configured = os.environ.get('SHERLOQ_TESSERACT')
    candidates = [configured] if configured else [str(root / 'native/bin/tesseract'), shutil.which('tesseract')]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(Path(candidate).resolve())
    raise RuntimeError('Text exclusion requires Tesseract and its English language data. Install Tesseract or set SHERLOQ_TESSERACT.')


def parse_boxes(tsv, origin, shape):
    """Conservative word evidence; keep weak OCR guesses out of microscopy."""
    h, w = shape[:2]
    boxes = []
    for row in csv.DictReader(tsv.splitlines(), delimiter='\t', quoting=csv.QUOTE_NONE):
        if row.get('level') != '5':
            continue
        word = row.get('text', '').strip()
        confidence = float(row['conf'])
        if confidence < 70 or not any(c.isalnum() for c in word):
            continue
        if len(word) == 1 and not (word.isupper() and confidence >= 90):
            continue
        x, y, bw, bh = (int(row[k]) for k in ('left', 'top', 'width', 'height'))
        if bw <= 0 or bh <= 0:
            continue
        pad = max(2, round(bh * .1))
        left, top = max(0, x + origin[0] - pad), max(0, y + origin[1] - pad)
        right, bottom = min(w - 1, x + origin[0] + bw - 1 + pad), min(h - 1, y + origin[1] + bh - 1 + pad)
        boxes.append(dict(bounds=(left, top, right, bottom), confidence=confidence))
    return boxes


def supported_boxes(image, boxes):
    """Require a flat label background, not OCR guesses on biological texture.

    Trim rows outside that background (OCR occasionally joins a caption to a
    branch below it). Deliberately conservative: labels on textured backgrounds
    may remain. This is an exclusion rule, not a scientific-content classifier.
    """
    accepted = []
    for box in boxes:
        x, y, r, b = box['bounds']
        gray = cv.cvtColor(image[y:b+1, x:r+1], cv.COLOR_BGR2GRAY)
        peak = int(np.bincount(gray.ravel(), minlength=256).argmax())
        flat = np.abs(gray.astype(np.int16) - peak) <= 5
        if flat.mean() < .45:
            continue
        rows = np.flatnonzero(flat.mean(axis=1) >= .2)
        if not len(rows):
            continue
        # Only the dominant contiguous background band may define an exclusion.
        bands = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
        band = max(bands, key=len)
        if len(band) < max(4, gray.shape[0] * .45):
            continue
        accepted.append(dict(box, bounds=(x, y + int(band[0]), r, y + int(band[-1]))))
    return accepted


def _tile(image, bounds, command, cancel):
    check(cancel)
    x, y, right, bottom = bounds
    with tempfile.TemporaryDirectory(prefix='sherloq-text-') as folder:
        path = Path(folder)
        gray = cv.cvtColor(image[y:bottom, x:right], cv.COLOR_BGR2GRAY)
        if not cv.imwrite(str(path / 'input.png'), gray):
            raise RuntimeError('Could not prepare text localization input.')
        args = [command, str(path / 'input.png'), str(path / 'output'), '-l', 'eng', '--psm', '11']
        data = Path(__file__).resolve().parents[4] / 'native/share/tessdata'
        if (data / 'eng.traineddata').is_file():
            args += ['--tessdata-dir', str(data)]
        args += ['tsv']
        with (path / 'stderr').open('wb') as log:
            process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=log,
                                       env={**os.environ, 'OMP_THREAD_LIMIT': '1'})
            try:
                while True:
                    check(cancel)
                    try:
                        process.wait(timeout=.1)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                check(cancel)
                if process.returncode:
                    raise RuntimeError('Tesseract text localization failed; check the executable and eng.traineddata installation.')
                return supported_boxes(image, parse_boxes((path / 'output.tsv').read_text(), (x, y), image.shape))
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()


def detect_text(image, cancel=lambda: False, progress=lambda *args: None):
    command = executable()
    h, w = image.shape[:2]
    edge, overlap = 1536, 128
    def starts(length):
        values = list(range(0, max(1, length - edge + 1), edge - overlap))
        if length > edge and values[-1] != length - edge:
            values.append(length - edge)
        return values
    tiles = [(x, y, min(w, x + edge), min(h, y + edge)) for y in starts(h) for x in starts(w)]
    abort = Event()
    stopped = lambda: cancel() or abort.is_set()
    boxes = []
    workers = min(4, os.cpu_count() or 1, len(tiles))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Submit a bounded window rather than a future for every huge-image tile.
        try:
            for start in range(0, len(tiles), workers):
                check(stopped)
                futures = [pool.submit(_tile, image, tile, command, stopped) for tile in tiles[start:start + workers]]
                for future in futures:
                    boxes.extend(future.result())
                progress(min(100, (start + workers) * 100 // len(tiles)), 'Locating text')
        finally:
            abort.set()
    # Suppress duplicate words from overlapping tiles, preserving stable order.
    selected = []
    for box in sorted(boxes, key=lambda b: (-b['confidence'], b['bounds'])):
        x, y, r, b = box['bounds']
        if any(max(0, min(r, rr) - max(x, xx) + 1) * max(0, min(b, bb) - max(y, yy) + 1)
               / max(1, min((r-x+1)*(b-y+1), (rr-xx+1)*(bb-yy+1))) > .8
               for xx, yy, rr, bb in (other['bounds'] for other in selected)):
            continue
        selected.append(box)
    return tuple(sorted(selected, key=lambda box: box['bounds']))


def polygons(boxes):
    return tuple(((x, y), (r, y), (r, b), (x, b)) for x, y, r, b in (box['bounds'] for box in boxes))
