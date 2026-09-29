"""Conservative aligned double-quantization trace detector (new, experimental).

Looks for depleted bins on quantization lattices, not a compression counter or
an authenticity test. Based on the double-quantization mechanism described by
Popescu/Farid (2004) and He et al. (2006); this is NOT their trained classifier.
All histogram counts are exact integers; no GPU or estimated pixel-domain DCT.
"""
from pathlib import Path
from contextlib import nullcontext
import hashlib
import json
import tempfile
import time
import numpy as np
from scipy.ndimage import gaussian_filter1d

FREQUENCIES = ((0,1),(1,0),(1,1),(0,2),(2,0),(1,2),(2,1),(0,3),(3,0))
THRESHOLD = .8
MIN_VOTES = 3
VERSION = 'aligned-lattice-v1'
LIMITATIONS = ('Experimental global detector for aligned JPEG grids and sufficiently separated quantization steps. '
               'Equal qualities, a coarser last compression, cropping, resampling, small or smooth images can hide traces. '
               'Periodic or synthetic content can imitate traces. A detected trace does not establish malicious editing; '
               'an inconclusive result does not establish a single compression.')


def lattice(q1, q2, bins=256):
    """Positive round-half-up lattice; +/-1 tolerance for final steps 1 or 2."""
    # Integer arithmetic avoids floating point tie ambiguity. Include enough
    # source levels to cover the complete histogram, regardless of step ratio.
    levels = np.arange(((bins+1) * q2 + q1 - 1) // q1 + 1, dtype=np.int64)
    mapped = (2 * levels * q1 + q2) // (2 * q2)
    allowed = np.zeros(bins+1, dtype=bool)
    allowed[mapped[mapped < bins+1]] = True
    if q2 < 3:
        allowed = np.convolve(allowed.astype(np.int8), [1,1,1], mode='same') > 0
    return allowed[:bins]


def histogram_evidence(histogram, q2):
    h = np.asarray(histogram, dtype=np.int64)
    if h.shape != (256,) or q2 < 1:
        raise ValueError('Expected 256 nonnegative histogram bins and a positive step.')
    if (h < 0).any():
        raise ValueError('Histogram counts must be nonnegative.')
    best = dict(score=0., candidate_step=None, current_step=int(q2), samples=0,
                expected_gap_fraction=None, observed_gap_fraction=None, eligible=False)
    k = np.arange(256)
    for q1 in range(max(3, 2*q2), min(128, 8*q2)+1):
        allowed = lattice(q1, q2)
        mask = (k >= max(3, (q1+q2-1)//q2)) & (k <= 128)
        n = int(h[mask].sum())
        if n < 1000 or np.count_nonzero(h[mask]) < 12:
            continue
        smooth = gaussian_filter1d(h.astype(np.float64), max(2., q1/q2), mode='reflect')
        expected = float(smooth[mask & ~allowed].sum() / max(1., smooth[mask].sum()))
        if expected < .25:
            continue
        observed = float(h[mask & ~allowed].sum()/n)
        score = float(max(0., 1-observed/expected))
        if not best['eligible'] or score > best['score']:
            best.update(score=score, candidate_step=q1, samples=n, eligible=True,
                        expected_gap_fraction=expected, observed_gap_fraction=observed)
    return best


def analyze(path, snapshot_directory=None):
    """Read a private byte snapshot, then analyze exact stored luminance DCTs.

    Run in a cancellable helper process: libjpeg parsing stays outside the UI.
    Input bytes are never rewritten. Candidate steps can be harmonics and must
    not be presented as recovered original JPEG qualities.
    """
    from PIL import Image
    import jpeglib
    started = time.perf_counter()
    source = Path(path)
    initial = source.stat()
    if initial.st_size > 512 * 1024**2:
        raise ValueError('JPEG file exceeds the 512 MiB analysis limit.')
    digest = hashlib.sha256()
    context = tempfile.TemporaryDirectory(prefix='sherloq-double-jpeg-') if snapshot_directory is None else nullcontext(snapshot_directory)
    with context as folder:
        snapshot = Path(folder)/'input.jpg'
        with source.open('rb') as reader, snapshot.open('wb') as writer:
            total = 0
            while chunk := reader.read(1024*1024):
                total += len(chunk)
                if total > 512 * 1024**2:
                    raise ValueError('JPEG file exceeds the 512 MiB analysis limit.')
                digest.update(chunk); writer.write(chunk)
        final = source.stat()
        if (initial.st_size,initial.st_mtime_ns,initial.st_ino) != (final.st_size,final.st_mtime_ns,final.st_ino):
            raise ValueError('The image file changed during analysis. Reload it and retry.')
        with Image.open(snapshot) as header:
            if header.format != 'JPEG':
                raise ValueError('This detector requires an actual JPEG file. The recompression curve remains available for other formats.')
            if header.mode not in ('L','RGB'):
                raise ValueError('This detector supports grayscale and RGB JPEGs, not CMYK JPEGs.')
            width,height = header.size
            if width*height > 100_000_000:
                raise ValueError('JPEG exceeds the 100 megapixel analysis limit.')
        jpeg = jpeglib.read_dct(str(snapshot))
        coefficients = jpeg.Y
        if coefficients is None:
            raise ValueError('No luminance DCT component is available.')
        # Exclude padded edge blocks: repeated edge pixels are not evidence.
        coefficients = coefficients[:height//8,:width//8]
        table = jpeg.qt[jpeg.quant_tbl_no[0]]
        records = []
        for row,col in FREQUENCIES:
            values = np.abs(coefficients[:,:,row,col].astype(np.int32)).ravel()
            histogram = np.bincount(values, minlength=256)[:256]
            evidence = histogram_evidence(histogram, int(table[row,col]))
            evidence.update(frequency=[row,col], histogram=histogram.tolist(),
                            ignored_tail=int(np.count_nonzero(values>=256)))
            records.append(evidence)
        votes = sum(r['eligible'] and r['score'] >= THRESHOLD for r in records)
        eligible = sum(r['eligible'] for r in records)
        detected = votes >= MIN_VOTES
        return dict(version=VERSION, verdict='compatible_traces' if detected else 'inconclusive',
                    reason='lattice_depletion' if detected else ('insufficient_histograms' if eligible<MIN_VOTES else 'no_strong_consensus'),
                    supporting_frequencies=votes, eligible_frequencies=eligible, tested_frequencies=len(records),
                    threshold=THRESHOLD, required_frequencies=MIN_VOTES,
                    source_sha256=digest.hexdigest(), dimensions=[width,height],
                    complete_blocks=int(coefficients.shape[0]*coefficients.shape[1]),
                    progressive=bool(jpeg.progressive_mode), jpeglib_version=__import__('importlib.metadata', fromlist=['version']).version('jpeglib'),
                    records=records, limitations=LIMITATIONS, seconds=time.perf_counter()-started)


if __name__ == '__main__':
    import sys
    try:
        print(json.dumps(analyze(sys.argv[1], sys.argv[2] if len(sys.argv)>2 else None), allow_nan=False))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
