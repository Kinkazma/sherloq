"""Composite numerical/functional regression. Development only, never a preflight.

Optional --large checks the existing 3 MP / 20 MP F32 residuals without running
their unchanged neural network again. Fixtures are synthetic and portable.
"""
from pathlib import Path
import sys, json, time, hashlib
import numpy as np
import cv2 as cv
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source/gui'))
from noiseprint.post_em import EMgu_img, getSpamFromNoiseprint, faetReduce
from noiseprint.noiseprint_blind import genMappUint8
from noiseprint.utility.stable_covariance import STATISTICS_POLICY, RELATIVE_VARIANCE_FLOOR, floor_covariance


def normalized(m):
    return (m-m.min()) / max(float(np.ptp(m)), np.finfo(float).tiny)


def fit(spam, valid, threads=1, workers=1, policy=STATISTICS_POLICY):
    with threadpool_limits(limits=threads, user_api='blas'):
        start = time.perf_counter()
        m, o = EMgu_img(spam, valid, workers=workers, numerical_policy=policy)
        seconds = time.perf_counter() - start
    assert np.isfinite(m).all()
    return m, o, seconds


def fixture(name, directory, filename):
    target = OUT/'fixtures'/f'{name}.npz'
    if not target.exists():
        raise FileNotFoundError(f'Missing supplied synthetic fixture: {target.name}')
    return np.load(target)


report = {'policy': STATISTICS_POLICY, 'relative_variance_floor': RELATIVE_VARIANCE_FLOOR, 'cases': []}
for name, directory, filename in [
    ('singular-small', 'composite-small-singular', 'reference.json'),
    ('singular-chain', 'composite', 'chain-reference.json'),
    ('well-conditioned', 'composite', 'reference.json'),
]:
    f = fixture(name, directory, filename)
    spam, valid, r0, r1, size = getSpamFromNoiseprint(f['noise'], f['gray'])
    with threadpool_limits(limits=2, user_api='blas'):
        old, _, _ = fit(spam, valid, threads=2, policy='legacy')
        assert np.array_equal(old, f['old_map']), name
    runs = []
    first = None
    for threads, workers in [(1, 1), (2, 1), (4, 1), (8, 1), (1, 4)]:
        m, o, seconds = fit(spam, valid, threads, workers)
        raster = genMappUint8(m, valid, r0, r1, size)
        if first is None:
            first = m, raster
        normalized_error = float(abs(normalized(m)-normalized(first[0])).max())
        raster_error = int(abs(raster.astype(int)-first[1]).max())
        assert normalized_error < 1e-5, (name, normalized_error)
        assert raster_error <= 1, (name, raster_error)
        if workers == 4:
            assert np.array_equal(m, first[0]), name
        if name == 'well-conditioned':
            assert np.array_equal(raster, f['old_raster'])
            if threads == 2:
                assert np.array_equal(m, old)
        runs.append(dict(threads=threads, workers=workers, seconds=seconds,
                         normalized_thread_error=normalized_error, raster_thread_error=raster_error,
                         raw_thread_error=float(abs(m-first[0]).max()),
                         raster_old_error=int(abs(raster.astype(int)-f['old_raster']).max()),
                         covariance_regularizations=o['covariance_regularizations'],
                         pca_regularized_components=o['pca_regularized_components']))
        if threads == 2:
            np.savez_compressed(OUT/'fixtures'/f'{name}-stable.npz', map=m, raster=raster,
                                Sigma=o['Sigma'], mu=o['mu'], L=o['L'], eigs=o['eigs'],
                                outliersProb=o['outliersProb'])
    row = dict(name=name, legacy_stored_map_exact=True, runs=runs)
    report['cases'].append(row)
    print(json.dumps(row), flush=True)

# Known changed residual region, away from the transition of overlapping windows.
rng = np.random.default_rng(763)
gray = rng.uniform(.1, .9, (512, 512)).astype(np.float32)
noise = rng.normal(0, 1, gray.shape).astype(np.float32)
noise[144:336, 160:352] *= 2.5
spam, valid, r0, r1, size = getSpamFromNoiseprint(noise, gray)
rr, cc = np.meshgrid(r0, r1, indexing='ij')
positive = (rr >= 176) & (rr < 304) & (cc >= 192) & (cc < 320) & valid
negative = ((rr < 112) | (rr >= 368) | (cc < 128) | (cc >= 384)) & valid
def auc(m):
    a, b = m[positive], m[negative]
    ranks = rankdata(np.r_[a, b])
    return float((ranks[:len(a)].sum()-len(a)*(len(a)+1)/2)/(len(a)*len(b)))
old, oo, old_seconds = fit(spam, valid, policy='legacy')
new, no, new_seconds = fit(spam, valid)
assert auc(new) >= auc(old)-.001
assert auc(new) > .9
row = dict(name='known-residual-anomaly', old_auc=auc(old), new_auc=auc(new),
           old_seconds=old_seconds, new_seconds=new_seconds, raw_exact=np.array_equal(old,new))
report['cases'].append(row); print(json.dumps(row), flush=True)

# Exact dependencies exercise whitening as well as covariance regularization.
base = rng.normal(size=(25*28, 5))
spam = (base @ rng.normal(size=(5, 40))).reshape(25, 28, 40)
valid = np.ones(spam.shape[:2], bool)
first = None
for threads in [1, 2, 4]:
    m, o, _ = fit(spam, valid, threads=threads)
    assert o['pca_regularized_components'] == 27
    if first is None: first = m
    assert abs(normalized(m)-normalized(first)).max() < 1e-5
report['low_rank_whitening_stable'] = True
with threadpool_limits(limits=1, user_api='blas'):
    try:
        faetReduce(np.ones((100, 40)), range(32), True)
    except ValueError as e:
        assert 'variation' in str(e)
    else: raise AssertionError('Constant features must not invent an anomaly map')
cov = np.diag([0., 1.])
fixed, _ = floor_covariance(cov)
scaled, _ = floor_covariance(cov*1e6)
assert np.allclose(scaled, fixed*1e6, rtol=1e-14, atol=0)
report['zero_variation_rejected'] = report['scale_equivariance'] = True

if '--large' in sys.argv:
    for name in ['small', 'large', '20mp']:
        image = np.load(ROOT/'tests/f32'/f'{name}-input.npy', mmap_mode='r')
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY).astype(np.float32)/255
        noise = np.load(ROOT/'tests/f32'/f'{name}-mps-noise.npy', mmap_mode='r')
        spam, valid, r0, r1, size = getSpamFromNoiseprint(noise, gray)
        old, _, old_seconds = fit(spam, valid, workers=4, policy='legacy')
        new, o, new_seconds = fit(spam, valid, workers=4)
        row = dict(name=name, pixels=int(gray.size), old_seconds=old_seconds,
                   new_seconds=new_seconds, raw_exact=np.array_equal(old,new),
                   normalized_old_error=float(abs(normalized(old)-normalized(new)).max()),
                   covariance_regularizations=o['covariance_regularizations'])
        assert row['raw_exact'], row
        report['cases'].append(row); print(json.dumps(row), flush=True)
        del spam, gray, noise, old, new
report['passed'] = True
report['sources'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
    ROOT/'source/gui/noiseprint/post_em.py', ROOT/'source/gui/noiseprint/utility/gaussianMixture.py',
    ROOT/'source/gui/noiseprint/utility/stable_covariance.py']}
(OUT/'results.json').write_text(json.dumps(report, indent=2)+'\n')
