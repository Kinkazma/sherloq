"""96 MP production worker: real MPS Noiseprint -> full global statistics -> PNG.

Development qualification only. No cached residual, resized input, reduced
features, skipped replicate or local substitute for the global model.
"""
from pathlib import Path
import sys, json, time, subprocess, shutil, resource
import numpy as np
import cv2 as cv
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
folder = OUT/'run96'
folder.mkdir(exist_ok=False)
image = np.lib.format.open_memmap(folder/'image.npy', mode='w+', dtype=np.uint8, shape=(8000,12000,3))
rng = np.random.default_rng(20261001)
for start in range(0,8000,128):
    count = min(128,8000-start)
    # Independently noisy pixels, with a repeated region added below.
    image[start:start+count] = rng.integers(16,240,size=(count,12000,3),dtype=np.uint8)
image[4500:5500,8500:9500] = image[1000:2000,1500:2500]
image.flush(); del image
command = [sys.executable, str(ROOT/'source/gui/sherloq_app/core/splicing_worker.py'), str(folder),'map','101','mps']
start = time.perf_counter()
with (OUT/'96mp-worker.log').open('w') as log:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log, text=True)
    stages = []
    for line in process.stdout:
        log.write(line);log.flush()
        try:
            value, text = json.loads(line)['progress']
            stages.append(dict(seconds=time.perf_counter()-start,progress=value,text=text))
            print(json.dumps(stages[-1]),flush=True)
        except (ValueError,KeyError): pass
    code = process.wait()
assert code == 0, (OUT/'96mp-worker.log').read_text()[-3000:]
metadata = json.loads((folder/'map-statistics.json').read_text())
assert metadata['statistics_policy'] == 'covariance-floor-v1'
noise = np.load(folder/'noise.npy',mmap_mode='r')
spam = np.load(folder/'spam.npy',mmap_mode='r')
valid = np.load(folder/'valid.npy',mmap_mode='r')
raw = np.load(folder/'map.npy',mmap_mode='r')
render = np.load(folder/'map-display.npy',mmap_mode='r')
assert noise.shape == (8000,12000) and render.shape == (8000,12000,3)
assert np.isfinite(raw).all() and raw.shape == valid.shape
assert raw.shape[0] > 900 and raw.shape[1] > 1400
assert cv.imwrite(str(folder/'export.png'),render)
decoded = cv.imread(str(folder/'export.png'))
assert np.array_equal(decoded,render)
report = dict(passed=True,pixels=96000000,image_shape=[8000,12000,3],
              backend='mps',full_noise_inference=True,global_statistics=True,
              features_shape=list(spam.shape),valid_cells=int(valid.sum()),
              full_resolution_render=True,export_exact=True,
              worker_seconds=stages[-1]['seconds'],total_seconds=time.perf_counter()-start,
              peak_worker_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
              statistics=metadata,stages=stages,
              scope='96 MP noisy synthetic image with copied region; capacity/lifecycle, not a forensic accuracy benchmark.')
(OUT/'96mp-results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='stages'}),flush=True)
del noise,spam,valid,raw,render,decoded
shutil.rmtree(folder)
