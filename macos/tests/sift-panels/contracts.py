"""Real OCR + SIFT tests; no author weights or user images required."""
import json
import os
from pathlib import Path
import sys
import time

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source'))
from gui.sherloq_app.core.cloning2 import Cloning2Engine, PANELS_TEXT, polygon_mask, render
from gui.sherloq_app.core.text_regions import detect_text, parse_boxes, supported_boxes, executable
from gui.sherloq_app.core.jpeg_curve import Cancelled

cv.setNumThreads(4)
rng = np.random.default_rng(3010)
image = np.full((350, 660, 3), 255, np.uint8)
a = cv.GaussianBlur(rng.integers(20, 230, (230, 280, 3), dtype=np.uint8), (3, 3), 0)
b = cv.GaussianBlur(rng.integers(20, 230, (230, 280, 3), dtype=np.uint8), (3, 3), 0)
b[60:170, 80:210] = a[50:160, 50:180]
image[90:320, 20:300] = a
image[90:320, 360:640] = b
for x in (30, 370):
    cv.putText(image, 'SAMPLE ABC', (x, 55), cv.FONT_HERSHEY_SIMPLEX, .9, (0, 0, 0), 2, cv.LINE_AA)
regions = (((20,90),(299,90),(299,319),(20,319)), ((360,90),(639,90),(639,319),(360,319)))
params = (PANELS_TEXT, 1200, 800., 10., .725, 50., 'Affine', 3., 8, 8, 8, False, 2., True, True, False, (), ())
engine = Cloning2Engine(image)
start = time.perf_counter()
automatic = engine.analyze(params)
assert len(automatic['regions']) == 3, automatic['regions']
assert len(automatic['preprocessing']['text_boxes']) >= 2
assert automatic['groups'], 'Known copied texture must survive text exclusion.'
assert any(box['bounds'][1] < 65 for box in automatic['preprocessing']['text_boxes'])
mask = polygon_mask(image.shape[:2], (), automatic['preprocessing']['text_exclusions'])
xy = np.rint(automatic['points'][:,:2]).astype(int)
assert np.all(mask[xy[:,1],xy[:,0]] == 255), 'Text exclusion centres must never be SIFT features.'
assert engine.analyze(params) is automatic
counts = engine.counts.copy()
style = (10.,800.,2,(),False,False,False,True,())
plain,_,_ = render(image, automatic, style)
shown,_,_ = render(image, automatic, (*style, True))
assert not np.array_equal(plain, shown) and engine.counts == counts
search = engine.analyze(params, regions)
for pair in search['pairs']:
    xs = search['points'][pair[:2].astype(int),0]
    assert (xs[0] < 320) == (xs[1] < 320), 'Search must never connect two distinct zones.'
compare = engine.analyze(params, regions, True)
assert len(compare['groups']) > 0
for pair in compare['pairs']:
    xs = compare['points'][pair[:2].astype(int),0]
    assert (xs[0] < 320) != (xs[1] < 320)
intact = image.copy()
intact[90:320, 360:640] = cv.GaussianBlur(rng.integers(20,230,(230,280,3),dtype=np.uint8),(3,3),0)
control = Cloning2Engine(intact).analyze(params)
assert not control['groups'], 'Repeated labels alone must not create a clone biome.'
excluded = list(params);excluded[16] = (regions[1],)
empty = engine.analyze(tuple(excluded), regions, True)
assert len(empty['pairs']) == 0
uniform = Cloning2Engine(np.full((100,120,3),128,np.uint8)).analyze(params)
assert not uniform['groups'] and not uniform['regions']
try:
    detect_text(image, lambda: True)
    raise AssertionError('Cancellation ignored')
except Cancelled:
    pass
# Cancellation while real OCR subprocesses are active, not only before launch.
deadline = time.monotonic() + .15
try:
    detect_text(np.tile(image,(5,5,1)), lambda:time.monotonic()>deadline)
    raise AssertionError('Active OCR cancellation ignored')
except Cancelled:
    assert time.monotonic()-deadline < 3
original = os.environ.get('SHERLOQ_TESSERACT')
try:
    os.environ['SHERLOQ_TESSERACT'] = str(ROOT/'tests/sift-panels/does-not-exist')
    try:
        executable()
        raise AssertionError('Missing OCR silently accepted')
    except RuntimeError:
        pass
finally:
    if original is None:os.environ.pop('SHERLOQ_TESSERACT', None)
    else:os.environ['SHERLOQ_TESSERACT'] = original
record = dict(passed=True, seconds=time.perf_counter()-start, automatic_zones=len(automatic['regions']),
              text_boxes=len(automatic['preprocessing']['text_boxes']), auto_pairs=len(automatic['pairs']),
              auto_biomes=len(automatic['groups']), compare_biomes=len(compare['groups']),
              isolated_search=True, exclusions=True, cache=True, cancellation=True,
              intact_repeated_labels_biomes=len(control['groups']))
Path(__file__).with_name('results.json').write_text(json.dumps(record,indent=2)+'\n')
cv.imwrite(str(Path(__file__).with_name('synthetic.png')),image)
print(record)
