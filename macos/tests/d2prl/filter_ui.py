"""Cached-grid filtering: boundaries, zones, UI races and actual TIFF result.

Run with QT_QPA_PLATFORM=offscreen. No checkpoint or neural inference needed.
"""
import json
import sys
import time
from pathlib import Path
import cv2 as cv
import numpy as np
from PySide6.QtWidgets import QApplication

R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'source'))
from gui.sherloq_app.core.d2prl import postprocess,refilter
from gui.sherloq_app.tools.tampering.clone_detectors import CloneDetectorsWidget,render
from gui.sherloq_app.ui.jobs import _POOL

raw=np.zeros((3,448,448),np.float32)
raw[:,10:14,10:15]=.9 # 20 connected pixels
raw[:,50:55,50:56]=.9 # 30 connected pixels
for minimum,count in ((0,50),(20,50),(21,30),(30,30),(31,0),(500,0)):
    assert int(postprocess(raw,minimum)[0].sum())==count
shape=(448,916)
result=dict(map=np.zeros(shape,np.float32),mask=np.zeros(shape,np.uint8),
    analyzed=np.ones(shape,np.uint8),candidates=np.zeros(shape,np.uint8),
    source=np.zeros(shape,np.float32),target=np.zeros(shape,np.float32),
    raw_probabilities=np.stack([raw,raw]),metadata=dict(variant='D2PRL',seconds=1,
        boxes=[(0,0,448,448),(468,0,916,448)],zones=[{},{}],status='empty'))
result['analyzed'][:,448:468]=0
out=refilter((result,21))
assert out['mask'].sum()==60 and not out['mask'][:,448:468].any()
assert not result['mask'].any() and 'min_component' not in result['metadata']
assert out['raw_probabilities'] is result['raw_probabilities']

app=QApplication([]);errors=[]
sys.excepthook=lambda typ,value,tb:errors.append(str(value))
w=CloneDetectorsWidget(np.full((*shape,3),100,np.uint8))
w.variant.setCurrentText('D2PRL')
assert not w.filter_controls.isHidden() and w.minimum.value()==500
def forbidden(*args):raise AssertionError('Filtering must not start inference')
w.job.request=forbidden
def drain():
    deadline=time.monotonic()+10
    while w.filter_job.is_busy or w.draw.is_busy:
        app.processEvents();time.sleep(.005)
        assert time.monotonic()<deadline and not errors,errors
w.complete(result);drain();assert not w.result['mask'].any()
for value in (0,500,21,0):w.minimum.setValue(value)
assert not w.save.isEnabled()
drain()
assert w.result['mask'].sum()==100 and w.minimum_value.value()==0 and w.save.isEnabled()
w.minimum_value.setValue(21);drain()
assert w.minimum.value()==21 and w.result['mask'].sum()==60
overlay=render((w.image,w.result,'Superposition'))
assert np.array_equal(overlay[~w.result['mask'].astype(bool)],w.image[~w.result['mask'].astype(bool)])
w.minimum.setValue(0);w.changed();drain()
assert w.result is None and not w.save.isEnabled()
w.variant.setCurrentIndex(0);assert w.filter_controls.isHidden()
w.close();_POOL.waitForDone(10000);app.processEvents();assert not errors,errors

report=dict(passed=True,boundary_cases=6,independent_zones=True,no_inference=True,
    last_slider_value_wins=True,invalidation=True,overlay_respects_mask=True)
folder=R/'tests/d2prl/user-images-round2'
if (folder/'results.json').exists():
    record=json.loads((folder/'results.json').read_text())[0]
    cached=dict(np.load(folder/'image-1-maps.npz'))
    cached['metadata']=record['metadata']
    for minimum in (500,403,100,0):
        start=time.perf_counter();filtered=refilter((cached,minimum))
        report[str(minimum)]=dict(mask_pixels=int(filtered['mask'].sum()),seconds=time.perf_counter()-start)
    assert report['500']['mask_pixels']==0 and report['403']['mask_pixels']>0
    assert report['0']['mask_pixels']>=report['100']['mask_pixels']>=report['403']['mask_pixels']
print(json.dumps(report,indent=2))
(R/'tests/d2prl/filter-ui.json').write_text(json.dumps(report,indent=2))
