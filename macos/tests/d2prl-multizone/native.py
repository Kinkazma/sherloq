"""Actual D2PRL service through automatic UI: two real panels plus enclosing ROI."""
import os,sys,time,json
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import cv2 as cv,numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'source'))
from gui.sherloq_app.tools.tampering.automatic_clones import AutomaticClonesWidget
from gui.sherloq_app.core.auto_zones import detect_panels
from gui.sherloq_app.core.clone_corroboration import context_counts
from gui.sherloq_app.core.d2prl import refilter
from gui.sherloq_app.ui.jobs import _POOL
app=QApplication([]);errors=[];sys.excepthook=lambda typ,value,tb:errors.append(str(value))
im=cv.imread(sys.argv[1]);panels=detect_panels(im);assert len(panels)==16
w=AutomaticClonesWidget(im,autostart=False)
w.job.compute=lambda _:dict(groups=(),group_algorithms=())
w.sift_job.compute=lambda _:dict(groups=())
w.forge.request=lambda *_:QTimer.singleShot(0,lambda:w.complete('forgeryscope',dict(metadata=dict(zones=[]))))
start=time.perf_counter();w.auto_ready(panels[:2]);last=None
end=time.monotonic()+240
while True:
 app.processEvents();time.sleep(.01)
 assert not errors and not w.errors,(errors,w.errors)
 assert time.monotonic()<end,w.states
 if w.states['d2prl']!=last:
  last=w.states['d2prl'];print(last,flush=True)
 if all(v=='Complete' for v in w.states.values()) and not any(j.is_busy for j in (w.prepare,w.draw)):break
result=w.results['d2prl'];assert len(result['metadata']['zones'])==3
assert result['raw_probabilities'].shape==(3,3,448,448)
assert np.array_equal(context_counts(im.shape,w.biomes),result['mask'])
w.resize(1400,900);w.show();app.processEvents();w.viewer.view.zoom_at(3,w.viewer.view.viewport().rect().center());app.processEvents()
v=w.viewer.view
before=(v.transform(),v.horizontalScrollBar().value(),v.verticalScrollBar().value())
command=w.d2_job.current
w.d2_slider.setValue(0)
while w.prepare.is_busy or w.draw.is_busy:app.processEvents();time.sleep(.01)
assert w.d2_job.current==command and np.array_equal(context_counts(im.shape,w.biomes),refilter((result,0))['mask'])
assert (v.transform(),v.horizontalScrollBar().value(),v.verticalScrollBar().value())==before
report=dict(passed=True,real_d2prl=True,other_detectors_stubbed=True,regions=3,native_grids=list(result['raw_probabilities'].shape),seconds=time.perf_counter()-start,mask_pixels=int(result['mask'].sum()),zero_filter_pixels=int(context_counts(im.shape,w.biomes).sum()),zoom_preserved=True,refilter_without_inference=True)
Path(__file__).with_name('native-results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
w.close();_POOL.waitForDone(10000);app.processEvents()
