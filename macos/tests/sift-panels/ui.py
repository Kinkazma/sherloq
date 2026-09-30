"""Offscreen real job, automatic zones, cached display and NPZ roundtrip."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import json,sys,time,tempfile
from pathlib import Path
from threading import Event
import cv2 as cv
import numpy as np
from PySide6.QtWidgets import QApplication
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.tools.tampering.cloning2 import Cloning2Widget,write_export,PANELS_TEXT
from gui.sherloq_app.ui.localization import choice
app=QApplication([])
im=cv.imread(str(Path(__file__).with_name('synthetic.png')))
w=Cloning2Widget(im)
assert choice(w.algorithm)=='PatchMatch Zernike'
w.algorithm.setCurrentText(PANELS_TEXT)
assert choice(w.algorithm)==PANELS_TEXT and w.cpu.isEnabled() and w.low_spin.value()==10
w.limit.setValue(1200);w.start(False)
def wait(predicate):
    end=time.monotonic()+90
    while not predicate():
        app.processEvents();time.sleep(.01)
        assert time.monotonic()<end,w.status.text()
wait(lambda:w.result is not None and w.export.isEnabled())
assert len(w.viewer.view.regions)==3 and not w.dirty
assert w.result['preprocessing']['text_boxes']
before=w.engine.counts.copy();result=w.result
w.show_text.setChecked(True)
wait(lambda:not w.render_job.is_busy and w.export.isEnabled())
assert w.result is result and w.engine.counts==before
with tempfile.TemporaryDirectory() as folder:
    filename=str(Path(folder)/'result.npz')
    write_export((filename,result,w.style(),tuple(w.visible),im,Event()))
    with np.load(filename) as data:
        metadata=json.loads(str(data['metadata_json']))
        assert metadata['algorithm']==PANELS_TEXT and metadata['preprocessing']['text_boxes']
        assert len(metadata['zones'])==3 and metadata['display'][-1] is True
        assert np.array_equal(data['pairs'],result['pairs'])
w.close();app.processEvents()
print('PASS: real UI worker, automatic zones, text overlay, cache, NPZ')
