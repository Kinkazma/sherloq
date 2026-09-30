"""Real resident worker, Auto selector, branch views and NPZ export."""
import sys,time,json,tempfile
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import QApplication
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'source'))
from gui.sherloq_app.tools.tampering.clone_detectors import CloneDetectorsWidget,render
from gui.sherloq_app.tools.noise.noisesniffer import export
from gui.sherloq_app.ui.extensions import EXTENSION_ROLE
from gui.sherloq_app.ui.jobs import _POOL
from gui.sherloq_app.core.clone_detectors import analyze
from gui.sherloq_app.core.automatic_clones import parameters
from gui.sherloq_app.core.cloning2 import EXTENDED_SYMMETRIC
app=QApplication([]);errors=[];sys.excepthook=lambda kind,value,tb:errors.append(str(value))
w=CloneDetectorsWidget(np.zeros((192,256,3),np.uint8));w.variant.setCurrentText('Forgeryscope Auto')
assert w.variant.currentData(EXTENSION_ROLE)=='green' and not w.cpu.isChecked()
assert all(w.view_mode.model().item(i).isEnabled() for i in range(6,10))
assert w.filter_controls.isHidden() and not w.compare.isEnabled()
w.start(False);end=time.monotonic()+120
while w.job.is_busy or w.draw.is_busy:
 app.processEvents();time.sleep(.02);assert time.monotonic()<end and not errors,(w.status.text(),errors)
assert w.result is not None,w.status.text()
assert w.result['metadata']['variant']=='Forgeryscope Auto'
assert w.result['metadata']['status']=='no_panels' and w.save.isEnabled()
for i in range(6,10):
 w.view_mode.setCurrentIndex(i)
 while w.draw.is_busy:app.processEvents();time.sleep(.005)
for key in ('branch_microscopy','branch_blots','branch_lanes','geometric'):
 assert key in w.result and w.result[key].shape==(192,256)
with tempfile.TemporaryDirectory() as folder:
 name=export((str(Path(folder)/'result.npz'),w.result))
 with np.load(name,allow_pickle=False) as saved:
  assert 'branch_lanes' in saved and json.loads(str(saved['metadata_json']))['variant']=='Forgeryscope Auto'
w.variant.setCurrentText('Forgeryscope microscopie')
assert all(not w.view_mode.model().item(i).isEnabled() for i in range(6,10))
assert not w.cpu.isChecked()
w.close();_POOL.waitForDone(10000);app.processEvents();assert not errors,errors
# The accepted mirror+scale default is untouched by this standalone addition.
rect=((0,0),(255,0),(255,191),(0,191))
assert parameters((192,256,3),(rect,),rect,set())[0][0]==EXTENDED_SYMMETRIC
report=dict(passed=True,worker=True,green=True,default_gpu=True,branch_views=True,npz_export=True,
 extended_mirror_profile_unchanged=True)
Path(__file__).with_name('ui-results.json').write_text(json.dumps(report,indent=2));print(report)
