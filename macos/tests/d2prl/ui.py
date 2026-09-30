import sys,time,json
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'source'))
import numpy as np,cv2 as cv
from PySide6.QtWidgets import QApplication
from gui.sherloq_app.tools.tampering.clone_detectors import CloneDetectorsWidget
from gui.sherloq_app.ui.extensions import EXTENSION_ROLE
from gui.sherloq_app.ui.jobs import _POOL
app=QApplication([]);errors=[];sys.excepthook=lambda t,v,b:errors.append(str(v))
w=CloneDetectorsWidget(cv.imread(str(R/'third_party/research/clone_detectors/01_d2prl/testpic/1.jpg')))
w.variant.setCurrentText('D2PRL');assert w.variant.currentData(EXTENSION_ROLE)=='green'
assert all(w.view_mode.model().item(i).isEnabled() for i in [4,5])
assert not w.compare.isEnabled();assert not w.cpu.isChecked() and w.cpu.isEnabled()
w.cpu.setChecked(False);w.start(False)
end=time.monotonic()+180
while w.job.is_busy or w.draw.is_busy:
 app.processEvents();time.sleep(.02);assert time.monotonic()<end,w.status.text();assert not errors,errors
assert w.result is not None,w.status.text();assert w.result['metadata']['variant']=='D2PRL'
assert w.save.isEnabled();assert all(k in w.result for k in ('source','target','raw_probabilities'))
w.view_mode.setCurrentIndex(4)
while w.draw.is_busy:app.processEvents();time.sleep(.02)
w.start(False);deadline=time.monotonic()+30
while w.progress.value()<1:
 app.processEvents();time.sleep(.02);assert time.monotonic()<deadline and not errors
w.changed();assert w.result is None and not w.save.isEnabled()
w.close();_POOL.waitForDone(30000);app.processEvents();assert not errors,errors
report=dict(passed=True,green_outline=True,resident_worker=True,source_target_views=True,cancellation=True)
(R/'tests/d2prl/ui.json').write_text(json.dumps(report,indent=2));print(report)
