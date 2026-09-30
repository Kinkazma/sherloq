"""Real SIFT layer in the complete panel; other clone jobs explicitly stubbed."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import sys,time,json,tempfile
from pathlib import Path
import cv2 as cv,numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer,Qt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.tools.tampering.complete_analysis import CompleteAnalysisWidget,SIFT_SOURCE
from gui.sherloq_app.core.automatic_clones import export
from gui.sherloq_app.ui.jobs import _POOL
app=QApplication([]);errors=[];sys.excepthook=lambda typ,value,tb:errors.append(str(value))
im=cv.imread(str(ROOT/'tests/sift-panels/synthetic.png'))
w=CompleteAnalysisWidget(im,autostart=False)
w.job.compute=lambda _:dict(groups=(),group_algorithms=())
w.forge.request=lambda *_:QTimer.singleShot(0,lambda:w.complete('forgeryscope',dict(metadata=dict(zones=[]))))
regions=(((20,90),(299,90),(299,319),(20,319)),((360,90),(639,90),(639,319),(360,319)))
w.auto_ready(regions)
def wait():
    end=time.monotonic()+90
    while True:
        app.processEvents();time.sleep(.01)
        assert not errors and not w.errors,(errors,w.errors)
        assert time.monotonic()<end,w.states
        if (all(s=='Complete' for s in w.states.values()) and
                not any(j.is_busy for j in (w.sift_job,w.job,w.ela_job,w.prepare,w.draw,w.export_job))):break
wait()
assert w.results['sift']['groups'] and w.results['sift']['biome_partitions']
assert w.submitted['sift_parameters'][0]==SIFT_SOURCE
assert w.submitted['sift_parameters'][3]==10 and not w.submitted['sift_parameters'][11]
rows=[e for e in w.biomes if e['source']==SIFT_SOURCE];assert rows
tokens=(w.sift_job.token,w.job.token,w.ela_job.token,w.prepare.token)
counts=w.sift_engine.counts.copy()
w.tabs.setCurrentIndex(w.sources.index(SIFT_SOURCE)+1);wait()
assert w.viewer.view.biomes and all(e['source']==SIFT_SOURCE for e in w.viewer.view.biomes)
key=w.model.rows[0]['id'];w.model.setData(w.model.index(0),Qt.Unchecked,Qt.CheckStateRole);w.clear_focus();wait()
assert key not in [e['id'] for e in w.viewer.view.biomes]
w.source_checks[SIFT_SOURCE].setChecked(False);wait();assert not w.viewer.view.biomes
w.source_checks[SIFT_SOURCE].setChecked(True);wait()
assert key in w.model.hidden and key not in [e['id'] for e in w.viewer.view.biomes]
assert tokens==(w.sift_job.token,w.job.token,w.ela_job.token,w.prepare.token)
assert counts==w.sift_engine.counts
with tempfile.TemporaryDirectory() as folder:
    path=export((str(Path(folder)/'complete.npz'),dict(results=w.results,configuration=w.submitted,biomes=w.biomes)))
    with np.load(path) as saved:
        assert np.array_equal(saved['root_results_sift_pairs'],w.results['sift']['pairs'])
        meta=json.loads(str(saved['metadata_json']))
        assert meta['results']['sift']['biome_partitions']
        assert meta['configuration']['sift_parameters'][0]==SIFT_SOURCE
w.close();_POOL.waitForDone(10000);app.processEvents();assert not errors
print('PASS: real SIFT layer, source checkbox/tab, hidden biome, no display inference, export provenance')
