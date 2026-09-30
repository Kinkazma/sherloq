import os,sys,json
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import numpy as np
from PySide6.QtWidgets import QApplication
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'source'))
from gui.sherloq_app.tools.tampering.automatic_clones import AutomaticClonesWidget
from gui.sherloq_app.tools.tampering.complete_analysis import CompleteAnalysisWidget
from gui.sherloq_app.ui.jobs import _POOL
app=QApplication([])
for cls in (AutomaticClonesWidget,CompleteAnalysisWidget):
 w=cls(np.zeros((800,1200,3),np.uint8),autostart=False);w.resize(1650,1000);w.show();app.processEvents()
 w.states=dict.fromkeys(w.states,'Running');w.update_status();app.processEvents()
 baseline=(w.viewer.geometry(),w.status.height(),w.detail.height(),w.size())
 w.progress('patchmatch',37,'Checking geometric consistency');app.processEvents()
 assert '37%' in w.status.text() and 'PatchMatch' in w.status.text() and 'Checking geometric consistency' in w.detail.text()
 w.d2_job.progress.emit(0,'En attente d’une autre analyse…');app.processEvents()
 assert 'D2PRL' in w.detail.text() and ('attente' in w.detail.text() or 'Waiting' in w.detail.text())
 w.forge.progress.emit(5,'Chargement du modèle…');app.processEvents()
 assert 'Forgeryscope' in w.detail.text() and ('Chargement' in w.detail.text() or 'Loading' in w.detail.text())
 w.states['patchmatch']='Complete';w.progress('d2prl',51,'Analyse 1100/3000');app.processEvents()
 assert '1/'+str(len(w.states)) in w.status.text() and '51%' in w.status.text()
 assert 'Analyse 1100/3000' in w.detail.text()
 assert baseline==(w.viewer.geometry(),w.status.height(),w.detail.height(),w.size())
 assert 'SIFT+G2NN+RANSAC' in w.status.text() and 'Forgeryscope' in w.status.text()
 w.states=dict.fromkeys(w.states,'Complete');w.update_status();app.processEvents()
 assert 'Calculations finished' in w.status.text() and not w.detail.text()
 assert baseline==(w.viewer.geometry(),w.status.height(),w.detail.height(),w.size())
 w.states['sift']='Failed';w.errors['sift']='example';w.update_status();app.processEvents()
 assert 'partial results (errors)' in w.status.text() and 'example' in w.detail.text()
 assert baseline==(w.viewer.geometry(),w.status.height(),w.detail.height(),w.size())
 w.close();_POOL.waitForDone(10000);app.processEvents()
print(json.dumps(dict(passed=True,both_panels=True,live_group_percentages=True,live_worker_messages=True,unchanged_viewer_geometry=True,clear_completion_errors=True)))
