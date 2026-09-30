"""Both automatic panels: full-chain identity, display-only heatmap/check state."""
import os,sys,time,json
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer,Qt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.tools.tampering.automatic_clones import AutomaticClonesWidget
from gui.sherloq_app.tools.tampering.complete_analysis import CompleteAnalysisWidget
from gui.sherloq_app.core.automatic_clones import SOURCES,SIFT_SOURCE,_entry
from gui.sherloq_app.core.clone_corroboration import votes,render_heat
from gui.sherloq_app.ui.jobs import _POOL
app=QApplication([]);errors=[];sys.excepthook=lambda typ,value,tb:errors.append(str(value))
image=np.full((160,360,3),120,np.uint8)
def rect(x):return [[x,30],[x+60,30],[x+60,90],[x,90]]
rows=tuple(_entry(s,[rect(20),rect(240)],12,{}) for s in SOURCES)
for cls in (AutomaticClonesWidget,CompleteAnalysisWidget):
 w=cls(image,autostart=False)
 w.job.compute=lambda _:dict(groups=(),group_algorithms=())
 w.sift_job.compute=lambda _:dict(groups=())
 w.forge.request=lambda *_,w=w:QTimer.singleShot(0,lambda:w.complete('forgeryscope',dict(metadata=dict(zones=[]))))
 w.d2_job.request=lambda *_,w=w:QTimer.singleShot(0,lambda:w.complete('d2prl',dict(mask=np.zeros(image.shape[:2],np.uint8),metadata=dict(boxes=[],zones=[]))))
 w.run_whole_image()
 def wait():
  deadline=time.monotonic()+45
  while True:
   app.processEvents();time.sleep(.01)
   assert not errors and not w.errors,(errors,w.errors)
   assert time.monotonic()<deadline,w.states
   jobs=[w.job,w.sift_job,w.prepare,w.draw,w.export_job]+([w.ela_job] if hasattr(w,'ela_job') else [])
   if all(s=='Complete' for s in w.states.values()) and not any(j.is_busy for j in jobs):break
 wait()
 assert w.submitted['sift_parameters'][0]==SIFT_SOURCE and w.submitted['sift_parameters'][11]
 tab=w.sources.index(SIFT_SOURCE)+1
 assert w.tabs.tabText(tab)=='SIFT + G2NN + RANSAC' and w.tabs.tabToolTip(tab)==SIFT_SOURCE
 AutomaticClonesWidget.biomes_ready(w,rows);wait()
 tokens=(w.job.token,w.sift_job.token,w.prepare.token)
 if hasattr(w,'ela_job'):tokens+= (w.ela_job.token,)
 w.presentation.setCurrentIndex(2);wait()
 assert votes(image.shape,w.viewer.view.biomes).max()==5
 w.source_checks[SOURCES[1]].setChecked(False);wait()
 assert votes(image.shape,w.viewer.view.biomes).max()==4
 w.tabs.setCurrentIndex(tab);wait()
 assert votes(image.shape,w.viewer.view.biomes).max()==1
 key=w.model.rows[0]['id'];w.model.setData(w.model.index(0),Qt.Unchecked,Qt.CheckStateRole);wait()
 assert not w.viewer.view.biomes
 w.tabs.setCurrentIndex(0);w.clear_focus();w.opacity.setValue(20);wait()
 assert key in w.model.hidden and votes(image.shape,w.viewer.view.biomes).max()==3
 after=(w.job.token,w.sift_job.token,w.prepare.token)
 if hasattr(w,'ela_job'):after+=(w.ela_job.token,)
 assert after==tokens,'Display controls reran analysis or segmentation'
 builds=w.renderer.builds
 w.opacity.setValue(30);wait();assert w.renderer.builds==builds
 w.opacity.setValue(20);wait();assert w.renderer.builds==builds
 w.presentation.setCurrentIndex(1);wait();assert w.renderer.builds==builds
 assert np.array_equal(w.viewer.processed,render_heat(image,w.viewer.view.biomes,(),'heat',.2))
 config=w.corroboration_snapshot();assert config['presentation']=='heat' and config['opacity']==.2
 assert set(e['source'] for e in config['entries'])=={SOURCES[0],SOURCES[2],SOURCES[4]}
 if hasattr(w,'ela_mode'):
  w.tabs.setCurrentIndex(w.sources.index('ELA biomes')+1);wait()
  assert not w.presentation.isEnabled() and np.array_equal(w.viewer.processed,image)
 w.close();_POOL.waitForDone(10000);app.processEvents()
assert not errors
print(json.dumps(dict(passed=True,both_panels=True,complete_chain_short_label=True,mirror_default=True,
 source_checkboxes=True,hidden_state=True,heat_counts=True,display_without_inference=True)))
