"""Per-tab presentation state, both automatic panels and independent AI layers."""
import os,sys,time,json
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import numpy as np,cv2 as cv
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'source'))
from gui.sherloq_app.tools.tampering.automatic_clones import AutomaticClonesWidget
from gui.sherloq_app.tools.tampering.complete_analysis import CompleteAnalysisWidget
from gui.sherloq_app.core.automatic_clones import SOURCES,D2PRL_SOURCE,_entry
from gui.sherloq_app.core.d2prl import postprocess
from gui.sherloq_app.core.d2prl_regions import regions
from gui.sherloq_app.core.clone_corroboration import context_counts
from gui.sherloq_app.ui.jobs import _POOL
app=QApplication([]);errors=[];sys.excepthook=lambda typ,v,tb:errors.append(str(v))
image=np.full((210,430,3),120,np.uint8)
def rect(x,y,w,h):return ((x,y),(x+w,y),(x+w,y+h),(x,y+h))
panels=(rect(10,10,180,180),rect(230,10,180,180));envelope=rect(10,10,400,180)
raw=np.zeros((3,448,448),np.float32);raw[0,30:90,30:90]=1;raw[0,50:60,50:60]=0;raw[0,120:130,120:130]=1
m,t,s=postprocess(raw)
d2=dict(mask=cv.resize(m,(430,210),interpolation=cv.INTER_NEAREST),target=cv.resize(t,(430,210),interpolation=cv.INTER_NEAREST),source=cv.resize(s,(430,210),interpolation=cv.INTER_NEAREST),analyzed=np.ones((210,430),np.uint8),raw_probabilities=raw[None],metadata=dict(boxes=[[0,0,430,210]],zones=[{}]))
for cls in (AutomaticClonesWidget,CompleteAnalysisWidget):
 w=cls(image,autostart=False);calls=[]
 w.job.compute=lambda _:dict(groups=(),group_algorithms=())
 w.sift_job.compute=lambda _:dict(groups=())
 w.forge.request=lambda *_,w=w:QTimer.singleShot(0,lambda:w.complete('forgeryscope',dict(metadata=dict(zones=[]))))
 def request(params,device):
  calls.append((params,device));QTimer.singleShot(0,lambda:w.complete('d2prl',d2))
 w.d2_job.request=request
 def wait():
  end=time.monotonic()+60
  while True:
   app.processEvents();time.sleep(.004)
   assert not errors and not w.errors,(errors,w.errors)
   assert time.monotonic()<end,w.states
   jobs=[w.job,w.sift_job,w.prepare,w.draw,w.export_job]+([w.ela_job] if hasattr(w,'ela_job') else [])
   if all(v=='Complete' or v.startswith('Disabled') for v in w.states.values()) and not any(j.is_busy for j in jobs):break
 w.auto_ready(panels)
 assert 'partial' in w.status.text().lower() or 'partiels' in w.status.text().lower()
 status_height=w.status.height();wait()
 assert w.status.height()==status_height and not w.status.wordWrap()
 assert ('finished' in w.status.text().lower() or 'terminés' in w.status.text().lower()),w.status.text()
 assert len(calls)==1 and calls[0][1]=='mps' and calls[0][0]['regions']==(*panels,envelope)
 assert calls[0][0]['variant']==D2PRL_SOURCE and w.d2_minimum.value()==500
 assert any(e['source']==D2PRL_SOURCE for e in w.biomes)
 tokens=(w.job.token,w.sift_job.token,len(calls))+((w.ela_job.token,) if hasattr(w,'ela_job') else ())
 # Overlay defaults persist independently of isolated tabs.
 assert w.presentation.currentData()=='overlay' and w.relations.currentData()=='within'
 w.opacity.setValue(57)
 w.tabs.setCurrentIndex(w.sources.index(SOURCES[1])+1);wait()
 assert w.presentation.currentData()=='biomes' and w.relations.currentData()=='all'
 w.relations.setCurrentIndex(1);w.presentation.setCurrentIndex(1);w.relations.setCurrentIndex(0);wait()
 w.tabs.setCurrentIndex(w.sources.index(SOURCES[2])+1);wait()
 assert w.presentation.currentData()=='biomes' and w.relations.currentData()=='all'
 w.tabs.setCurrentIndex(0);wait()
 assert w.presentation.currentData()=='overlay' and w.relations.currentData()=='within' and w.opacity.value()==57
 w.tabs.setCurrentIndex(w.sources.index(SOURCES[1])+1);wait()
 assert w.presentation.currentData()=='heat' and w.relations.currentData()=='within'
 w.presentation.setCurrentIndex(0);wait();assert w.relations.currentData()=='between'
 w.tabs.setCurrentIndex(w.sources.index(D2PRL_SOURCE)+1);wait()
 assert w.presentation.currentData()=='biomes' and not w.relations.isEnabled()
 assert all(e['source']==D2PRL_SOURCE for e in w.viewer.view.biomes) and w.viewer.view.biomes
 n=len(w.viewer.view.biomes);w.d2_minimum.setValue(0);wait();assert len(w.viewer.view.biomes)>n and len(calls)==1
 # No geometric minimum/overlap gate is applied to D2PRL segmentation.
 w.minimum.setValue(200);w.overlap.setValue(1);wait();assert w.viewer.view.biomes
 w.tabs.setCurrentIndex(0);wait()
 assert not w.d2_controls.isHidden()
 before=context_counts(image.shape,w.viewer.view.biomes)
 w.d2_slider.setValue(500);wait()
 assert w.d2_minimum.value()==500 and len(calls)==1
 after_map=context_counts(image.shape,w.viewer.view.biomes)
 assert np.any(before>after_map)
 for index in range(3):
  w.relations.setCurrentIndex(index);wait();assert any(e['source']==D2PRL_SOURCE for e in w.viewer.view.biomes)
 w.source_checks[D2PRL_SOURCE].setChecked(False);wait();assert not any(e['source']==D2PRL_SOURCE for e in w.viewer.view.biomes)
 w.source_checks[D2PRL_SOURCE].setChecked(True);wait()
 after=(w.job.token,w.sift_job.token,len(calls))+((w.ela_job.token,) if hasattr(w,'ela_job') else ())
 assert after==tokens,(tokens,after)
 # Disabling the enclosing pass must not disable D2PRL's local searches.
 w.zone_model.disabled.add(2);w.start();wait()
 assert calls[-1][0]['regions']==panels and w.states['forgeryscope'].startswith('Disabled')
 w.zone_model.disabled.add(0);w.start();wait()
 assert calls[-1][0]['regions']==(panels[1],) and calls[-1][0]['excluded']==(panels[0],)
 # A failed method must not be reported as a complete success.
 w.failed('d2prl','test failure');assert 'partial' in w.status.text().lower() or 'partiels' in w.status.text().lower()
 w.close();_POOL.waitForDone(10000);app.processEvents()
print(json.dumps(dict(passed=True,both_panels=True,per_tab_preferences=True,d2_integration=True,refilter_without_inference=True,status_fixed_height=True)))
