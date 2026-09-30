"""Two panels, exact display-only scope filters; no automatic research reruns."""
import os,sys,time,json
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer,Qt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.tools.tampering.automatic_clones import AutomaticClonesWidget
from gui.sherloq_app.tools.tampering.complete_analysis import CompleteAnalysisWidget
from gui.sherloq_app.core.automatic_clones import SOURCES,_entry
from gui.sherloq_app.core.clone_relations import polygon_key
from gui.sherloq_app.core.clone_corroboration import context_counts
from gui.sherloq_app.ui.jobs import _POOL
app=QApplication([]);errors=[];sys.excepthook=lambda typ,value,tb:errors.append(str(value))
image=np.full((210,430,3),120,np.uint8)
def rect(x,y,w,h):return ((x,y),(x+w,y),(x+w,y+h),(x,y+h))
panels=(rect(10,10,180,180),rect(230,10,180,180));envelope=rect(10,10,400,180)
def entry(source,polygons,context):return _entry(source,polygons,12,dict(search_region=context,search_context='roi:'+polygon_key(np.asarray(context))))
rows=(entry(SOURCES[3],(rect(15,15,165,165),rect(235,15,165,165)),envelope),
      entry(SOURCES[3],(rect(30,40,8,8),rect(80,90,8,8)),panels[0]),
      entry(SOURCES[3],(rect(30,40,8,8),rect(80,90,8,8)),envelope),
      entry(SOURCES[0],(rect(15,15,165,165),rect(235,15,165,165)),envelope))
for cls in (AutomaticClonesWidget,CompleteAnalysisWidget):
 w=cls(image,autostart=False)
 w.job.compute=lambda _:dict(groups=(),group_algorithms=())
 w.sift_job.compute=lambda _:dict(groups=())
 w.forge.request=lambda *_,w=w:QTimer.singleShot(0,lambda:w.complete('forgeryscope',dict(metadata=dict(zones=[]))))
 w.d2_job.request=lambda *_,w=w:QTimer.singleShot(0,lambda:w.complete('d2prl',dict(mask=np.zeros(image.shape[:2],np.uint8),metadata=dict(boxes=[],zones=[]))))
 w.auto_ready(panels)
 def wait():
  end=time.monotonic()+60
  while True:
   app.processEvents();time.sleep(.005)
   assert not errors and not w.errors,(errors,w.errors)
   assert time.monotonic()<end,w.states
   jobs=[w.job,w.sift_job,w.prepare,w.draw,w.export_job]+([w.ela_job] if hasattr(w,'ela_job') else [])
   if all(s=='Complete' for s in w.states.values()) and not any(j.is_busy for j in jobs):break
 wait()
 AutomaticClonesWidget.biomes_ready(w,rows);wait()
 assert w.presentation.currentData()=='overlay' and w.relations.currentData()=='within'
 assert w.relations.count()==3 and w.opacity.value()==70
 assert len(w.viewer.view.biomes)==3 # two local search contexts + AI
 assert context_counts(image.shape,w.viewer.view.biomes)[42,32]==3
 tokens=(w.job.token,w.sift_job.token,w.prepare.token)+((w.ela_job.token,) if hasattr(w,'ela_job') else ())
 n=w.renderer.builds;w.opacity.setValue(25);wait();assert w.renderer.builds==n
 w.presentation.setCurrentIndex(0);wait()
 assert w.relations.currentData()=='all' and len(w.viewer.view.biomes)==4
 w.relations.setCurrentIndex(1);wait() # remember biomes: between
 assert len(w.viewer.view.biomes)==2
 w.presentation.setCurrentIndex(2);wait()
 assert w.relations.currentData()=='within' and len(w.viewer.view.biomes)==3
 w.relations.setCurrentIndex(2);wait() # remember corroboration: all
 assert len(w.viewer.view.biomes)==4
 w.presentation.setCurrentIndex(1);wait() # heat and overlay share relation selection
 assert w.relations.currentData()=='all' and len(w.viewer.view.biomes)==4
 w.presentation.setCurrentIndex(0);wait()
 assert w.relations.currentData()=='between' and len(w.viewer.view.biomes)==2
 w.presentation.setCurrentIndex(2);wait();assert w.relations.currentData()=='all'
 for i,expected in ((0,3),(1,2),(2,4)):
  w.relations.setCurrentIndex(i);wait();assert len(w.viewer.view.biomes)==expected
  assert any(e['source']==SOURCES[0] for e in w.viewer.view.biomes)
 w.source_checks[SOURCES[0]].setChecked(False);wait()
 assert len(w.viewer.view.biomes)==3
 w.source_checks[SOURCES[0]].setChecked(True);wait()
 # AI evidence with unknown endpoints is still independent of the biome filter.
 unknown=entry(SOURCES[0],(rect(0,0,200,200),rect(210,0,210,205)),envelope)
 AutomaticClonesWidget.biomes_ready(w,rows+(unknown,));wait()
 w.relations.setCurrentIndex(0);wait()
 assert sum(e['source']==SOURCES[0] for e in w.viewer.view.biomes)==2
 w.tabs.setCurrentIndex(w.sources.index(SOURCES[3])+1);wait()
 assert all(e['source']==SOURCES[3] for e in w.viewer.view.biomes)
 w.tabs.setCurrentIndex(0);wait()
 if hasattr(w,'ela_mode'):
  from gui.sherloq_app.core.complete_analysis import ELA_SOURCE,render as render_complete
  ela=dict(id='ela-test',source=ELA_SOURCE,polygons=(rect(20,20,30,30),),count=1,
           color=(0,255,255),block=10,cells=((3,3),),provenance={})
  AutomaticClonesWidget.biomes_ready(w,rows+(ela,));wait()
  w.ela_mode.setCurrentIndex(1);wait()
  assert w.effective_ela_mode()==0 and not any(e['source']==ELA_SOURCE for e in w.viewer.view.biomes)
  assert w.source_checks[ELA_SOURCE].isChecked()
  w.presentation.setCurrentIndex(0);wait()
  assert w.effective_ela_mode()==1 and any(e['source']==ELA_SOURCE for e in w.viewer.view.biomes)
  w.presentation.setCurrentIndex(2);wait()
  assert w.ela_mode.currentIndex()==1 and w.effective_ela_mode()==0
  # Core renderer also refuses the ELA background and polygons in heat modes.
  from gui.sherloq_app.core.automatic_clones import render as render_clones
  preview=np.full_like(image,230)
  assert np.array_equal(render_complete((image,rows+(ela,),(),preview,1,'overlay',.7)),render_clones((image,rows,(),'overlay',.7)))
  w.tabs.setCurrentIndex(w.sources.index(ELA_SOURCE)+1);wait()
  assert not w.presentation.isEnabled() and w.effective_ela_mode()==1
  w.tabs.setCurrentIndex(0);w.presentation.setCurrentIndex(0);wait()
  w.source_checks[ELA_SOURCE].setChecked(False);wait()
  w.presentation.setCurrentIndex(2);w.presentation.setCurrentIndex(0);wait()
  assert not w.source_checks[ELA_SOURCE].isChecked() and not any(e['source']==ELA_SOURCE for e in w.viewer.view.biomes)
 after=(w.job.token,w.sift_job.token,w.prepare.token)+((w.ela_job.token,) if hasattr(w,'ela_job') else ())
 assert after==tokens,(tokens,after)
 config=w.corroboration_snapshot();assert not config['exclude_enclosing_search']
 w.close();_POOL.waitForDone(10000);app.processEvents()
print(json.dumps(dict(passed=True,both_panels=True,envelope_retained=True,per_view_preferences=True,ela_temporarily_hidden=True,ia_retained=True,
 relations_separate=True,no_inference_for_display=True,no_size_weighting=True)))
