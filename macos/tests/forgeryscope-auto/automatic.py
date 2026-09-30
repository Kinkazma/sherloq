"""Both automatic panels: routing, evidence, display filters and exports."""
import json,sys,time,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication,QFileDialog
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'source'))
from gui.sherloq_app.core.automatic_clones import entries,SOURCES,render
from gui.sherloq_app.core.cloning2 import EXTENDED_SYMMETRIC
from gui.sherloq_app.tools.tampering.automatic_clones import AutomaticClonesWidget
from gui.sherloq_app.tools.tampering.complete_analysis import CompleteAnalysisWidget
from gui.sherloq_app.ui.jobs import _POOL
def poly(x,y=5):return [[x,y],[x+10,y],[x+10,y+10],[x,y+10]]
micro=dict(polygon0=poly(5),polygon1=poly(40),score=.9,inliers=12,supported=True,accepted=True,branch='microscopy')
blot=dict(polygon0=poly(60),polygon1=poly(100),score=.9,inliers=0,supported=False,accepted=True,branch='blots')
lanes=dict(polygon0=poly(120),polygon1=poly(160),score=.9,display_polygon0=poly(125),display_polygon1=poly(175))
forge=dict(metadata=dict(variant='Forgeryscope Auto',zones=[dict(origin=[10,20],
 comparisons=[micro,blot,dict(blot,accepted=False)],lane_pairs=[lanes],lane_matches=1,embedding_candidates=[{},{}])]))
rows=entries(forgeryscope=forge);assert len(rows)==3
assert {e['provenance']['branch'] for e in rows}=={'microscopy','blots','lanes'}
assert rows[0]['polygons'][0][0]==[15.,25.]
assert rows[2]['polygons'][0][0]==[135.,25.] # expanded lane geometry, translated once
assert rows[1]['count_kind']=='pairs' and rows[1]['provenance']['evidence']=='similarity'
assert not entries(forgeryscope=forge,low=80)
assert [r['id'] for r in entries(forgeryscope=forge,low=0)]==[r['id'] for r in rows]
near=dict(metadata=dict(zones=[dict(lane_pairs=[dict(lanes,display_polygon0=poly(0),display_polygon1=poly(2))])]))
assert not entries(forgeryscope=near,low=0)
assert len(entries(forgeryscope=near,low=0,maximum_overlap=.9))==1

app=QApplication([]);errors=[];sys.excepthook=lambda typ,value,tb:errors.append(str(value))
image=np.full((128,256,3),90,np.uint8)
regions=(((0,0),(110,0),(110,127),(0,127)),((130,0),(255,0),(255,127),(130,127)))
pm=dict(points=np.empty((0,2),np.float32),pairs=np.empty((0,4)),groups=[],group_algorithms=[])
def settle(w):
 end=time.monotonic()+10
 while any(j.is_busy for j in ((w.prepare,w.draw,w.export_job,w.sift_job) if hasattr(w,'sift_job') else (w.prepare,w.draw,w.export_job))):
  app.processEvents();time.sleep(.005);assert time.monotonic()<end and not errors,errors
for cls in (AutomaticClonesWidget,CompleteAnalysisWidget):
 w=cls(image,autostart=False)
 if hasattr(w,'sift_job'):w.sift_job.compute=lambda _:dict(groups=())
 with patch.object(w,'start'):w.auto_ready(regions)
 with patch.object(w.job,'request') as p,patch.object(w.forge,'request') as f:
  if hasattr(w,'queue_ela'):
   with patch.object(w,'queue_ela'):w.start()
  else:w.start()
  args=f.call_args.args[0]
  assert args['variant']=='Forgeryscope Auto' and args['regions']==(w.envelope,)
  assert p.call_args.args[0][0][0]==EXTENDED_SYMMETRIC and p.call_args.args[0][0][11] is True
  assert w.minimum.value()==10 and w.overlap.value()==80
  w.results=dict(patchmatch=pm,forgeryscope=forge);w.states=dict.fromkeys(w.states,'Complete');w.prepare_biomes();settle(w)
  assert len(w.biomes)==3 and w.sources[0]=='Forgeryscope Auto'
  tokens=(w.job.token,w.prepare.token,w.forge.current,w.ela_job.token if hasattr(w,'ela_job') else None)
  w.forge_branch.setCurrentIndex(2);settle(w)
  assert len(w.viewer.view.biomes)==1 and w.viewer.view.biomes[0]['provenance']['branch']=='blots'
  w.legend.setCurrentIndex(w.model.index(0));hidden=w.model.rows[0]['id']
  w.model.setData(w.model.index(0),Qt.Unchecked,Qt.CheckStateRole);w.clear_focus();settle(w)
  assert not w.viewer.view.biomes
  w.forge_branch.setCurrentIndex(0);settle(w)
  assert len(w.viewer.view.biomes)==2 and hidden in w.model.hidden
  if hasattr(w,'source_checks'):
   w.source_checks[SOURCES[0]].setChecked(False);settle(w);assert not w.viewer.view.biomes
   w.source_checks[SOURCES[0]].setChecked(True);settle(w);assert len(w.viewer.view.biomes)==2
  assert tokens==(w.job.token,w.prepare.token,w.forge.current,w.ela_job.token if hasattr(w,'ela_job') else None)
  w.forge_branch.setCurrentIndex(3);settle(w)
  with tempfile.TemporaryDirectory() as folder:
   target=str(Path(folder)/'result.npz')
   with patch.object(QFileDialog,'getSaveFileName',return_value=(target,'')):w.export_results();settle(w)
   with np.load(target,allow_pickle=False) as data:
    meta=json.loads(str(data['metadata_json']))
    assert meta['configuration']['forgeryscope_parameters']['variant']=='Forgeryscope Auto'
    assert meta['display']['forgeryscope_branch']=='lanes' and hidden in meta['display']['hidden']
    assert any(e['provenance']['evidence']=='similarity' for e in meta['biomes'])
  # Disabling the enclosing zone skips Forge; disabled subimages still exclude pixels.
  w.zone_model.disabled={0};w.sync_zones()
  if hasattr(w,'queue_ela'):
   with patch.object(w,'queue_ela'):w.start()
  else:w.start()
  assert f.call_args.args[0]['excluded']==(regions[0],)
  previous=f.call_count;w.zone_model.disabled.add(2)
  if hasattr(w,'queue_ela'):
   with patch.object(w,'queue_ela'):w.start()
  else:w.start()
  assert f.call_count==previous and w.states['forgeryscope'].startswith('Disabled')
 w.close();app.processEvents()
_POOL.waitForDone(10000);assert not errors,errors
excluded=(((0,0),(80,0),(80,100),(0,100)),)
output=render((image,rows,excluded));assert np.array_equal(output[:101,:81],image[:101,:81])
report=dict(passed=True,panels=2,all_branches=True,evidence_distinct=True,lane_expansion=True,
 display_minimum10=True,overlap80=True,mirror_scale_profile_preserved=True,enclosing_only=True,
 exclusions=True,display_changes_no_inference=True,hidden_state_persists=True,export=True)
Path(__file__).with_name('automatic-results.json').write_text(json.dumps(report,indent=2));print(report)
