import sys,json,time,tempfile
from pathlib import Path
from unittest.mock import patch
import cv2 as cv,numpy as np
from PySide6.QtCore import Qt,QTimer
from PySide6.QtWidgets import QApplication,QFileDialog
from PySide6.QtTest import QTest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core.complete_analysis import ela_entries,render,ELA_SOURCE
from gui.sherloq_app.core.automatic_clones import visible
from gui.sherloq_app.tools.tampering.complete_analysis import CompleteAnalysisWidget
from gui.sherloq_app.tools.tampering.automatic_clones import AutomaticView
from gui.sherloq_app.ui.tools import ToolTree
from gui.sherloq_app.ui.extensions import EXTENSION_ROLE,BLUE
from gui.sherloq_app.ui.jobs import _POOL
from gui.sherloq_app.ui.localization import t
from gui.sherloq_app.core.jpeg import compress_jpg
app=QApplication([]);errors=[];
def exception(*a):
 errors.append(str(a[1]));sys.__excepthook__(*a)
sys.excepthook=exception
def wait(w):
 end=time.monotonic()+60
 while True:
  app.processEvents();time.sleep(.004)
  if not any(j.is_busy for j in (w.auto_job,w.job,w.sift_job,w.ela_job,w.prepare,w.draw,w.export_job)) and not w.restart.isActive():break
  assert time.monotonic()<end,(w.states,errors)
 assert not errors,errors
# Exact cells with a hole: no convex hull, no painting/picking outside evidence.
b=16;score=np.full((8,8),3.,np.float32);score[3:5,3:5]=0
base=dict(score=score,legacy_score=score,supported=np.ones((8,8),bool),signed_scores=np.zeros((8,8,3,5),np.float32),metadata=dict(block=b))
zone=(((0,0),(127,0),(127,127),(0,127)),)
e,r=ela_entries(base,2,3,zone,(),(128,128,3));im=np.full((128,128,3),127,np.uint8);out=render((im,e,()))
assert np.array_equal(out[48:80,48:80],im[48:80,48:80]);assert not AutomaticView.contains(e[0],60,60)
assert AutomaticView.contains(e[0],20,20)
excluded=(((0,0),(31,0),(31,127),(0,127)),)
e,r=ela_entries(base,2,3,zone,excluded,im.shape);out=render((im,e,excluded));assert np.array_equal(out[:,:32],im[:,:32])
assert not r['supported'][:,:2].any()
# Real ELA through the complete panel, cheap explicit stand-ins for clone workers.
rng=np.random.default_rng(411);yy,xx=np.indices((512,512));gray=100+20*np.sin(xx/80)+15*np.sin(yy/95)
im=np.clip(gray[:,:,None]+rng.normal(0,6,(512,512,3)),0,255).astype(np.uint8);im[160:352,160:352]=compress_jpg(im,45)[160:352,160:352]
w=CompleteAnalysisWidget(im,autostart=True);w.ela_auto_thresholds.setChecked(False);w.resize(1400,900);w.show()
whole=(((0.,0.),(511.,0.),(511.,511.),(0.,511.)),)
w.auto_job.compute=lambda _:whole
points=np.array([[10,10],[20,10],[20,20],[50,10],[60,10],[60,20]],np.float32)
pairs=np.array([[0,3,.1,40],[1,4,.1,40],[2,5,.1,40]])
pm=dict(points=points,pairs=pairs,groups=[np.arange(3),np.arange(3)],group_algorithms=['PatchMatch Zernike','PatchMatch SIFT'])
w.job.compute=lambda _:pm
w.sift_job.compute=lambda _:dict(points=np.empty((0,7)),pairs=np.empty((0,4)),groups=(),bases=())
polys=[[[60,60],[90,60],[90,90],[60,90]],[[120,120],[150,120],[150,150],[120,150]]]
forge=dict(metadata=dict(zones=[dict(comparisons=[dict(supported=True,polygon0=polys[0],polygon1=polys[1],inliers=12,score=.9)])]))
calls=[]
def request(params,device):
 calls.append((params,device));QTimer.singleShot(0,lambda:w.complete('forgeryscope',forge))
w.forge.request=request
wait(w)
assert w.ela_legacy_visible.isChecked()
w.ela_legacy_visible.setChecked(True);wait(w)
assert all(s=='Complete' for s in w.states.values()),w.states
assert set(w.results)=={'ela','patchmatch','forgeryscope','sift'} and len(w.sources)==5
assert any(e['source']==ELA_SOURCE for e in w.biomes)
assert w.ela_params()==(32,0,True,False,True,.01,.99,False,'sensitive') and w.ela_threshold.value()==2 and w.ela_minimum.value()==3
assert w.minimum.value()==10 and w.overlap.value()==80
assert calls[0][0]['regions']==whole and w.blue_outline.color==BLUE
tree=ToolTree();assert tree.tool_item(7, 8).data(0,EXTENSION_ROLE)=='blue'
assert t('Complete Automatic Analysis','fr')=='Analyse automatique complète'
assert w.results['ela']['metadata']['background']['enabled']
old=(w.job.token,len(calls));n=w.ela_engine.encodes
w.ela_background.setChecked(False);wait(w);assert w.results['ela']['metadata']['version']==3
w.ela_background.setChecked(True);wait(w);assert w.results['ela']['metadata']['version']==5
assert w.ela_engine.encodes==n and old==(w.job.token,len(calls))
w.ela_threshold.setValue(2.5);w.ela_minimum.setValue(4);wait(w)
assert w.ela_engine.encodes==n and old==(w.job.token,len(calls))
w.ela_block.setCurrentText('16');wait(w);assert w.ela_engine.encodes==n+3 and old==(w.job.token,len(calls))
w.ela_block.setCurrentText('32');w.ela_threshold.setValue(2);w.ela_minimum.setValue(3);wait(w)
w.tabs.setCurrentIndex(w.sources.index(ELA_SOURCE)+1);wait(w);assert all(e['source']==ELA_SOURCE for e in w.model.rows)
key=w.model.rows[0]['id'];w.legend.setCurrentIndex(w.model.index(0));w.model.setData(w.model.index(0),Qt.Unchecked,Qt.CheckStateRole);w.clear_focus();wait(w)
assert key in w.model.hidden and all(e['id']!=key for e in w.viewer.view.biomes)
for i in range(w.tabs.count()):w.tabs.setCurrentIndex(i)
w.tabs.setCurrentIndex(0);wait(w);assert key in w.model.hidden and old==(w.job.token,len(calls))
# Display-only source checkboxes and ELA views never rerun analysis/segmentation.
tokens=(w.job.token,w.ela_job.token,w.prepare.token,len(calls),w.ela_engine.encodes)
assert all(c.isChecked() for c in w.source_checks.values()) and w.overlay_check.isChecked()
source=w.sources[1]
QTest.mouseClick(w.source_checks[source],Qt.LeftButton);wait(w)
assert w.tabs.currentIndex()==0 and source not in w.enabled_sources()
assert all(e['source']!=source for e in w.model.rows+w.viewer.view.biomes)
assert w.overlay_check.checkState()==Qt.PartiallyChecked
w.tabs.setCurrentIndex(2);wait(w)
assert not w.model.rows and not w.viewer.view.biomes
w.tabs.setCurrentIndex(0)
w.source_checks[source].setChecked(True);wait(w)
assert key in w.model.hidden and all(e['id']!=key for e in w.viewer.view.biomes)
assert w.overlay_check.checkState()==Qt.Checked
w.overlay_check.click();wait(w)
assert not w.enabled_sources() and not w.viewer.view.biomes
assert np.array_equal(w.viewer.processed,w.image)
w.overlay_check.click();wait(w)
assert len(w.enabled_sources())==5
w.ela_mode.setCurrentIndex(1);wait(w)
expected=render((w.image,w.viewer.view.biomes,(),w.ela_base['ela'],1))
assert np.array_equal(w.viewer.processed,expected)
# A clone-only tab retains the original background despite the ELA preference.
w.tabs.setCurrentIndex(1);wait(w)
assert w.effective_ela_mode()==0
assert np.array_equal(w.viewer.processed,render((w.image,w.viewer.view.biomes,())))
w.tabs.setCurrentIndex(0)
w.ela_mode.setCurrentIndex(2);wait(w)
assert np.array_equal(w.viewer.processed,w.ela_base['ela'])
assert not w.viewer.view.biomes and not w.model.rows and not w.viewer.view.show_regions
w.source_checks[ELA_SOURCE].setChecked(False);wait(w)
assert w.effective_ela_mode()==0 and w.viewer.view.show_regions
assert all(e['source']!=ELA_SOURCE for e in w.viewer.view.biomes)
w.source_checks[ELA_SOURCE].setChecked(True);wait(w)
assert np.array_equal(w.viewer.processed,w.ela_base['ela'])
assert tokens==(w.job.token,w.ela_job.token,w.prepare.token,len(calls),w.ela_engine.encodes)
assert key in w.model.hidden
w.ela_mode.setCurrentIndex(0);wait(w)
assert w.viewer.view.show_regions and key in w.model.hidden
with tempfile.TemporaryDirectory() as tmp:
 out=Path(tmp)/'hybrid.npz'
 with patch.object(QFileDialog,'getSaveFileName',return_value=(str(out),'')):
  w.export_results();wait(w)
 with np.load(out,allow_pickle=False) as data:
  meta=json.loads(str(data['metadata_json']));assert meta['method']=='complete_automatic_analysis'
  assert meta['display']['background']=='original'
  assert np.array_equal(data['root_results_ela_labels'],w.results['ela']['labels'])
  assert np.array_equal(data['root_results_patchmatch_pairs'],pairs)
 w.ela_mode.setCurrentIndex(1);w.source_checks[source].setChecked(False);wait(w)
 with patch.object(QFileDialog,'getSaveFileName',return_value=(str(out),'')):
  w.export_results();wait(w)
 with np.load(out,allow_pickle=False) as data:
  meta=json.loads(str(data['metadata_json']))
  assert meta['display']['background']=='ela' and meta['display']['ela_view']=='ela_biomes'
  assert source not in meta['display']['enabled_sources'] and key in meta['display']['hidden']
w.ela_mode.setCurrentIndex(0);w.source_checks[source].setChecked(True);wait(w)
w.grab().save(str(ROOT/'tests/complete-analysis/gui.png'))
w.stop();wait(w);w.close();app.processEvents();assert w.ela_job.closed and w.ela_cancel.is_set()
_POOL.waitForDone(30000)
result=dict(passed=True,platform=app.platformName(),exact_cell_geometry=True,holes_and_exclusions=True,autostart=True,defaults_32_2_3=True,clones_stand_ins=True,ela_real=True,threshold_cache=True,cell_auto=True,clones_not_restarted_for_ela_settings=True,source_isolation=True,persistent_hidden=True,npz_roundtrip=True,blue_marker=True,shutdown=True)
result.update(tab_checkboxes=True,master_checkbox=True,ela_views=True,raw_ela_exact=True,
              no_recalculation_for_display=True,display_state_exported=True)
(ROOT/'tests/complete-analysis/contracts-results.json').write_text(json.dumps(result,indent=2));print(result)
