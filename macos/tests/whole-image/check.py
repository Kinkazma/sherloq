"""Cocoa lifecycle regression; real jobs/ELA, explicit lightweight clone fixtures."""
import sys,time,json,hashlib,logging
from pathlib import Path
from threading import Event
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.tools.tampering.automatic_clones import AutomaticClonesWidget
from gui.sherloq_app.tools.tampering.complete_analysis import CompleteAnalysisWidget
from gui.sherloq_app.ui.jobs import _POOL,STAGING_POOL
from gui.sherloq_app.ui.localization import t
app=QApplication([]);errors=[];records=[]
sys.excepthook=lambda *a:(errors.append(str(a[1])),sys.__excepthook__(*a))
im=np.random.default_rng(24).integers(0,256,(160,240,3),dtype=np.uint8)
whole=(((0.,0.),(239.,0.),(239.,159.),(0.,159.)),)
regions=(((10.,10.),(95.,10.),(95.,145.),(10.,145.)),((125.,10.),(225.,10.),(225.,145.),(125.,145.)))
pm=dict(points=np.empty((0,2),np.float32),pairs=np.empty((0,4)),groups=[],group_algorithms=[])
forge=dict(metadata=dict(zones=[]))
def pump_until(predicate,limit=25):
 end=time.monotonic()+limit
 while not predicate():
  app.processEvents();time.sleep(.004)
  assert time.monotonic()<end,'timeout'
 assert not errors,errors
 for _ in range(3):app.processEvents()
def jobs(w):
 return [w.auto_job,w.job,w.draw,w.prepare,w.forge.preparer,w.forge.loader]+([w.ela_job] if hasattr(w,'ela_job') else [])
def idle(w):
 pump_until(lambda:all(not j.is_busy and j.active is None for j in jobs(w)) and not w.restart.isActive())
def make(cls,auto=True):
 w=cls(im,autostart=auto);w.resize(1300,800);w.calls={'detect':0,'pm':[],'forge':[]}
 def detect(_):w.calls['detect']+=1;return regions
 def patch(request):w.calls['pm'].append(request);return pm
 w.auto_job.compute=detect;w.job.compute=patch
 original=w.forge.request
 def cached(params,device):
  w.calls['forge'].append((params,device))
  key=hashlib.sha256(json.dumps([params,device],sort_keys=True).encode()).hexdigest()
  w.forge.cache.put(key,forge);original(params,device)
 w.forge.request=cached;return w

def assert_whole(w):
 assert w.viewer.view.snapshot()==whole,w.viewer.view.snapshot()
 assert w.envelope==whole[0] and not w.zone_model.disabled
 assert w.viewer.view.enabled_regions=={0}
 assert w.submitted['zones']==whole and w.submitted['disabled_zones']==[]
 params,used,compare,cancel=w.calls['pm'][-1]
 assert used==whole and params[-2:]==((),()) and not compare
 assert w.calls['forge'][-1][0]['regions']==whole and w.calls['forge'][-1][0]['excluded']==()
 assert set(w.results)==set(w.initial_states()) and all(s=='Complete' for s in w.states.values()),w.states

for cls in (AutomaticClonesWidget,CompleteAnalysisWidget):
 w=make(cls);w.show();idle(w)
 assert w.calls['detect']==1 and len(w.viewer.view.snapshot())==3
 # Exclusions include the old envelope; whole-image must undo all of them.
 w.zone_model.disabled={0,2};w.sync_zones();w.viewer.view.focus_regions={1}
 before={k:len(w.calls[k]) for k in ('pm','forge')}
 QTest.mouseClick(w.run_whole,Qt.LeftButton);idle(w);assert_whole(w)
 assert all(len(w.calls[k])==before[k]+1 for k in before)
 w.detect.click();idle(w);assert w.calls['detect']==2 and len(w.viewer.view.snapshot())==3
 w.auto_job.compute=lambda _:();w.detect.click();idle(w);assert_whole(w)
 # Controlled detector error: button remains usable, no silent simulation.
 def fail(_):raise RuntimeError('intentional detector failure')
 w.auto_job.compute=fail;w.detect.click();idle(w)
 assert w.run_whole.isEnabled() and 'intentional' in w.detail.text()
 w.run_whole.click();idle(w);assert_whole(w)
 w.close();app.processEvents()
 # Explicit click before the singleShot startup is consumed wins over it.
 w=make(cls);w.run_whole.click();idle(w);assert_whole(w);assert w.calls['detect']==0
 w.close();app.processEvents()
 records.append(dict(panel=cls.__name__,default_detection=True,whole_image_clears_exclusions=True,all_branches_restarted=True,redetect=True,empty_detection=True,detection_error_recovery=True,early_click=True))

# Force genuine asynchronous old worker completions AFTER the override. They must
# not reach consumer slots. Clone math is fixture-only; job token lifecycle is real.
w=make(CompleteAnalysisWidget,False);w.run_whole.click();idle(w)
for name in ('auto_job','job','ela_job','prepare','draw','forge.loader'):
 job=w.forge.loader if name=='forge.loader' else getattr(w,name)
 original=job.compute;entered=Event();release=Event();sentinel=object();received=[]
 def blocked(_):entered.set();assert release.wait(10);return sentinel
 receiver=received.append
 job.compute=blocked;job.result.connect(receiver)
 job.request(('obsolete',));pump_until(entered.is_set)
 old_token=job.token
 old_cancel=w.auto_cancel;old_analysis=w.cancel;old_ela=w.ela_cancel
 w.run_whole.click()
 assert old_cancel.is_set() and old_analysis.is_set() and old_ela.is_set()
 assert job.token>old_token
 # The active _Work already captured blocked; pending fresh work uses original.
 job.compute=original;release.set();idle(w)
 assert all(x is not sentinel for x in received),name
 assert_whole(w);job.result.disconnect(receiver)
 records.append(dict(stale_branch=name,late_result_rejected=True))
# Old detection failure is also discarded after whole-image action.
entered=Event();release=Event()
def old_error(_):entered.set();assert release.wait(10);raise RuntimeError('obsolete detection failure')
w.auto_job.compute=old_error;w.detect.click();pump_until(entered.is_set)
w.run_whole.click();release.set();idle(w);assert_whole(w)
assert 'obsolete' not in w.detail.text()
w.auto_job.compute=lambda _:regions;w.detect.click();idle(w)
assert len(w.viewer.view.snapshot())==3
assert t('Run on whole image','fr')=='Exécuter sur toute l’image'
w.grab().save(str(Path(__file__).with_name('native.png')));w.close();app.processEvents()
_POOL.waitForDone(30000);STAGING_POOL.waitForDone(30000)
result=dict(passed=True,platform=app.platformName(),real_ela=True,clone_math='explicit empty fixtures through real jobs/cache; no model accuracy claim',stale_detection_error_rejected=True,records=records)
Path(__file__).with_name('results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
