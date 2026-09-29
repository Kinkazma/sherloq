"""Real Qt input and delayed-worker regressions for both ELA controllers.

Run with the project's Python from the workspace root. No clone models run.
"""
import sys,time,json,tempfile
from pathlib import Path
from threading import Event
import cv2 as cv
from PySide6.QtWidgets import QApplication,QStyle,QStyleOptionSlider
from PySide6.QtCore import Qt,QPoint,QPointF,QSettings
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.ui.ela_biomes import ElaBiomesPanel
from gui.sherloq_app.tools.tampering.complete_analysis import CompleteAnalysisWidget
from gui.sherloq_app.ui.ela_profiles import ProfileStore
from gui.sherloq_app.ui.localization import install
from gui.sherloq_app.ui.jobs import _POOL
app=QApplication([]);errors=[]
sys.excepthook=lambda *a:(errors.append(str(a[1])),sys.__excepthook__(*a))
def until(predicate):
 end=time.monotonic()+60
 while not predicate():
  app.processEvents();time.sleep(.005);assert time.monotonic()<end,'Timed out'
 assert not errors,errors

def wait(jobs):
 QTest.qWait(30);until(lambda:not any(j.is_busy or j.active is not None for j in jobs));QTest.qWait(30)

def values(profiles):return [s.value() for s in profiles.sliders]
def geometry(sliders):return [(s.mapTo(s.window(),QPoint()).toTuple(),s.size().toTuple()) for s in sliders]
def handle(slider):
 option=QStyleOptionSlider();slider.initStyleOption(option)
 return slider.style().subControlRect(QStyle.CC_Slider,option,QStyle.SC_SliderHandle,slider).center()

def wheel(slider,delta):
 local=QPointF(handle(slider));global_pos=QPointF(slider.mapToGlobal(local.toPoint()))
 event=QWheelEvent(local,global_pos,QPoint(),QPoint(0,delta),Qt.NoButton,Qt.NoModifier,Qt.NoScrollPhase,False)
 QApplication.sendEvent(slider,event)

rows=[]
with tempfile.TemporaryDirectory() as tmp:
 app._ela_profile_store=ProfileStore(QSettings(str(Path(tmp)/'settings.ini'),QSettings.IniFormat),app)
 lang=install();lang.set_mode('fr',persist=False)
 im=cv.imread(str(ROOT/'tests/sample.jpg'))
 p=ElaBiomesPanel(im);p.ghost.setChecked(False);p.resize(1400,850);p.show();wait((p.job,p.draw))
 w=CompleteAnalysisWidget(im,autostart=False);w.ela_ghost.setChecked(False);w.resize(1400,850);w.show();w.tabs.setCurrentIndex(4)
 zones=(((0,0),(im.shape[1]-1,0),(im.shape[1]-1,im.shape[0]-1),(0,im.shape[0]-1)),)
 w.viewer.view.set_regions(zones);w.zone_model.update(zones,None);w.sync_zones();w.queue_ela();wait((w.ela_job,w.prepare,w.draw))
 for panel,job,jobs in ((p,p.job,(p.job,p.draw)),(w,w.ela_job,(w.ela_job,w.prepare,w.draw))):
  profiles=panel.profiles
  saved=profiles.store.add('Input regression',[2,997,41,42])
  for profile in ('standard','conservative','sensitive',saved):
   for index,slider in enumerate(profiles.sliders):
    profiles.combo.setCurrentIndex(profiles.combo.findData(profile));wait(jobs)
    before=values(profiles);bounds=geometry(profiles.sliders);start=handle(slider)
    direction=-30 if slider.value()>(slider.minimum()+slider.maximum())/2 else 30
    QTest.mousePress(slider,Qt.LeftButton,pos=start)
    QTest.mouseMove(slider,start+QPoint(direction,0),20)
    QTest.mouseRelease(slider,Qt.LeftButton,pos=start+QPoint(direction,0));wait(jobs)
    after=values(profiles)
    assert after[index]!=before[index],(profile,index,before,after)
    assert all(after[i]==before[i] for i in range(4) if i!=index),(profile,index,before,after)
    assert geometry(profiles.sliders)==bounds
    assert profiles.combo.currentData()=='manual' and not profiles.automatic.isChecked()
    rows.append(dict(panel=type(panel).__name__,action='drag',profile=profile if not profile.startswith('user:') else 'saved',index=index,before=before,after=after))
  profiles.store.remove(saved)
  # Block a real automatic computation AFTER it computed its answer. Release
  # that obsolete answer only after input that leaves the selected value intact.
  compute=job.compute
  for action in ('key_at_limit','wheel_at_limit','groove_at_limit','handle_press'):
   for index,slider in enumerate(profiles.sliders):
    profiles.combo.setCurrentIndex(profiles.combo.findData('standard'));wait(jobs)
    upper=index==1;slider.setValue(slider.maximum() if upper else slider.minimum());wait(jobs)
    ready=Event();release=Event()
    def delayed(request):
     result=compute(request)
     if request[0][7]:
      ready.set();assert release.wait(30),'Gate not released'
     return result
    job.compute=delayed
    profiles.combo.setCurrentIndex(profiles.combo.findData('sensitive'));until(ready.is_set)
    before=values(profiles);token=job.token
    try:
     if action=='key_at_limit':QTest.keyClick(slider,Qt.Key_End if upper else Qt.Key_Home)
     elif action=='wheel_at_limit':wheel(slider,120 if upper else -120)
     elif action=='groove_at_limit':
      pos=QPoint(slider.width()-1 if upper else 0,slider.height()//2)
      QTest.mousePress(slider,Qt.LeftButton,pos=pos);QTest.mouseRelease(slider,Qt.LeftButton,pos=pos)
     else:
      pos=handle(slider);QTest.mousePress(slider,Qt.LeftButton,pos=pos);QTest.mouseRelease(slider,Qt.LeftButton,pos=pos)
     assert values(profiles)==before,(action,index,before,values(profiles))
     assert profiles.combo.currentData()=='manual' and not profiles.automatic.isChecked(),(action,index)
     assert job.token>token,(action,index,'automatic result still valid')
    finally:release.set()
    wait(jobs);assert values(profiles)==before,(action,index,'late auto moved controls',before,values(profiles))
    job.compute=compute
    rows.append(dict(panel=type(panel).__name__,action=action,index=index,before=before,after=values(profiles)))
  # Explicit automatic selection remains available after manual interaction.
  profiles.combo.setCurrentIndex(profiles.combo.findData('conservative'));wait(jobs)
  assert profiles.automatic.isChecked() and profiles.combo.currentData()=='conservative'
  # Right-click and merely focusing a slider do not select manual mode.
  slider=profiles.sliders[0];slider.setFocus();QTest.mouseClick(slider,Qt.RightButton,pos=handle(slider));wait(jobs)
  assert profiles.automatic.isChecked()
 p.grab().save(str(Path(__file__).with_name('native.png')))
 lang.shutdown();p.close();w.close();app.processEvents();_POOL.waitForDone(30000)
result=dict(passed=True,platform=app.platformName(),cases=len(rows),real_qt_drag=True,delayed_automatic_result_rejected=True,geometry_stable=True,formulas_unchanged=True,rows=rows)
Path(__file__).with_name('results.json').write_text(json.dumps(result,indent=2)+'\n')
print({k:v for k,v in result.items() if k!='rows'},flush=True)
