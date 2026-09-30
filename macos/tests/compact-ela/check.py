"""Compact native layouts, source-specific controls and profile semantics."""
import sys,time,json,tempfile
from pathlib import Path
import cv2 as cv
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.ui.ela_biomes import ElaBiomesPanel
from gui.sherloq_app.tools.tampering.complete_analysis import CompleteAnalysisWidget
from gui.sherloq_app.ui.ela_profiles import ProfileStore
from gui.sherloq_app.ui.localization import LanguageManager
from gui.sherloq_app.ui.jobs import _POOL
app=QApplication([]);errors=[];sys.excepthook=lambda *a:(errors.append(str(a[1])),sys.__excepthook__(*a))
def events():
 for _ in range(5):app.processEvents();time.sleep(.005)
def wait(*jobs):
 end=time.monotonic()+90
 while True:
  events()
  if not any(j.is_busy for j in jobs):break
  assert time.monotonic()<end
 assert not errors,errors
with tempfile.TemporaryDirectory() as tmp:
 app._ela_profile_store=ProfileStore(QSettings(str(Path(tmp)/'settings.ini'),QSettings.IniFormat),app)
 manager=LanguageManager(app);manager.set_mode('fr',persist=False)
 im=cv.imread(str(ROOT/'tests/sample.jpg'));p=ElaBiomesPanel(im);p.resize(1200,800);p.show();wait(p.job,p.draw)
 w=CompleteAnalysisWidget(im,autostart=False);w.resize(1200,800);w.show();events();manager.refresh()
 zone=(((0,0),(im.shape[1]-1,0),(im.shape[1]-1,im.shape[0]-1),(0,im.shape[0]-1)),)
 w.viewer.view.set_regions(zone);w.zone_model.update(zone,None);w.sync_zones();w.ela_engine=p.engine;w.queue_ela();wait(w.ela_job,w.prepare,w.draw)
 for panel in (p,w):
  profiles=panel.profiles
  assert profiles.combo.currentData()=='standard'
  assert profiles.combo.currentText()=='Conservateur'
  assert [s.value() for s in profiles.sliders]==[10,990,50,50]
  assert [profiles.combo.itemText(i) for i in range(3)]==['Agressif','Sensible','Conservateur']
  before=[s.value() for s in profiles.sliders]
  profiles.sliders[2].setValue(51)
  assert profiles.combo.currentData()=='manual' and not profiles.automatic.isChecked()
  assert [s.value() for s in profiles.sliders]==[before[0],before[1],51,before[3]]
  profiles.combo.setCurrentIndex(profiles.combo.findData('standard'))
 wait(p.job,p.draw,w.ela_job,w.prepare,w.draw)
 tokens=(w.job.token,w.ela_job.token,w.prepare.token,p.job.token,p.draw.token)
 sizes=[]
 for width in (1000,1200,1600):
  p.resize(width,800);w.resize(width,800);events()
  ela_index=w.sources.index('ELA biomes')+1
  for index in (0,ela_index,1,ela_index,0):
   w.tabs.setCurrentIndex(index);events()
   assert w.ela_settings.isVisible()==(index==ela_index)
   assert w.clone_intervals.isVisible()==(index!=ela_index)
   assert w.width()==width and p.width()==width
  for settings in (p.settings,w.ela_settings):
   for index in (1,0):settings.pages.setCurrentIndex(index);events()
   settings.toggle.setChecked(False);events();assert settings.pages.isHidden()
   settings.toggle.setChecked(True);events()
  sizes.append(dict(width=width,standalone_controls_height=p.settings.height(),standalone_image_height=p.viewer.height(),overlay_image_height=w.viewer.height()))
 assert tokens==(w.job.token,w.ela_job.token,w.prepare.token,p.job.token,p.draw.token),tokens
 p.resize(1200,800);w.resize(1200,800);events()
 p.grab().save(str(Path(__file__).with_name('ela.png')));w.grab().save(str(Path(__file__).with_name('overlay.png')))
 w.tabs.setCurrentIndex(w.sources.index('ELA biomes')+1);events();w.grab().save(str(Path(__file__).with_name('complete-ela.png')))
 # Profile names and menu actions follow live language changes without losing data.
 manager.set_mode('en',persist=False);manager.refresh();events()
 assert p.profiles.combo.currentText()=='Conservative'
 assert [p.profiles.combo.itemText(i) for i in range(3)]==['Aggressive','Sensitive','Conservative']
 manager.set_mode('fr',persist=False);manager.refresh();events()
 assert p.profiles.combo.currentText()=='Conservateur'
 assert p.profiles.save.text()=='Enregistrer un profil…'
 manager.shutdown();p.close();w.close();events();_POOL.waitForDone(30000)
r=dict(passed=True,platform=app.platformName(),conservative_fixed_default_both=True,renames_without_formula_changes=True,manual_after_edit=True,other_sliders_unchanged=True,source_specific_controls=True,layout_no_analysis=True,live_translation=True,sizes=sizes)
Path(__file__).with_name('results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
