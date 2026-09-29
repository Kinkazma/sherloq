"""Render actual installed SHERLOQ tools on controlled, non-generative examples."""
import os,sys,time,json,hashlib,importlib,traceback
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
OUT=Path(__file__).resolve().parent
ROOT=Path(os.environ['SHERLOQ_NATIVE_ROOT']).expanduser().resolve()
sys.path.insert(0,str(ROOT/'source'))
import cv2 as cv
import numpy as np
from PySide6.QtWidgets import QApplication,QWidget,QVBoxLayout,QLabel,QComboBox,QSpinBox,QDoubleSpinBox,QCheckBox
from PySide6.QtGui import QFont
from gui.sherloq_app.ui.jobs import LatestJob
from gui.sherloq_app.ui.viewer import ImageViewer
SPECS={
 'cloning2':('tampering.cloning2','Cloning2Widget','Copy-Move Forgery 2','coffee-copy.png'),
 'automatic_clones':('tampering.automatic_clones','AutomaticClonesWidget','Automatic Clone Search','coffee-copy.png'),
 'ela':('jpeg.ela','ElaWidget','Error Level Analysis','coffee-copy.jpg'),
 'multiple':('jpeg.multiple','MultipleWidget','Multiple Compression','coffee-double.jpg'),
 'illuminant':('various.illuminant','IlluminantWidget','Illuminant Map','coffee-original.png'),
 'defect_pixels':('various.defect_pixels','DefectWidget','Dead / Hot Pixels','coffee-pixels.png'),
 'noisesniffer':('noise.noisesniffer','NoisesnifferWidget','Noisesniffer','coffee-copy.png'),
 'zero':('jpeg.zero','ZeroWidget','ZERO JPEG Grids','coffee-copy.png'),
 'adaptive_cfa':('tampering.adaptive_cfa','AdaptiveCFAWidget','Adaptive CFA','coffee-copy.png'),
 'trufor':('various.trufor','TruForWidget','TruFor','coffee-copy.jpg'),
 'catnet':('various.catnet','CatNetWidget','CAT-Net v2','coffee-copy.jpg'),
 'safire':('various.safire','SafireWidget','SAFIRE','coffee-copy.png'),
 'focal':('various.focal','FocalWidget','FOCAL','coffee-copy.png'),
 'adaifl':('various.adaifl','AdaIFLWidget','AdaIFL','coffee-copy.png'),
}
kind=sys.argv[1];module,cls,title,filename=SPECS[kind]
TAG=("-"+sys.argv[3]) if len(sys.argv)>3 else ""
if len(sys.argv)>2:filename=sys.argv[2]
app=QApplication([]);app.setOrganizationName('SHERLOQExampleRendering');app.setApplicationName('ExtendedDocumentation');app.setStyle('Fusion');app.setPalette(app.style().standardPalette());app.setFont(QFont('Arial',11))
errors=[];sys.excepthook=lambda typ,exc,tb:(errors.append(str(exc)),traceback.print_exception(typ,exc,tb))
module=importlib.import_module('gui.sherloq_app.tools.'+module);Class=getattr(module,cls)
path=OUT/filename;image=cv.imread(str(path));assert image is not None
if kind in ('trufor','catnet'):tool=Class(str(path),image)
elif kind in ('ela','multiple'):tool=Class(image,filename=str(path))
else:tool=Class(image)
frame=QWidget();layout=QVBoxLayout(frame);header=QLabel(title+' · SHERLOQ');header.setFont(QFont('Arial',18,QFont.Weight.Bold));layout.addWidget(header);layout.addWidget(tool,1)
caption=QLabel(('Source: my edited texture · ' if TAG else 'Source: Rachel Michetti · CC0 · ')+filename+' · actual SHERLOQ output');layout.addWidget(caption)
frame.resize(1600,1100 if kind=='automatic_clones' else 1050);frame.show()
if kind=='zero':tool.layout().setStretch(2,1)
for obj in tool.findChildren(__import__('PySide6.QtCore',fromlist=['QObject']).QObject):
 if hasattr(obj,'failed'):
  try:obj.failed.connect(errors.append)
  except (AttributeError,TypeError):pass

def wait():
 start=time.monotonic();idle=None;last=0
 while time.monotonic()-start<900:
  app.processEvents()
  if errors:raise RuntimeError(errors)
  busy=any(bool(getattr(o,'is_busy',False)) for o in tool.findChildren(__import__('PySide6.QtCore',fromlist=['QObject']).QObject) if not callable(getattr(o,'is_busy',None)))
  busy=busy or any(j.timer.isActive() for j in tool.findChildren(LatestJob))
  if kind=='automatic_clones':busy=busy or any(v in ('Waiting','Running') for v in tool.states.values())
  if busy:idle=None
  elif idle is None:idle=time.monotonic()
  elif time.monotonic()-idle>.8:return
  if time.monotonic()-start-last>20:
   labels=[x.text()[:200] for x in tool.findChildren(QLabel) if any(k in x.text().lower() for k in ['analys','loading','chargement','progress','%','complete'])]
   print(json.dumps({'tool':kind,'elapsed':round(time.monotonic()-start),'progress':labels[:4]}),flush=True);last=time.monotonic()-start
  time.sleep(.01)
 raise TimeoutError(kind)

def capture(suffix=''):
 wait()
 for viewer in tool.findChildren(ImageViewer):
  viewer.view.zoom_fit()
  h,w=viewer.original.shape[:2]
  scale=min(viewer.view.viewport().width()/w,viewer.view.viewport().height()/h)*.97
  if scale>1:
   viewer.view.set_scaling(min(scale,3.0));viewer.view.notify_change()
 app.processEvents()
 for n,viewer in enumerate(tool.findChildren(ImageViewer)):
  if viewer.processed is not None:cv.imwrite(str(OUT/(kind+TAG+suffix+'-view'+str(n)+'.png')),viewer.processed)
 target=OUT/(kind+TAG+suffix+'.png');assert frame.grab().save(str(target))
 controls=[]
 for c in tool.findChildren(QComboBox):controls.append({'type':'choice','value':c.currentText()})
 for c in tool.findChildren(QSpinBox)+tool.findChildren(QDoubleSpinBox):controls.append({'type':'number','value':c.value()})
 for c in tool.findChildren(QCheckBox):controls.append({'type':'check','label':c.text(),'value':c.isChecked()})
 labels=[c.text() for c in tool.findChildren(QLabel) if c.text()]
 record={'tool':kind,'title':title,'input':filename,'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'controls':controls,'labels':labels,'screenshot':target.name}
 result=getattr(tool,'result',None)
 if isinstance(result,dict):
  record['result_metadata']=result.get('metadata',{})
  arrays={k:v for k,v in result.items() if isinstance(v,np.ndarray)}
  if arrays:np.savez_compressed(OUT/(kind+TAG+'-arrays.npz'),**arrays)
 if kind=='automatic_clones':record['states']=tool.states;record['errors']=tool.errors;record['biomes']=len(tool.biomes)
 (OUT/(kind+TAG+suffix+'.json')).write_text(json.dumps(record,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')
 print('CAPTURED '+target.name,flush=True)

try:
 if kind=='cloning2':tool.start(False)
 elif kind in ('noisesniffer','zero','adaptive_cfa','catnet','safire','focal','adaifl'):tool.start()
 elif kind=='trufor':tool.trufor_process()
 elif kind=='illuminant':tool.block_combo.setCurrentText('32')
 elif kind=='defect_pixels':tool.mode_combo.setCurrentIndex(0)
 capture()
 if kind=='ela':
  tool.tabs.setCurrentIndex(1);tool.biomes.mode.setCurrentIndex(1);capture('-layers')
 if kind=='multiple':
  tool.tabs.setCurrentIndex(1);tool.detector.start();capture('-double-jpeg')
 if kind=='zero':tool.mode.setCurrentIndex(1);capture('-grids')
 if kind=='defect_pixels':tool.mode_combo.setCurrentIndex(1);capture('-mask')
 if kind=='trufor':tool.map_combo.setCurrentIndex(1);capture('-confidence')
except Exception:
 (OUT/(kind+TAG+'-ERROR.txt')).write_text(traceback.format_exc());traceback.print_exc();sys.exit(1)
finally:
 tool.shutdown();frame.close();app.processEvents()
 from gui.sherloq_app.ui.research_service import SERVICES
 for service in SERVICES.values():service.shutdown()
