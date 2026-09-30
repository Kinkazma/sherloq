import os,sys,tempfile
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'source'))
from gui.sherloq_app.ui.viewer import ImageViewer
from gui.sherloq_app.ui.selection_view import SelectionView
app=QApplication([])
for view_class in (None,SelectionView):
 for h,w in ((1800,2400),(1800,800)):
  image=np.zeros((h,w,3),np.uint8)
  viewer=ImageViewer(image,image,**({'view_class':view_class} if view_class else {}));viewer.resize(800,650);viewer.show();app.processEvents()
  v=viewer.view;v.zoom_at(4.,QPointF(217,139));v.horizontalScrollBar().setValue(v.horizontalScrollBar().value()+27);v.verticalScrollBar().setValue(v.verticalScrollBar().value()+36)
  app.processEvents()
  def snapshot():return v.transform(),v.sceneRect(),v.horizontalScrollBar().value(),v.verticalScrollBar().value(),v.mapToScene(217,139)
  before=snapshot()
  assert v.sceneRect()!=v.scene.sceneRect()
  for n in range(5):
   viewer.update_processed(np.full_like(image,n));app.processEvents();assert snapshot()==before
   viewer.original_radio.setChecked(True);app.processEvents();assert snapshot()==before
   viewer.process_radio.setChecked(True);app.processEvents();assert snapshot()==before
  with tempfile.TemporaryDirectory() as folder:
   mapped=np.memmap(Path(folder)/'image.bin',mode='w+',shape=image.shape,dtype=np.uint8)
   viewer.update_processed(mapped);app.processEvents();assert snapshot()==before
   viewer.update_processed(image);app.processEvents();assert snapshot()==before
   del mapped
  viewer.update_processed(np.zeros((300,400,3),np.uint8));app.processEvents()
  assert v.scene.sceneRect().width()==400 and v.transform().m11()==v.fit_scale
  viewer.close();app.processEvents()
print('PASS: redraws, original/result toggles and tiled transitions preserve zoom, pan and cursor scene coordinates; dimension changes still fit.')
