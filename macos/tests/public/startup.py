"""Small offscreen Qt launch using isolated settings and a generated image."""
import sys,os,tempfile,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
with tempfile.TemporaryDirectory(prefix='sherloq-settings-') as tmp:
 os.environ['XDG_CONFIG_HOME']=tmp;os.environ['MPLCONFIGDIR']=tmp
 from PySide6.QtCore import QSettings
 from PySide6.QtWidgets import QApplication
 QSettings.setDefaultFormat(QSettings.IniFormat);QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,tmp)
 app=QApplication([])
 from gui.sherloq_app.main import MainWindow
 import numpy as np,cv2
 window=MainWindow();window.show();app.processEvents()
 image=np.random.default_rng(56).integers(0,256,(97,101,3),dtype=np.uint8)
 path=Path(tmp)/'synthetic.png';cv2.imwrite(str(path),image)
 window.initialize(str(path),path.name,image);app.processEvents()
 assert window.image is not None
 window.close();app.processEvents()
 print(json.dumps({'passed':True,'main_window':True,'synthetic_image_initialized':True,'rendering':'offscreen','all_panels_tested':False}))
