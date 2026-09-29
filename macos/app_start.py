"""Local macOS entry point: Finder document events, logging, diagnostics."""
import os,sys,time,traceback,logging
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.chdir(ROOT/'source')
sys.path.insert(0,str(ROOT/'source'))
for key in ('QT_PLUGIN_PATH','QT_QPA_PLATFORM_PLUGIN_PATH','PYTHONPATH','PYTHONHOME'):
    os.environ.pop(key,None)
os.environ['PATH']=str(ROOT/'venv/bin')+':/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin'
os.environ['DYLD_FALLBACK_LIBRARY_PATH']=str(ROOT/'native/runtime')
os.environ['MAGIC']=str(ROOT/'native/runtime/magic.mgc')
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['OMP_NUM_THREADS']='8'
os.environ['OPENBLAS_NUM_THREADS']='8'
(ROOT/'logs').mkdir(exist_ok=True)
from logging.handlers import RotatingFileHandler
handler=RotatingFileHandler(ROOT/'logs/application.log',maxBytes=2_000_000,backupCount=3)
logging.basicConfig(handlers=[handler],level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
logging.info('Starting SHERLOQ; Python %s',sys.version.split()[0])
from PySide6.QtCore import QEvent,QTimer,QSettings,Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication,QMessageBox
verify=any(arg.startswith('--verify') for arg in sys.argv)
if verify and '--verify-localization' not in sys.argv:
    os.environ['SHERLOQ_LANGUAGE']='en'
preview_3d='--preview-3d' in sys.argv
if verify or preview_3d:
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(ROOT/'tests/bundle-settings'))
class MacApplication(QApplication):
    window=None
    pending=[]
    def event(self,event):
        if event.type()==QEvent.FileOpen:
            filename=event.file()
            if self.window: self.open_file(filename)
            else: self.pending.append(filename)
            return True
        return super().event(event)
    def open_file(self,filename):
        if verify:
            # Existing per-tool diagnostics use a synchronous fixture setup.
            # The T02 diagnostic explicitly tests window.open_image instead.
            from gui.sherloq_app.core.utility import load_image
            result=load_image(self.window,filename)
            if result[0] is not None:self.window.initialize(*result)
        else:
            self.window.open_image(filename)
        if self.window.isMinimized():
            self.window.setWindowState(self.window.windowState() & ~Qt.WindowMinimized)
        self.window.show();self.window.raise_();self.window.activateWindow()
QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
app=MacApplication([sys.argv[0]])
app.setFont(QFont('Helvetica Neue',12))
def report_error(kind,value,tb):
    logging.error('Application error',exc_info=(kind,value,tb))
    if verify:
        (ROOT/'tests/bundle-error.txt').write_text(''.join(traceback.format_exception(kind,value,tb)))
        app.exit(1)
    else:
        QMessageBox.critical(app.window,'SHERLOQ — Erreur',f"L’opération n’a pas abouti :\n{value}\n\nDétails : {ROOT / 'logs/application.log'}")
sys.excepthook=report_error
from gui.sherloq_app.main import MainWindow
app.window=MainWindow()
for filename in app.pending+[arg for arg in sys.argv[1:] if not arg.startswith('-') and Path(arg).is_file()]:
    app.open_file(filename)
def verify_bundle():
    if '--verify-compact-ela' in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location('compact_ela_bundle',ROOT/'tests/compact-ela/bundle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-complete-analysis' in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location('complete_analysis_bundle',ROOT/'tests/complete-analysis/bundle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-extended-mirror' in sys.argv or '--verify-extended-mirror-auto' in sys.argv:
        import importlib.util
        filename='automatic_bundle.py' if '--verify-extended-mirror-auto' in sys.argv else 'bundle.py'
        spec=importlib.util.spec_from_file_location('extended_mirror_bundle',ROOT/'tests/extended-mirror'/filename)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-mirror-recovery' in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location('mirror_recovery_bundle',ROOT/'tests/mirror-recovery/bundle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-sift-extension' in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location('sift_extension_bundle',ROOT/'tests/sift-extension/bundle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-research-round2' in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location('research_round2_bundle',ROOT/'tests/research-round2/bundle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-localization' in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location('localization_bundle',ROOT/'tests/localization/bundle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-automatic-clones' in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location('automatic_clones_bundle',ROOT/'tests/automatic-clones/bundle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if "--verify-clone-detectors" in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location("clone_detectors_bundle",ROOT/"tests/clone-detectors/bundle.py")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-cm2-metal' in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location('cm2_metal',ROOT/'tests/cm2-compute/bundle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-cm2-zones-icon' in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location('cm2_zones_icon',ROOT/'tests/cm2-zones/icon.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-x11' in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location('x11_bundle',ROOT/'tests/x11/bundle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-x12' in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location('x12_bundle',ROOT/'tests/x12/bundle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-x10' in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location('x10_bundle',ROOT/'tests/x10/bundle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-x09' in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location('x09_bundle',ROOT/'tests/x09/bundle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-x08' in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location('x08_bundle',ROOT/'tests/x08/bundle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-x07' in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location('x07_bundle',ROOT/'tests/x07/bundle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-x06' in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location('x06_bundle',ROOT/'tests/x06/bundle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-x05' in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location('x05_bundle',ROOT/'tests/x05/bundle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-x04' in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location('x04_bundle',ROOT/'tests/x04/bundle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-x03' in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location('x03_bundle',ROOT/'tests/x03/bundle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-x02' in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location('x02_bundle',ROOT/'tests/x02/bundle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-cm2-plots' in sys.argv:
        sys.path.insert(0,str(ROOT/'tests/cm2'))
        from bundle_plots import run
        run(app)
        return
    if '--verify-cm2' in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location('cm2_bundle',ROOT/'tests/cm2/bundle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-f28-detector' in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location('f28_detector_bundle',ROOT/'tests/f28-detector/bundle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-t02' in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location('t02_bundle',ROOT/'tests/t02/bundle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-f38' in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location('f38_bundle',ROOT/'tests/f38/bundle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-f37' in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location('f37_bundle',ROOT/'tests/f37/bundle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-f36' in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location('f36_bundle',ROOT/'tests/f36/bundle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-f35' in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location('f35_bundle',ROOT/'tests/f35/bundle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-f34' in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location('f34_bundle',ROOT/'tests/f34/bundle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-f33' in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location('f33_bundle',ROOT/'tests/f33/bundle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-f32' in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location('f32_bundle',ROOT/'tests/f32/bundle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-f31' in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location('f31_bundle',ROOT/'tests/f31/bundle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-f30' in sys.argv:
        import importlib.util
        spec=importlib.util.spec_from_file_location('f30_bundle',ROOT/'tests/f30/bundle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.run(app);return
    if '--verify-f29' in sys.argv:
        sys.path.insert(0,str(ROOT/'tests/f29'))
        from bundle import run
        run(app)
        return
    if '--verify-f28' in sys.argv:
        sys.path.insert(0,str(ROOT/'tests/f28'))
        from bundle import run
        run(app)
        return
    if '--verify-f27' in sys.argv:
        sys.path.insert(0,str(ROOT/'tests/f27'))
        from bundle import run
        run(app)
        return
    if '--verify-f26' in sys.argv:
        sys.path.insert(0,str(ROOT/'tests/f26'))
        from bundle import run
        run(app)
        return
    if '--verify-f25' in sys.argv:
        sys.path.insert(0,str(ROOT/'tests/f25'))
        from bundle import run
        run(app)
        return
    if '--verify-f24' in sys.argv:
        sys.path.insert(0,str(ROOT/'tests/f24'))
        from bundle import run
        run(app)
        return
    if '--verify-f23' in sys.argv:
        sys.path.insert(0,str(ROOT/'tests/f23'))
        from bundle import run
        run(app)
        return
    if '--verify-f22' in sys.argv:
        sys.path.insert(0,str(ROOT/'tests/f22'))
        from bundle import run
        run(app)
        return
    if '--verify-f21' in sys.argv:
        sys.path.insert(0,str(ROOT/'tests/f21'))
        from bundle import run
        run(app)
        return
    if '--verify-f20' in sys.argv:
        sys.path.insert(0,str(ROOT/'tests/f20'))
        from bundle import run
        run(app)
        return
    if '--verify-f19' in sys.argv:
        sys.path.insert(0,str(ROOT/'tests/f19'))
        from bundle import run
        run(app)
        return
    if '--verify-f18' in sys.argv:
        sys.path.insert(0,str(ROOT/'tests/f18'))
        from bundle import run
        run(app)
        return
    if '--verify-f17' in sys.argv:
        sys.path.insert(0,str(ROOT/'tests/f17'))
        from bundle import run
        run(app)
        return
    if '--verify-f16' in sys.argv:
        sys.path.insert(0,str(ROOT/'tests/f16'))
        from bundle import run
        run(app)
        return
    if '--verify-f15' in sys.argv:
        sys.path.insert(0,str(ROOT/'tests/f15'))
        from bundle import run
        run(app)
        return
    if '--verify-f14' in sys.argv:
        sys.path.insert(0,str(ROOT/'tests/f14'))
        from bundle import run
        run(app)
        return
    if '--verify-f13' in sys.argv:
        sys.path.insert(0,str(ROOT/'tests/f13'))
        from bundle import run
        run(app)
        return
    if '--verify-f12' in sys.argv:
        sys.path.insert(0,str(ROOT/'tests/f12'))
        from bundle import run
        run(app)
        return
    if '--verify-f11' in sys.argv:
        sys.path.insert(0,str(ROOT/'tests/f11'))
        from bundle import run
        run(app)
        return
    if '--verify-f10' in sys.argv:
        sys.path.insert(0,str(ROOT/'tests/f10'))
        from bundle import run
        run(app)
        return
    if '--verify-f09' in sys.argv:
        sys.path.insert(0,str(ROOT/'tests/f09'))
        from bundle import run
        run(app)
        return
    if '--verify-f08' in sys.argv:
        sys.path.insert(0,str(ROOT/'tests/f08'))
        from bundle import run
        run(app)
        return
    if '--verify-f07' in sys.argv:
        sys.path.insert(0,str(ROOT/'tests/f07'))
        from bundle import run
        run(app)
        return
    if '--verify-f04-network' in sys.argv:
        sys.path.insert(0,str(ROOT/'tests/f04'))
        from bundle_network import run
        run(app)
        return
    if '--verify-f03-network' in sys.argv:
        sys.path.insert(0,str(ROOT/'tests/f03'))
        from bundle_network import run
        run(app)
        return
    if '--verify-f06' in sys.argv:
        sys.path.insert(0,str(ROOT/'tests/f06'))
        from bundle import run
        run(app)
        return
    if '--verify-f05' in sys.argv:
        sys.path.insert(0,str(ROOT/'tests/f05'))
        from bundle import run
        run(app)
        return
    if '--verify-f03' in sys.argv:
        sys.path.insert(0,str(ROOT/'tests/f03'))
        from bundle import run
        run(app)
        return
    if '--verify-f02' in sys.argv:
        sys.path.insert(0,str(ROOT/'tests/f02'))
        from bundle import run
        run(app)
        return
    if '--verify-f01' in sys.argv:
        sys.path.insert(0,str(ROOT/'tests/f01'))
        from bundle import run
        run(app)
        return
    if '--verify-performance' in sys.argv:
        sys.path.insert(0,str(ROOT/'tests'))
        from bundle_performance import run
        run(app)
        return
    import json,platform
    if '--verify-finder' not in sys.argv:
        app.open_file(str(ROOT/'tests/sample.jpg'))
    assert app.window.image is not None, 'Finder did not deliver the image'
    app.window.open_tool(app.window.tree_widget.topLevelItem(6).child(1),None)
    app.processEvents()
    assert app.window.image is not None
    assert len(app.window.mdi_area.subWindowList())==2
    app.window.grab().save(str(ROOT/'tests/bundle-interface.png'))
    (ROOT/'tests/bundle-results.json').write_text(json.dumps({'passed':True,'architecture':platform.machine(),'executable':sys.executable,'windows':len(app.window.mdi_area.subWindowList())},indent=2))
    if '--verify-trufor' in sys.argv:
        app.window.open_tool(app.window.tree_widget.topLevelItem(8).child(0),None)
        widget=[s.widget() for s in app.window.mdi_area.subWindowList() if s.windowTitle()=='TruFor'][0]
        def check_trufor(*_):
            assert widget.result is not None, widget.output
            (ROOT/'tests/bundle-trufor-results.json').write_text(json.dumps({'passed':True,'device':str(widget.result['device']),'seconds':float(widget.result['seconds']),'score':float(widget.result['score'])},indent=2))
            app.window.close();app.quit()
        widget.ready.connect(check_trufor)
        widget.trufor_process()
    else:
        app.window.close();app.quit()
def preview_plot():
    if app.window.image is None:
        app.open_file(str(ROOT/'tests/sample.jpg'))
    app.window.open_tool(app.window.tree_widget.topLevelItem(4).child(0),None)
    sub=[s for s in app.window.mdi_area.subWindowList() if s.windowTitle()=='RGB/HSV Plots'][0]
    sub.showMaximized()
    sub.widget().tab_widget.setCurrentIndex(1)
    sub.widget().colors_check.setChecked(True)
    app.window.showNormal();app.window.raise_();app.window.activateWindow()
if preview_3d: QTimer.singleShot(500,preview_plot)
if verify: QTimer.singleShot(5000 if '--verify-finder' in sys.argv else 1500,verify_bundle)
logging.info('Main window ready')
result=app.exec()
logging.info('Normal exit (%s)',result)
sys.exit(result)
