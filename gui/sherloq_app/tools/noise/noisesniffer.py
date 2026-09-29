from gui.sherloq_app.ui.localization import choice, t
"""Noisesniffer panel: isolated cancellable analysis and retained full outputs."""
import json,sys,tempfile,os
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import QLabel,QPushButton,QComboBox,QSpinBox,QDoubleSpinBox,QHBoxLayout,QVBoxLayout,QFileDialog
from gui.sherloq_app.core.interactive import ArrayCache
from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.ui.viewer import ImageViewer
from gui.sherloq_app.ui.jobs import LatestJob
from gui.sherloq_app.ui.process import ProcessJob


def prepare(request):
    folder,image,lifetime=request;path=Path(folder)/'image.npy'
    if not path.exists():np.save(path,image,allow_pickle=False)
    return folder


def read_result(request):
    folder,lifetime=request
    p=Path(folder);result={name:np.load(p/f'{name}.npy',allow_pickle=False) for name in ('mask','distribution','all_blocks','low_noise_blocks','selected','low_noise')}
    result['metadata']=json.loads((p/'result.json').read_text());return result


def display(request):
    image,result,mode=request
    if mode==1:return np.repeat(result['mask'][:,:,None],3,axis=2)
    if mode==2:return result['distribution']
    output=image.copy();selected=result['mask']>0;output[selected]=(output[selected].astype(np.float32)*.55+np.array([0,0,255])*.45).astype(np.uint8);return output


def export(request):
    filename,result=request;temporary=None
    try:
        with tempfile.NamedTemporaryFile('wb',dir=Path(filename).parent,delete=False) as f:
            temporary=f.name;np.savez_compressed(f,metadata_json=json.dumps(result['metadata']),**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
        os.replace(temporary,filename);return filename
    finally:
        if temporary and os.path.exists(temporary):os.unlink(temporary)


class NoisesnifferWidget(ToolWidget):
    def __init__(self,image,parent=None):
        super().__init__(parent);self.image=image;self.directory=tempfile.TemporaryDirectory(prefix='sherloq-noisesniffer-');self.folder=self.directory.name;self.cache=ArrayCache(384);self.result=None;self.key=None;self.active=None;self.closed=False
        self.block=QComboBox();self.block.addItems(['3','5','7','8']);self.cell=QSpinBox();self.cell.setRange(10,500);self.cell.setValue(100)
        self.samples=QSpinBox();self.samples.setRange(100,200000);self.samples.setValue(20000)
        self.fraction=QDoubleSpinBox();self.fraction.setRange(.01,1.);self.fraction.setValue(.1);self.fraction.setSingleStep(.01)
        self.low=QDoubleSpinBox();self.low.setRange(.01,.99);self.low.setValue(.5);self.low.setSingleStep(.05)
        self.mode=QComboBox();self.mode.addItems(['Régions','Masque','Distribution des blocs'])
        self.run=QPushButton('Analyser');self.save=QPushButton('Exporter NPZ');self.save.setEnabled(False);self.status=QLabel('Anomalies locales de bruit ; elles ne suffisent pas à établir une retouche.');self.status.setWordWrap(True)
        self.viewer=ImageViewer(image,image);layout=QVBoxLayout(self);row=QHBoxLayout()
        self.controls=[self.block,self.cell,self.samples,self.fraction,self.low]
        for label,control in zip(['Bloc','Cellule','Échantillons / classe','Fraction','Bruit faible'],self.controls):row.addWidget(QLabel(label));row.addWidget(control)
        layout.addLayout(row);row=QHBoxLayout()
        for control in (self.mode,self.run,self.save):row.addWidget(control)
        row.addStretch();layout.addLayout(row);layout.addWidget(self.status);layout.addWidget(self.viewer)
        self.prepare_job=LatestJob(self,prepare,delay=0);self.read_job=LatestJob(self,read_result,delay=0);self.draw_job=LatestJob(self,display,delay=0);self.export_job=LatestJob(self,export,delay=0)
        self.process=ProcessJob(self,timeout_ms=1200000,heavy=True)
        self.prepare_job.result.connect(self.launch);self.process.result.connect(lambda _:self.read_job.request((self.folder,self.directory)));self.read_job.result.connect(self.complete);self.draw_job.result.connect(self.viewer.update_processed);self.export_job.result.connect(lambda p:self.status.setText(f'Exporté : {p}'))
        for job in (self.prepare_job,self.read_job,self.process):job.failed.connect(self.failed)
        self.draw_job.failed.connect(self.status.setText);self.export_job.failed.connect(self.status.setText)
        self.process.waiting.connect(lambda waiting:self.status.setText('En attente d’une autre analyse…' if waiting else 'Analyse du bruit…'))
        self.run.clicked.connect(self.start);self.save.clicked.connect(self.export_data);self.mode.currentIndexChanged.connect(self.redraw)
        for control in self.controls:
            signal=control.currentIndexChanged if isinstance(control,QComboBox) else control.valueChanged;signal.connect(self.settings_changed)

    def params(self):return (int(choice(self.block)),self.cell.value(),self.samples.value(),self.fraction.value(),self.low.value())
    def busy(self):return self.active is not None
    def settings_changed(self):
        self.save.setEnabled(self.result is not None and self.key==self.params() and not self.busy());self.viewer.set_busy(self.key!=self.params())
    def state(self):
        busy=self.busy();self.run.setText('Annuler' if busy else 'Analyser')
        for c in self.controls:c.setEnabled(not busy)
        self.save.setEnabled(not busy and self.key==self.params() and self.result is not None)
        self.viewer.set_busy(busy or self.key!=self.params())
    def start(self):
        if self.busy():
            self.prepare_job.invalidate();self.read_job.invalidate();self.process.cancel();self.active=None;self.state();self.status.setText('Annulé.');return
        if self.process.is_busy:return
        self.draw_job.invalidate()
        key=self.params();cached=self.cache.get(key)
        if cached is not None:self.active=key;self.complete(cached);return
        self.active=key;self.state();self.status.setText('Préparation de l’image…');self.prepare_job.request((self.folder,self.image,self.directory))
    def launch(self,folder):
        if self.active is None:return
        from gui.sherloq_app.core import noisesniffer
        self.process.start(sys.executable,[str(Path(noisesniffer.__file__)),folder,json.dumps(self.active)])
    def complete(self,result):
        if self.active is None:return
        self.result=result;self.key=self.active;self.cache.put(self.key,result);self.active=None;self.state();self.redraw();meta=result['metadata']
        self.status.setText('Analyse indéterminée : pas assez de blocs exploitables.' if meta['inconclusive'] else f"{len(meta['regions'])} régions ; {meta['selected_blocks']:,} blocs retenus — {meta['seconds']:.2f} s.")
    def failed(self,error):
        self.active=None;self.state();self.status.setText(error)
    def redraw(self):
        if self.result is not None:self.draw_job.request((self.image,self.result,self.mode.currentIndex()))
    def export_data(self):
        if not self.save.isEnabled():return
        path,_=QFileDialog.getSaveFileName(self,t('Exporter Noisesniffer'),'noisesniffer.npz',t('NumPy (*.npz)'))
        if path:self.export_job.request((str(Path(path).with_suffix('.npz')),self.result))
    def shutdown(self):
        if self.closed:return
        self.closed=True;self.process.shutdown();super().shutdown()
        directory=self.directory;self.directory=None
        pending=[j.active for j in (self.prepare_job,self.read_job) if j.active is not None]
        count=[len(pending)]
        def finished(*args):
            count[0]-=1
            if count[0]==0:directory.cleanup()
        if pending:
            for work in pending:work.signals.finished.connect(finished)
        else:directory.cleanup()
