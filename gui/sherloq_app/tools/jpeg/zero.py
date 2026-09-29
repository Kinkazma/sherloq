from gui.sherloq_app.ui.localization import t
"""ZERO JPEG grids, with isolated native CPU execution and cached displays."""
import json,sys,tempfile
from pathlib import Path
import numpy as np
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QLabel,QPushButton,QComboBox,QCheckBox,QHBoxLayout,QVBoxLayout,QFileDialog,QListWidget,QListWidgetItem,QSplitter
from gui.sherloq_app.core.interactive import ArrayCache
from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.ui.viewer import ImageViewer
from gui.sherloq_app.ui.jobs import LatestJob
from gui.sherloq_app.ui.process import ProcessJob
from gui.sherloq_app.tools.noise.noisesniffer import prepare,export


def read_result(request):
    folder,lifetime=request;p=Path(folder)
    names=('luminance','luminance_jpeg','votes','votes_jpeg','mask_f','mask_f_reg','mask_m','mask_m_reg','grid_log10_nfa')
    result={name:np.load(p/f'{name}.npy',allow_pickle=False) for name in names};result['metadata']=json.loads((p/'result.json').read_text());return result


def palette():
    from matplotlib import colormaps
    colors=[]
    for i in range(64):
        j=4 if i==0 else 0 if i==4 else i
        name,offset,scale=('tab20',0,20) if j<20 else ('tab20b',20,20) if j<40 else ('tab20c',40,20) if j<60 else ('Set3',60,20)
        colors.append((np.asarray(colormaps[name]((j-offset)/scale)[:3])*255).astype(np.uint8)[::-1])
    return np.asarray(colors)


def display(request):
    result,mode=request
    if mode in (1,2):
        votes=result['votes' if mode==1 else 'votes_jpeg'];out=palette()[np.clip(votes,0,63)];out[votes<0]=0;return out
    if mode in (3,4):return np.repeat(result['mask_f_reg' if mode==3 else 'mask_m_reg'][:,:,None],3,axis=2).astype(np.uint8)
    f=result['mask_f_reg']>0;m=(result['mask_m_reg']>0)&~f
    gray=result['luminance'].astype(np.uint8)
    if f.any() or m.any():gray=(gray*.2).astype(np.uint8)
    out=np.repeat(gray[:,:,None],3,axis=2);out[f,2]=255;out[m,0]=255;return out


class ZeroWidget(ToolWidget):
    def __init__(self,image,parent=None):
        super().__init__(parent);self.image=image;self.directory=tempfile.TemporaryDirectory(prefix='sherloq-zero-');self.folder=self.directory.name;self.cache=ArrayCache(384);self.result=None;self.key=None;self.active=None;self.closed=False
        self.missing=QCheckBox('Rechercher les grilles manquantes');self.missing.setChecked(True);self.cpu=QCheckBox('CPU');self.cpu.setToolTip('Référence sérielle')
        self.mode=QComboBox();self.mode.addItems(['Régions','Votes de grille','Votes JPEG99','Grilles différentes','Grilles manquantes'])
        self.run=QPushButton('Analyser');self.save=QPushButton('Exporter NPZ');self.save.setEnabled(False);self.status=QLabel('Rouge : grille différente ; bleu : grille manquante. Un indice de traitement, pas un verdict d’authenticité.');self.status.setWordWrap(True)
        self.viewer=ImageViewer(image,image);self.legend=QListWidget();self.legend.setMaximumWidth(260)
        colors=palette()
        for i,c in enumerate(colors):
            item=QListWidgetItem(f'■ {i:02d} — origine ({i%8}, {i//8})');item.setForeground(QColor(int(c[2]),int(c[1]),int(c[0])));self.legend.addItem(item)
        layout=QVBoxLayout(self);row=QHBoxLayout()
        for control in (self.missing,self.cpu,self.mode,self.run,self.save):row.addWidget(control)
        row.addStretch();layout.addLayout(row);layout.addWidget(self.status);split=QSplitter();split.addWidget(self.viewer);split.addWidget(self.legend);layout.addWidget(split)
        self.prepare_job=LatestJob(self,prepare,delay=0);self.read_job=LatestJob(self,read_result,delay=0);self.draw_job=LatestJob(self,display,delay=0);self.export_job=LatestJob(self,export,delay=0);self.process=ProcessJob(self,timeout_ms=1200000,heavy=True)
        self.prepare_job.result.connect(self.launch);self.process.result.connect(lambda _:self.read_job.request((self.folder,self.directory)));self.read_job.result.connect(self.complete);self.draw_job.result.connect(self.viewer.update_processed);self.export_job.result.connect(lambda p:self.status.setText(f'Exporté : {p}'))
        for job in (self.prepare_job,self.read_job,self.process):job.failed.connect(self.failed)
        for job in (self.draw_job,self.export_job):job.failed.connect(self.status.setText)
        self.process.waiting.connect(lambda waiting:self.status.setText('En attente d’une autre analyse…' if waiting else 'Analyse des grilles JPEG…'))
        self.run.clicked.connect(self.start);self.save.clicked.connect(self.export_data);self.mode.currentIndexChanged.connect(self.redraw);self.missing.toggled.connect(self.settings_changed);self.cpu.toggled.connect(self.settings_changed)

    def params(self):return (self.missing.isChecked(),self.cpu.isChecked())
    def busy(self):return self.active is not None
    def settings_changed(self):
        self.save.setEnabled(self.result is not None and self.key==self.params() and not self.busy());self.viewer.set_busy(self.key!=self.params())
    def state(self):
        busy=self.busy();self.run.setText('Annuler' if busy else 'Analyser');self.cpu.setEnabled(not busy);self.missing.setEnabled(not busy);self.save.setEnabled(not busy and self.key==self.params() and self.result is not None);self.viewer.set_busy(busy or self.key!=self.params())
    def start(self):
        if self.busy():
            self.prepare_job.invalidate();self.read_job.invalidate();self.process.cancel();self.active=None;self.state();self.status.setText('Annulé.');return
        if self.process.is_busy:return
        self.draw_job.invalidate();key=self.params();cached=self.cache.get(key)
        if cached is not None:self.active=key;self.complete(cached);return
        self.active=key;self.state();self.status.setText('Préparation de l’image…');self.prepare_job.request((self.folder,self.image,self.directory))
    def launch(self,folder):
        if self.active is None:return
        from gui.sherloq_app.core import zero
        self.process.start(sys.executable,[str(Path(zero.__file__)),folder,json.dumps(self.active)])
    def complete(self,result):
        if self.active is None:return
        self.result=result;self.key=self.active;self.cache.put(self.key,result);self.active=None;self.state();self.redraw();meta=result['metadata'];grid=meta['main_grid'];main=f'({grid%8}, {grid//8})' if grid>=0 else 'aucune'
        self.status.setText(f"Grille principale : {main} ; {len(meta['foreign_regions'])} régions différentes, {len(meta['missing_regions'])} manquantes — {meta['seconds']:.2f} s.")
        for i in range(64):
            item=self.legend.item(i);item.setToolTip(f"log10 NFA : {result['grid_log10_nfa'][i]:.6g}")
    def failed(self,error):self.active=None;self.state();self.status.setText(error)
    def redraw(self):
        self.legend.setVisible(self.mode.currentIndex() in (1,2))
        if self.result is not None:self.draw_job.request((self.result,self.mode.currentIndex()))
    def export_data(self):
        if not self.save.isEnabled():return
        path,_=QFileDialog.getSaveFileName(self,t('Exporter ZERO'),'zero.npz',t('NumPy (*.npz)'))
        if path:self.export_job.request((str(Path(path).with_suffix('.npz')),self.result))
    def shutdown(self):
        if self.closed:return
        self.closed=True;self.process.shutdown();super().shutdown();directory=self.directory;self.directory=None
        pending=[j.active for j in (self.prepare_job,self.read_job) if j.active is not None];count=[len(pending)]
        def finished(*args):
            count[0]-=1
            if count[0]==0:directory.cleanup()
        if pending:
            for work in pending:work.signals.finished.connect(finished)
        else:directory.cleanup()
