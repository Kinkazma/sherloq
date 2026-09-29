from gui.sherloq_app.ui.localization import choice, t
"""Common cached map viewer for pretrained forensic methods."""
import cv2 as cv
import numpy as np
from pathlib import Path
from PySide6.QtWidgets import QLabel,QPushButton,QComboBox,QSpinBox,QCheckBox,QVBoxLayout,QHBoxLayout,QFileDialog,QProgressBar
from .tools import ToolWidget
from .viewer import ImageViewer
from .jobs import LatestJob
from .research_job import ResearchJob
from ..tools.noise.noisesniffer import export

def display(request):
    image,result,mode=request;meta=result['metadata'];output=image.copy()
    if meta['method']=='adaptive_cfa':
        block=meta['block'];y,x=meta['origin'];h,w=meta['valid_shape']
        if mode==2:
            colors=np.array([[220,90,210],[70,190,255],[255,190,50],[90,220,110]],np.uint8);heat=colors[result['local_grid']]
        else:heat=cv.applyColorMap(np.rint(result['suspicion']*255).astype(np.uint8),cv.COLORMAP_INFERNO)
        heat=np.repeat(np.repeat(heat,block,0),block,1)
        output[y:y+h,x:x+w]=cv.addWeighted(image[y:y+h,x:x+w],.5,heat,.5,0) if mode==0 else heat
    else:
        if mode==2 and 'mask' in result:
            return cv.resize(np.repeat((result['mask']*255)[:,:,None],3,2),(image.shape[1],image.shape[0]),interpolation=cv.INTER_NEAREST)
        values=result['map'];heat=cv.applyColorMap(np.rint(np.clip(values,0,1)*255).astype(np.uint8),cv.COLORMAP_INFERNO);heat=cv.resize(heat,(image.shape[1],image.shape[0]),interpolation=cv.INTER_NEAREST)
        output=cv.addWeighted(image,.5,heat,.5,0) if mode==0 else heat
    return output

class ResearchPanel(ToolWidget):
    def __init__(self,image,method,variants,note,parent=None,filename=None):
        super().__init__(parent);self.image=image;self.method=method;self.result=None;self.result_settings=None;self.closed=False
        self.variant=QComboBox();self.variant.addItems(variants);self.cpu=QCheckBox('CPU');self.block=QSpinBox();self.block.setRange(8,128);self.block.setSingleStep(2);self.block.setValue(32)
        self.run=QPushButton('Analyser');self.save=QPushButton('Exporter NPZ');self.save.setEnabled(False);self.mode=QComboBox();self.mode.addItems(['Superposition','Carte'])
        if method=='adaptive_cfa':self.mode.addItem('Grille CFA locale')
        if method=='adaifl':self.mode.addItem('Masque')
        self.status=QLabel(note);self.status.setWordWrap(True);self.legend=QLabel('Carte : noir = 0 · jaune = 1'+('   |   Grilles : violet 00 · orange 01 · bleu 10 · vert 11' if method=='adaptive_cfa' else ''));self.legend.setWordWrap(True)
        self.progress=QProgressBar();self.viewer=ImageViewer(image,image);layout=QVBoxLayout(self);row=QHBoxLayout();row.addWidget(QLabel('Modèle'));row.addWidget(self.variant)
        if method=='adaptive_cfa':row.addWidget(QLabel('Bloc'));row.addWidget(self.block)
        else:self.block.hide()
        for c in (self.cpu,self.mode,self.run,self.save):row.addWidget(c)
        row.addStretch();layout.addLayout(row);layout.addWidget(self.status);layout.addWidget(self.legend);layout.addWidget(self.viewer);layout.addWidget(self.progress)
        self.job=ResearchJob(self,image,method,filename);self.draw_job=LatestJob(self,display,delay=0);self.export_job=LatestJob(self,export,delay=0)
        self.job.result.connect(self.complete);self.job.busy.connect(self.state);self.job.failed.connect(self.failed);self.job.progress.connect(self.progressed)
        self.draw_job.result.connect(self.viewer.update_processed);self.draw_job.failed.connect(self.status.setText);self.export_job.failed.connect(self.status.setText);self.export_job.result.connect(lambda p:self.status.setText(f'Exporté : {p}'))
        self.run.clicked.connect(self.start);self.save.clicked.connect(self.export_data);self.mode.currentIndexChanged.connect(self.redraw)
        self.variant.currentIndexChanged.connect(self.changed);self.block.valueChanged.connect(self.changed);self.cpu.toggled.connect(self.changed)
    def params(self):return dict(variant=choice(self.variant),**(dict(block=self.block.value()) if self.method=='adaptive_cfa' else {}))
    def settings(self):return self.params(),'cpu' if self.cpu.isChecked() else 'mps'
    def changed(self,*_):
        current=self.result is not None and self.result_settings==self.settings();self.save.setEnabled(current and not self.job.is_busy);self.viewer.set_busy(not current)
    def state(self,busy):
        self.run.setText('Annuler' if busy else 'Analyser')
        for c in (self.variant,self.block,self.cpu):c.setEnabled(not busy)
        self.save.setEnabled(not busy and self.result is not None and self.result_settings==self.settings());self.viewer.set_busy(busy or self.result_settings!=self.settings())
    def start(self):
        if self.job.is_busy:self.job.cancel();self.status.setText('Annulé.');return
        self.draw_job.invalidate();self.progress.setValue(0);self.status.setText('Préparation de l’image…');self.job.request(*self.settings())
    def complete(self,result):
        self.result=result;self.result_settings=self.settings();self.progress.setValue(100);self.state(False);self.redraw();m=result['metadata'];self.status.setText(f"Analyse terminée — {m['seconds']:.2f} s."+(' Les bords non analysés conservent l’image originale.' if self.method=='adaptive_cfa' else ''))
    def failed(self,error):self.status.setText(error);self.state(False)
    def progressed(self,value,text):self.progress.setValue(value);self.status.setText(text)
    def redraw(self,*_):
        if self.result is not None:self.draw_job.request((self.image,self.result,self.mode.currentIndex()))
    def export_data(self):
        if not self.save.isEnabled():return
        path,_=QFileDialog.getSaveFileName(self,t('Exporter les résultats'),self.method+'.npz',t('NumPy (*.npz)'))
        if path:self.export_job.request((str(Path(path).with_suffix('.npz')),self.result))
    def shutdown(self):
        if self.closed:return
        self.closed=True;self.job.shutdown();super().shutdown()
