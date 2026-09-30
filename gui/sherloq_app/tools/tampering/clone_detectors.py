from gui.sherloq_app.ui.localization import choice, t
"""Green research integrations: segmentation maps and public panel comparisons."""
from pathlib import Path
import cv2 as cv
import numpy as np
from PySide6.QtCore import Qt,QSignalBlocker
from PySide6.QtWidgets import (QWidget,QSlider,QSpinBox,QLabel,QPushButton,QComboBox,QCheckBox,QVBoxLayout,QHBoxLayout,QFileDialog,QProgressBar,QListWidget,QListWidgetItem)
from ...ui.tools import ToolWidget
from ...ui.viewer import ImageViewer
from ...ui.selection_view import SelectionView
from ...ui.extensions import mark_combo
from ...ui.research_job import ResearchJob
from ...ui.jobs import LatestJob
from ...core.clone_detectors import VARIANTS,FORGERYSCOPE
from ...core.auto_zones import detect_panels,enclosing
from ...core.d2prl import refilter
from ..noise.noisesniffer import export

def render(request):
    image,result,mode=request;mask=result['analyzed'].astype(bool)
    if mode=='Suggestions':values=result['candidates'].astype(np.float32)
    elif mode=='Source':values=result.get('source',np.zeros(image.shape[:2],np.float32))
    elif mode=='Cible':values=result.get('target',np.zeros(image.shape[:2],np.float32))
    elif mode=='Masque':return np.repeat((result['mask']*255)[:,:,None],3,2)
    else:values=result['map']
    if mode=='Superposition' and result['metadata']['variant']=='D2PRL':
        mask &= result['mask'].astype(bool)
    heat=cv.applyColorMap(np.rint(np.clip(values,0,1)*255).astype(np.uint8),cv.COLORMAP_INFERNO)
    output=image.copy();output[mask]=cv.addWeighted(image,.55,heat,.45,0)[mask] if mode=='Superposition' else heat[mask]
    return output

class CloneDetectorsWidget(ToolWidget):
    def __init__(self,image,parent=None):
        super().__init__(parent);self.image=image;self.result=None;self.raw_result=None;self.closed=False;self.submitted=None;self._rebuilding=False;self.envelope=None
        self.viewer=ImageViewer(image,image,view_class=SelectionView)
        self.variant=QComboBox();self.variant.addItems(VARIANTS);mark_combo(self.variant,[],green_items=VARIANTS)
        self.cpu=QCheckBox('CPU');self.view_mode=QComboBox();self.view_mode.addItems(['Superposition','Carte','Masque','Suggestions','Source','Cible'])
        self.search=QPushButton('Rechercher');self.compare=QPushButton('Comparer');self.save=QPushButton('Exporter NPZ');self.save.setEnabled(False)
        self.selection=QComboBox();self.selection.addItems(['Déplacer','Rectangle']);self.auto=QPushButton('Détecter les sous-images');self.delete=QPushButton('Supprimer la zone');self.clear=QPushButton('Effacer les zones')
        self.zones=QListWidget();self.zones.setMaximumWidth(220);self.zones.setMinimumWidth(160)
        self.note=QLabel();self.note.setWordWrap(True);self.status=QLabel('Prêt.');self.status.setWordWrap(True);self.progress=QProgressBar()
        layout=QVBoxLayout(self);top=QHBoxLayout()
        for w in (self.variant,self.cpu,self.view_mode,self.search,self.compare,self.save):top.addWidget(w)
        layout.addLayout(top)
        self.filter_controls=QWidget();filter_row=QHBoxLayout(self.filter_controls);filter_row.setContentsMargins(0,0,0,0)
        self.minimum=QSlider(Qt.Horizontal);self.minimum.setRange(0,5000);self.minimum.setValue(500)
        self.minimum_value=QSpinBox();self.minimum_value.setRange(0,5000);self.minimum_value.setValue(500)
        filter_row.addWidget(QLabel('Minimum region size (448 × 448 grid)'));filter_row.addWidget(self.minimum,1);filter_row.addWidget(self.minimum_value)
        filter_row.addWidget(QLabel('0 = no size filter'))
        layout.addWidget(self.filter_controls);layout.addWidget(self.note);controls=QHBoxLayout()
        for w in (self.selection,self.auto,self.delete,self.clear):controls.addWidget(w)
        controls.addStretch();layout.addLayout(controls);body=QHBoxLayout();body.addWidget(self.viewer,1);zone_layout=QVBoxLayout();zone_layout.addWidget(QLabel('Zones'));zone_layout.addWidget(self.zones);body.addLayout(zone_layout);layout.addLayout(body,1);layout.addWidget(self.status);layout.addWidget(self.progress)
        self.job=ResearchJob(self,image,'clone_detectors');self.draw=LatestJob(self,render,delay=0);self.export_job=LatestJob(self,export,delay=0);self.auto_job=LatestJob(self,lambda im:detect_panels(im),delay=0)
        self.job.result.connect(self.complete);self.job.failed.connect(self.failed);self.job.busy.connect(self.state);self.job.progress.connect(lambda n,t:(self.progress.setValue(n),self.status.setText(t)))
        self.draw.result.connect(self.viewer.update_processed);self.draw.failed.connect(self.status.setText);self.export_job.failed.connect(self.status.setText);self.export_job.result.connect(lambda p:self.status.setText('Exporté : '+str(p)))
        self.auto_job.result.connect(self.auto_complete);self.auto_job.failed.connect(self.status.setText)
        self.filter_job=LatestJob(self,refilter,delay=80)
        self.filter_job.result.connect(self.filtered);self.filter_job.failed.connect(self.failed)
        self.minimum.valueChanged.connect(self.filter_changed);self.minimum_value.valueChanged.connect(self.minimum.setValue)
        self.search.clicked.connect(lambda:self.start(False));self.compare.clicked.connect(lambda:self.start(True));self.save.clicked.connect(self.export_data)
        self.variant.currentIndexChanged.connect(self.variant_changed);self.cpu.toggled.connect(self.changed);self.view_mode.currentIndexChanged.connect(self.redraw)
        self.selection.currentIndexChanged.connect(lambda i:self.viewer.view.set_mode('Rectangle' if i else 'Pan'))
        self.viewer.view.regionsChanged.connect(self.regions_changed);self.zones.itemChanged.connect(self.changed);self.zones.currentRowChanged.connect(self.focus_zone)
        self.auto.clicked.connect(self.auto_start);self.delete.clicked.connect(self.delete_zone);self.clear.clicked.connect(self.viewer.view.clear_regions)
        self.variant_changed()
    def active(self):
        snapshot=self.viewer.view.snapshot()
        return tuple(p for i,p in enumerate(snapshot) if i<self.zones.count() and self.zones.item(i).checkState()==Qt.Checked)
    def params(self,compare=False):
        return dict(variant=choice(self.variant),regions=self.active(),selection_present=bool(self.viewer.view.snapshot()),compare=compare)
    def variant_changed(self,*_):
        variant=choice(self.variant);forced=variant=='MGCFDN VIG 16×16'
        self.filter_controls.setVisible(variant=='D2PRL')
        with QSignalBlocker(self.cpu):self.cpu.setChecked(forced)
        self.cpu.setEnabled(not forced)
        self.view_mode.model().item(3).setEnabled(variant in FORGERYSCOPE)
        for i in (4,5):self.view_mode.model().item(i).setEnabled(variant in ('MGCFDN source/cible','D2PRL'))
        self.view_mode.setCurrentIndex(3 if variant.endswith('pistes') else 0)
        self.note.setText('Compare des panneaux, pas les retouches internes d’un panneau. Rechercher traite chaque zone séparément ; Ensemble couvre la planche. Comparer utilise deux rectangles. Suggestions = similarité sans confirmation géométrique.' if variant in FORGERYSCOPE else 'Carte de segmentation. Chaque rectangle est analysé séparément à la résolution du modèle ; comparaison entre zones et rayon de recherche non disponibles.')
        self.changed()
    def regions_changed(self,regions):
        self.auto_job.invalidate()
        self._rebuilding=True;checks=[self.zones.item(i).checkState() for i in range(self.zones.count())];self.zones.clear()
        for i,p in enumerate(regions):
            item=QListWidgetItem('Ensemble' if p==self.envelope else f'Zone {i+1}');item.setFlags(item.flags()|Qt.ItemIsUserCheckable);item.setCheckState(checks[i] if i<len(checks) else Qt.Checked);self.zones.addItem(item)
        self._rebuilding=False;self.changed()
    def focus_zone(self,i):
        self.viewer.view.focus_regions={i} if i>=0 else set();self.viewer.view.viewport().update()
    def delete_zone(self):
        i=self.zones.currentRow()
        if i<0:return
        del self.viewer.view.regions[i]
        with QSignalBlocker(self.zones):self.zones.takeItem(i)
        self.regions_changed(self.viewer.view.snapshot());self.viewer.view.viewport().update()
    def auto_start(self):
        self.auto_job.request(self.image)
        self.status.setText('Détection des sous-images…')
    def auto_complete(self,regions):
        self.envelope=enclosing(regions) if len(regions)>1 else None
        self.viewer.view.regions=list(regions)+([self.envelope] if self.envelope else [])
        # New proposals are all active, independently of previously unchecked rows.
        with QSignalBlocker(self.zones):self.zones.clear()
        self.regions_changed(self.viewer.view.snapshot());self.viewer.view.viewport().update()
        self.status.setText(f'{len(regions)} sous-images proposées'+(' + Ensemble : recherche entre sous-images dans cette zone.' if self.envelope else '.'))
    def changed(self,*_):
        if self._rebuilding:return
        self.auto_job.invalidate()
        if self.job.is_busy:self.job.cancel()
        self.filter_job.invalidate();self.raw_result=None
        self.draw.invalidate();self.result=None;self.save.setEnabled(False);self.viewer.set_busy(True)
        self.viewer.view.enabled_regions={i for i in range(self.zones.count()) if self.zones.item(i).checkState()==Qt.Checked};self.viewer.view.viewport().update();self.state(False)
    def state(self,busy):
        self.search.setText('Annuler' if busy else 'Rechercher');self.compare.setEnabled(not busy and choice(self.variant) in FORGERYSCOPE and len(self.active())==2)
        self.compare.setToolTip('Compare exactement deux rectangles avec Forgeryscope.')
        self.save.setEnabled(not busy and not self.filter_job.is_busy and self.result is not None)
    def start(self,compare):
        if self.job.is_busy:self.job.cancel();self.status.setText('Annulé.');return
        if self.viewer.view.snapshot() and not self.active():self.status.setText('Cocher au moins une zone.');return
        self.filter_job.invalidate();self.raw_result=None
        self.draw.invalidate();self.result=None;self.save.setEnabled(False);self.submitted=self.params(compare);self.progress.setValue(0)
        self.job.request(self.submitted,'cpu' if self.cpu.isChecked() else 'mps')
    def complete(self,result):
        self.raw_result=result
        if result['metadata']['variant']=='D2PRL':
            self.progress.setValue(100);self.state(False);self.filter_changed(self.minimum.value());return
        self.filtered(result)
    def filter_changed(self,value):
        with QSignalBlocker(self.minimum_value):self.minimum_value.setValue(value)
        if self.raw_result is None or self.raw_result['metadata']['variant']!='D2PRL':return
        self.draw.invalidate();self.save.setEnabled(False)
        self.filter_job.request((self.raw_result,value))
    def filtered(self,result):
        if result['metadata']['variant']=='D2PRL':self.raw_result=result
        self.result=result;self.progress.setValue(100);self.state(False);self.viewer.set_busy(False);self.redraw();m=result['metadata']
        messages={'insufficient_panels':'Aucune comparaison possible : chaque zone contient moins de deux panneaux adaptés. Activer Ensemble ou comparer deux rectangles ; pour les retouches internes, utiliser Copy-Move Forgery 2.', 'no_panels':'Aucun panneau adapté reconnu. Choisir deux rectangles pour comparer, ou une méthode de segmentation.','empty':'Aucune zone retenue par ce réglage.','candidates':'Suggestions à examiner ; aucune zone confirmée par la géométrie.','ok':'Analyse terminée.'}
        detail=''
        if m['variant'] in FORGERYSCOPE:
            candidates=sum(len(z.get('embedding_candidates',())) for z in m['zones'])
            kept=sum(sum(bool(c.get('supported')) for c in z.get('comparisons',())) for z in m['zones'])
            if not m['variant'].endswith('pistes'):detail=f' Paires candidates : {candidates} ; géométries retenues : {kept}.'
        self.status.setText(messages[m['status']]+detail+f" {m['seconds']:.2f} s.")
    def failed(self,error):self.status.setText(error);self.state(False)
    def redraw(self,*_):
        if self.result is not None:self.draw.request((self.image,self.result,choice(self.view_mode)))
    def export_data(self):
        if self.result is None or not self.save.isEnabled():return
        path,_=QFileDialog.getSaveFileName(self,t('Exporter les cartes et paramètres'),'clone-detectors.npz',t('NumPy (*.npz)'))
        if path:self.export_job.request((str(Path(path).with_suffix('.npz')),self.result))
    def shutdown(self):
        if self.closed:return
        self.closed=True;self.job.shutdown();super().shutdown()
