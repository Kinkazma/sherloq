from gui.sherloq_app.ui.localization import choice, t
"""ELA biome exploration, native file export and worker-only image processing."""
import colorsys
from functools import partial
from pathlib import Path
from threading import Event
import cv2 as cv
import numpy as np
from PySide6.QtCore import Qt, QTimer, QSignalBlocker
from PySide6.QtWidgets import (QComboBox,QDoubleSpinBox,QSpinBox,QPushButton,QLabel,
    QVBoxLayout,QHBoxLayout,QListWidget,QSplitter,QFileDialog,QCheckBox,QSlider)
from .tools import ToolWidget
from .viewer import ImageViewer
from .jobs import LatestJob
from .ela_profiles import ElaProfiles,compute_profile
from .percentile_slider import PercentileSlider
from .ela_settings import ElaSettings
from ..core.ela_biomes import (ElaBiomeEngine,Cancelled,segment,DEFAULT_BLOCK,
    DEFAULT_THRESHOLD,DEFAULT_MINIMUM_CELLS)
from ..tools.noise.noisesniffer import export
from ..core.ela_energy import energy_color


def analyze(engine,request):
    params,cancel=request
    try:return params,engine.prepare(*params[:2],cancel=cancel.is_set,ghost=bool(params[2]) if len(params)>2 else False,all_grids=bool(params[3]) if len(params)>3 else False,background=bool(params[4]) if len(params)>4 else False,energy=bool(params[4]) if len(params)>4 else False,energy_quantiles=tuple(params[5:7]) if len(params)>5 else (.1,.9),energy_auto=bool(params[7]) if len(params)>7 else False,energy_auto_profile=params[8] if len(params)>8 else 'sensitive')
    except Cancelled:return params,None


def color(i):
    rgb=colorsys.hsv_to_rgb((.73+(i-1)*.61803398875)%1,.7,.95)
    return tuple(round(x*255) for x in rgb[::-1])


def energy_controls():
    """Shared display thresholds: values are independent of the profile threshold."""
    row=QHBoxLayout();sliders=[]
    for name in ('Shadow deviation','Highlight deviation'):
        slider=QSlider(Qt.Horizontal);slider.setRange(0,200);slider.setValue(20)
        slider.setMinimumWidth(120);slider.setMaximumWidth(280)
        slider.setAccessibleName(name)
        slider.setToolTip('Higher: require a larger energy difference. Lower: include smaller differences.')
        value=QLabel('2.0');value.setFixedWidth(64);slider.setProperty('valueLabel',value)
        slider.valueChanged.connect(lambda n,label=value:label.setText(f'{n/10:.1f}'))
        row.addWidget(QLabel(name));row.addWidget(slider);row.addWidget(value);sliders.append(slider)
    row.addStretch()
    return row,sliders


def apply_automatic(base,sliders):
    automatic=base['metadata'].get('energy',{}).get('automatic')
    if not automatic:return
    values=(*[round(x*1000) for x in automatic['quantiles']],*[round(x*10) for x in automatic['thresholds']])
    for slider,value in zip(sliders,values):
        # Do not schedule another analysis, but refresh the numeric label.
        with QSignalBlocker(slider):slider.setValue(value)
        slider.setProperty('automaticValue',value)
        label=slider.property('valueLabel')
        if label is not None:label.setText(f'{value/10:.1f}'+(' %' if slider.property('histogramPercentile') else ''))


def histogram_controls():
    row=QHBoxLayout();sliders=[]
    for name,upper,default in (('Lower histogram bound',False,100),('Upper histogram bound',True,900)):
        slider=PercentileSlider(upper);slider.setValue(default)
        slider.setMinimumWidth(120);slider.setMaximumWidth(280);slider.setAccessibleName(name)
        slider.setToolTip('Percentile of pixels ranked by ELA energy; defines the central reference interval. Red marks values below 90%.' if upper else 'Percentile of pixels ranked by ELA energy; defines the central reference interval. Red marks values above 10%.')
        value=QLabel(f'{default/10:.1f} %');value.setFixedWidth(80);slider.setProperty('valueLabel',value)
        slider.valueChanged.connect(lambda _,label=value,control=slider:label.setText(f'{control.value()/10:.1f} %'))
        row.addWidget(QLabel(name));row.addWidget(slider);row.addWidget(value);sliders.append(slider)
    row.addStretch()
    return row,sliders


def render(request,result=None):
    image,base,threshold,minimum,mode,selection=request[:6]
    energy_thresholds=request[6:8] if len(request)>6 else None
    energy_visible,legacy_visible=request[8] if len(request)>8 else (True,True)
    if result is None:result=segment(base,threshold,minimum,energy_thresholds=energy_thresholds)
    block=result['metadata']['block']
    h,w=result['metadata']['valid_shape'];labels=result['labels']
    if mode==4:
        heat=cv.applyColorMap(np.rint(np.clip(result['score']/max(threshold*2,1),0,1)*255).astype(np.uint8),cv.COLORMAP_INFERNO)
        heat[~result['supported']]=(100,100,100)
    elif mode==5:
        heat=np.where(result['supported'][:,:,None],np.array([100,170,100]),np.array([100,100,100])).astype(np.uint8)
    else:
        heat=np.zeros((*labels.shape,3),np.uint8)
        for r in result['metadata']['regions']:heat[labels==r['id']]=color(r['id'])
    output=(base['ela'] if mode in (0,3,6,7) else image).copy()
    if mode==3:return result,output,request[2:]
    shown=(labels>0)&((labels==selection) if selection else True)&(mode not in (6,7))&legacy_visible
    if mode==2:output[:]=32
    large=np.repeat(np.repeat(heat,block,0),block,1)
    if mode in (4,5):
        output[:]=100
        output[:h,:w]=large
    else:
        mask=np.repeat(np.repeat(shown,block,0),block,1)
        mixed=cv.addWeighted(output[:h,:w],.55,large,.45,0) if mode in (0,1) else large
        output[:h,:w][mask]=mixed[mask]
        # Draw the cell boundary last: energy fills used to erase pieces of it.
        cell_contours,_=cv.findContours(mask.astype(np.uint8),cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
        for r in result['metadata']['regions']:
            if not energy_visible or 'kind' not in r or (selection and selection!=r['id']):continue
            if (mode==6 and r['kind']!='low') or (mode==7 and r['kind']!='high'):continue
            x,y,rw,rh=r['bbox'];mask=(result['energy_labels'][y:y+rh,x:x+rw]==r['id'])
            roi=output[y:y+rh,x:x+rw];solid=np.full_like(roi,energy_color(r))
            mixed=cv.addWeighted(roi,.55,solid,.45,0) if mode!=2 else solid
            roi[mask]=mixed[mask]
            contours,_=cv.findContours(mask.astype(np.uint8),cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
            border=np.zeros(mask.shape,np.uint8);cv.drawContours(border,contours,-1,1,1)
            roi[mask&border.astype(bool)]=(245,245,245)
        cv.drawContours(output,cell_contours,-1,(245,245,245),1)
    return result,output,request[2:]


class ElaRenderer:
    """Keep segmentation across display-only changes; one serial LatestJob."""
    def __init__(self):self.key=None;self.base=None;self.result=None;self.segmentations=0
    def __call__(self,request):
        key=(id(request[1]),request[2],request[3],tuple(request[6:8]))
        if key!=self.key:
            result=segment(request[1],request[2],request[3],energy_thresholds=request[6:8] or None)
            self.key=key;self.base=request[1];self.result=result;self.segmentations+=1
        return render(request,self.result)


class ElaBiomesPanel(ToolWidget):
    def __init__(self,image,filename=None,parent=None):
        super().__init__(parent);self.image=image;self.engine=ElaBiomeEngine(image,filename)
        self.base=None;self.base_key=None;self.result=None;self.cancel=Event();self.started=False
        self.quality=QSpinBox();self.quality.setRange(0,100);self.quality.setSpecialValueText('Auto JPEG');self.quality.setValue(0);self.quality.setMinimumWidth(140)
        self.quality.setToolTip('Auto : table JPEG la plus proche ; sans table exploitable, référence 75. Vérification à trois qualités voisines.')
        self.block=QComboBox();self.block.addItems(['16','32','64','96']);self.block.setCurrentText(str(DEFAULT_BLOCK))
        self.threshold=QDoubleSpinBox();self.threshold.setRange(1,20);self.threshold.setSingleStep(.25);self.threshold.setValue(DEFAULT_THRESHOLD)
        self.threshold.setToolTip('Distance robuste entre profils comparables ; ce seuil exploratoire n’est pas une probabilité de retouche.')
        self.minimum=QSpinBox();self.minimum.setRange(1,1000);self.minimum.setValue(DEFAULT_MINIMUM_CELLS)
        self.mode=QComboBox();self.mode.addItems(['Biomes sur ELA','Biomes sur image','Biomes seuls','ELA linéaire','Écarts','Zones comparables','Low ELA energy','High ELA energy'])
        self.ghost=QCheckBox('JPEG Ghosts');self.ghost.setChecked(True)
        self.all_grids=QCheckBox('All JPEG grids');self.all_grids.setChecked(False)
        self.all_grids.setToolTip('Test all 64 JPEG grid positions.')
        self.background=QCheckBox('Energy disparities');self.background.setChecked(True)
        self.background.setToolTip('Map low and high residual energy relative to the central histogram levels.')
        self.run=QPushButton('Analyser');self.save=QPushButton('Exporter NPZ');self.save.setEnabled(False)
        self.status=QLabel('ELA linéaire · Scale 50 % · Contrast 20 % · Biomes expérimentaux');self.status.setWordWrap(True)
        self.note=QLabel('ELA linéaire · Scale 50 % · Contrast 20 %\nBiomes expérimentaux = profils ELA atypiques, sans attribution de retouche. Gris dans les cartes : comparaison insuffisante.');self.note.setWordWrap(True)
        self.legend=QListWidget();self.legend.setMinimumWidth(210);self.legend.setMaximumWidth(340)
        self.viewer=ImageViewer(image,image)
        self.split=QSplitter(Qt.Horizontal);self.split.addWidget(self.viewer);self.split.addWidget(self.legend);self.split.setStretchFactor(0,1)
        layout=QVBoxLayout(self);layout.setContentsMargins(6,6,6,6);layout.setSpacing(4)
        self.energy_visible=QCheckBox('Energy biomes');self.energy_visible.setChecked(True)
        self.legacy_visible=QCheckBox('Cell biomes (legacy)');self.legacy_visible.setChecked(True)
        histogram_row,(self.histogram_low,self.histogram_high)=histogram_controls()
        self.auto_thresholds=QCheckBox('Auto thresholds');self.auto_thresholds.setChecked(True)
        self.auto_profile=QComboBox();self.auto_profile.addItem('Conservative','standard')
        energy_row,(self.shadow_threshold,self.highlight_threshold)=energy_controls()
        self.settings=ElaSettings(self.mode,self.auto_profile,self.auto_thresholds,histogram_row,energy_row,
            [('Qualité',self.quality),('Cellule (px)',self.block),('Seuil',self.threshold),('Cellules / biome',self.minimum)],
            (self.ghost,self.all_grids,self.background),(self.energy_visible,self.legacy_visible),(self.run,self.save))
        layout.addWidget(self.settings)
        layout.addWidget(self.status);layout.addWidget(self.split,1);layout.addWidget(self.note)
        self.job=LatestJob(self,partial(analyze,self.engine),delay=120)
        self.renderer=ElaRenderer();self.draw=LatestJob(self,self.renderer,delay=50);self.export_job=LatestJob(self,export,delay=0)
        self.job.result.connect(self.complete);self.job.failed.connect(self.failed)
        self.draw.result.connect(self.painted);self.draw.failed.connect(self.failed)
        self.export_job.failed.connect(self.status.setText);self.export_job.result.connect(lambda p:self.status.setText(f'Exporté : {p}'))
        self.run.clicked.connect(self.start);self.save.clicked.connect(self.export_data)
        self.ghost.toggled.connect(self.changed);self.all_grids.toggled.connect(self.changed)
        self.background.toggled.connect(self.changed)
        self.quality.valueChanged.connect(self.changed);self.block.currentIndexChanged.connect(self.changed)
        self.auto_thresholds.toggled.connect(self.changed)
        self.auto_profile.currentIndexChanged.connect(self.profile_changed)
        for slider in (self.histogram_low,self.histogram_high,self.shadow_threshold,self.highlight_threshold):
            slider.sliderPressed.connect(self.take_manual_control)
        self.histogram_low.valueChanged.connect(self.manual_bounds);self.histogram_high.valueChanged.connect(self.manual_bounds)
        self.shadow_threshold.valueChanged.connect(self.manual_deviation);self.highlight_threshold.valueChanged.connect(self.manual_deviation)
        self.threshold.valueChanged.connect(self.refilter);self.minimum.valueChanged.connect(self.refilter)
        self.mode.currentIndexChanged.connect(self.refilter);self.legend.currentRowChanged.connect(self.redraw)
        self.energy_visible.toggled.connect(self.refilter);self.legacy_visible.toggled.connect(self.refilter)
        self.profiles=ElaProfiles(self,self.auto_profile,self.auto_thresholds,(self.histogram_low,self.histogram_high,self.shadow_threshold,self.highlight_threshold),self.changed,self.settings.toolbar)
        self.profiles.editStarted.connect(self.take_manual_control)
        self.profiles.apply()
        QTimer.singleShot(0, self.activate)

    def params(self):return int(choice(self.block)),self.quality.value(),self.ghost.isChecked(),self.all_grids.isChecked(),self.background.isChecked(),self.histogram_low.value()/1000,self.histogram_high.value()/1000,self.auto_thresholds.isChecked(),compute_profile(self.auto_profile)
    def current(self):return self.base is not None and self.base_key==self.params() and not self.job.is_busy
    def selection(self):
        item=self.legend.currentItem()
        return int(item.data(Qt.UserRole) or 0) if item is not None else 0
    def activate(self):
        if not self.started and not self.job.closed:self.queue_analysis()
    def profile_changed(self,*_):
        self.profiles.apply()
    def take_manual_control(self):
        was_auto=self.auto_thresholds.isChecked()
        self.profiles.manual()
        if was_auto:
            if self.job.is_busy:self.changed()
            elif self.base is not None:self.base_key=self.params()
    def manual_bounds(self,*_):
        self.profiles.manual()
        self.changed()
    def manual_deviation(self,*_):
        self.profiles.manual()
        if self.base is not None and not self.job.is_busy:
            self.base={**self.base,'metadata':{**self.base['metadata'],'energy':{**self.base['metadata'].get('energy',{}),'automatic':None}}}
            self.base_key=self.params();self.refilter()
        else:self.changed()
    def changed(self,*_):
        self.queue_analysis()
    def queue_analysis(self):
        if self.job.closed:return
        self.started=True;self.cancel.set();self.cancel=Event()
        self.draw.invalidate();self.save.setEnabled(False);self.viewer.set_busy(True)
        self.run.setText('Annuler');self.run.setEnabled(True)
        self.status.setText('Analyse des profils ELA et des zones comparables…')
        # LatestJob replaces pending work and rejects results of older requests.
        # Controls remain editable while a previous calculation winds down.
        self.job.request((self.params(),self.cancel))
    def start(self):
        if self.job.is_busy:
            self.cancel.set();self.job.invalidate();self.draw.invalidate()
            self.run.setText('Analyser');self.run.setEnabled(True)
            self.viewer.set_busy(not self.current());self.status.setText('Annulé.')
            if self.current():self.redraw()
            return
        self.queue_analysis()
    def complete(self,answer):
        params,base=answer;self.run.setText('Analyser');self.run.setEnabled(True);self.quality.setEnabled(True);self.block.setEnabled(True)
        if base is None:
            self.viewer.set_busy(not self.current());self.status.setText('Annulé.')
            if self.current():self.redraw()
            return
        if params!=self.params():return
        if self.auto_thresholds.isChecked():apply_automatic(base,(self.histogram_low,self.histogram_high,self.shadow_threshold,self.highlight_threshold))
        self.base=base;self.base_key=self.params();self.refilter()
    def failed(self,error):
        self.run.setText('Analyser');self.run.setEnabled(True);self.quality.setEnabled(True);self.block.setEnabled(True)
        self.save.setEnabled(False);self.viewer.set_busy(True);self.status.setText(error)
    def refilter(self,*_):
        self.legend.blockSignals(True);self.legend.clear();self.legend.addItem('Tous les biomes');self.legend.setCurrentRow(0);self.legend.blockSignals(False)
        self.redraw()
    def display_settings(self):
        return (self.threshold.value(),self.minimum.value(),self.mode.currentIndex(),self.selection(),self.shadow_threshold.value()/10,self.highlight_threshold.value()/10,(self.energy_visible.isChecked(),self.legacy_visible.isChecked()))
    def redraw(self,*_):
        if not self.current():return
        self.save.setEnabled(False);self.viewer.set_busy(True)
        self.draw.request((self.image,self.base,*self.display_settings()))
    def painted(self,answer):
        if not self.current():return
        result,output,settings=answer
        if settings!=self.display_settings():return
        self.result=result;self.viewer.update_processed(output);self.viewer.set_busy(False);self.save.setEnabled(True)
        regions=result['metadata']['regions']
        kind='low' if self.mode.currentIndex()==6 else 'high' if self.mode.currentIndex()==7 else None
        listed=[r for r in regions if (kind is None or r.get('kind')==kind) and (self.energy_visible.isChecked() if 'kind' in r else self.legacy_visible.isChecked())]
        if self.mode.currentIndex() in (3,4,5):listed=[]
        identifiers=[0]+[r['id'] for r in listed]
        if (self.legend.count()!=len(listed)+1 or getattr(self,'_legend_ids',None)!=identifiers):
            from PySide6.QtGui import QColor,QIcon,QPixmap
            self.legend.blockSignals(True);self.legend.clear();self.legend.addItem('Tous les biomes')
            names={'luminance':'intensité','chroma_red':'couleur rouge','chroma_blue':'couleur bleue','grain_1px':'grain 1 px','grain_2px':'grain 2 px'}
            for r in listed:
                description=t(r['sources'][0]) if 'kind' in r else names[r['dominant_descriptor']]
                sources='' if 'kind' in r else ' · '+', '.join(r.get('sources', ['ELA']))
                self.legend.addItem(f"Biome {r['id']} · {r['pixels']:,} px\nÉcart {r['score']:.2f} · {description}{sources}")
                self.legend.item(self.legend.count()-1).setData(Qt.UserRole,r['id'])
                swatch=QPixmap(14,14);swatch.fill(QColor(*(energy_color(r) if 'kind' in r else color(r['id']))[::-1]))
                self.legend.item(self.legend.count()-1).setIcon(QIcon(swatch))
            self._legend_ids=identifiers
            self.legend.setCurrentRow(identifiers.index(settings[3]) if settings[3] in identifiers else 0);self.legend.blockSignals(False)
        m=result['metadata'];coverage=100*np.count_nonzero(result['supported'])*m['block']**2/(self.image.shape[0]*self.image.shape[1])
        origin='tables JPEG' if m['quality_origin']=='jpeg_tables' else 'référence manuelle' if m['quality_origin']=='manual' else 'repli sans tables JPEG'
        if m['quality_origin']=='jpeg_tables' and m['table_deviation']:
            origin='tables JPEG approchées'
        self.status.setText(f"{len(listed)} biomes · Q {m['quality']} ({origin}) · cellules {m['block']} px · {coverage:.1f} % comparables · {m['seconds']:.2f} s")
    def export_data(self):
        if not self.save.isEnabled():return
        path,_=QFileDialog.getSaveFileName(self,t('Exporter les biomes ELA'),'ela-biomes.npz',t('NumPy (*.npz)'))
        if path:
            result={**self.result,'metadata':{**self.result['metadata'],'profile':self.profiles.snapshot(),'display':dict(energy_biomes=self.energy_visible.isChecked(),legacy_biomes=self.legacy_visible.isChecked())}}
            self.export_job.request((str(Path(path).with_suffix('.npz')),result))
    def shutdown(self):
        self.cancel.set();super().shutdown()
