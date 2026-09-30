from gui.sherloq_app.ui.localization import choice, localized_data, t
"""Copy-Move Forgery 2: bounded search, arbitrary ROIs and biome exploration."""
from functools import partial
from threading import Event
import json,math,os,tempfile,hashlib
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt,QObject,Signal,QEvent,QAbstractListModel,QModelIndex,QItemSelectionModel,QSignalBlocker
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QLabel,QPushButton,QComboBox,QSpinBox,QDoubleSpinBox,QCheckBox,QSlider,QListView,QAbstractItemView,QVBoxLayout,QHBoxLayout,QGridLayout,QSplitter,QWidget,QProgressBar,QFileDialog,QTabWidget)
from gui.sherloq_app.core.cloning2 import Cloning2Engine,ALGORITHMS,COMBINED,EXTENDED,SYMMETRIC,EXTENDED_SYMMETRIC,PANELS_TEXT,render
from gui.sherloq_app.core.jpeg_curve import Cancelled
from gui.sherloq_app.ui.viewer import ImageViewer
from gui.sherloq_app.ui.selection_view import SelectionView
from gui.sherloq_app.ui.jobs import LatestJob
from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.ui.check_list import CheckList
from gui.sherloq_app.core.auto_zones import detect_panels,enclosing,diagonal,distance_policy


class Progress(QObject):
    changed=Signal(object,int,str)


def analyze(engine,updates,request):
    params,regions,compare,event=request
    from gui.sherloq_app.core.model_store import Store,feature_for,Cancelled as DownloadCancelled
    try:
        feature=feature_for('copy_move',params[0])
        if feature!='none':
            Store().ensure(feature,event.is_set,lambda n,total,text:updates.changed.emit(event,100*n//max(1,total),text))
        return engine.analyze(params,regions,compare,event.is_set,lambda n,s:updates.changed.emit(event,n,s))
    except DownloadCancelled:return None
    except Cancelled:return None


def draw(image,request):
    result,style,event=request
    try:return render(image,result,style,event.is_set)
    except Cancelled:return None


def write_export(request):
    """Write an immutable analysis/display snapshot off the GUI thread."""
    filename,r,style,visible,image,cancel=request
    if cancel.is_set():return None
    data=dict(version=2,algorithm=r['params'][0],parameters=r['params'],zones=r['regions'],compare=r['compare'],
              image_shape=image.shape,decoded_bgr8_sha256=hashlib.sha256(memoryview(np.ascontiguousarray(image))).hexdigest(),
              points=r['points'].tolist(),pairs=r['pairs'].tolist(),pair_columns=['point_a','point_b','descriptor_distance','length_px'],
              biomes=[g.tolist() for g in r['groups']],colors_bgr=r['colors'].tolist(),biome_colors_bgr=r['bases'],geometric_models=r.get('models',()),source_algorithms=r.get('source_algorithms'),group_algorithms=r.get('group_algorithms'),pair_algorithms=r.get('pair_algorithms',np.empty(0,np.uint8)).tolist(),workers_per_algorithm=r.get('workers_per_algorithm'),display=style,visible_biomes=visible,
              distance_policy=r.get('distance_policy'),zone_configuration=r.get('zone_configuration'),preprocessing=r.get('preprocessing'),biome_partitions=r.get('biome_partitions'),
              self_match_filter=r.get('self_match_filter'),rejected_biomes=r.get('rejected_biomes',()),
              feature_policy=r.get('feature_policy'),backend=r.get('backend','cpu'),descriptor_backend=r.get('descriptor_backend'),descriptor_backends=r.get('descriptor_backends'),dense_correspondences=r.get('dense_count'),dense_consistent_correspondences=r.get('dense_consistent_count'),
              limits='Potential correspondences, not proof of forgery. Biome hulls are not exact segmentations.')
    if r.get('mirror_policy'):
        data.update(mirror_policy=r['mirror_policy'],group_variants=r['group_variants'])
    if r.get('extension'):
        data.update(extension=r['extension'],extension_bins=r['extension_bins'],
                    group_frames=r['group_frames'],orientation_energy_ratio=r['orientation_energy_ratio'],
                    extension_geometry=r['extension_geometry'],frame_workers=r['frame_workers'],
                    extension_detail=r.get('extension_detail'))
    temporary=None
    try:
        if Path(filename).suffix.lower()=='.npz':
            arrays=dict(points=r['points'],pairs=r['pairs'],colors_bgr=r['colors'])
            descriptions=[]
            for i,field in enumerate(r.get('dense_maps',())):
                for key,value in field.items():
                    if isinstance(value,np.ndarray):arrays[f'{key}_{i}']=value
                descriptions.append({k:v for k,v in field.items() if not isinstance(v,np.ndarray)})
            data['dense_fields']=descriptions;arrays['metadata_json']=np.asarray(json.dumps(data,ensure_ascii=False))
            with tempfile.NamedTemporaryFile('wb',dir=Path(filename).parent,delete=False) as stream:
                temporary=stream.name;np.savez_compressed(stream,**arrays)
        else:
            with tempfile.NamedTemporaryFile('w',dir=Path(filename).parent,delete=False,encoding='utf-8') as stream:
                temporary=stream.name;json.dump(data,stream,ensure_ascii=False)
        if cancel.is_set():return None
        os.replace(temporary,filename)
        return filename
    finally:
        if temporary and os.path.exists(temporary):os.unlink(temporary)


def auto_detect(request):
    image,event=request
    try:return detect_panels(image,event.is_set)
    except Cancelled:return None


class ZoneModel(QAbstractListModel):
    changed=Signal()
    def __init__(self,parent):super().__init__(parent);self.regions=();self.disabled=set();self.envelope=None
    def rowCount(self,parent=QModelIndex()):return 0 if parent.isValid() else len(self.regions)
    @localized_data
    def data(self,index,role=Qt.DisplayRole):
        if not index.isValid():return None
        i=index.row();name='Ensemble' if self.regions[i]==self.envelope else f'Zone {i+1}'
        if role==Qt.DisplayRole:return f'{name} · diagonale {diagonal(self.regions[i]):.0f} px'
        if role==Qt.CheckStateRole:return Qt.Unchecked if i in self.disabled else Qt.Checked
        if role==Qt.ToolTipRole:return 'Coche : inclure dans la recherche. Clic : isoler le contour ; clic dans l’image : tous les contours actifs.'
    def flags(self,index):return super().flags(index)|Qt.ItemIsUserCheckable
    def setData(self,index,value,role=Qt.EditRole):
        if role!=Qt.CheckStateRole or not index.isValid():return False
        if value in (Qt.Checked,Qt.Checked.value):self.disabled.discard(index.row())
        else:self.disabled.add(index.row())
        self.dataChanged.emit(index,index,[Qt.CheckStateRole]);self.changed.emit();return True
    def update(self,regions,envelope):
        disabled={i for i in self.disabled if i<len(regions) and i<len(self.regions) and regions[i]==self.regions[i]}
        self.beginResetModel();self.regions=regions;self.envelope=envelope;self.disabled=disabled;self.endResetModel()


class BiomeModel(QAbstractListModel):
    changed=Signal()
    def __init__(self,parent):super().__init__(parent);self.rows=[];self.bases=();self.hidden=set();self.algorithms=()
    def rowCount(self,parent=QModelIndex()):return 0 if parent.isValid() else len(self.rows)
    @localized_data
    def data(self,index,role=Qt.DisplayRole):
        if not index.isValid():return None
        i,count,total=self.rows[index.row()]
        if role==Qt.DisplayRole:return (self.algorithms[i].replace('PatchMatch ','')+' · ' if self.algorithms else '')+f'Biome {i+1} · {count}/{total} liens'
        if role==Qt.DecorationRole:return QColor(*self.bases[i][::-1])
        if role==Qt.CheckStateRole:return Qt.Unchecked if i in self.hidden else Qt.Checked
        if role==Qt.ToolTipRole:return 'Même couleur : deux ensembles de points associés. Enveloppes indicatives, pas une segmentation prouvée.'
    def flags(self,index):return super().flags(index)|Qt.ItemIsUserCheckable
    def setData(self,index,value,role=Qt.EditRole):
        if role!=Qt.CheckStateRole or not index.isValid():return False
        i=self.rows[index.row()][0]
        if value in (Qt.Checked,Qt.Checked.value):self.hidden.discard(i)
        else:self.hidden.add(i)
        self.dataChanged.emit(index,index,[Qt.CheckStateRole]);self.changed.emit();return True
    def update(self,rows,bases):self.beginResetModel();self.rows=rows;self.bases=bases;self.endResetModel()


class Cloning2Widget(ToolWidget):
    def __init__(self,image,parent=None):
        super().__init__(parent);self.image=image;self.engine=Cloning2Engine(image);self.result=None;self.dirty=True;self.cancel=Event();self.draw_cancel=Event();self.chosen=set();self.visible=[];self.envelope=None;self.auto_cancel=Event()
        self.diagonal=math.hypot(*image.shape[:2])
        self.viewer=ImageViewer(image,image,view_class=SelectionView)
        self.algorithm=QComboBox();self.algorithm.addItems(ALGORITHMS)
        self.algorithm.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon);self.algorithm.setMinimumContentsLength(16)
        from gui.sherloq_app.ui.extensions import mark_combo
        mark_combo(self.algorithm,[name for name in ALGORITHMS if name not in ('AKAZE','BRISK','ORB')],green_items=['SIFT + G2NN + RANSAC',PANELS_TEXT,EXTENDED,SYMMETRIC,EXTENDED_SYMMETRIC])
        self.limit=QSpinBox();self.limit.setRange(100,20000);self.limit.setValue(6000);self.limit.setSingleStep(500)
        self.radius=QDoubleSpinBox();self.radius.setRange(.01,self.diagonal);self.radius.setDecimals(2);self.radius.setValue(min(600,self.diagonal))
        self.auto_radius=QCheckBox('Automatique par zone');self.auto_radius.setChecked(True)
        self.ignore_gap=QCheckBox('Ignorer les intervalles entre zones');self.ignore_gap.setChecked(True)
        self.units=QComboBox();self.units.addItems(['px','%']);self._unit=0
        self.minimum=QDoubleSpinBox();self.minimum.setRange(0,self.diagonal);self.minimum.setValue(10);self.minimum.setSuffix(' px')
        self.threshold=QDoubleSpinBox();self.threshold.setRange(.001,1);self.threshold.setDecimals(3);self.threshold.setSingleStep(.025);self.threshold.setValue(.3)
        self.tolerance=QDoubleSpinBox();self.tolerance.setRange(1,self.diagonal);self.tolerance.setValue(min(50,self.diagonal));self.tolerance.setSuffix(' px')
        self.support=QSpinBox();self.support.setRange(1,100000);self.support.setValue(3)
        self.geometry=QComboBox();self.geometry.addItems(['Aucune','Similitude','Affine','Homographie']);self.geometry.setCurrentIndex(1)
        mark_combo(self.geometry,['Similitude','Affine','Homographie'])
        self.geometry_error=QDoubleSpinBox();self.geometry_error.setRange(.1,100);self.geometry_error.setValue(3);self.geometry_error.setSuffix(' px')
        self.geometry_support=QSpinBox();self.geometry_support.setRange(4,10000);self.geometry_support.setValue(6)
        self.geometry.currentIndexChanged.connect(self.invalidate);self.geometry_error.valueChanged.connect(self.invalidate);self.geometry_support.valueChanged.connect(self.invalidate)
        self.patch=QSpinBox();self.patch.setRange(2,32);self.patch.setValue(8)
        self.iterations=QSpinBox();self.iterations.setRange(1,32);self.iterations.setValue(8)
        self.reflection=QCheckBox('Miroir dense')
        self.independent_sift=QCheckBox('Per-zone SIFT');self.independent_sift.toggled.connect(self.invalidate)
        self.independent_sift.setToolTip('Extract each selected zone independently, with its own point limit, scale and contrast. This can increase computation substantially.')
        from gui.sherloq_app.ui.extensions import ExtensionOutline,GREEN
        self.independent_sift.extension_outline=ExtensionOutline(self.independent_sift);self.independent_sift.extension_outline.color=GREEN
        self.cpu=QCheckBox('CPU');self.cpu.setEnabled(False);self.cpu.toggled.connect(self.invalidate)
        self.texture=QDoubleSpinBox();self.texture.setRange(0,255);self.texture.setValue(2)
        for c in (self.patch,self.iterations,self.texture):c.valueChanged.connect(self.invalidate)
        self.reflection.toggled.connect(self.invalidate)
        self.mode=QComboBox();self.mode.addItems(['Déplacer','Rectangle','Lasso','Polygone'])
        self.mode.currentIndexChanged.connect(lambda i:self.viewer.view.set_mode(('Pan','Rectangle','Lasso','Polygon')[i]))
        self.undo=QPushButton('Retirer la dernière zone');self.undo.clicked.connect(self.viewer.view.undo_region)
        self.clear=QPushButton('Effacer les zones');self.clear.clicked.connect(self.viewer.view.clear_regions)
        self.zone_label=QLabel('Image entière')
        self.auto_zones=QPushButton('Détecter les sous-images');self.auto_zones.clicked.connect(self.detect_zones)
        self.search=QPushButton('Rechercher');self.compare=QPushButton('Comparer 2 zones');self.compare.setEnabled(False)
        self.cancel_button=QPushButton('Annuler');self.cancel_button.setEnabled(False)
        self.search.clicked.connect(lambda:self.start(False));self.compare.clicked.connect(lambda:self.start(True));self.cancel_button.clicked.connect(self.stop)
        self.circles=QCheckBox('Cercles');self.circles.setChecked(False)
        self.lines=QCheckBox('Lignes');self.lines.setChecked(False)
        self.points=QCheckBox('Points centraux');self.areas=QCheckBox('Biomes');self.areas.setChecked(True)
        self.show_zones=QCheckBox('Zones de recherche');self.show_zones.setChecked(True)
        self.show_text=QCheckBox('Text exclusions');self.show_text.setChecked(False)
        self.show_zones.toggled.connect(self.toggle_zones)
        self.preset=QPushButton('Biomes seuls');self.preset.clicked.connect(self.biomes_only)
        self.low=QSlider(Qt.Horizontal);self.high=QSlider(Qt.Horizontal)
        self.low_spin=QDoubleSpinBox();self.high_spin=QDoubleSpinBox()
        for slider in (self.low,self.high):slider.setRange(0,math.ceil(self.diagonal))
        for spin in (self.low_spin,self.high_spin):spin.setRange(0,math.ceil(self.diagonal));spin.setDecimals(1);spin.setSuffix(' px')
        self.high.setValue(math.ceil(self.diagonal));self.high_spin.setValue(math.ceil(self.diagonal))
        self.low.setValue(min(10,math.ceil(self.diagonal)));self.low_spin.setValue(min(10,math.ceil(self.diagonal)))
        self.low.valueChanged.connect(lambda v:self.low_spin.setValue(v));self.high.valueChanged.connect(lambda v:self.high_spin.setValue(v))
        self.low_spin.valueChanged.connect(lambda v:self.length_changed(True,v));self.high_spin.valueChanged.connect(lambda v:self.length_changed(False,v))
        self.legend=CheckList();self.model=BiomeModel(self.legend);self.legend.setModel(self.model);self.legend.selectionModel().selectionChanged.connect(self.choose)
        self.model.changed.connect(self.redraw)
        self.zone_list=CheckList();self.zone_model=ZoneModel(self.zone_list);self.zone_list.setModel(self.zone_model)
        self.zone_model.changed.connect(self.zones_toggled);self.zone_list.selectionModel().selectionChanged.connect(self.focus_zones)
        self.zone_list.focusCleared.connect(self.focus_zones);self.legend.focusCleared.connect(self.show_all_biomes)
        self.viewer.view.interactionStarted.connect(self.clear_focus)
        self.all_biomes=QPushButton('Tous les biomes cochés');self.all_biomes.clicked.connect(self.show_all_biomes)
        self.export=QPushButton('Exporter les correspondances');self.export.setEnabled(False);self.export.clicked.connect(self.export_data)
        self.status=QLabel('Choisir des zones ou rechercher dans toute l’image. Les détections sont des correspondances potentielles.');self.status.setWordWrap(True)
        self.progress=QProgressBar();self.progress.setRange(0,100)
        self.updates=Progress();self.updates.changed.connect(self.progressed)
        self.job=LatestJob(self,partial(analyze,self.engine,self.updates),delay=0);self.job.result.connect(self.ready);self.job.failed.connect(self.failed);self.job.busy.connect(self.busy)
        self.export_cancel=Event();self.export_job=LatestJob(self,write_export,delay=0);self.export_job.result.connect(self.exported);self.export_job.failed.connect(self.export_failed)
        self.render_job=LatestJob(self,partial(draw,image),delay=60);self.render_job.result.connect(self.rendered);self.render_job.failed.connect(self.failed)
        self.auto_job=LatestJob(self,auto_detect,delay=0);self.auto_job.result.connect(self.auto_ready);self.auto_job.failed.connect(self.auto_failed)
        self.viewer.view.regionsChanged.connect(self.regions_changed)
        self.auto_radius.toggled.connect(self.radius_mode_changed);self.ignore_gap.toggled.connect(self.invalidate)
        for control in (self.limit,self.radius,self.minimum,self.threshold,self.tolerance):control.valueChanged.connect(self.invalidate)
        self.algorithm.currentIndexChanged.connect(self.algorithm_changed);self.units.currentIndexChanged.connect(self.unit_changed)
        self.support.valueChanged.connect(self.redraw)
        for box in (self.circles,self.lines,self.points,self.areas,self.show_text):box.toggled.connect(self.redraw)
        settings=QGridLayout();settings.setVerticalSpacing(2)
        radius_controls=QWidget();radius_layout=QHBoxLayout(radius_controls);radius_layout.setContentsMargins(0,0,0,0);radius_layout.addWidget(self.radius);radius_layout.addWidget(self.units)
        self.units.setToolTip('Unit: pixels or percentage of the image diagonal.')
        for col,(label,widget) in enumerate([('Détecteur',self.algorithm),('Points / links',self.limit),('Écartement maximum',radius_controls)]):settings.addWidget(QLabel(label),0,2*col);settings.addWidget(widget,0,2*col+1)
        for col,(label,widget) in enumerate([('Séparation minimum',self.minimum),('Tolérance descripteur',self.threshold),('Proximité des biomes',self.tolerance)]):settings.addWidget(QLabel(label),1,2*col);settings.addWidget(widget,1,2*col+1)
        for col,(label,widget) in enumerate([('Géométrie',self.geometry),('Erreur maximum',self.geometry_error),('Témoins minimum',self.geometry_support)]):settings.addWidget(QLabel(label),2,2*col);settings.addWidget(widget,2,2*col+1)
        for col,(label,widget) in enumerate([('Rayon / bin dense',self.patch),('Itérations denses',self.iterations),('Texture minimum',self.texture)]):settings.addWidget(QLabel(label),3,2*col);settings.addWidget(widget,3,2*col+1)
        settings.addWidget(QLabel('Points minimum / côté'),4,0);settings.addWidget(self.support,4,1)
        settings.addWidget(self.reflection,4,2,1,2);settings.addWidget(self.cpu,4,4)
        settings.addWidget(self.independent_sift,5,0,1,2)
        radius_label=settings.itemAtPosition(0,4).widget()
        font=radius_label.font();font.setBold(True);radius_label.setFont(font);self.radius.setFont(font);self.radius.setMinimumWidth(130)
        settings.addWidget(self.auto_radius,5,2,1,2);settings.addWidget(self.ignore_gap,5,4,1,2)
        self.patch.setToolTip('Zernike : rayon du voisinage. SIFT dense : taille des cellules. Pixels originaux, sans réduction.')
        self.texture.setToolTip('Écart-type minimum du voisinage, sur la moyenne des canaux 8 bits ; zéro désactive ce filtre.')
        self.limit.setToolTip('Détecteurs : maximum de points extraits. PatchMatch : maximum de liens dessinés par algorithme, échantillonnés sur le champ dense complet. Mode combiné : jusqu’à quatre tâches par moteur, sur les zones indépendantes.')
        self.geometry.setToolTip('Vérifie une ou plusieurs transformations par biome. Aucune conserve tous les liens descriptifs. Affine autorise aussi les réflexions si les descripteurs les retrouvent.')
        self.reflection.setToolTip('Horizontal reflection matching instead of ordinary matching. Use Affine geometry for reflections. Dense SIFT uses a fixed scale and orientation.')
        selections=QVBoxLayout();selections.setSpacing(2)
        for widgets in ((self.mode,self.auto_zones,self.undo,self.clear),(self.zone_label,self.search,self.compare,self.cancel_button)):
            row=QHBoxLayout()
            for w in widgets:row.addWidget(w)
            selections.addLayout(row)
        styles=QHBoxLayout()
        for w in (self.circles,self.lines,self.points,self.areas,self.preset,self.show_zones,self.show_text):styles.addWidget(w)
        styles.addStretch()
        lengths=QGridLayout();lengths.addWidget(QLabel('Longueur affichée minimum'),0,0);lengths.addWidget(self.low,0,1);lengths.addWidget(self.low_spin,0,2);lengths.addWidget(QLabel('maximum'),0,3);lengths.addWidget(self.high,0,4);lengths.addWidget(self.high_spin,0,5)
        side=QWidget();side_layout=QVBoxLayout(side);self.side_tabs=QTabWidget()
        biome_page=QWidget();biome_layout=QVBoxLayout(biome_page);biome_layout.addWidget(self.legend);biome_layout.addWidget(self.all_biomes)
        zone_page=QWidget();zone_layout=QVBoxLayout(zone_page);zone_layout.addWidget(self.zone_list)
        all_zones=QPushButton('Toutes les zones cochées');all_zones.clicked.connect(self.clear_focus);zone_layout.addWidget(all_zones)
        self.delete_zone=QPushButton('Supprimer la zone sélectionnée');self.delete_zone.setEnabled(False);self.delete_zone.clicked.connect(self.remove_selected_zone);zone_layout.addWidget(self.delete_zone)
        self.side_tabs.addTab(biome_page,'Biomes');self.side_tabs.addTab(zone_page,'Zones');side_layout.addWidget(self.side_tabs,1);side_layout.addWidget(self.export)
        note=QLabel('Une famille de teintes par biome ; une nuance par lien, partagée par ses deux extrémités. Plus sombre : descripteurs plus proches. Les enveloppes sont indicatives.');note.setWordWrap(True);side_layout.addWidget(note)
        splitter=QSplitter();splitter.addWidget(self.viewer);splitter.addWidget(side);splitter.setStretchFactor(0,1);splitter.setSizes([1100,280])
        layout=QVBoxLayout(self);layout.addLayout(settings);layout.addLayout(selections);layout.addLayout(styles);layout.addLayout(lengths);layout.addWidget(splitter,1);layout.addWidget(self.progress);layout.addWidget(self.status)
        self.search.setToolTip('Recherche indépendante à l’intérieur de chaque zone. Aucune correspondance entre des zones différentes.')
        self.compare.setToolTip('Recherche uniquement entre les deux zones. Disponible avec exactement deux zones.')
        self.algorithm.setToolTip('XFeat conserve le prétraitement officiel : dimensions ramenées au multiple de 32 inférieur, puis coordonnées rétablies. ALIKED : grand côté ramené à 1024 px ; SIFT + LightGlue : pleine résolution et RootSIFT. Les matchers Glue utilisent uniquement les liens spatiaux admissibles.')
        for spin in (self.low_spin,self.high_spin):spin.setToolTip('Longueur entre centres ; les intervalles sont retranchés si l’option correspondante était activée lors de l’analyse.')
        self.minimum.setToolTip('Distance minimum entre centres, en pixels de l’image originale, appliquée avant l’appariement. Défaut : 5 px pour PatchMatch Zernike, 10 px pour les autres méthodes. XFeat/ALIKED : biomes recouvrants à 80 % écartés automatiquement ; les liens bruts restent exportables.')
        self.radius.setToolTip('Automatique : diagonale de chaque zone active ; l’ensemble utilise sa propre diagonale. En comparaison, les intervalles peuvent être retranchés sans déplacer les pixels.')
        self.ignore_gap.setToolTip('Ignore les bandes entre les boîtes des zones. Pour l’ensemble automatique : bandes hors de toutes les sous-images actives. Les pixels et les transformations géométriques restent dans l’image originale.')
        self.auto_zones.setToolTip('Remplace les sélections par les sous-images rectangulaires détectées et leur rectangle englobant. Aucun nombre de zones imposé.')
        self.threshold.setToolTip('SIFT/RootSIFT : distance L2 entre descripteurs normalisés. Autres : fraction de bits différents. Plus faible = plus strict.')
        self.tolerance.setToolTip('Deux liens appartiennent au même biome si leurs extrémités respectives sont proches, directement ou par une chaîne de liens.')
        self.mode.setToolTip('Lasso : relâcher pour fermer. Polygone : clic droit pour relier le dernier point au premier. Échap annule le tracé.')
        self.viewer.set_busy(True)
        self.radius_mode_changed()
        self.algorithm.setCurrentText('PatchMatch Zernike')
        self.status.setText('Choisir des zones ou rechercher dans toute l’image. Les détections sont des correspondances potentielles.')
        for widget in self.findChildren(QWidget):widget.installEventFilter(self)

    def eventFilter(self,watched,event):
        if event.type()==QEvent.MouseButtonPress and isinstance(watched,QWidget):
            if watched not in (getattr(self,'delete_zone',None),getattr(self,'export',None)) and not any(watched is view or view.isAncestorOf(watched) for view in (self.legend,self.zone_list)):
                self.clear_focus()
        return super().eventFilter(watched,event)

    def toggle_zones(self,checked):
        self.viewer.view.show_regions=checked;self.viewer.view.viewport().update()

    def parameters(self):
        radius=self.radius.value()*(self.diagonal/100 if self.units.currentIndex() else 1)
        return (choice(self.algorithm),self.limit.value(),radius,self.minimum.value(),self.threshold.value(),self.tolerance.value(),('None','Similarity','Affine','Homography')[self.geometry.currentIndex()],self.geometry_error.value(),self.geometry_support.value(),self.patch.value(),self.iterations.value(),self.reflection.isChecked(),self.texture.value(),self.cpu.isChecked(),self.auto_radius.isChecked(),self.ignore_gap.isChecked(),tuple(r for i,r in enumerate(self.viewer.view.snapshot()) if i in self.zone_model.disabled and r!=self.envelope),tuple(r for r in self.active_regions() if r!=self.envelope) if self.envelope in self.viewer.view.snapshot() else ()) + ((1,self.independent_sift.isChecked()) if choice(self.algorithm)=='SIFT + G2NN + RANSAC' else ())

    def algorithm_changed(self):
        dense=choice(self.algorithm).startswith('PatchMatch ') or choice(self.algorithm) in (EXTENDED,SYMMETRIC,EXTENDED_SYMMETRIC)
        for control in (self.patch,self.iterations,self.reflection,self.texture):control.setEnabled(dense)
        algorithm=choice(self.algorithm)
        previous=getattr(self,'_previous_algorithm',None)
        if algorithm in (SYMMETRIC,EXTENDED_SYMMETRIC) and previous not in (SYMMETRIC,EXTENDED_SYMMETRIC):
            self._manual_reflection=self.reflection.isChecked()
            with QSignalBlocker(self.reflection):self.reflection.setChecked(True)
        elif previous in (SYMMETRIC,EXTENDED_SYMMETRIC) and algorithm not in (SYMMETRIC,EXTENDED_SYMMETRIC):
            with QSignalBlocker(self.reflection):self.reflection.setChecked(self._manual_reflection)
        self._previous_algorithm=algorithm
        self.reflection.setEnabled(dense and algorithm not in (SYMMETRIC,EXTENDED_SYMMETRIC))
        self.independent_sift.setVisible(algorithm=='SIFT + G2NN + RANSAC')
        self.show_text.setVisible(algorithm==PANELS_TEXT)
        self.patch.setMinimum(3 if algorithm in ('PatchMatch SIFT',COMBINED,EXTENDED,SYMMETRIC,EXTENDED_SYMMETRIC) else 2)
        previous_default=getattr(self,'_spacing_default',10)
        self._spacing_default=5 if algorithm in ('PatchMatch Zernike',COMBINED,EXTENDED,SYMMETRIC,EXTENDED_SYMMETRIC) else 10
        # Follow algorithm defaults only while the values are still defaults;
        # preserve manually chosen search/display distances across switches.
        if self.minimum.value()==previous_default:self.minimum.setValue(self._spacing_default)
        previous_display=getattr(self,'_display_spacing_default',previous_default)
        self._display_spacing_default=0 if algorithm=='SIFT + G2NN + RANSAC' else self._spacing_default
        if self.low_spin.value()==previous_display:self.low_spin.setValue(self._display_spacing_default)
        with QSignalBlocker(self.cpu):self.cpu.setChecked(False)
        from gui.sherloq_app.core.metal_dense import sift_enabled
        self.cpu.setEnabled(algorithm in ('PatchMatch Zernike',COMBINED,EXTENDED,SYMMETRIC,EXTENDED_SYMMETRIC,'SIFT + G2NN + RANSAC',PANELS_TEXT) or (algorithm=='PatchMatch SIFT' and sift_enabled()) or algorithm.startswith(('XFeat','ALIKED')) or 'Glue' in algorithm)
        self.algorithm.setToolTip('Extended: adds quarter-turn and multiple-scale SIFT passes to the existing duo. More computation; arbitrary rotations are not covered.' if algorithm==EXTENDED else '')
        if algorithm==SYMMETRIC:self.algorithm.setToolTip('Keeps the normal duo and adds reflected copies, with geometric and pixel-detail checks.')
        if algorithm==EXTENDED_SYMMETRIC:self.algorithm.setToolTip('Keeps the extended duo and adds reflected copies at multiple scales, with geometric and pixel-detail checks.')
        if 'Glue' in algorithm:self.limit.setValue(min(2000,self.limit.value()))
        self.threshold.setValue(.9 if 'Glue' in algorithm else .7 if algorithm.startswith('ALIKED') or algorithm=='SIFT + G2NN + RANSAC' else .12 if algorithm in ('BRISK','ORB','AKAZE') else .3)
        self.threshold.setToolTip('Glue : 1 − confiance minimale. SIFT/RootSIFT/XFeat/ALIKED et dense : distance L2 normalisée. BRISK/ORB/AKAZE : fraction de bits différents.')
        self.limit.setToolTip('Point limit applies per zone with Per-zone SIFT enabled; otherwise keypoint methods share a global budget. Dense methods cap displayed links per pass.')
        if algorithm=='SIFT + G2NN + RANSAC':
            # User's 2026-09-28 fallback preset: speed improved, sensitivity not established.
            self.limit.setValue(20000);self.minimum.setValue(10);self.threshold.setValue(.725);self.tolerance.setValue(50)
            self.auto_radius.setChecked(False);self.ignore_gap.setChecked(False);self.units.setCurrentIndex(0);self.radius.setValue(min(self.diagonal,808.95))
            self.geometry.setCurrentIndex(2);self.geometry_support.setValue(10);self.geometry_error.setValue(5.)
            self.support.setValue(2);self.low_spin.setValue(0);self.high_spin.setValue(self.high_spin.maximum())
            self.threshold.setToolTip('G2NN : rapport de rupture entre les voisins autorisés (défaut 0,725). Profil classique, sans YOLO.')
        if algorithm==PANELS_TEXT:
            self.limit.setValue(6000);self.minimum.setValue(10);self.threshold.setValue(.725);self.tolerance.setValue(50)
            self.auto_radius.setChecked(True);self.ignore_gap.setChecked(False)
            self.geometry.setCurrentIndex(2);self.geometry_support.setValue(10);self.geometry_error.setValue(5.)
            self.support.setValue(2);self.low_spin.setValue(10);self.high_spin.setValue(self.high_spin.maximum())
            self.algorithm.setToolTip('SHERLOQ alternative: automatic panels when no zones are selected, text exclusion, per-zone SIFT and geometric verification. Not the luc_pub competition ensemble.')
        self.invalidate()

    def unit_changed(self,index):
        px=self.radius.value()*(self.diagonal/100 if self._unit else 1);self._unit=index
        with QSignalBlocker(self.radius):self.radius.setMaximum(100 if index else self.diagonal);self.radius.setValue(px/self.diagonal*100 if index else px)
        self.invalidate()

    def invalidate(self,*_):
        self.cancel.set();self.job.invalidate();self.draw_cancel.set();self.render_job.invalidate();self.dirty=True;self.export.setEnabled(False);self.viewer.set_busy(True);self.status.setText('Réglages modifiés — lancer Rechercher ou Comparer.')

    def active_regions(self):
        return tuple(r for i,r in enumerate(self.viewer.view.snapshot()) if i not in self.zone_model.disabled)

    def regions_changed(self,regions):
        self.auto_cancel.set();self.auto_job.invalidate();self.auto_zones.setEnabled(True)
        self.zone_model.update(regions,self.envelope);self.zones_toggled()

    def zones_toggled(self):
        view=self.viewer.view;view.enabled_regions=set(range(len(view.regions)))-self.zone_model.disabled
        view.region_names=['Ensemble' if r==self.envelope else f'Zone {i+1}' for i,r in enumerate(view.snapshot())]
        self.focus_zones();self.radius_mode_changed();n=len(self.active_regions())
        self.zone_label.setText(f'{n}/{len(view.regions)} zones actives' if view.regions else 'Image entière')
        self.compare.setEnabled(n==2 and not self.job.is_busy)
        self.search.setEnabled(not self.job.is_busy and (n>0 or not view.regions))

    def remove_selected_zone(self):
        selected=self.zone_list.selectionModel().selectedRows()
        if not selected:return
        index=selected[0].row();old=self.viewer.view.snapshot();remaining=tuple(r for i,r in enumerate(old) if i!=index)
        disabled={i-(i>index) for i in self.zone_model.disabled if i!=index}
        if old[index]==self.envelope:self.envelope=None
        # Preserve check states of all remaining zones, even after index shifts.
        self.zone_model.update(remaining,self.envelope);self.zone_model.disabled=disabled
        self.viewer.view.set_regions(remaining)

    def focus_zones(self,*_):
        self.viewer.view.focus_regions={i.row() for i in self.zone_list.selectionModel().selectedRows()}
        if hasattr(self,'delete_zone'):self.delete_zone.setEnabled(bool(self.viewer.view.focus_regions))
        self.viewer.view.viewport().update()

    def clear_focus(self):
        with QSignalBlocker(self.zone_list.selectionModel()):self.zone_list.clearSelection()
        self.focus_zones();self.show_all_biomes()

    def radius_mode_changed(self,*_):
        regions=self.active_regions();automatic=self.auto_radius.isChecked() and bool(regions)
        self.radius.setReadOnly(automatic);self.units.setEnabled(not automatic)
        if automatic:
            value=max(diagonal(r) for r in regions)
            with QSignalBlocker(self.radius):self.radius.setValue(value*100/self.diagonal if self.units.currentIndex() else value)
        self.invalidate()

    def detect_zones(self):
        self.auto_cancel.set();self.auto_cancel=Event();self.auto_zones.setEnabled(False)
        self.status.setText('Détection des sous-images rectangulaires…');self.auto_job.request((self.image,self.auto_cancel))

    def auto_ready(self,regions):
        self.auto_zones.setEnabled(True)
        if regions is None:return
        if not regions:self.status.setText('Aucune sous-image rectangulaire détectée ; sélections conservées.');return
        self.zone_model.disabled.clear();self.envelope=enclosing(regions)
        self.viewer.view.set_regions((*regions,self.envelope));self.side_tabs.setCurrentIndex(1)
        self.status.setText(f'{len(regions)} sous-images + 1 ensemble détectés. Rechercher analyse chaque zone ; l’ensemble permet aussi les liens entre sous-images.')

    def auto_failed(self,message):
        self.auto_zones.setEnabled(True);self.status.setText(message)

    def start(self,compare):
        self.dirty=True;self.cancel.set();self.cancel=Event();self.draw_cancel.set();self.render_job.invalidate();self.export.setEnabled(False);self.viewer.set_busy(True)
        regions=self.active_regions()
        if self.viewer.view.regions and not regions:
            self.status.setText('Aucune zone active.');return
        if regions and self.auto_radius.isChecked():
            value=distance_policy(self.parameters(),regions,compare)[0]
            with QSignalBlocker(self.radius):self.radius.setValue(value*100/self.diagonal if self.units.currentIndex() else value)
        self.job.request((self.parameters(),regions,compare,self.cancel))

    def stop(self):
        self.cancel.set();self.job.invalidate();self.status.setText('Annulé. Les étapes terminées restent en mémoire.')

    def busy(self,busy):
        self.cancel_button.setEnabled(busy);self.search.setEnabled(not busy and (bool(self.active_regions()) or not self.viewer.view.regions));self.compare.setEnabled(not busy and len(self.active_regions())==2)

    def progressed(self,event,n,text):
        if event is self.cancel and not event.is_set():self.progress.setValue(n);self.status.setText(text)

    def failed(self,message):
        self.status.setText(message);self.viewer.set_busy(True)

    def ready(self,result):
        if result is None:return
        prep=result.get('preprocessing',{})
        if prep.get('automatic_panels') and result['regions'] and not self.viewer.view.regions:
            self.envelope=prep['envelope'];self.zone_model.disabled.clear()
            self.viewer.view.set_regions(result['regions'])
        self.result={**result,'zone_configuration':dict(regions=self.viewer.view.snapshot(),disabled=sorted(self.zone_model.disabled),envelope=self.envelope)};self.dirty=False;self.chosen=set();self.model.hidden.clear();self.redraw()

    def length_changed(self,lower,value):
        slider=self.low if lower else self.high
        with QSignalBlocker(slider):slider.setValue(round(value))
        if lower and value>self.high_spin.value():self.high_spin.setValue(value)
        if not lower and value<self.low_spin.value():self.low_spin.setValue(value)
        self.redraw()

    def choose(self,*_):
        self.chosen={self.model.rows[i.row()][0] for i in self.legend.selectionModel().selectedRows()};self.redraw()

    def show_all_biomes(self):
        changed=bool(self.chosen)
        with QSignalBlocker(self.legend.selectionModel()):self.legend.clearSelection()
        self.chosen=set()
        if changed:self.redraw()

    def biomes_only(self):
        self.circles.setChecked(False);self.lines.setChecked(False);self.points.setChecked(False);self.areas.setChecked(True)

    def style(self):
        return (self.low_spin.value(),self.high_spin.value(),self.support.value(),tuple(sorted(self.chosen)),self.circles.isChecked(),self.lines.isChecked(),self.points.isChecked(),self.areas.isChecked(),tuple(sorted(self.model.hidden)),self.show_text.isChecked())

    def redraw(self,*_):
        if self.result is None or self.dirty:return
        self.draw_cancel.set();self.draw_cancel=Event();self.viewer.set_busy(True);self.export.setEnabled(False);self.render_job.request((self.result,self.style(),self.draw_cancel))

    def rendered(self,value):
        if value is None or self.dirty:return
        image,visible,rows=value;self.visible=visible;self.viewer.update_processed(image);self.viewer.set_busy(False);self.export.setEnabled(not self.export_job.is_busy)
        # Legend includes all supported biomes, even if a biome selection hides
        # them, so the user can change selection without rerunning detection.
        selection=self.legend.selectionModel()
        with QSignalBlocker(selection):
            self.model.algorithms=self.result.get('group_algorithms',());self.model.update(rows,self.result['bases'])
            for row,(i,_,_) in enumerate(rows):
                if i in self.chosen:selection.select(self.model.index(row),QItemSelectionModel.Select|QItemSelectionModel.Rows)
        r=self.result
        prefix=f"Champ dense : {r['dense_count']:,} liens, {r['dense_consistent_count']:,} cohérents ; aperçu {len(r['pairs']):,}. " if 'dense_count' in r else ''
        if r.get('rejected_biomes'):prefix+=f"{len(r['rejected_biomes'])} biomes auto-superposés écartés. "
        if r.get('preprocessing'):prefix+=t('Text exclusions: {0}. ').format(len(r['preprocessing']['text_boxes']))
        self.status.setText(prefix+f"{len(r['points'])}/{r['total_features']} points · {r['candidate_comparisons']:,} comparaisons locales · {len(r['pairs']):,} liens candidats · {len(visible)} biomes affichés ({sum(n for _,n in visible)} liens). Analyse {self.job.seconds:.2f} s ; affichage {self.render_job.seconds:.3f} s.")

    def export_data(self):
        if self.result is None or self.dirty or self.render_job.is_busy or self.export_job.is_busy:return
        filename,selected_filter=QFileDialog.getSaveFileName(self,t('Exporter Copy-Move Forgery 2'),'copy-move-2.json',t('JSON — liens affichables (*.json);;NPZ — champs denses et liens (*.npz)'))
        if not filename:return
        suffix = ".npz" if selected_filter.startswith("NPZ") else ".json"
        if not filename.lower().endswith(suffix):
            filename = str(Path(filename).with_suffix(suffix))
        self.export.setEnabled(False)
        self.export_job.request((filename,self.result,self.style(),tuple(self.visible),self.image,self.export_cancel))

    def exported(self,filename):
        if filename:self.status.setText(f'Correspondances exportées : {filename}')
        self.export.setEnabled(not self.dirty and not self.render_job.is_busy)

    def export_failed(self,message):
        self.status.setText(message);self.export.setEnabled(not self.dirty and not self.render_job.is_busy)

    def shutdown(self):
        self.cancel.set();self.draw_cancel.set();self.export_cancel.set();self.auto_cancel.set();super().shutdown()
