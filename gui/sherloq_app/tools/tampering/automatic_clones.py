from gui.sherloq_app.ui.localization import localized_data, t
"""Automatic panel selection and independent classical/AI clone layers."""
from functools import partial
import hashlib
from pathlib import Path
from threading import Event
import math

import cv2 as cv
import numpy as np
from PySide6.QtCore import (Qt, QTimer, Signal, QEvent, QAbstractListModel,
                            QModelIndex, QSignalBlocker)
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
    QCheckBox, QComboBox, QVBoxLayout, QHBoxLayout, QTabWidget, QTabBar, QFileDialog, QSlider, QDoubleSpinBox, QSpinBox, QSizePolicy)

from ...core.automatic_clones import SOURCES, CLASSICAL_SOURCES, D2PRL_SOURCE, SIFT_SOURCE, source_label, parameters, entries, visible, render, export
from ...core.auto_zones import enclosing, diagonal
from ...core.clone_corroboration import CachedRenderer
from ...core.clone_relations import annotate
from ...core.cloning2 import Cloning2Engine, EXTENDED_SYMMETRIC
from ...ui.tools import ToolWidget
from ...ui.check_list import CheckList
from ...ui.jobs import LatestJob
from ...ui.research_job import ResearchJob
from ...ui.selection_view import SelectionView
from ...ui.viewer import ImageViewer
from .cloning2 import ZoneModel, Progress, analyze, auto_detect


class AutomaticView(SelectionView):
    biomeClicked = Signal(object)

    def __init__(self, image, parent=None):
        super().__init__(image, parent)
        self.biomes = ()
        self.press_position = None

    def mousePressEvent(self, event):
        self.press_position = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if (event.button() == Qt.LeftButton and self.press_position is not None
                and (event.position().toPoint()-self.press_position).manhattanLength() < 4):
            p = self.mapToScene(event.position().toPoint())
            candidates=[e for e in self.biomes if self.contains(e,p.x(),p.y())]
            smallest=min(candidates,key=lambda e:sum(cv.contourArea(np.asarray(poly,np.float32)) for poly in e['polygons'])) if candidates else None
            match=smallest['id'] if smallest else None
            self.biomeClicked.emit(match)


    @staticmethod
    def contains(entry, x, y):
        if 'pixel_mask' in entry:
            ox,oy=entry['origin'];col,row=int(math.floor(x-ox)),int(math.floor(y-oy))
            mask=entry['pixel_mask']
            return 0<=row<mask.shape[0] and 0<=col<mask.shape[1] and bool(mask[row,col])
        if 'cells' in entry:
            block=entry['block'];cell=(int(y//block),int(x//block))
            return any(tuple(c)==cell for c in entry['cells'])
        return any(cv.pointPolygonTest(np.asarray(poly,np.float32),(x,y),False)>=0
                   for poly in entry['polygons'])


class EnglishZones(ZoneModel):
    @localized_data
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            name = 'Enclosing zone' if self.regions[index.row()] == self.envelope else f'Zone {index.row()+1}'
            return f'{name} · diagonal {diagonal(self.regions[index.row()]):.0f} px'
        if role == Qt.ToolTipRole:
            return 'Check to include in analysis. Click to isolate the outline; click outside to show all checked zones.'
        return super().data(index, role)


class Biomes(QAbstractListModel):
    changed = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.rows = ()
        self.hidden = set()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    @localized_data
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        e = self.rows[index.row()]
        if role == Qt.DisplayRole:
            unit=e.get('count_kind','pixels' if 'pixel_mask' in e else 'cells' if 'cells' in e else 'matches')
            return f"Biome {index.row()+1} · {e['count']} {unit}\n{t(e.get('label',e['source']))}"
        if role == Qt.CheckStateRole:
            return Qt.Unchecked if e['id'] in self.hidden else Qt.Checked
        if role == Qt.DecorationRole:
            return QColor(*e['color'][::-1])
        if role == Qt.ToolTipRole:
            if 'relation' in e:
                zones=' ↔ '.join('?' if i is None else str(i+1) for i in e['endpoint_zones'])
                return t({'within':'Within zones','between':'Between zones','unassigned':'Unassigned relations'}[e['relation']])+' · '+zones+'\n'+t('Search context')+': '+t(e.get('search_label','Unspecified search'))+'\n'+t('Check to show; click to isolate.')
            return ('Check to show; click to isolate.' if 'cells' in e or 'pixel_mask' in e else
                    'The same color links both parts. Check to show; click to isolate.')

    def flags(self, index):
        return super().flags(index) | Qt.ItemIsUserCheckable

    def setData(self, index, value, role=Qt.EditRole):
        if role != Qt.CheckStateRole or not index.isValid():
            return False
        key = self.rows[index.row()]['id']
        if value in (Qt.Checked, Qt.Checked.value):
            self.hidden.discard(key)
        else:
            self.hidden.add(key)
        self.dataChanged.emit(index, index, [Qt.CheckStateRole])
        self.changed.emit()
        return True

    def update(self, rows):
        self.beginResetModel(); self.rows = rows; self.endResetModel()


class AutomaticClonesWidget(ToolWidget):
    sources = SOURCES
    def __init__(self, image, parent=None, *, autostart=True):
        super().__init__(parent)
        self.image = image
        self.closed = False
        self.envelope = None
        self.results = {}
        self.states = self.initial_states()
        self.errors = {}
        self.progress_messages = {}
        self.biomes = ()
        self.focused = None
        self.cancel = Event(); self.auto_cancel = Event()
        self.submitted = None
        self.engine = Cloning2Engine(image)
        self.viewer = ImageViewer(image, image, view_class=AutomaticView)
        self.tabs = QTabBar()
        for label in ('Overlay', *self.sources):
            index=self.tabs.addTab(source_label(label))
            self.tabs.setTabToolTip(index,label)
        self.source_checks={}
        for index,source in enumerate(self.sources,1):
            check=QCheckBox();check.setChecked(True)
            check.setAccessibleName(source)
            check.setToolTip(source+' — '+t('Show or hide this source without recalculating'))
            self.source_checks[source]=check
            self.tabs.setTabButton(index,QTabBar.LeftSide,check)
            check.toggled.connect(self.sources_changed)
        self.overlay_check=QCheckBox();self.overlay_check.setTristate(True)
        self.overlay_check.setCheckState(Qt.Checked)
        self.overlay_check.setAccessibleName('All sources')
        self.overlay_check.setToolTip('Show or hide all sources')
        self.tabs.setTabButton(0,QTabBar.LeftSide,self.overlay_check)
        self.overlay_check.clicked.connect(self.toggle_all_sources)
        self.cpu = QCheckBox('CPU')
        self.forge_branch=QComboBox()
        for label,key in (('Forgeryscope: all branches',''),('Microscopy','microscopy'),('Western blots','blots'),('Lanes','lanes')):
            self.forge_branch.addItem(label,key)
        self.forge_branch.setToolTip('Filter Forgeryscope display only; other analyses stay visible.')
        self.detect = QPushButton('Detect subimages')
        self.run = QPushButton('Run again')
        self.run_whole = QPushButton('Run on whole image')
        self.run_whole.setToolTip('Ignore detected subimages and excluded zones, then restart all analyses on the whole image.')
        self.stop_button = QPushButton('Cancel')
        self.save = QPushButton('Export results')
        self.show_zones = QCheckBox('Search zones'); self.show_zones.setChecked(True)
        self.minimum = QDoubleSpinBox(); self.maximum = QDoubleSpinBox()
        self.low = QSlider(Qt.Horizontal); self.high = QSlider(Qt.Horizontal)
        limit = math.ceil(math.hypot(*image.shape[:2]))
        for spin in (self.minimum, self.maximum):
            spin.setRange(0, limit); spin.setDecimals(1); spin.setSuffix(' px')
            spin.setToolTip('Display only. PatchMatch: match length; Forgeryscope: distance between polygon centers.')
        for slider in (self.low, self.high):
            slider.setRange(0, limit)
        self.minimum.setValue(min(10, limit)); self.low.setValue(min(10, limit))
        self.maximum.setValue(limit); self.high.setValue(limit)
        self.overlap = QDoubleSpinBox(); self.overlap.setRange(1, 100); self.overlap.setDecimals(0); self.overlap.setValue(80)
        self.overlap.setSuffix(' %'); self.overlap.setToolTip('Hide biomes at or above this overlap, measured against the smaller polygon.')
        self.legend = CheckList(); self.model = Biomes(self.legend); self.legend.setModel(self.model)
        self.zones = CheckList(); self.zone_model = EnglishZones(self.zones); self.zones.setModel(self.zone_model)
        self.side = QTabWidget(); self.side.setMinimumWidth(255); self.side.setMaximumWidth(420)
        self.side.addTab(self.legend, 'Biomes'); self.side.addTab(self.zones, 'Zones')
        self.status = QLabel('Detecting subimages…' if autostart else 'Ready.'); self.status.setWordWrap(False); self.status.setMinimumWidth(0); self.status.setFixedHeight(self.status.fontMetrics().height()+6)
        self.detail = QLabel(); self.detail.setWordWrap(False)
        self.detail.setFixedHeight(self.detail.fontMetrics().height())
        for label in (self.status,self.detail):
            label.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Fixed)
        layout = QVBoxLayout(self); layout.addWidget(self.tabs)
        controls = QHBoxLayout()
        for w in (self.detect, self.run, self.run_whole, self.stop_button, self.cpu, self.show_zones, self.save):
            controls.addWidget(w)
        controls.addStretch(); layout.addLayout(controls)
        self.presentation=QComboBox()
        for label,key in (('Biomes','biomes'),('Corroboration heatmap','heat'),('Corroboration on image','overlay')):
            self.presentation.addItem(label,key)
        self.presentation.setCurrentIndex(2)
        # Heatmap and overlay show the same evidence; biome browsing has its
        # own scope. Switching views must not overwrite a user's scope choice.
        self._relation_views={'biomes':'all','corroboration':'within'}
        self._relation_view='corroboration'
        self._display_tab=None;self._tab_displays={}
        self.relations=QComboBox()
        for label,key in (('Within zones','within'),('Between zones','between'),('All relations','all')):
            self.relations.addItem(label,key)
        self.relations.setCurrentIndex(0)
        self.relations.setToolTip('Separate copies inside a subimage from correspondences between subimages. Display only.')
        self.opacity=QSlider(Qt.Horizontal);self.opacity.setRange(0,100);self.opacity.setValue(70)
        self.opacity.setMaximumWidth(160);self.opacity.setToolTip('Heatmap opacity')
        self.heat_legend=QLabel('1 → 6+ search contexts · ELA excluded')
        self.heat_legend.setToolTip('Integer count of methods and search zones. No size weighting; not a probability.')
        display_row=QHBoxLayout();display_row.addWidget(self.forge_branch)
        display_row.addWidget(self.presentation);display_row.addWidget(self.relations);display_row.addWidget(self.opacity);display_row.addWidget(self.heat_legend)
        layout.addLayout(display_row)
        self.d2_minimum=QSpinBox();self.d2_minimum.setRange(0,5000);self.d2_minimum.setValue(500)
        self.d2_minimum.setToolTip('Minimum region size (448 × 448 grid)')
        self.d2_minimum.setPrefix('D2PRL ≥ ');self.d2_minimum.setSuffix(' px')
        self.d2_minimum.setAccessibleName('D2PRL minimum region size')
        self.d2_slider=QSlider(Qt.Horizontal);self.d2_slider.setRange(0,5000);self.d2_slider.setValue(500)
        self.d2_slider.setMaximumWidth(140);self.d2_slider.setMinimumWidth(60)
        self.d2_slider.setAccessibleName('D2PRL minimum region size')
        self.d2_slider.setToolTip('Minimum region size (448 × 448 grid)')
        self.d2_controls=QWidget();d2_layout=QHBoxLayout(self.d2_controls);d2_layout.setContentsMargins(0,0,0,0)
        d2_layout.addWidget(self.d2_minimum);d2_layout.addWidget(self.d2_slider)
        controls.insertWidget(controls.indexOf(self.save)+1,self.d2_controls)
        self.clone_intervals=QWidget()
        intervals = QHBoxLayout(self.clone_intervals);intervals.setContentsMargins(0,0,0,0)
        for widget in (QLabel('Displayed length — minimum'), self.low, self.minimum,
                       QLabel('maximum'), self.high, self.maximum, QLabel('Exclude overlap ≥'), self.overlap):
            intervals.addWidget(widget)
        layout.addWidget(self.clone_intervals)
        body = QHBoxLayout(); body.addWidget(self.viewer, 1); body.addWidget(self.side); layout.addLayout(body, 1)
        layout.addWidget(self.status); layout.addWidget(self.detail)

        self.updates = Progress(self)
        self.job = LatestJob(self, partial(analyze, self.engine, self.updates), delay=0)
        self.sift_engine=Cloning2Engine(image);self.sift_updates=Progress()
        self.sift_job=LatestJob(self,partial(analyze,self.sift_engine,self.sift_updates),delay=0)
        self.sift_job.result.connect(lambda result:self.complete('sift',result))
        self.sift_job.failed.connect(lambda error:self.failed('sift',error))
        self.sift_updates.changed.connect(lambda event,value,message:self.progress('sift',value,message)
                                         if event is self.cancel and not event.is_set() else None)
        self.forge = ResearchJob(self, image, 'clone_detectors')
        self.d2_job=ResearchJob(self,image,'clone_detectors')
        self.d2_job.result.connect(lambda r:self.complete('d2prl',r))
        self.d2_job.failed.connect(lambda e:self.failed('d2prl',e))
        self.d2_job.progress.connect(lambda n,message:self.progress('d2prl',n,message))
        self.auto_job = LatestJob(self, auto_detect, delay=0)
        self.renderer=CachedRenderer(render)
        self.draw = LatestJob(self, self.renderer, delay=0)
        from ...core.d2prl_regions import RegionCache
        self.d2_regions=RegionCache()
        self.prepare = LatestJob(self, lambda args: entries(*args[:6])+self.d2_regions(*args[6:]), delay=60)
        self.export_job = LatestJob(self, export, delay=0)
        self.restart = QTimer(self); self.restart.setSingleShot(True); self.restart.setInterval(300)
        self.restart.timeout.connect(self.start)
        self.job.result.connect(lambda r: self.complete('patchmatch', r))
        self.forge.result.connect(lambda r: self.complete('forgeryscope', r))
        self.job.failed.connect(lambda e: self.failed('patchmatch', e))
        self.forge.failed.connect(lambda e: self.failed('forgeryscope', e))
        self.updates.changed.connect(self.patch_progress)
        self.forge.progress.connect(lambda n, message: self.progress('forgeryscope', n, message))
        self.auto_job.result.connect(self.auto_ready)
        self.auto_job.failed.connect(self.auto_failed)
        self.draw.result.connect(self.viewer.update_processed)
        self.draw.failed.connect(lambda e: self.detail.setText('Display failed: '+e))
        self.prepare.result.connect(self.biomes_ready)
        self.prepare.failed.connect(lambda e: self.detail.setText('Display preparation failed: '+e))
        self.export_job.result.connect(lambda p: self.detail.setText('Exported: '+p))
        self.export_job.failed.connect(lambda e: self.detail.setText('Export failed: '+e))
        self.detect.clicked.connect(self.detect_zones); self.run.clicked.connect(self.start)
        self.run_whole.clicked.connect(self.run_whole_image)
        self.stop_button.clicked.connect(self.stop); self.save.clicked.connect(self.export_results)
        self.cpu.toggled.connect(self.changed); self.show_zones.toggled.connect(self.toggle_zones)
        self.zone_model.changed.connect(self.changed)
        self.zones.selectionModel().selectionChanged.connect(self.focus_zones)
        self.zones.focusCleared.connect(self.clear_focus)
        self.legend.selectionModel().selectionChanged.connect(self.choose)
        self.legend.focusCleared.connect(self.clear_focus)
        self.model.changed.connect(self.redraw)
        self.viewer.view.biomeClicked.connect(self.pick)
        self.relations.currentIndexChanged.connect(self.relation_changed)
        self.presentation.currentIndexChanged.connect(self.presentation_changed)
        self.opacity.valueChanged.connect(self.redraw)
        self.tabs.currentChanged.connect(self.source_tab_changed)
        self.forge_branch.currentIndexChanged.connect(self.tab_changed)
        self.low.valueChanged.connect(self.minimum.setValue); self.high.valueChanged.connect(self.maximum.setValue)
        self.minimum.valueChanged.connect(lambda v: self.length_changed(True, v))
        self.maximum.valueChanged.connect(lambda v: self.length_changed(False, v))
        self.overlap.valueChanged.connect(self.prepare_biomes)
        self.d2_minimum.valueChanged.connect(self.d2_filter_changed)
        self.d2_slider.valueChanged.connect(self.d2_minimum.setValue)
        self.prepare.busy.connect(self.update_status)
        self.draw.busy.connect(self.update_status)
        QApplication.instance().installEventFilter(self)
        self.update_status()
        if autostart:
            QTimer.singleShot(0, self.initial_detection)

    def enabled_sources(self):
        return tuple(s for s in self.sources if not hasattr(self,'source_checks') or self.source_checks[s].isChecked())

    def sources_changed(self,*_):
        enabled=self.enabled_sources()
        with QSignalBlocker(self.overlay_check):
            self.overlay_check.setCheckState(Qt.Checked if len(enabled)==len(self.sources)
                                             else Qt.PartiallyChecked if enabled else Qt.Unchecked)
        self.tab_changed()

    def toggle_all_sources(self,enabled):
        for check in self.source_checks.values():
            with QSignalBlocker(check):check.setChecked(enabled)
        self.sources_changed()

    def initial_detection(self):
        # A user action before the queued startup must not be overwritten.
        if not self.closed and not self.viewer.view.snapshot():
            self.detect_zones()

    def run_whole_image(self):
        if self.closed:
            return
        # Invalidate detection as well as analysis: a late detector answer must
        # not restore the subimages after this explicit override.
        self.cancel_work()
        self.auto_ready(())

    def initial_states(self):
        return {'patchmatch': 'Waiting', 'forgeryscope': 'Waiting', 'sift':'Waiting','d2prl':'Waiting'}

    def state_labels(self):
        return {'patchmatch': t(EXTENDED_SYMMETRIC), 'forgeryscope': 'Forgeryscope Auto', 'sift':source_label(SIFT_SOURCE),'d2prl':D2PRL_SOURCE}

    def source(self):
        return self.sources[self.tabs.currentIndex()-1] if self.tabs.currentIndex() else None

    def eventFilter(self, obj, event):
        if (event.type() == QEvent.MouseButtonPress and isinstance(obj, QWidget)
                and self.isAncestorOf(obj) and obj != self.side and not self.side.isAncestorOf(obj)
                and obj != self.viewer.view.viewport()):
            self.clear_focus()
        return super().eventFilter(obj, event)

    def cancel_work(self):
        self.restart.stop(); self.cancel.set(); self.auto_cancel.set()
        self.job.invalidate(); self.auto_job.invalidate(); self.forge.cancel(); self.d2_job.cancel(); self.draw.invalidate()
        self.prepare.invalidate();self.sift_job.invalidate()

    def stop(self):
        self.cancel_work()
        for key in self.states:
            if self.states[key].startswith(('Running', 'Waiting')):
                self.states[key] = 'Cancelled'
        self.detect.setEnabled(True); self.run.setEnabled(True)
        self.prepare_biomes(); self.update_status()

    def detect_zones(self):
        if self.closed:
            return
        self.cancel_work(); self.clear_results()
        self.auto_cancel = Event(); self.detect.setEnabled(False); self.run.setEnabled(False)
        self.stop_button.setEnabled(True); self.status.setText('Detecting subimages…')
        self.auto_job.request((self.image, self.auto_cancel))

    def auto_ready(self, regions):
        if regions is None or self.closed:
            return
        self.envelope = enclosing(regions) if regions else None
        if not regions:
            h,w = self.image.shape[:2]
            regions = (((0.,0.), (w-1.,0.), (w-1.,h-1.), (0.,h-1.)),)
            self.envelope = regions[0]
        elif self.envelope not in regions:
            regions = (*regions, self.envelope)
        self.viewer.view.set_regions(regions)
        self.zone_model.disabled.clear(); self.zone_model.update(tuple(regions), self.envelope)
        self.detect.setEnabled(True); self.run.setEnabled(True)
        self.sync_zones(); self.start()

    def auto_failed(self, error):
        self.detect.setEnabled(True); self.run.setEnabled(True); self.stop_button.setEnabled(False)
        self.status.setText('Subimage detection failed.'); self.detail.setText(error)

    def sync_zones(self):
        view = self.viewer.view
        view.enabled_regions = set(range(len(view.regions))) - self.zone_model.disabled
        view.region_names = ['Enclosing zone' if r == self.envelope else f'Zone {i+1}'
                             for i,r in enumerate(view.snapshot())]
        view.viewport().update()

    def changed(self, *_):
        self.cancel_work(); self.clear_results(); self.sync_zones()
        self.status.setText('Zones or CPU setting changed; restarting…')
        self.restart.start()

    def clear_results(self):
        self.results = {}; self.biomes = (); self.errors = {}; self.focused = None; self.progress_messages.clear()
        self.states = self.initial_states()
        self.model.update(()); self.viewer.view.biomes = ()
        self.save.setEnabled(False); self.viewer.update_processed(self.image)
        self.detail.clear()

    def start(self):
        if self.closed:
            return
        if not self.viewer.view.snapshot():
            self.detect_zones(); return
        self.cancel_work(); self.clear_results()
        params, regions, forge = parameters(self.image.shape, self.viewer.view.snapshot(),
                                           self.envelope, self.zone_model.disabled, self.cpu.isChecked())
        if not regions:
            self.states = dict.fromkeys(self.states, 'No active zones'); self.update_status(); return
        self.submitted = dict(patchmatch_parameters=params, forgeryscope_parameters=forge,
                              zones=self.viewer.view.snapshot(), disabled_zones=sorted(self.zone_model.disabled),
                              requested_device='cpu' if self.cpu.isChecked() else 'mps')
        self.cancel = Event()
        self.states = dict.fromkeys(self.states, 'Running')
        if not forge['regions']:
            self.states['forgeryscope'] = 'Disabled (enclosing zone unchecked)'
        self.update_status()
        self.job.request((params, regions, False, self.cancel))
        if forge['regions']:
            self.forge.request(forge, self.submitted['requested_device'])
        # Each active subimage gets its own native 448px analysis, just as in
        # AI Clone Detection. Keep the enclosing pass too when it is enabled.
        d2=dict(forge,variant=D2PRL_SOURCE,regions=tuple(dict.fromkeys(regions)))
        self.submitted['d2prl_parameters']=d2
        self.d2_job.request(d2,self.submitted['requested_device'])

        if not self.closed and self.states.get('sift')=='Running':
            p=list(self.submitted['patchmatch_parameters']);p[0]=SIFT_SOURCE;p[3]=10.;p[4]=.725
            p[6:9]=['Affine',5.,10];p[11]=True;p[15]=False
            zones=tuple(r for i,r in enumerate(self.submitted['zones']) if i not in self.submitted['disabled_zones'])
            self.submitted['sift_parameters']=tuple(p)
            self.sift_job.request((tuple(p),zones,False,self.cancel))

    def patch_progress(self, event, value, message):
        if event is self.cancel and not event.is_set():
            self.progress('patchmatch', value, message)

    def progress(self, key, value, message=None):
        if self.states[key].startswith('Running'):
            self.states[key] = f'Running {max(0,min(100,value))}%'
            if message:
                self.progress_messages.pop(key,None)
                self.progress_messages[key]=message
            self.update_status()

    def complete(self, key, result):
        if result is None or self.closed:
            return
        self.results[key] = result; self.states[key] = 'Complete'
        self.prepare_biomes(); self.update_status()

    def length_changed(self, lower, value):
        spin, slider = (self.maximum, self.low) if lower else (self.minimum, self.high)
        with QSignalBlocker(slider):
            slider.setValue(round(value))
        if (lower and value > spin.value()) or (not lower and value < spin.value()):
            spin.setValue(value)
        self.prepare_biomes()

    def prepare_biomes(self, *_):
        if not self.closed:
            self.draw.invalidate()
            self.save.setEnabled(False)
            self.prepare.request((self.results.get('patchmatch'), self.results.get('forgeryscope'),
                                  self.minimum.value(), self.maximum.value(), self.overlap.value()/100, self.results.get('sift'),self.results.get('d2prl'),self.d2_minimum.value()))

    def biomes_ready(self, result):
        clones=tuple(e for e in result if e['source'] in SOURCES)
        regions=self.viewer.view.snapshot()
        key=(tuple(id(e) for e in clones),regions,self.envelope)
        if key!=getattr(self,'_relation_key',None):
            self._relation_rows=annotate(clones,regions,self.envelope,self.image.shape)
            self._relation_inputs=clones;self._relation_key=key
        self.biomes=self._relation_rows+tuple(e for e in result if e['source'] not in SOURCES)
        self.rebuild_legend();self.update_status();self.redraw()

    def failed(self, key, error):
        self.states[key] = 'Failed'; self.errors[key] = error
        self.update_status()

    def update_status(self, *_):
        pending = any(v.startswith('Running') for v in self.states.values())
        self.stop_button.setEnabled(pending)
        self.save.setEnabled(bool(self.results) and not pending and not self.prepare.is_busy)
        labels = self.state_labels()
        details=' | '.join(labels[k]+': '+v for k,v in self.states.items())
        waiting=any(v=='Waiting' for v in self.states.values())
        finishing=self.prepare.is_busy or self.draw.is_busy
        if pending or waiting:
            title=t('Calculations in progress — partial results')
        elif finishing:
            title=t('Finalizing display — partial results')
        elif self.errors:
            title=t('Calculations finished — partial results (errors)')
        elif any(v=='Cancelled' for v in self.states.values()):
            title=t('Calculations stopped — partial results')
        else:title=t('Calculations finished')
        completed=sum(v=='Complete' for v in self.states.values())
        total=sum(not v.startswith('Disabled') for v in self.states.values())
        counter=t('{0}/{1} groups finished').replace('{0}',str(completed)).replace('{1}',str(total))
        compact=dict(patchmatch='PatchMatch Zernike + SIFT',forgeryscope='Forgeryscope',
                     sift='SIFT+G2NN+RANSAC',d2prl='D2PRL',ela='ELA')
        states=' | '.join(compact.get(k,labels[k])+': '+(v[8:] if v.startswith('Running ') else t(v))
                          for k,v in self.states.items())
        self.status.setText(f'{title} · {counter} | {states}')
        explanation=t('The counter tracks calculation groups, not individual algorithms or subimages.')
        self.status.setToolTip(explanation+'\n'+details+f' | {len(self.biomes)} regions')
        notes = [labels[k]+': '+v for k,v in self.errors.items()]
        if self.results and any(v != 'Complete' and not v.startswith('Disabled') for v in self.states.values()):
            notes.insert(0, 'Partial results: not all methods have completed.')
        if 'forgeryscope' in self.results:
            meta = self.results['forgeryscope']['metadata']
            candidates = sum(len(z.get('embedding_candidates', ())) for z in meta['zones'])
            accepted = sum(bool(m['supported']) for z in meta['zones'] for m in z.get('comparisons', ()))
            notes.append(f'Forgeryscope: {accepted} geometrically supported pairs / {candidates} candidate pairs.')
            similarities=sum(m.get('accepted',False) and not m.get('supported',False) for z in meta['zones'] for m in z.get('comparisons',()))
            lanes=sum(z.get('lane_matches',0) for z in meta['zones'])
            notes.append(t('Forgeryscope Auto: {0} similarity pairs; {1} lane pairs.').replace('{0}',str(similarities)).replace('{1}',str(lanes)))
        phases=[labels[k]+': '+t(message) for k,message in self.progress_messages.items()
                if self.states.get(k,'').startswith('Running')]
        # Reuse the existing detail line, prioritizing the latest active phase.
        self.detail.setText(phases[-1] if phases else ' '.join(notes))
        self.detail.setToolTip('\n'.join(phases+notes))

    def display_entries(self):
        branch=self.forge_branch.currentData()
        return tuple(e for e in self.biomes if (e['source'] not in CLASSICAL_SOURCES or self.relations.currentData()=='all' or e.get('relation','unassigned')==self.relations.currentData())

                     and e['source'] in self.enabled_sources() and (not self.source() or e['source'] == self.source())
                     and (not branch or e['source']!=SOURCES[0] or e['provenance'].get('branch')==branch))

    def rebuild_legend(self):
        with QSignalBlocker(self.legend.selectionModel()):
            self.model.update(self.display_entries())
        if self.focused:
            for i,e in enumerate(self.model.rows):
                if e['id'] == self.focused:
                    with QSignalBlocker(self.legend.selectionModel()):
                        self.legend.setCurrentIndex(self.model.index(i))
                    break

    def tab_changed(self, *_):
        self.forge_branch.setVisible(self.source() in (None,SOURCES[0]))
        self.d2_controls.setVisible(self.source() in (None,D2PRL_SOURCE))
        self.focused = None; self.rebuild_legend(); self.redraw()

    def d2_filter_changed(self, value):
        with QSignalBlocker(self.d2_slider):self.d2_slider.setValue(value)
        self.prepare_biomes()

    def source_tab_changed(self, *_):
        # Store controls under the tab being left, not the newly selected source.
        self._tab_displays[self._display_tab]=(self.presentation.currentData(),
                                               dict(self._relation_views),self.opacity.value())
        self._display_tab=self.source()
        mode,scopes,opacity=self._tab_displays.get(self._display_tab,
            ('biomes',{'biomes':'all','corroboration':'all'},70))
        self._relation_views=dict(scopes)
        self._relation_view='biomes' if mode=='biomes' else 'corroboration'
        with QSignalBlocker(self.presentation),QSignalBlocker(self.relations),QSignalBlocker(self.opacity):
            self.presentation.setCurrentIndex(self.presentation.findData(mode))
            self.relations.setCurrentIndex(self.relations.findData(scopes[self._relation_view]))
            self.opacity.setValue(opacity)
        self.tab_changed()

    def relation_changed(self, *_):
        self._relation_views[self._relation_view]=self.relations.currentData()
        self.tab_changed()

    def presentation_changed(self, *_):
        self._relation_view=('biomes' if self.presentation.currentData()=='biomes'
                             else 'corroboration')
        with QSignalBlocker(self.relations):
            self.relations.setCurrentIndex(self.relations.findData(self._relation_views[self._relation_view]))
        self.tab_changed()

    def choose(self, *_):
        selected = self.legend.selectionModel().selectedIndexes()
        self.focused = self.model.rows[selected[0].row()]['id'] if selected else None
        self.redraw()

    def pick(self, key):
        self.clear_focus()
        if key:
            for i,e in enumerate(self.model.rows):
                if e['id'] == key:
                    self.legend.setCurrentIndex(self.model.index(i)); break

    def clear_focus(self, *_):
        self.focused = None
        with QSignalBlocker(self.legend.selectionModel()):
            self.legend.clearSelection(); self.legend.setCurrentIndex(QModelIndex())
        with QSignalBlocker(self.zones.selectionModel()):
            self.zones.clearSelection(); self.zones.setCurrentIndex(QModelIndex())
        self.viewer.view.focus_regions = set(); self.viewer.view.viewport().update(); self.redraw()

    def focus_zones(self, *_):
        self.viewer.view.focus_regions = {i.row() for i in self.zones.selectionModel().selectedIndexes()}
        self.viewer.view.viewport().update()

    def toggle_zones(self, enabled):
        self.viewer.view.show_regions = enabled; self.viewer.view.viewport().update()

    def corroboration_active(self):
        return (self.presentation.currentData()!='biomes' and
                (self.source() is None or self.source() in SOURCES))

    def presentation_settings(self):
        clone_view=self.source() is None or self.source() in SOURCES
        self.presentation.setEnabled(clone_view)
        self.heat_legend.setToolTip(t('Integer counts without size weighting. Relation filters affect PatchMatch/SIFT only; AI sources remain included. ELA is hidden in this view.'))
        self.relations.setEnabled(self.source() is None or self.source() in CLASSICAL_SOURCES)
        mode=self.presentation.currentData() if clone_view else 'biomes'
        self.opacity.setEnabled(mode=='overlay')
        self.heat_legend.setVisible(mode!='biomes')
        return mode,self.opacity.value()/100

    def corroboration_snapshot(self):
        shown=visible(self.display_entries(),None,self.model.hidden,self.focused)
        excluded=tuple(r for i,r in enumerate(self.viewer.view.snapshot())
                       if i in self.zone_model.disabled and r!=self.envelope)
        return dict(entries=tuple(e for e in shown if e['source'] in SOURCES),excluded=excluded,
                    sources=SOURCES,metric='integer_method_search_context_count_not_probability',
                    relation_view=self.relations.currentData(),exclude_enclosing_search=False,
                    presentation=self.presentation.currentData(),opacity=self.opacity.value()/100,
                    biome_opacity_policy='visual_only_area_dependent_non_ai',
                    ela_suppressed=self.corroboration_active(),d2prl_min_component=self.d2_minimum.value())

    def redraw(self):
        if self.closed:
            return
        shown = visible(self.display_entries(), None, self.model.hidden, self.focused)
        self.viewer.view.biomes = shown
        excluded = tuple(r for i,r in enumerate(self.viewer.view.snapshot())
                         if i in self.zone_model.disabled and r != self.envelope)
        self.draw.request((self.image, shown, excluded, *self.presentation_settings()))

    def export_results(self):
        if not self.save.isEnabled():
            return
        path, _ = QFileDialog.getSaveFileName(self, t('Export clone search'), 'automatic-clones.npz', t('NumPy (*.npz)'))
        if path:
            snapshot = dict(version=1, configuration=self.submitted, results=dict(self.results),
                states=dict(self.states), errors=dict(self.errors), biomes=self.biomes,
                display=dict(source=self.source(), forgeryscope_branch=self.forge_branch.currentData(), hidden=sorted(self.model.hidden), focused=self.focused,
                             minimum_length_px=self.minimum.value(), maximum_length_px=self.maximum.value(),
                             maximum_overlap=self.overlap.value()/100),
                corroboration=self.corroboration_snapshot(),
                image_shape=self.image.shape, decoded_bgr8_sha256=hashlib.sha256(memoryview(np.ascontiguousarray(self.image))).hexdigest())
            self.export_job.request((str(Path(path).with_suffix('.npz')), snapshot))

    def shutdown(self):
        if self.closed:
            return
        self.closed = True; self.cancel_work(); self.forge.shutdown(); self.d2_job.shutdown()
        QApplication.instance().removeEventFilter(self)
        super().shutdown()
