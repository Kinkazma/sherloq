from gui.sherloq_app.ui.localization import localized_data, t
"""Automatic panel selection and three-source clone exploration."""
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
    QCheckBox, QVBoxLayout, QHBoxLayout, QTabWidget, QTabBar, QFileDialog, QSlider, QDoubleSpinBox)

from ...core.automatic_clones import SOURCES, parameters, entries, visible, render, export
from ...core.auto_zones import enclosing, diagonal
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
            match = next((e['id'] for e in reversed(self.biomes)
                          if self.contains(e, p.x(), p.y())), None)
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
            unit='pixels' if 'pixel_mask' in e else 'cells' if 'cells' in e else 'matches'
            return f"Biome {index.row()+1} · {e['count']} {unit}\n{t(e.get('label',e['source']))}"
        if role == Qt.CheckStateRole:
            return Qt.Unchecked if e['id'] in self.hidden else Qt.Checked
        if role == Qt.DecorationRole:
            return QColor(*e['color'][::-1])
        if role == Qt.ToolTipRole:
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
        self.biomes = ()
        self.focused = None
        self.cancel = Event(); self.auto_cancel = Event()
        self.submitted = None
        self.engine = Cloning2Engine(image)
        self.viewer = ImageViewer(image, image, view_class=AutomaticView)
        self.tabs = QTabBar()
        for label in ('Overlay', *self.sources):
            self.tabs.addTab(label)
        self.cpu = QCheckBox('CPU')
        self.detect = QPushButton('Detect subimages')
        self.run = QPushButton('Run again')
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
        self.status = QLabel('Detecting subimages…' if autostart else 'Ready.'); self.status.setWordWrap(True)
        self.detail = QLabel(); self.detail.setWordWrap(True)
        layout = QVBoxLayout(self); layout.addWidget(self.tabs)
        controls = QHBoxLayout()
        for w in (self.detect, self.run, self.stop_button, self.cpu, self.show_zones, self.save):
            controls.addWidget(w)
        controls.addStretch(); layout.addLayout(controls)
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
        self.forge = ResearchJob(self, image, 'clone_detectors')
        self.auto_job = LatestJob(self, auto_detect, delay=0)
        self.draw = LatestJob(self, render, delay=0)
        self.prepare = LatestJob(self, lambda args: entries(*args), delay=60)
        self.export_job = LatestJob(self, export, delay=0)
        self.restart = QTimer(self); self.restart.setSingleShot(True); self.restart.setInterval(300)
        self.restart.timeout.connect(self.start)
        self.job.result.connect(lambda r: self.complete('patchmatch', r))
        self.forge.result.connect(lambda r: self.complete('forgeryscope', r))
        self.job.failed.connect(lambda e: self.failed('patchmatch', e))
        self.forge.failed.connect(lambda e: self.failed('forgeryscope', e))
        self.updates.changed.connect(self.patch_progress)
        self.forge.progress.connect(lambda n, _: self.progress('forgeryscope', n))
        self.auto_job.result.connect(self.auto_ready)
        self.auto_job.failed.connect(self.auto_failed)
        self.draw.result.connect(self.viewer.update_processed)
        self.draw.failed.connect(lambda e: self.detail.setText('Display failed: '+e))
        self.prepare.result.connect(self.biomes_ready)
        self.prepare.failed.connect(lambda e: self.detail.setText('Display preparation failed: '+e))
        self.export_job.result.connect(lambda p: self.detail.setText('Exported: '+p))
        self.export_job.failed.connect(lambda e: self.detail.setText('Export failed: '+e))
        self.detect.clicked.connect(self.detect_zones); self.run.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop); self.save.clicked.connect(self.export_results)
        self.cpu.toggled.connect(self.changed); self.show_zones.toggled.connect(self.toggle_zones)
        self.zone_model.changed.connect(self.changed)
        self.zones.selectionModel().selectionChanged.connect(self.focus_zones)
        self.zones.focusCleared.connect(self.clear_focus)
        self.legend.selectionModel().selectionChanged.connect(self.choose)
        self.legend.focusCleared.connect(self.clear_focus)
        self.model.changed.connect(self.redraw)
        self.viewer.view.biomeClicked.connect(self.pick)
        self.tabs.currentChanged.connect(self.tab_changed)
        self.low.valueChanged.connect(self.minimum.setValue); self.high.valueChanged.connect(self.maximum.setValue)
        self.minimum.valueChanged.connect(lambda v: self.length_changed(True, v))
        self.maximum.valueChanged.connect(lambda v: self.length_changed(False, v))
        self.overlap.valueChanged.connect(self.prepare_biomes)
        QApplication.instance().installEventFilter(self)
        self.update_status()
        if autostart:
            QTimer.singleShot(0, self.detect_zones)

    def initial_states(self):
        return {'patchmatch': 'Waiting', 'forgeryscope': 'Waiting'}

    def state_labels(self):
        return {'patchmatch': t(EXTENDED_SYMMETRIC), 'forgeryscope': 'Forgeryscope microscopy'}

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
        self.job.invalidate(); self.auto_job.invalidate(); self.forge.cancel(); self.draw.invalidate()
        self.prepare.invalidate()

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
        self.results = {}; self.biomes = (); self.errors = {}; self.focused = None
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

    def patch_progress(self, event, value, _):
        if event is self.cancel and not event.is_set():
            self.progress('patchmatch', value)

    def progress(self, key, value):
        if self.states[key].startswith('Running'):
            self.states[key] = f'Running {max(0,min(100,value))}%'; self.update_status()

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
                                  self.minimum.value(), self.maximum.value(), self.overlap.value()/100))

    def biomes_ready(self, result):
        self.biomes = result; self.rebuild_legend(); self.update_status(); self.redraw()

    def failed(self, key, error):
        self.states[key] = 'Failed'; self.errors[key] = error
        self.update_status()

    def update_status(self):
        pending = any(v.startswith('Running') for v in self.states.values())
        self.stop_button.setEnabled(pending)
        self.save.setEnabled(bool(self.results) and not pending and not self.prepare.is_busy)
        labels = self.state_labels()
        self.status.setText(' | '.join(labels[k]+': '+v for k,v in self.states.items()) + f' | {len(self.biomes)} biomes')
        notes = [labels[k]+': '+v for k,v in self.errors.items()]
        if self.results and any(v != 'Complete' and not v.startswith('Disabled') for v in self.states.values()):
            notes.insert(0, 'Partial results: not all methods have completed.')
        if 'forgeryscope' in self.results:
            meta = self.results['forgeryscope']['metadata']
            candidates = sum(len(z.get('embedding_candidates', ())) for z in meta['zones'])
            accepted = sum(bool(m['supported']) for z in meta['zones'] for m in z.get('comparisons', ()))
            notes.append(f'Forgeryscope: {accepted} geometrically supported pairs / {candidates} candidate pairs.')
        self.detail.setText(' '.join(notes))

    def display_entries(self):
        return tuple(e for e in self.biomes if not self.source() or e['source'] == self.source())

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
        self.focused = None; self.rebuild_legend(); self.redraw()

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

    def redraw(self):
        if self.closed:
            return
        shown = visible(self.biomes, self.source(), self.model.hidden, self.focused)
        self.viewer.view.biomes = shown
        excluded = tuple(r for i,r in enumerate(self.viewer.view.snapshot())
                         if i in self.zone_model.disabled and r != self.envelope)
        self.draw.request((self.image, shown, excluded))

    def export_results(self):
        if not self.save.isEnabled():
            return
        path, _ = QFileDialog.getSaveFileName(self, t('Export clone search'), 'automatic-clones.npz', t('NumPy (*.npz)'))
        if path:
            snapshot = dict(version=1, configuration=self.submitted, results=dict(self.results),
                states=dict(self.states), errors=dict(self.errors), biomes=self.biomes,
                display=dict(source=self.source(), hidden=sorted(self.model.hidden), focused=self.focused,
                             minimum_length_px=self.minimum.value(), maximum_length_px=self.maximum.value(),
                             maximum_overlap=self.overlap.value()/100),
                image_shape=self.image.shape, decoded_bgr8_sha256=hashlib.sha256(memoryview(np.ascontiguousarray(self.image))).hexdigest())
            self.export_job.request((str(Path(path).with_suffix('.npz')), snapshot))

    def shutdown(self):
        if self.closed:
            return
        self.closed = True; self.cancel_work(); self.forge.shutdown()
        QApplication.instance().removeEventFilter(self)
        super().shutdown()
