from gui.sherloq_app.ui.localization import t
from os.path import splitext
from functools import partial
from threading import Event

import cv2 as cv
import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QToolButton,
    QMessageBox,
    QSpinBox,
    QCheckBox,
    QComboBox,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QProgressBar,
)

from gui.sherloq_app.ui.jobs import LatestJob
from gui.sherloq_app.core.cloning import CloningEngine
from gui.sherloq_app.core.jpeg_curve import Cancelled
from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.core.utility import modify_font, load_image
from gui.sherloq_app.ui.viewer import ImageViewer


class _Progress(QObject):
    changed = Signal(object, int, str)


def _run(engine, updates, request):
    params, mask_id, mask, event = request
    try:
        output, stats = engine.analyze(params, mask_id, mask, event.is_set,
            lambda value,text:updates.changed.emit(event,value,text))
        return (params,mask_id),output,stats
    except Cancelled:
        return None


class CloningWidget(ToolWidget):
    def __init__(self, image, parent=None):
        super(CloningWidget, self).__init__(parent)

        self.detector_combo = QComboBox()
        self.detector_combo.addItems(
            [self.tr("BRISK"), self.tr("ORB"), self.tr("AKAZE")]
        )
        self.detector_combo.setCurrentIndex(0)
        self.detector_combo.setToolTip(
            self.tr("Algorithm used for localization and description")
        )
        self.response_spin = QSpinBox()
        self.response_spin.setRange(0, 100)
        self.response_spin.setSuffix(self.tr("%"))
        self.response_spin.setValue(90)
        self.response_spin.setToolTip(
            self.tr("Higher values retain more keypoints, including weaker responses.")
        )
        self.matching_spin = QSpinBox()
        self.matching_spin.setRange(1, 100)
        self.matching_spin.setSuffix(self.tr("%"))
        self.matching_spin.setValue(20)
        self.matching_spin.setToolTip(
            self.tr("Higher values allow larger Hamming descriptor differences.")
        )
        self.distance_spin = QSpinBox()
        self.distance_spin.setRange(1, 100)
        self.distance_spin.setSuffix(self.tr("%"))
        self.distance_spin.setValue(15)
        self.distance_spin.setToolTip(
            self.tr("Spatial clustering tolerance; also excludes close source/destination pairs.")
        )
        self.cluster_spin = QSpinBox()
        self.cluster_spin.setRange(1, 20)
        self.cluster_spin.setValue(5)
        self.cluster_spin.setToolTip(
            self.tr("Minimum number of keypoints to create a new cluster")
        )
        self.kpts_check = QCheckBox(self.tr("Show keypoints"))
        self.kpts_check.setToolTip(self.tr("Show keypoint coverage"))
        self.nolines_check = QCheckBox(self.tr("Hide lines"))
        self.nolines_check.setToolTip(self.tr("Disable match line drawing"))
        self.process_button = QToolButton()
        self.process_button.setText(self.tr("Process"))
        self.process_button.setToolTip(self.tr("Perform automatic detection"))
        modify_font(self.process_button, bold=True)
        self.status_label = QLabel()
        self.mask_label = QLabel()
        self.mask_button = QToolButton()
        self.mask_button.setText(self.tr("Load mask..."))
        self.mask_button.setToolTip(self.tr("Load an image to be used as mask"))
        self.onoff_button = QToolButton()
        self.onoff_button.setText(self.tr("OFF"))
        self.onoff_button.setCheckable(True)
        self.onoff_button.setToolTip(self.tr("Toggle keypoint detection mask"))

        self.image = image
        self.viewer = ImageViewer(self.image, self.image)
        self.engine = CloningEngine(image)
        self.mask = None
        self.mask_id = 0
        self.cancel_event = Event()
        self._requested = self._displayed = None
        self.stats = None
        self.progress = QProgressBar()
        self.progress.setRange(0,100)
        self.updates = _Progress()
        self.updates.changed.connect(self._progress)
        self.job = LatestJob(self,partial(_run,self.engine,self.updates),delay=0)
        self.job.result.connect(self.show_result)
        self.job.failed.connect(self.show_error)
        self.job.busy.connect(self.set_busy)
        self.viewer.set_busy(True)

        self.detector_combo.currentIndexChanged.connect(self.update_detector)
        self.response_spin.valueChanged.connect(self.update_detector)
        self.matching_spin.valueChanged.connect(self.update_matching)
        self.distance_spin.valueChanged.connect(self.update_cluster)
        self.cluster_spin.valueChanged.connect(self.update_cluster)
        self.nolines_check.stateChanged.connect(self.update_style)
        self.kpts_check.stateChanged.connect(self.update_style)
        self.process_button.clicked.connect(self.process)
        self.mask_button.clicked.connect(self.load_mask)
        self.onoff_button.toggled.connect(self.toggle_mask)
        self.onoff_button.setEnabled(False)

        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel(self.tr("Detector:")))
        top_layout.addWidget(self.detector_combo)
        top_layout.addWidget(QLabel(self.tr("Response:")))
        top_layout.addWidget(self.response_spin)
        top_layout.addWidget(QLabel(self.tr("Matching:")))
        top_layout.addWidget(self.matching_spin)
        top_layout.addWidget(QLabel(self.tr("Distance:")))
        top_layout.addWidget(self.distance_spin)
        top_layout.addWidget(QLabel(self.tr("Cluster:")))
        top_layout.addWidget(self.cluster_spin)
        top_layout.addWidget(self.nolines_check)
        top_layout.addWidget(self.kpts_check)
        top_layout.addStretch()

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.process_button)
        bottom_layout.addWidget(self.status_label)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.mask_button)
        bottom_layout.addWidget(self.onoff_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addLayout(bottom_layout)
        main_layout.addWidget(self.progress)
        main_layout.addWidget(self.viewer)
        self.setLayout(main_layout)

    def parameters(self):
        return (self.detector_combo.currentIndex(), self.response_spin.value(),
                self.matching_spin.value(), self.distance_spin.value(),
                self.cluster_spin.value(), self.kpts_check.isChecked(),
                self.nolines_check.isChecked())

    def current_key(self):
        return self.parameters(), self.mask_id if self.onoff_button.isChecked() else 0

    def _invalidate(self):
        self.cancel_event.set()
        self.job.invalidate()
        self._requested = None
        self.set_busy(False)

    def update_detector(self):
        self._invalidate()
        self.status_label.setText('Settings changed. Process to update.')

    update_matching = update_detector
    update_cluster = update_detector

    def update_style(self):
        self._invalidate()
        self._start()

    def toggle_mask(self, checked):
        self.onoff_button.setText('ON' if checked else 'OFF')
        self._invalidate()
        preview = self.image if not checked else self.image*self.mask[:,:,None]
        self.viewer.update_processed(preview)
        self._displayed = None
        self.set_busy(False)
        self.status_label.setText('Detection mask changed. Process to update.')

    def load_mask(self):
        filename, basename, mask = load_image(self)
        if filename is None:return
        if self.image.shape[:2] != mask.shape[:2]:
            QMessageBox.critical(self,t('Error'),t('Image and mask must have the same size.'))
            return
        _, self.mask = cv.threshold(cv.cvtColor(mask,cv.COLOR_BGR2GRAY),0,1,cv.THRESH_BINARY)
        self.mask_id += 1
        self.onoff_button.setEnabled(True)
        if self.onoff_button.isChecked():
            self.toggle_mask(True)
        else:
            self.onoff_button.setChecked(True)
        self.mask_button.setText(f'"{splitext(basename)[0]}"')
        self.mask_button.setToolTip('Current detection mask image')

    def process(self):
        if self.job.is_busy:
            self._invalidate()
            self.status_label.setText('Cancelled. Completed analysis stages retained.')
            return
        if self._displayed == self.current_key():return
        self._start()

    def _start(self):
        self.cancel_event.set()
        self.cancel_event = Event()
        self._requested = self.current_key()
        params, mask_id = self._requested
        mask = self.mask if mask_id else None
        self.job.request((params,mask_id,mask,self.cancel_event))

    def set_busy(self,busy):
        self.process_button.setEnabled(True)
        self.process_button.setText('Cancel' if busy else 'Process')
        self.viewer.set_busy(busy or self._displayed != self.current_key())
        if busy:self.status_label.setText('Processing…')

    def _progress(self,event,value,text):
        if not self.job.closed and event is self.cancel_event and not event.is_set():
            self.progress.setValue(value)
            self.status_label.setText(text)

    def show_result(self,result):
        if result is None:return
        key,output,stats = result
        if key != self.current_key():return
        self._displayed = key
        self.stats = stats
        self.viewer.update_processed(output)
        self.set_busy(False)
        self.progress.setValue(100)
        if not stats['filtered']:
            text = 'No keypoints retained with these settings.'
        else:
            text = (f"Keypoints: {stats['total']} → Filtered: {stats['filtered']} → "
                    f"Matches: {stats['matches']} → Clusters: {stats['clusters']} → "
                    f"Direction groups (estimate): {stats['regions']}")
        self.status_label.setText(text)
        self.info_message.emit(f'Copy-Move Forgery = {self.job.seconds:.3f} s')

    def show_error(self,message):
        self.status_label.setText(message)
        self.set_busy(False)

    def shutdown(self):
        self.cancel_event.set()
        super().shutdown()
