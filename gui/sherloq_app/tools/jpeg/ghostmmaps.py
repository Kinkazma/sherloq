"""JPEG Ghost Maps: cached numeric maps, isolated rendering and safe cancellation."""
from functools import partial
from threading import Event
from PySide6.QtCore import QObject, Signal, QSignalBlocker
from PySide6.QtWidgets import (QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
                              QCheckBox, QPushButton, QProgressBar)
from gui.sherloq_app.core.ghost_maps import GhostEngine
from gui.sherloq_app.core.jpeg_curve import Cancelled
from gui.sherloq_app.ui.jobs import LatestJob
from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.ui.viewer import ImageViewer


class _Progress(QObject):
    changed = Signal(object, int, str)


def _run_ghost(engine, updates, request):
    params, cancel = request
    try:
        plot, maps = engine.compute(params, cancel.is_set,
            lambda value, text: updates.changed.emit(cancel, value, text))
        return params, plot, maps
    except Cancelled:
        return None


class GhostmapWidget(ToolWidget):
    # tool layout
    def __init__(self, filename, image, parent=None):
        super(GhostmapWidget, self).__init__(parent)

        # save variables to self
        self.filename = filename
        self.image = image

        # store different xy-offsets so user can quickly cycle different maps and inspect changes
        self.engine = GhostEngine(image)
        self.cancel_event = Event()
        self._requested = None
        self._displayed = None
        self.maps = None

        # prepare user interface - input variables
        # qmin
        self.qmin_spin = QSpinBox()
        self.qmin_spin.setRange(0, 100)
        self.qmin_spin.setValue(50)
        # qmax
        self.qmax_spin = QSpinBox()
        self.qmax_spin.setRange(0, 100)
        self.qmax_spin.setValue(90)
        # qstep
        self.qstep_spin = QSpinBox()
        self.qstep_spin.setRange(1, 20)
        self.qstep_spin.setValue(5)
        # lattice X offset
        self.xoffset_spin = QSpinBox()
        self.xoffset_spin.setRange(0, 7)
        self.xoffset_spin.setValue(0)
        # lattice Y offset
        self.yoffset_spin = QSpinBox()
        self.yoffset_spin.setRange(0, 7)
        self.yoffset_spin.setValue(0)
        # grayscale
        self.showgray_check = QCheckBox(self.tr("Grayscale"))
        self.showgray_check.setChecked(True)
        # plot original above the maps?
        self.includeoriginal_check = QCheckBox(self.tr("Include original in plot?"))
        self.includeoriginal_check.setChecked(False)
        # calculate ghost maps
        self.process_button = QPushButton(self.tr("Calculate Maps"))
        # calculate next offset ghost maps
        self.process_next_offset_button = QPushButton(self.tr("Next offset"))
        # calculate previous offset ghost maps
        self.process_previous_offset_button = QPushButton(self.tr("Previous offset"))

        # combine top layout
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel(self.tr("Lower Quality:")))
        top_layout.addWidget(self.qmin_spin)
        top_layout.addWidget(QLabel(self.tr("Upper Quality:")))
        top_layout.addWidget(self.qmax_spin)
        top_layout.addWidget(QLabel(self.tr("Quality Step:")))
        top_layout.addWidget(self.qstep_spin)
        top_layout.addWidget(self.showgray_check)
        top_layout.addWidget(self.includeoriginal_check)
        top_layout.addWidget(self.process_button)
        offset_layout = QHBoxLayout()
        offset_layout.addWidget(QLabel(self.tr("Offset X:")))
        offset_layout.addWidget(self.xoffset_spin)
        offset_layout.addWidget(QLabel(self.tr("Offset Y:")))
        offset_layout.addWidget(self.yoffset_spin)
        offset_layout.addWidget(self.process_previous_offset_button)
        offset_layout.addWidget(self.process_next_offset_button)
        offset_layout.addStretch()
        top_layout.addStretch()

        self.viewer = ImageViewer(image, image, None)

        self.status_label = QLabel()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.updates = _Progress()
        self.updates.changed.connect(self._progress)
        self.job = LatestJob(self, partial(_run_ghost, self.engine, self.updates), delay=0)
        self.job.busy.connect(self.set_busy)
        self.job.result.connect(self.show_result)
        self.job.failed.connect(self.show_error)
        for control in (self.qmin_spin, self.qmax_spin, self.qstep_spin,
                        self.xoffset_spin, self.yoffset_spin):
            control.valueChanged.connect(self.invalidate)
        self.showgray_check.stateChanged.connect(self.invalidate)
        self.includeoriginal_check.stateChanged.connect(self.invalidate)
        self.processGhostmaps()

        self.process_button.clicked.connect(self.processGhostmaps)
        self.process_previous_offset_button.clicked.connect(
            self.calculate_previous_offset
        )
        self.process_next_offset_button.clicked.connect(self.calculate_next_offset)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addLayout(offset_layout)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.progress)
        main_layout.addWidget(self.viewer)
        self.setLayout(main_layout)

    def parameters(self):
        return (self.qmin_spin.value(), self.qmax_spin.value(), self.qstep_spin.value(),
                self.xoffset_spin.value(), self.yoffset_spin.value(),
                self.showgray_check.isChecked(), self.includeoriginal_check.isChecked())

    def invalidate(self):
        self.cancel_event.set()
        self.job.invalidate()
        self._requested = None
        self.set_busy(False)
        self.status_label.setText("Parameters changed. Calculate Maps to update.")

    def calculate_next_offset(self):
        self._move_offset(1)

    def calculate_previous_offset(self):
        self._move_offset(-1)

    def _move_offset(self, direction):
        index = (self.xoffset_spin.value() + 8*self.yoffset_spin.value() + direction) % 64
        blockers = [QSignalBlocker(self.xoffset_spin), QSignalBlocker(self.yoffset_spin)]
        self.xoffset_spin.setValue(index % 8)
        self.yoffset_spin.setValue(index // 8)
        del blockers
        self.processGhostmaps()

    def processGhostmaps(self):
        params = self.parameters()
        if self.job.is_busy and params == self._requested:
            self.cancel_event.set()
            self.job.invalidate()
            self._requested = None
            self.set_busy(False)
            self.status_label.setText("Cancelled. Completed quality maps are retained.")
            return
        if params == self._displayed:
            self.set_busy(False)
            return
        self.cancel_event.set()
        self.cancel_event = Event()
        self._requested = params
        self.job.request((params, self.cancel_event))

    def set_busy(self, busy):
        self.process_button.setText("Cancel" if busy else "Calculate Maps")
        self.viewer.set_busy(busy or self._displayed != self.parameters())
        if busy:
            self.status_label.setText("Calculating…")

    def _progress(self, event, value, text):
        if not self.job.closed and event is self.cancel_event and not event.is_set():
            self.progress.setValue(value)
            self.status_label.setText(text)

    def show_result(self, result):
        if result is None:
            self._requested = None
            self.set_busy(False)
            return
        params, plot, maps = result
        if params != self.parameters():
            return
        self._displayed = params
        self.maps = maps
        self.viewer.update_processed(plot)
        self.set_busy(False)
        self.progress.setValue(100)
        self.status_label.setText(f"JPEG Ghost Maps = {self.job.seconds:.3f} s")
        if maps.shape[2] == 1:
            self.status_label.setText("One quality selected: normalization across qualities gives zero maps. Select a range to compare.")
        self.info_message.emit(self.status_label.text())

    def show_error(self, message):
        self._requested = None
        self.set_busy(False)
        self.status_label.setText(message)

    def shutdown(self):
        self.cancel_event.set()
        super().shutdown()
