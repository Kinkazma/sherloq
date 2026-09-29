from gui.sherloq_app.core.minmax import MinMaxEngine
from gui.sherloq_app.ui.jobs import LatestJob
from PySide6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QSpinBox,
    QPushButton,
    QLabel,
)

from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.ui.viewer import ImageViewer


class MinMaxWidget(ToolWidget):
    def __init__(self, image, parent=None):
        super(MinMaxWidget, self).__init__(parent)

        self.chan_combo = QComboBox()
        self.chan_combo.addItems(
            [
                self.tr("Luminance"),
                self.tr("Red"),
                self.tr("Green"),
                self.tr("Blue"),
                self.tr("RGB Norm"),
            ]
        )
        colors = [
            self.tr("Red"),
            self.tr("Green"),
            self.tr("Blue"),
            self.tr("White"),
            self.tr("Black"),
        ]
        self.process_button = QPushButton(self.tr("Process"))

        self.min_combo = QComboBox()
        self.min_combo.addItems(colors)
        self.min_combo.setCurrentIndex(1)

        self.max_combo = QComboBox()
        self.max_combo.addItems(colors)
        self.max_combo.setCurrentIndex(0)

        self.filter_spin = QSpinBox()
        self.filter_spin.setRange(0, 5)
        self.filter_spin.setSpecialValueText(self.tr("Off"))

        self.image = image
        self.viewer = ImageViewer(self.image, self.image)
        self.low = self.high = None
        self.engine = MinMaxEngine(image)
        self.job = LatestJob(self, self.engine.compute, delay=0)
        self.status_label = QLabel()
        self._requested = None
        self._available = False
        self.job.result.connect(self.show_result)
        self.job.busy.connect(self.set_busy)
        self.job.failed.connect(self.show_error)
        self.change()

        self.process_button.clicked.connect(self.preprocess)
        self.chan_combo.currentIndexChanged.connect(self.change)
        self.min_combo.currentIndexChanged.connect(self.process)
        self.max_combo.currentIndexChanged.connect(self.process)
        self.filter_spin.valueChanged.connect(self.process)

        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel(self.tr("Channel:")))
        top_layout.addWidget(self.chan_combo)
        top_layout.addWidget(self.process_button)
        top_layout.addWidget(QLabel(self.tr("Minimum:")))
        top_layout.addWidget(self.min_combo)
        top_layout.addWidget(QLabel(self.tr("Maximum:")))
        top_layout.addWidget(self.max_combo)
        top_layout.addWidget(QLabel(self.tr("Filter:")))
        top_layout.addWidget(self.filter_spin)
        top_layout.addWidget(self.status_label)
        top_layout.addStretch()
        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.viewer)
        self.setLayout(main_layout)

    def change(self):
        self.job.invalidate()
        self._requested = None
        self._available = False
        self.low = self.high = None
        self.min_combo.setEnabled(False)
        self.max_combo.setEnabled(False)
        self.filter_spin.setEnabled(False)
        self.process_button.setEnabled(True)
        self.process_button.setText(self.tr("Process"))
        self.status_label.clear()
        self.viewer.update_processed(self.image)
        self.viewer.set_busy(True)

    def preprocess(self):
        if self.job.is_busy:
            self.cancel()
            return
        self._request()

    def cancel(self):
        self.job.invalidate()
        self._requested = None
        self.process_button.setText(self.tr("Process"))
        self.process_button.setEnabled(True)
        self.status_label.setText(self.tr("Cancelled"))
        self.viewer.set_busy(True)

    def process(self):
        if self._available:
            self._request()

    def _request(self):
        params = (self.chan_combo.currentIndex(), self.min_combo.currentIndex(),
                  self.max_combo.currentIndex(), self.filter_spin.value())
        if params != self._requested:
            self._requested = params
            self.job.request(params)

    def set_busy(self, busy):
        self.viewer.set_busy(busy)
        self.process_button.setText(self.tr("Cancel") if busy else self.tr("Process"))
        self.process_button.setEnabled(busy or not self._available)
        if busy:
            self.status_label.setText(self.tr("Calculating…"))

    def show_result(self, result):
        image, self.low, self.high = result
        self._available = True
        self.min_combo.setEnabled(True)
        self.max_combo.setEnabled(True)
        self.filter_spin.setEnabled(True)
        self.process_button.setEnabled(False)
        self.viewer.update_processed(image)
        self.status_label.setText(f"{self.job.seconds:.3f} s")

    def show_error(self, error):
        self._requested = None
        self.process_button.setEnabled(True)
        self.viewer.set_busy(True)
        self.status_label.setText(error)
