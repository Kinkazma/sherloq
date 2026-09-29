from gui.sherloq_app.ui.localization import choice
from gui.sherloq_app.core.interactive import WaveletEngine
from gui.sherloq_app.ui.jobs import LatestJob

import pywt
from PySide6.QtWidgets import QSpinBox, QComboBox, QHBoxLayout, QVBoxLayout, QLabel

from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.ui.viewer import ImageViewer


class WaveletWidget(ToolWidget):
    def __init__(self, image, parent=None):
        super(WaveletWidget, self).__init__(parent)

        self.family_combo = QComboBox()
        self.family_combo.addItems(
            [
                self.tr("Daubechies"),
                self.tr("Symlets"),
                self.tr("Coiflets"),
                self.tr("Biorthogonal"),
            ]
        )
        self.wavelet_combo = QComboBox()
        self.wavelet_combo.setMinimumWidth(70)
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 100)
        self.threshold_spin.setSuffix("%")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(
            [
                self.tr("Soft"),
                self.tr("Hard"),
                self.tr("Garrote"),
                self.tr("Greater"),
                self.tr("Less"),
            ]
        )
        self.level_spin = QSpinBox()

        self.image = image
        self.engine = WaveletEngine(image)
        self.job = LatestJob(self, self.engine.compute, delay=40)
        self.status_label = QLabel()
        self.job.result.connect(self.show_result)
        self.job.failed.connect(lambda error: self.status_label.setText("Error: " + error))
        self.job.busy.connect(self.set_busy)
        self.viewer = ImageViewer(self.image, self.image)
        self._requested = None
        self.update_wavelet()

        self.family_combo.activated.connect(self.update_wavelet)
        self.wavelet_combo.activated.connect(self.update_level)
        self.threshold_spin.valueChanged.connect(self.compute_idwt)
        self.mode_combo.activated.connect(self.compute_idwt)
        self.level_spin.valueChanged.connect(self.compute_idwt)

        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel(self.tr("Family:")))
        top_layout.addWidget(self.family_combo)
        top_layout.addWidget(QLabel(self.tr("Wavelet:")))
        top_layout.addWidget(self.wavelet_combo)
        top_layout.addWidget(QLabel(self.tr("Threshold:")))
        top_layout.addWidget(self.threshold_spin)
        top_layout.addWidget(QLabel(self.tr("Mode:")))
        top_layout.addWidget(self.mode_combo)
        top_layout.addWidget(QLabel(self.tr("Level:")))
        top_layout.addWidget(self.level_spin)
        top_layout.addWidget(self.status_label)
        top_layout.addStretch()

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.viewer)
        self.setLayout(main_layout)

    def update_wavelet(self):
        self.wavelet_combo.clear()
        family = self.family_combo.currentIndex()
        if family == 0:
            self.wavelet_combo.addItems([f"db{i}" for i in range(1, 21)])
        elif family == 1:
            self.wavelet_combo.addItems([f"sym{i}" for i in range(2, 21)])
        elif family == 2:
            self.wavelet_combo.addItems([f"coif{i}" for i in range(1, 6)])
        else:
            types = [
                "1.1",
                "1.3",
                "1.5",
                "2.2",
                "2.4",
                "2.6",
                "2.8",
                "3.1",
                "3.3",
                "3.5",
                "3.7",
                "3.9",
                "4.4",
                "5.5",
                "6.8",
            ]
            self.wavelet_combo.addItems([f"bior{t}" for t in types])
        self.update_level()

    def update_level(self):
        wavelet = choice(self.wavelet_combo)
        max_level = pywt.dwtn_max_level(self.image.shape[:-1], wavelet)
        self.level_spin.blockSignals(True)
        self.level_spin.setRange(0 if max_level == 0 else 1, max_level)
        self.level_spin.setValue(max_level // 2)
        self.level_spin.blockSignals(False)
        self.compute_dwt()

    def compute_dwt(self):
        self.compute_idwt()

    def compute_idwt(self):
        threshold, level = self.threshold_spin.value(), self.level_spin.value()
        mode = choice(self.mode_combo).lower()
        if threshold == 0 or level == 0:
            threshold, level, mode = 0, 0, 'soft'
        params = choice(self.wavelet_combo), threshold, level, mode
        if params != self._requested:
            self._requested = params
            self.job.request(params)

    def set_busy(self, busy):
        self.viewer.set_busy(busy)
        if busy:
            self.status_label.setText(self.tr("Calculating…"))

    def show_result(self, image):
        self.viewer.update_processed(image)
        self.status_label.setText(f"{self.job.seconds:.3f} s")
