from gui.sherloq_app.core.noise import NoiseEngine
from gui.sherloq_app.ui.jobs import LatestJob
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QCheckBox,
    QLabel,
    QVBoxLayout,
    QSpinBox,
)

from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.ui.viewer import ImageViewer


class NoiseWidget(ToolWidget):
    def __init__(self, image, parent=None):
        super(NoiseWidget, self).__init__(parent)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(
            [
                self.tr("Median"),
                self.tr("Gaussian"),
                self.tr("BoxBlur"),
                self.tr("Bilateral"),
                self.tr("NonLocal"),
            ]
        )

        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(1, 10)
        self.radius_spin.setSuffix(self.tr(" px"))
        self.radius_spin.setValue(1)

        self.sigma_spin = QSpinBox()
        self.sigma_spin.setRange(1, 200)
        self.sigma_spin.setValue(3)

        self.levels_spin = QSpinBox()
        self.levels_spin.setRange(0, 255)
        self.levels_spin.setSpecialValueText(self.tr("Equalized"))
        self.levels_spin.setValue(32)

        self.gray_check = QCheckBox(self.tr("Grayscale"))
        self.denoised_check = QCheckBox(self.tr("Denoised"))

        self.image = image
        self.viewer = ImageViewer(self.image, self.image)
        self.engine = NoiseEngine(image)
        self.job = LatestJob(self, self.engine.compute, delay=40)
        self.status_label = QLabel()
        self.radius_label = QLabel(self.tr("Radius:"))
        self._requested = None
        self.job.result.connect(self.show_result)
        self.job.failed.connect(self.show_error)
        self.job.busy.connect(self.set_busy)
        self.process()

        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel(self.tr("Mode:")))
        params_layout.addWidget(self.mode_combo)
        params_layout.addWidget(self.radius_label)
        params_layout.addWidget(self.radius_spin)
        params_layout.addWidget(QLabel(self.tr("Sigma:")))
        params_layout.addWidget(self.sigma_spin)
        params_layout.addWidget(QLabel(self.tr("Levels:")))
        params_layout.addWidget(self.levels_spin)
        params_layout.addWidget(self.gray_check)
        params_layout.addWidget(self.denoised_check)
        params_layout.addWidget(self.status_label)
        params_layout.addStretch()

        main_layout = QVBoxLayout()
        main_layout.addLayout(params_layout)
        main_layout.addWidget(self.viewer)
        self.setLayout(main_layout)

        self.mode_combo.currentTextChanged.connect(self.process)
        self.radius_spin.valueChanged.connect(self.process)
        self.sigma_spin.valueChanged.connect(self.process)
        self.levels_spin.valueChanged.connect(self.process)
        self.gray_check.stateChanged.connect(self.process)
        self.denoised_check.stateChanged.connect(self.process)

    def process(self):
        mode = self.mode_combo.currentIndex()
        denoised = self.denoised_check.isChecked()
        self.sigma_spin.setEnabled(mode == 3)
        self.levels_spin.setEnabled(not denoised)
        self.radius_label.setText(self.tr("Strength step:") if mode == 4 else self.tr("Radius:"))
        self.radius_spin.setSuffix("" if mode == 4 else self.tr(" px"))
        self.radius_spin.setToolTip(self.tr("Non-local strength h = 2 × step + 1; default search and template windows") if mode == 4 else "")
        params = self.engine.parameters((mode, self.radius_spin.value(), self.sigma_spin.value(),
                                         self.gray_check.isChecked(), denoised, self.levels_spin.value()))
        if params != self._requested:
            display_only = self._requested is not None and params[:4] == self._requested[:4]
            self.job.timer.setInterval(0 if display_only else 40)
            self._requested = params
            self.job.request(params)

    def set_busy(self, busy):
        self.viewer.set_busy(busy)
        if busy:
            self.status_label.setText(self.tr("Calculating…"))

    def show_result(self, image):
        self.viewer.update_processed(image)
        self.status_label.setText(f"{self.job.seconds:.3f} s")
        self.info_message.emit(f"Signal Separation = {self.job.seconds:.3f} s")

    def show_error(self, error):
        self._requested = None
        self.status_label.setText(error)
