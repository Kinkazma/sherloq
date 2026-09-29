from gui.sherloq_app.core.gradient import GradientEngine
from gui.sherloq_app.ui.jobs import LatestJob
from PySide6.QtWidgets import (
    QSpinBox,
    QComboBox,
    QCheckBox,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
)

from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.ui.viewer import ImageViewer


class GradientWidget(ToolWidget):
    def __init__(self, image, parent=None):
        super(GradientWidget, self).__init__(parent)
        self.intensity_spin = QSpinBox()
        self.intensity_spin.setRange(0, 100)
        self.intensity_spin.setValue(95)
        self.intensity_spin.setSuffix(self.tr(" %"))
        self.intensity_spin.setToolTip(self.tr("Tonality compression amount"))
        self.blue_combo = QComboBox()
        self.blue_combo.addItems(
            [self.tr("None"), self.tr("Flat"), self.tr("Abs"), self.tr("Norm")]
        )
        self.blue_combo.setCurrentIndex(2)
        self.blue_combo.setToolTip(self.tr("Blue component inclusion mode"))
        self.invert_check = QCheckBox(self.tr("Invert"))
        self.invert_check.setToolTip(self.tr("Reverse lighting direction"))
        self.equalize_check = QCheckBox(self.tr("Equalize"))
        self.equalize_check.setToolTip(self.tr("Apply histogram equalization"))

        self.image = image
        self.viewer = ImageViewer(self.image, self.image)
        self.engine = GradientEngine(image)
        self.job = LatestJob(self, self.engine.compute, delay=40)
        self.status_label = QLabel()
        self.job.result.connect(self.show_result)
        self.job.failed.connect(lambda error: self.status_label.setText(error))
        self.job.busy.connect(self.set_busy)
        self._requested = None
        self.process()

        self.intensity_spin.valueChanged.connect(self.process)
        self.blue_combo.currentIndexChanged.connect(self.process)
        self.invert_check.stateChanged.connect(self.process)
        self.equalize_check.stateChanged.connect(self.process)

        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel(self.tr("Intensity:")))
        top_layout.addWidget(self.intensity_spin)
        top_layout.addWidget(QLabel(self.tr("Blue channel:")))
        top_layout.addWidget(self.blue_combo)
        top_layout.addWidget(self.invert_check)
        top_layout.addWidget(self.equalize_check)
        top_layout.addWidget(self.status_label)
        top_layout.addStretch()

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.viewer)

        self.setLayout(main_layout)

    def process(self):
        equalize = self.equalize_check.isChecked()
        self.intensity_spin.setEnabled(not equalize)
        params = (0 if equalize else self.intensity_spin.value(),
                  self.blue_combo.currentIndex(), self.invert_check.isChecked(), equalize)
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
        self.info_message.emit(f"Luminance Gradient = {self.job.seconds:.3f} s")
