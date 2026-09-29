from gui.sherloq_app.core.interactive import EchoEngine
from gui.sherloq_app.ui.jobs import LatestJob

from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QCheckBox, QSpinBox, QLabel

from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.ui.viewer import ImageViewer


class EchoWidget(ToolWidget):
    def __init__(self, image, parent=None):
        super(EchoWidget, self).__init__(parent)

        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(1, 15)
        self.radius_spin.setSuffix(self.tr(" px"))
        self.radius_spin.setValue(2)
        self.radius_spin.setToolTip(self.tr("Laplacian filter radius"))

        self.contrast_spin = QSpinBox()
        self.contrast_spin.setRange(0, 100)
        self.contrast_spin.setSuffix(self.tr(" %"))
        self.contrast_spin.setValue(85)
        self.contrast_spin.setToolTip(self.tr("Output tonality compression"))

        self.gray_check = QCheckBox(self.tr("Grayscale"))
        self.gray_check.setToolTip(self.tr("Desaturated output mode"))

        self.image = image
        self.viewer = ImageViewer(self.image, self.image, None)
        self.engine = EchoEngine(image)
        self.job = LatestJob(self, self.engine.compute, delay=40)
        self.status_label = QLabel()
        self.job.result.connect(self.show_result)
        self.job.failed.connect(lambda error: self.status_label.setText("Error: " + error))
        self.job.busy.connect(self.set_busy)
        self._requested = None
        self.process()

        self.radius_spin.valueChanged.connect(self.process)
        self.contrast_spin.valueChanged.connect(self.process)
        self.gray_check.stateChanged.connect(self.process)

        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel(self.tr("Radius:")))
        params_layout.addWidget(self.radius_spin)
        params_layout.addWidget(QLabel(self.tr("Contrast:")))
        params_layout.addWidget(self.contrast_spin)
        params_layout.addWidget(self.gray_check)
        params_layout.addWidget(self.status_label)
        params_layout.addStretch()

        main_layout = QVBoxLayout()
        main_layout.addLayout(params_layout)
        main_layout.addWidget(self.viewer)
        self.setLayout(main_layout)

    def process(self):
        params = (self.radius_spin.value(), self.contrast_spin.value(), self.gray_check.isChecked())
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
        self.info_message.emit(f"Echo Edge Filter = {self.job.seconds:.3f} s")
