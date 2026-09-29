from gui.sherloq_app.core.bit_planes import PlanesEngine
from gui.sherloq_app.ui.jobs import LatestJob
from PySide6.QtWidgets import QHBoxLayout, QComboBox, QLabel, QVBoxLayout, QSpinBox

from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.ui.viewer import ImageViewer


class PlanesWidget(ToolWidget):
    def __init__(self, image, parent=None):
        super(PlanesWidget, self).__init__(parent)

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
        self.chan_combo.setToolTip(self.tr("RGB Norm: truncated square root of R²+G²+B², modulo 256 (original convention)."))
        self.plane_spin = QSpinBox()
        self.plane_spin.setPrefix(self.tr("Bit "))
        self.plane_spin.setRange(0, 7)
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(
            [self.tr("Disabled"), self.tr("Median"), self.tr("Gaussian")]
        )

        self.image = image
        self.viewer = ImageViewer(self.image, self.image)
        self.engine = PlanesEngine(image)
        self.job = LatestJob(self, self.engine.compute, delay=0)
        self.status_label = QLabel()
        self._requested = None
        self.job.result.connect(self.show_result)
        self.job.busy.connect(self.set_busy)
        self.job.failed.connect(self.show_error)
        self.process()

        self.chan_combo.currentIndexChanged.connect(self.process)
        self.plane_spin.valueChanged.connect(self.process)
        self.filter_combo.currentIndexChanged.connect(self.process)

        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel(self.tr("Channel:")))
        top_layout.addWidget(self.chan_combo)
        top_layout.addWidget(QLabel(self.tr("Plane:")))
        top_layout.addWidget(self.plane_spin)
        top_layout.addWidget(QLabel(self.tr("Filter:")))
        top_layout.addWidget(self.filter_combo)
        top_layout.addWidget(self.status_label)
        top_layout.addStretch()

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.viewer)
        self.setLayout(main_layout)

    def process(self):
        params = self.chan_combo.currentIndex(), self.plane_spin.value(), self.filter_combo.currentIndex()
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

    def show_error(self, error):
        self._requested = None
        self.status_label.setText(error)
