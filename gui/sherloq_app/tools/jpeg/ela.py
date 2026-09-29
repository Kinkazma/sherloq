from PySide6.QtCore import QSignalBlocker
from gui.sherloq_app.core.ela import ElaEngine
from gui.sherloq_app.ui.jobs import LatestJob

from PySide6.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QSpinBox,
    QLabel,
    QWidget,
    QTabWidget,
)

from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.ui.viewer import ImageViewer


class ElaWidget(ToolWidget):
    def __init__(self, image, parent=None, filename=None):
        super(ElaWidget, self).__init__(parent)

        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setSuffix(self.tr(" %"))
        self.quality_spin.setToolTip(self.tr("JPEG reference quality level"))
        self.scale_spin = QSpinBox()
        self.scale_spin.setRange(1, 100)
        self.scale_spin.setSuffix(" %")
        self.scale_spin.setToolTip(self.tr("Output multiplicative gain"))
        self.contrast_spin = QSpinBox()
        self.contrast_spin.setRange(0, 100)
        self.contrast_spin.setSuffix(" %")
        self.contrast_spin.setToolTip(self.tr("Output tonality compression"))
        self.linear_check = QCheckBox(self.tr("Linear"))
        self.linear_check.setToolTip(self.tr("Absolute compression error, without square-root amplification"))
        self.gray_check = QCheckBox(self.tr("Grayscale"))
        self.gray_check.setToolTip(self.tr("Desaturated output"))
        default_button = QPushButton(self.tr("Default"))
        default_button.setToolTip(self.tr("Revert to default parameters"))

        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel(self.tr("Quality:")))
        params_layout.addWidget(self.quality_spin)
        params_layout.addWidget(QLabel(self.tr("Scale:")))
        params_layout.addWidget(self.scale_spin)
        params_layout.addWidget(QLabel(self.tr("Contrast:")))
        params_layout.addWidget(self.contrast_spin)
        params_layout.addWidget(self.linear_check)
        params_layout.addWidget(self.gray_check)
        params_layout.addWidget(default_button)
        params_layout.addStretch()

        self.image = image
        self.viewer = ImageViewer(self.image, self.image)
        self.engine = ElaEngine(image)
        self.job = LatestJob(self, self.engine.compute, delay=40)
        self._requested = None
        self.status_label = QLabel()
        self.job.busy.connect(self.set_busy)
        self.job.result.connect(self.show_result)
        self.job.failed.connect(self.show_error)
        self.default()

        self.quality_spin.valueChanged.connect(self.preprocess)
        self.scale_spin.valueChanged.connect(self.process)
        self.contrast_spin.valueChanged.connect(self.process)
        self.linear_check.stateChanged.connect(self.process)
        self.gray_check.stateChanged.connect(self.process)
        default_button.clicked.connect(self.default)

        main_layout = QVBoxLayout()
        main_layout.addLayout(params_layout)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.viewer)
        classic = QWidget()
        classic.setLayout(main_layout)
        from gui.sherloq_app.ui.ela_biomes import ElaBiomesPanel
        from gui.sherloq_app.ui.extensions import mark_tab
        self.biomes = ElaBiomesPanel(image, filename, self)
        self.tabs = QTabWidget()
        self.tabs.addTab(classic, 'ELA')
        self.tabs.addTab(self.biomes, 'Biomes ELA')
        mark_tab(self.tabs, 1)
        outer = QVBoxLayout(self)
        outer.addWidget(self.tabs)
        self.tabs.currentChanged.connect(lambda index: self.biomes.activate() if index == 1 else None)

    def preprocess(self):
        self.process()

    def process(self):
        params = (self.quality_spin.value(), self.scale_spin.value(),
                  self.contrast_spin.value(), self.linear_check.isChecked(),
                  self.gray_check.isChecked())
        if params != self._requested:
            self.job.timer.setInterval(0 if self._requested and params[0] == self._requested[0] else 40)
            self._requested = params
            self.job.request(params)

    def set_busy(self, busy):
        self.viewer.set_busy(busy)
        if busy:
            self.status_label.setText("Calculating…")

    def show_result(self, result):
        self.viewer.update_processed(result)
        self.status_label.setText(f"Error Level Analysis = {self.job.seconds:.3f} s")
        self.info_message.emit(self.status_label.text())

    def show_error(self, message):
        self._requested = None
        self.status_label.setText(message)

    def default(self):
        controls = (self.quality_spin, self.scale_spin, self.contrast_spin,
                    self.linear_check, self.gray_check)
        blockers = [QSignalBlocker(control) for control in controls]
        self.linear_check.setChecked(False)
        self.gray_check.setChecked(False)
        self.quality_spin.setValue(75)
        self.scale_spin.setValue(50)
        self.contrast_spin.setValue(20)
        del blockers
        self.process()

    def shutdown(self):
        self.biomes.shutdown()
        super().shutdown()
