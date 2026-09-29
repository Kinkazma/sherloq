from gui.sherloq_app.core.pixel_stats import StatsEngine
from gui.sherloq_app.ui.jobs import LatestJob
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QRadioButton

from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.ui.viewer import ImageViewer


class StatsWidget(ToolWidget):
    def __init__(self, image, parent=None):
        super(StatsWidget, self).__init__(parent)

        self.min_radio = QRadioButton(self.tr("Minimum"))
        self.min_radio.setToolTip(self.tr("RGB channel with smallest value"))
        self.min_radio.setChecked(True)
        self.last_radio = self.min_radio
        self.avg_radio = QRadioButton(self.tr("Middle"))
        self.avg_radio.setToolTip(self.tr("RGB channel with middle value (median)"))
        self.max_radio = QRadioButton(self.tr("Maximum"))
        self.max_radio.setToolTip(self.tr("RGB channel with largest value"))
        self.incl_check = QCheckBox(self.tr("Inclusive"))
        self.incl_check.setToolTip(self.tr("Include equal values; ties use the original priority: red, then green, then blue."))

        self.image = image
        self.viewer = ImageViewer(self.image, self.image)
        self.engine = StatsEngine(image)
        self.job = LatestJob(self, self.engine.compute, delay=0)
        self.status_label = QLabel()
        self.job.result.connect(self.show_result)
        self.job.busy.connect(self.set_busy)
        self.job.failed.connect(self.show_error)
        self._requested = None
        self.process()

        self.min_radio.clicked.connect(self.process)
        self.avg_radio.clicked.connect(self.process)
        self.max_radio.clicked.connect(self.process)
        self.incl_check.stateChanged.connect(self.process)

        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel(self.tr("Mode:")))
        params_layout.addWidget(self.min_radio)
        params_layout.addWidget(self.avg_radio)
        params_layout.addWidget(self.max_radio)
        params_layout.addWidget(self.incl_check)
        params_layout.addWidget(self.status_label)
        params_layout.addStretch()

        main_layout = QVBoxLayout()
        main_layout.addLayout(params_layout)
        main_layout.addWidget(self.viewer)
        self.setLayout(main_layout)

    def process(self):
        mode = next((name for name in ('min', 'avg', 'max')
                     if getattr(self, name+'_radio').isChecked()), None)
        if mode is None:
            self.last_radio.setChecked(True)
            return
        self.last_radio = getattr(self, mode+'_radio')
        params = mode, self.incl_check.isChecked()
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
