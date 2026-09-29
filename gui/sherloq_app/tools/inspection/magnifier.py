from PySide6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSpinBox,
    QCheckBox,
)

from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.core.magnifier import MagnifierEngine
from gui.sherloq_app.ui.region_viewer import RegionViewer
from gui.sherloq_app.ui.jobs import LatestJob


class MagnifierWidget(ToolWidget):
    def __init__(self, image, parent=None):
        super(MagnifierWidget, self).__init__(parent)

        self.equalize_radio = QRadioButton(self.tr("Equalization"))
        self.equalize_radio.setToolTip(self.tr("RGB histogram equalization"))
        self.contrast_radio = QRadioButton(self.tr("Auto Contrast"))
        self.contrast_radio.setToolTip(self.tr("Compress luminance tonality"))
        self.centile_spin = QSpinBox()
        self.centile_spin.setRange(0, 100)
        self.centile_spin.setValue(20)
        self.centile_spin.setSuffix(self.tr(" %"))
        self.centile_spin.setToolTip(self.tr("Histogram percentile amount"))
        self.channel_check = QCheckBox(self.tr("By channel"))
        self.channel_check.setToolTip(self.tr("Independent RGB compression"))
        self.equalize_radio.setChecked(True)
        self.last_radio = self.equalize_radio

        self.image = image
        self.viewer = RegionViewer(self.image)
        self.engine = MagnifierEngine(image)
        self.job = LatestJob(self, self.engine.compute, delay=40)
        self.requested = None
        self.status = QLabel()
        self.job.result.connect(self.viewer.update_region)
        self.job.busy.connect(self._busy)
        self.job.failed.connect(self._failed)

        self.viewer.viewChanged.connect(self.process)
        self.equalize_radio.clicked.connect(self.change)
        self.contrast_radio.clicked.connect(self.change)
        self.centile_spin.valueChanged.connect(self.change)
        self.channel_check.stateChanged.connect(self.change)

        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel(self.tr("Mode:")))
        top_layout.addWidget(self.equalize_radio)
        top_layout.addWidget(self.contrast_radio)
        top_layout.addWidget(self.centile_spin)
        top_layout.addWidget(self.channel_check)
        top_layout.addStretch()

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.status)
        main_layout.addWidget(self.viewer, 1)
        self.setLayout(main_layout)
        self.change()

    def _busy(self, busy):
        self.viewer.set_busy(busy)
        self.status.setText(self.tr("Updating visible region…") if busy else "")

    def _failed(self, message):
        self.status.setText(message)
        self.viewer.set_busy(True)
        self.requested = None

    def process(self, rect):
        equalize = self.equalize_radio.isChecked()
        self.centile_spin.setEnabled(not equalize)
        self.channel_check.setEnabled(not equalize)
        params = (self.engine.bounds(rect.getRect()),
                  'equalize' if equalize else 'contrast',
                  0 if equalize else self.centile_spin.value(),
                  False if equalize else self.channel_check.isChecked())
        if params != self.requested:
            self.requested = params
            self.job.request(params)

    def change(self):
        self.process(self.viewer.get_rect())
