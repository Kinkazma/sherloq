from PySide6.QtWidgets import (
    QVBoxLayout,
    QLabel,
    QGridLayout,
    QCheckBox,
    QPushButton,
    QComboBox,
)

from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.core.utility import ParamSlider
from gui.sherloq_app.core.adjust import AdjustEngine
from gui.sherloq_app.ui.jobs import LatestJob
from gui.sherloq_app.ui.viewer import ImageViewer


class AdjustWidget(ToolWidget):
    def __init__(self, image, parent=None):
        super(AdjustWidget, self).__init__(parent)

        self.bright_slider = ParamSlider([-255, +255], 16, 0)
        self.sat_slider = ParamSlider([-255, +255], 16, 0)
        self.hue_slider = ParamSlider([0, 180], 10, 0, "°")
        self.gamma_slider = ParamSlider([1, 50], 10, 10)
        self.shadow_slider = ParamSlider([-100, +100], 10, 0, "%")
        self.high_slider = ParamSlider([-100, +100], 10, 0, "%")
        self.sweep_slider = ParamSlider([0, 255], 8, 127)
        self.width_slider = ParamSlider([0, 255], 8, 255)
        self.sharpen_slider = ParamSlider([0, 100], 10, 0, "%")
        self.thr_slider = ParamSlider([0, 255], 16, 255, special=self.tr("Auto"))
        self.equalize_combo = QComboBox()
        self.equalize_combo.addItems(
            [
                self.tr("No EQ"),
                self.tr("Hist EQ"),
                self.tr("CLAHE L1"),
                self.tr("CLAHE L2"),
                self.tr("CLAHE L3"),
                self.tr("CLAHE L4"),
            ]
        )
        self.equalize_combo.setToolTip(self.tr("Histogram equalization mode"))
        self.invert_check = QCheckBox(self.tr("Invert values"))
        self.invert_check.setToolTip(self.tr("Apply bitwise complement"))
        self.reset_button = QPushButton(self.tr("Reset"))

        self.image = image
        self.viewer = ImageViewer(self.image, self.image)
        self.engine = AdjustEngine(image)
        self.job = LatestJob(self, self.engine.compute, delay=40)
        self.job.result.connect(self.viewer.update_processed)
        self.job.busy.connect(self._busy)
        self.job.failed.connect(self._failed)
        self.status = QLabel()
        self.requested = None
        self._resetting = False
        self.reset()

        self.bright_slider.valueChanged.connect(self.process)
        self.sat_slider.valueChanged.connect(self.process)
        self.hue_slider.valueChanged.connect(self.process)
        self.gamma_slider.valueChanged.connect(self.process)
        self.shadow_slider.valueChanged.connect(self.process)
        self.high_slider.valueChanged.connect(self.process)
        self.sweep_slider.valueChanged.connect(self.process)
        self.width_slider.valueChanged.connect(self.process)
        self.thr_slider.valueChanged.connect(self.process)
        self.sharpen_slider.valueChanged.connect(self.process)
        self.equalize_combo.currentIndexChanged.connect(self.process)
        self.invert_check.stateChanged.connect(self.process)
        self.reset_button.clicked.connect(self.reset)

        params_layout = QGridLayout()
        params_layout.addWidget(QLabel(self.tr("Brightness")), 0, 0)
        params_layout.addWidget(QLabel(self.tr("Saturation")), 1, 0)
        params_layout.addWidget(QLabel(self.tr("Hue")), 2, 0)
        params_layout.addWidget(QLabel(self.tr("Gamma")), 3, 0)
        params_layout.addWidget(self.bright_slider, 0, 1)
        params_layout.addWidget(self.sat_slider, 1, 1)
        params_layout.addWidget(self.hue_slider, 2, 1)
        params_layout.addWidget(self.gamma_slider, 3, 1)
        params_layout.addWidget(QLabel(self.tr("Shadows")), 0, 2)
        params_layout.addWidget(QLabel(self.tr("Highlights")), 1, 2)
        params_layout.addWidget(QLabel(self.tr("Sweep")), 2, 2)
        params_layout.addWidget(QLabel(self.tr("Width")), 3, 2)
        params_layout.addWidget(self.shadow_slider, 0, 3)
        params_layout.addWidget(self.high_slider, 1, 3)
        params_layout.addWidget(self.sweep_slider, 2, 3)
        params_layout.addWidget(self.width_slider, 3, 3)
        params_layout.addWidget(QLabel(self.tr("Sharpen")), 0, 4)
        params_layout.addWidget(self.sharpen_slider, 0, 5)
        params_layout.addWidget(QLabel(self.tr("Threshold")), 1, 4)
        params_layout.addWidget(self.thr_slider, 1, 5)
        params_layout.addWidget(self.equalize_combo, 2, 4)
        params_layout.addWidget(self.invert_check, 2, 5)
        params_layout.addWidget(self.reset_button, 3, 4, 1, 2)

        main_layout = QVBoxLayout()
        main_layout.addLayout(params_layout)
        main_layout.addWidget(self.status)
        main_layout.addWidget(self.viewer, 1)
        self.setLayout(main_layout)

    def _busy(self, busy):
        self.viewer.set_busy(busy)
        self.status.setText(self.tr("Calculating…") if busy else "")

    def _failed(self, message):
        self.status.setText(message)
        self.viewer.set_busy(True)
        self.requested = None

    def process(self):
        if self._resetting:
            return
        params = tuple(slider.value() for slider in (
            self.bright_slider, self.sat_slider, self.hue_slider,
            self.gamma_slider, self.shadow_slider, self.high_slider,
            self.sweep_slider, self.width_slider, self.sharpen_slider,
            self.thr_slider,
        )) + (self.equalize_combo.currentIndex(), self.invert_check.isChecked())
        key = params[:8] + (params[8] // 4,) + params[9:]
        if key != self.requested:
            self.requested = key
            self.job.request(params)

    def reset(self):
        self._resetting = True
        self.bright_slider.reset_value()
        self.sat_slider.reset_value()
        self.hue_slider.reset_value()
        self.gamma_slider.reset_value()
        self.shadow_slider.reset_value()
        self.high_slider.reset_value()
        self.sweep_slider.reset_value()
        self.width_slider.reset_value()
        self.sharpen_slider.reset_value()
        self.thr_slider.reset_value()
        self.equalize_combo.setCurrentIndex(0)
        self.invert_check.setChecked(False)
        self._resetting = False
        self.process()
