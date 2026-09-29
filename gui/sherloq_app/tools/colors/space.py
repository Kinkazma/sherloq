from gui.sherloq_app.core.color_spaces import SpaceEngine
from gui.sherloq_app.ui.jobs import LatestJob
from PySide6.QtWidgets import (
    QVBoxLayout,
    QGridLayout,
    QRadioButton,
    QComboBox,
    QHBoxLayout,
    QLabel,
)

from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.core.utility import modify_font
from gui.sherloq_app.ui.viewer import ImageViewer


class SpaceWidget(ToolWidget):
    def __init__(self, image, parent=None):
        super(SpaceWidget, self).__init__(parent)
        self.rgb_radio = QRadioButton(self.tr("RGB"))
        self.rgb_radio.setChecked(True)
        self.rgb_combo = QComboBox()
        self.rgb_combo.addItem(self.tr("Red"))
        self.rgb_combo.addItem(self.tr("Green"))
        self.rgb_combo.addItem(self.tr("Blue"))
        self.last_radio = self.rgb_radio

        self.cmyk_radio = QRadioButton(self.tr("CMYK"))
        self.cmyk_radio.setToolTip(self.tr("Uncalibrated CMYK decomposition; no printer ICC profile is applied."))
        self.cmyk_combo = QComboBox()
        self.cmyk_combo.addItem(self.tr("Cyan"))
        self.cmyk_combo.addItem(self.tr("Magenta"))
        self.cmyk_combo.addItem(self.tr("Yellow"))
        self.cmyk_combo.addItem(self.tr("Black"))

        self.gray_radio = QRadioButton(self.tr("Grayscale"))
        self.gray_combo = QComboBox()
        self.gray_combo.addItem(self.tr("Lightness"))
        self.gray_combo.addItem(self.tr("Luminance"))
        self.gray_combo.addItem(self.tr("Average"))
        self.gray_combo.addItem(self.tr("Perceptual"))

        self.hsv_radio = QRadioButton(self.tr("HSV"))
        self.hsv_combo = QComboBox()
        self.hsv_combo.addItem(self.tr("Hue"))
        self.hsv_combo.addItem(self.tr("Saturation"))
        self.hsv_combo.addItem(self.tr("Value"))

        self.hls_radio = QRadioButton(self.tr("HLS"))
        self.hls_combo = QComboBox()
        self.hls_combo.addItem(self.tr("Hue"))
        self.hls_combo.addItem(self.tr("Luminance"))
        self.hls_combo.addItem(self.tr("Saturation"))

        self.ycrcb_radio = QRadioButton(self.tr("YCrCb"))
        self.ycrcb_combo = QComboBox()
        self.ycrcb_combo.addItem(self.tr("Luminance"))
        self.ycrcb_combo.addItem(self.tr("Chroma Red"))
        self.ycrcb_combo.addItem(self.tr("Chroma Blue"))

        self.xyz_radio = QRadioButton(self.tr("CIE XYZ"))
        self.xyz_combo = QComboBox()
        self.xyz_combo.addItem(self.tr("X"))
        self.xyz_combo.addItem(self.tr("Y"))
        self.xyz_combo.addItem(self.tr("Z"))

        self.lab_radio = QRadioButton(self.tr("CIE Lab"))
        self.lab_combo = QComboBox()
        self.lab_combo.addItem(self.tr("Luminosity"))
        self.lab_combo.addItem(self.tr("Green-Red"))
        self.lab_combo.addItem(self.tr("Blue-Yellow"))

        self.luv_radio = QRadioButton(self.tr("CIE Luv"))
        self.luv_combo = QComboBox()
        self.luv_combo.addItem(self.tr("Luminosity"))
        self.luv_combo.addItem(self.tr("Chroma U"))
        self.luv_combo.addItem(self.tr("Chroma V"))

        self.rgb_radio.clicked.connect(self.process)
        self.rgb_combo.currentIndexChanged.connect(self.process)
        self.cmyk_radio.clicked.connect(self.process)
        self.cmyk_combo.currentIndexChanged.connect(self.process)
        self.gray_radio.clicked.connect(self.process)
        self.gray_combo.currentIndexChanged.connect(self.process)
        self.hsv_radio.clicked.connect(self.process)
        self.hsv_combo.currentIndexChanged.connect(self.process)
        self.hls_radio.clicked.connect(self.process)
        self.hls_combo.currentIndexChanged.connect(self.process)
        self.ycrcb_radio.clicked.connect(self.process)
        self.ycrcb_combo.currentIndexChanged.connect(self.process)
        self.xyz_radio.clicked.connect(self.process)
        self.xyz_combo.currentIndexChanged.connect(self.process)
        self.lab_radio.clicked.connect(self.process)
        self.lab_combo.currentIndexChanged.connect(self.process)
        self.luv_radio.clicked.connect(self.process)
        self.luv_combo.currentIndexChanged.connect(self.process)

        self.viewer = ImageViewer(image, image)
        self.engine = SpaceEngine(image)
        self.job = LatestJob(self, self.engine.compute, delay=0)
        self.status_label = QLabel()
        self.job.result.connect(self.show_result)
        self.job.failed.connect(self.show_error)
        self.job.busy.connect(self.set_busy)
        self._requested = None
        self.process()

        grid_layout = QGridLayout()
        grid_layout.addWidget(self.rgb_radio, 0, 0)
        grid_layout.addWidget(self.rgb_combo, 0, 1)
        grid_layout.addWidget(self.cmyk_radio, 0, 2)
        grid_layout.addWidget(self.cmyk_combo, 0, 3)
        grid_layout.addWidget(self.gray_radio, 0, 4)
        grid_layout.addWidget(self.gray_combo, 0, 5)
        grid_layout.addWidget(self.hsv_radio, 1, 0)
        grid_layout.addWidget(self.hsv_combo, 1, 1)
        grid_layout.addWidget(self.hls_radio, 1, 2)
        grid_layout.addWidget(self.hls_combo, 1, 3)
        grid_layout.addWidget(self.ycrcb_radio, 1, 4)
        grid_layout.addWidget(self.ycrcb_combo, 1, 5)
        grid_layout.addWidget(self.xyz_radio, 2, 0)
        grid_layout.addWidget(self.xyz_combo, 2, 1)
        grid_layout.addWidget(self.lab_radio, 2, 2)
        grid_layout.addWidget(self.lab_combo, 2, 3)
        grid_layout.addWidget(self.luv_radio, 2, 4)
        grid_layout.addWidget(self.luv_combo, 2, 5)
        top_layout = QHBoxLayout()
        top_layout.addLayout(grid_layout)
        top_layout.addWidget(self.status_label)
        top_layout.addStretch()

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.viewer)
        self.setLayout(main_layout)

    def process(self):
        spaces = ('rgb', 'cmyk', 'gray', 'hsv', 'hls', 'ycrcb', 'xyz', 'lab', 'luv')
        active = next((name for name in spaces if getattr(self, name+'_radio').isChecked()), None)
        if active is None:
            self.last_radio.setChecked(True)
            return
        self.last_radio = getattr(self, active+'_radio')
        for name in spaces:
            modify_font(getattr(self, name+'_radio'), bold=name == active)
        params = active, getattr(self, active+'_combo').currentIndex()
        if params != self._requested:
            self._requested = params
            self.job.request(params)

    def set_busy(self, busy):
        self.viewer.set_busy(busy)
        if busy:
            self.status_label.setText(self.tr("Converting…"))

    def show_result(self, image):
        self.viewer.update_processed(image)
        self.status_label.setText(f"{self.job.seconds:.3f} s")

    def show_error(self, error):
        self._requested = None
        self.status_label.setText(error)
