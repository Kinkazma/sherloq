from gui.sherloq_app.core.interactive import FrequencyEngine
from gui.sherloq_app.ui.jobs import LatestJob

from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QGridLayout, QSpinBox

from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.core.utility import modify_font
from gui.sherloq_app.ui.viewer import ImageViewer


class FrequencyWidget(ToolWidget):
    def __init__(self, image, parent=None):
        super(FrequencyWidget, self).__init__(parent)

        self.split_spin = QSpinBox()
        self.split_spin.setRange(0, 100)
        self.split_spin.setValue(15)
        self.split_spin.setSuffix(self.tr(" %"))

        self.smooth_spin = QSpinBox()
        self.smooth_spin.setRange(0, 100)
        self.smooth_spin.setValue(25)
        self.smooth_spin.setSuffix(self.tr(" %"))
        self.smooth_spin.setSpecialValueText(self.tr("Off"))

        self.thr_spin = QSpinBox()
        self.thr_spin.setRange(0, 100)
        self.thr_spin.setValue(0)
        self.thr_spin.setSuffix(self.tr(" %"))
        self.thr_spin.setSpecialValueText(self.tr("Off"))

        self.zero_label = QLabel()
        modify_font(self.zero_label, italic=True)

        self.filter_spin = QSpinBox()
        self.filter_spin.setRange(0, 15)
        self.filter_spin.setValue(0)
        self.filter_spin.setSuffix(self.tr(" px"))
        self.filter_spin.setSpecialValueText(self.tr("Off"))

        self.split_spin.valueChanged.connect(self.process)
        self.smooth_spin.valueChanged.connect(self.process)
        self.thr_spin.valueChanged.connect(self.process)
        self.filter_spin.valueChanged.connect(self.postprocess)

        self.image = image
        self.engine = FrequencyEngine(image)
        self.low_viewer = ImageViewer(
            self.image, self.image, self.tr("Low frequency"), export=True
        )
        self.high_viewer = ImageViewer(
            self.image, self.image, self.tr("High frequency"), export=True
        )
        self.mag_viewer = ImageViewer(
            self.image, None, self.tr("DFT Magnitude"), export=True
        )
        self.phase_viewer = ImageViewer(
            self.image, None, self.tr("DFT Phase"), export=True
        )
        self.job = LatestJob(self, self.engine.compute, delay=40)
        self.job.result.connect(self.show_result)
        self.job.failed.connect(lambda error: self.zero_label.setText("Error: " + error))
        self.job.busy.connect(self.set_busy)
        self._requested = None
        self._shown_images = (None, None, None, None)
        self.process()

        self.low_viewer.viewChanged.connect(self.high_viewer.changeView)
        self.high_viewer.viewChanged.connect(self.low_viewer.changeView)
        self.mag_viewer.viewChanged.connect(self.phase_viewer.changeView)
        self.phase_viewer.viewChanged.connect(self.mag_viewer.changeView)

        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel(self.tr("Separation:")))
        top_layout.addWidget(self.split_spin)
        top_layout.addWidget(QLabel(self.tr("Smooth:")))
        top_layout.addWidget(self.smooth_spin)
        top_layout.addWidget(QLabel(self.tr("Threshold:")))
        top_layout.addWidget(self.thr_spin)
        top_layout.addWidget(QLabel(self.tr("Filter:")))
        top_layout.addWidget(self.filter_spin)
        top_layout.addWidget(self.zero_label)
        top_layout.addStretch()

        center_layout = QGridLayout()
        center_layout.addWidget(self.low_viewer, 0, 0)
        center_layout.addWidget(self.high_viewer, 0, 1)
        center_layout.addWidget(self.mag_viewer, 1, 0)
        center_layout.addWidget(self.phase_viewer, 1, 1)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addLayout(center_layout)
        self.setLayout(main_layout)

    def process(self):
        params = (self.split_spin.value(), self.smooth_spin.value(),
                  self.thr_spin.value(), self.filter_spin.value())
        if params != self._requested:
            display_only = self._requested is not None and params[:3] == self._requested[:3]
            self.job.timer.setInterval(0 if display_only else 40)
            self._requested = params
            self.job.request(params)

    def postprocess(self):
        self.process()

    def set_busy(self, busy):
        for viewer in (self.low_viewer, self.high_viewer, self.mag_viewer, self.phase_viewer):
            viewer.set_busy(busy)
        if busy:
            self.zero_label.setText(self.tr("Calculating…"))

    def show_result(self, result):
        low, high, magnitude, phase, zeros = result
        images = low, high, magnitude, phase
        updates = (self.low_viewer.update_processed, self.high_viewer.update_processed,
                   self.mag_viewer.update_original, self.phase_viewer.update_original)
        for previous, image, update in zip(self._shown_images, images, updates):
            if image is not previous:
                update(image)
        self._shown_images = images
        self.zero_label.setText(self.tr("(zeroed coefficients = {:.2f}%)").format(zeros))
        self.info_message.emit(f"Frequency Split = {self.job.seconds:.3f} s")
