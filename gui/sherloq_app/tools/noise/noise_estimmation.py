""" 
This code implements local noise estimation based on on high pass wavelet coefficients and subsequent grid blocking (median based).
The technique is explain in the following paper:
"Using noise inconsistencies for blind image forensics" by Babak Mahdian & Stanislav Saic
A block merging step has not been included because all attempts yielded unsastifactory results and made analysis more difficult.

In the paper the merging step appears highly effective, but this could be because of the isolated test conditions where the only obserable
noise in the image was the one added by the researchers for testing purposes.
"""

from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QPushButton
from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.ui.viewer import ImageViewer

#algorithm necessary imports
import pywt
from gui.sherloq_app.core.wavelet_blocking import WaveletBlockingEngine
from gui.sherloq_app.ui.jobs import LatestJob

class NoiseWaveletBlockingWidget(ToolWidget):
    #tool layout
    def __init__(self, filename, image, parent=None):
        super(NoiseWaveletBlockingWidget, self).__init__(parent)

        #save variables to self
        self.filename = filename
        self.image = image

        #prepare user interface - input variables
        self.blocksize_spin = QSpinBox()
        self.blocksize_spin.setRange(1, min(100, *(pywt.dwt_coeff_len(size, 16, "symmetric") for size in image.shape[:2])))
        self.blocksize_spin.setValue(8)

        self.process_button = QPushButton(self.tr("Process image"))

        #combine top layout
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel(self.tr("Block size:")))
        top_layout.addWidget(self.blocksize_spin)
        top_layout.addWidget(self.process_button)
        self.status_label = QLabel()
        top_layout.addWidget(self.status_label)
        top_layout.addStretch()

        self.viewer = ImageViewer(image, image, None)

        self.engine = WaveletBlockingEngine(filename, image)
        self.job = LatestJob(self, self.engine.compute, delay=0)
        self._requested = None
        self.job.result.connect(self.show_result)
        self.job.busy.connect(self.set_busy)
        self.job.failed.connect(self.show_error)
        self.calculate_noise_map()
        self.process_button.clicked.connect(self.calculate_noise_map)
        self.blocksize_spin.valueChanged.connect(self.parameter_changed)

        #main layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.viewer)
        self.setLayout(main_layout)




    def parameter_changed(self):
        self.job.invalidate()
        self._requested = None
        self.process_button.setText(self.tr("Process image"))
        self.viewer.set_busy(True)
        self.status_label.setText(self.tr("Ready to process"))

    def calculate_noise_map(self):
        if self.job.is_busy:
            self.job.invalidate()
            self._requested = None
            self.viewer.set_busy(True)
            self.status_label.setText(self.tr("Cancelled"))
            return
        blocksize = self.blocksize_spin.value()
        if blocksize != self._requested:
            self._requested = blocksize
            self.job.request(blocksize)

    def set_busy(self, busy):
        self.process_button.setText(self.tr("Cancel") if busy else self.tr("Process image"))
        self.viewer.set_busy(busy)
        if busy:
            self.status_label.setText(self.tr("Calculating…"))

    def show_result(self, result):
        image, self.noise_map = result
        self.viewer.update_processed(image)
        self.status_label.setText(f"{self.job.seconds:.3f} s")

    def show_error(self, error):
        self._requested = None
        self.viewer.set_busy(True)
        self.status_label.setText(error)
