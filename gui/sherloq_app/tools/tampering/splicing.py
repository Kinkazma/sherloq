"""Noiseprint UI: process isolation, model selection and retained scientific maps."""
import sys
import platform
import numpy as np
from PySide6.QtWidgets import QPushButton,QGridLayout,QLabel,QComboBox,QProgressBar
from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.ui.viewer import ImageViewer
from gui.sherloq_app.ui.splicing_job import SplicingJob


class SplicingWidget(ToolWidget):
    def __init__(self,image,parent=None):
        super().__init__(parent)
        self.image=image
        self.noise=self.map=self.raw_map=None
        self.model_combo=QComboBox()
        self.model_combo.addItem('Automatic JPEG quality estimate',0)
        for quality in range(51,101):self.model_combo.addItem(f'JPEG model {quality}',quality)
        self.model_combo.addItem('Uncompressed model (101)',101)
        self.model_combo.setToolTip('Select the model explicitly when the automatic recompression estimate is unsuitable.')
        self.cpu_button=QPushButton('CPU')
        self.cpu_button.setCheckable(True)
        self.cpu_button.setChecked(sys.platform!='darwin' or platform.machine()!='arm64')
        self.noise_button=QPushButton('(1/2) Estimate noise')
        self.map_button=QPushButton('(2/2) Compute heatmap')
        self.map_button.setEnabled(False)
        self.cancel_button=QPushButton('Cancel');self.cancel_button.setEnabled(False)
        self.status_label=QLabel('Noiseprint and its statistical heatmap are indicators, not a calibrated probability.')
        self.status_label.setWordWrap(True)
        self.progress=QProgressBar();self.progress.setRange(0,100)
        gray=np.full_like(image,127)
        self.noise_viewer=ImageViewer(image,gray,'Estimated noise print',export=True)
        self.map_viewer=ImageViewer(image,gray,'Splicing indicator heatmap')
        self.noise_viewer.set_busy(True);self.map_viewer.set_busy(True)
        self.job=SplicingJob(self,image)
        self.job.result.connect(self.show_result);self.job.failed.connect(self.show_error)
        self.job.busy.connect(self.set_busy);self.job.progress.connect(self.show_progress)
        self.noise_button.clicked.connect(self.estimate_noise)
        self.map_button.clicked.connect(self.compute_map)
        self.cancel_button.clicked.connect(self.job.cancel)
        self.model_combo.currentIndexChanged.connect(self.change_model)
        self.cpu_button.toggled.connect(self.change_model)
        # Connect view synchronization once, never on every heatmap button click.
        self.noise_viewer.viewChanged.connect(self.map_viewer.changeView)
        self.map_viewer.viewChanged.connect(self.noise_viewer.changeView)
        layout=QGridLayout(self)
        layout.addWidget(self.model_combo,0,0);layout.addWidget(self.cpu_button,0,1)
        layout.addWidget(self.cancel_button,5,0,1,2)
        layout.addWidget(self.status_label,1,0,1,2)
        layout.addWidget(self.progress,2,0,1,2)
        layout.addWidget(self.noise_viewer,3,0);layout.addWidget(self.map_viewer,3,1)
        layout.addWidget(self.noise_button,4,0);layout.addWidget(self.map_button,4,1)

    def change_model(self):
        self.job.invalidate()
        self.noise=self.map=self.raw_map=None
        self.noise_button.setText('(1/2) Estimate noise')
        self.map_button.setText('(2/2) Compute heatmap')
        self.set_busy(False)
        self.status_label.setText('Model or backend changed. Estimate noise to update both views.')

    def estimate_noise(self):
        if self.noise is not None:return
        self.job.request('noise',self.model_combo.currentData(),'cpu' if self.cpu_button.isChecked() else 'mps')

    def compute_map(self):
        if self.map is not None:return
        self.job.request('map',self.model_combo.currentData(),'cpu' if self.cpu_button.isChecked() else 'mps')

    def set_busy(self,busy):
        self.noise_button.setEnabled(not busy)
        self.map_button.setEnabled(not busy and self.noise is not None)
        self.cancel_button.setEnabled(busy)
        self.noise_viewer.set_busy(self.noise is None)
        self.map_viewer.set_busy(self.map is None)

    def show_progress(self,value,text):
        self.progress.setValue(value);self.status_label.setText(text)

    def show_result(self,result):
        self.noise=result['noise']
        self.noise_viewer.update_processed(result['noise_display'])
        self.noise_button.setText('Noise estimate retained')
        if 'map' in result:
            self.map=result['map'];self.raw_map=result['raw_map']
            self.map_viewer.update_processed(self.map)
            self.map_button.setText('Heatmap retained')
        self.set_busy(False);self.progress.setValue(100)
        selection='estimated' if result['automatic'] else 'selected'
        self.status_label.setText(f"Noiseprint model {result['model']} ({selection}), {result.get('backend','cpu').upper()}. Completed results retained.")

    def show_error(self,message):
        self.status_label.setText(message.splitlines()[-1] if message else "Analysis failed.")
        self.set_busy(False)

    def shutdown(self):
        self.job.shutdown()
        super().shutdown()
