from gui.sherloq_app.ui.localization import t
"""Asynchronous exact-file JPEG trace analysis and cached report view."""
import json
import os
import sys
import tempfile
from pathlib import Path
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QPushButton,QLabel,QFileDialog,QComboBox)
from gui.sherloq_app.ui.plot_canvas import FigureCanvas
from matplotlib.figure import Figure
from gui.sherloq_app.core.double_jpeg import LIMITATIONS, lattice
from .process import ProcessJob


class DoubleJpegPanel(QWidget):
    def __init__(self, filename, parent=None):
        super().__init__(parent)
        self.filename = filename
        self.report = None
        self.folder = None
        self.job = ProcessJob(self, max_output=1024*1024, timeout_ms=120000, heavy=True)
        self.job.result.connect(self._result)
        self.job.failed.connect(self._error)
        self.job.waiting.connect(lambda waiting: self.status.setText('Waiting for another analysis…') if waiting else None)
        self.status = QLabel('Analyze the JPEG file’s stored DCT coefficients for aligned double-quantization traces.')
        self.status.setWordWrap(True)
        self.start_button = QPushButton('Detect double compression')
        self.start_button.setEnabled(bool(filename))
        self.start_button.clicked.connect(self.start)
        self.export_button = QPushButton('Export report')
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export)
        self.frequency = QComboBox()
        self.frequency.currentIndexChanged.connect(self.draw)
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(250)
        self.axes = self.figure.subplots()
        self.axes.set_title('Stored luminance DCT histogram')
        self.axes.set_xlabel('Absolute quantized coefficient')
        self.axes.set_ylabel('Block count')
        self.details = QLabel('')
        self.details.setWordWrap(True)
        self.note = QLabel(LIMITATIONS)
        self.note.setWordWrap(True)
        controls = QHBoxLayout()
        controls.addWidget(self.start_button); controls.addWidget(self.export_button); controls.addWidget(self.frequency,1)
        layout = QVBoxLayout(self)
        layout.addWidget(self.status);layout.addWidget(self.canvas,1);layout.addWidget(self.details);layout.addLayout(controls);layout.addWidget(self.note)
        if not filename:
            self.status.setText('Open a JPEG file to use this detector.')

    def start(self):
        if self.job.is_busy:
            self.job.cancel()
            return
        if self.report is not None:
            return  # Presentation changes never repeat file parsing or analysis.
        self.status.setText('Analyzing JPEG coefficients…')
        self.start_button.setText('Cancel')
        script = Path(__file__).resolve().parents[1]/'core/double_jpeg.py'
        self.folder = tempfile.TemporaryDirectory(prefix='sherloq-double-jpeg-job-')
        self.job.start(sys.executable,[str(script),str(self.filename),self.folder.name])

    def _result(self, data):
        self._cleanup()
        try:
            report = json.loads(data)
            assert report['version']=='aligned-lattice-v1' and len(report['records'])==9
        except (ValueError,KeyError,AssertionError):
            self._error('The JPEG worker returned an invalid report.')
            return
        self.report = report
        found = report['verdict']=='compatible_traces'
        title = 'Traces compatible with double JPEG compression detected.' if found else 'Inconclusive — no strong consensus of double-quantization traces.'
        self.status.setText(f"{title} {report['supporting_frequencies']}/9 frequencies support the result; {report['seconds']:.3f} s.")
        self.start_button.setText('Analysis retained'); self.start_button.setEnabled(False)
        self.export_button.setEnabled(True)
        for r in report['records']:
            self.frequency.addItem(f"DCT ({r['frequency'][0]}, {r['frequency'][1]})",r)
        self.frequency.setCurrentIndex(max(range(len(report['records'])),key=lambda i:report['records'][i]['score']))
        self.draw()

    def _error(self, message):
        self._cleanup()
        self.status.setText(message)
        self.start_button.setText('Retry detection')
        self.start_button.setEnabled(True)

    def draw(self, *_):
        record = self.frequency.currentData()
        if record is None:
            return
        self.axes.clear()
        self.axes.bar(range(129),record['histogram'][:129],width=1,color='#007fae')
        self.axes.set_xlim(0,128)
        self.axes.set_yscale('symlog',linthresh=1)
        self.axes.set_xlabel('Absolute quantized coefficient')
        self.axes.set_ylabel('Block count (symlog)')
        if record['eligible']:
            support=lattice(record['candidate_step'],record['current_step'])
            import numpy as np
            edges=np.arange(130)-.5
            self.axes.fill_between(edges,0,1,where=np.r_[~support[:129],~support[128]],step='post',transform=self.axes.get_xaxis_transform(),color='#e69f00',alpha=.15)
            self.details.setText(f"Current step: {record['current_step']}; candidate earlier step: {record['candidate_step']}. "
                                 f"Gap depletion score: {record['score']:.3f}; {record['samples']} eligible coefficients. "
                                 'Orange bins are predicted gaps. The score is not a probability; the candidate step is not a recovered JPEG quality.')
        else:
            self.details.setText(f"Current step: {record['current_step']}. Insufficient populated bins for this frequency.")
        self.canvas.draw_idle()

    def export(self):
        if self.report is None:
            return
        path,_=QFileDialog.getSaveFileName(self,t('Export double JPEG report'),'double-jpeg.json',t('JSON (*.json)'))
        if not path:
            return
        temporary=None
        try:
            with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=Path(path).parent,delete=False) as stream:
                temporary=stream.name
                json.dump(self.report,stream,indent=2,allow_nan=False)
            os.replace(temporary,path)
        except OSError as exc:
            self.status.setText(f'Export failed: {exc}')
        finally:
            if temporary and os.path.exists(temporary):os.unlink(temporary)

    def _cleanup(self):
        if self.folder is not None:
            self.folder.cleanup()
            self.folder = None

    def shutdown(self):
        self.job.shutdown()
        self._cleanup()
