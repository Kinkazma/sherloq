"""Median-filter traces with retained exact model scores and background work."""
from functools import partial
from threading import Event
from PySide6.QtCore import QObject,Signal
from PySide6.QtWidgets import QDoubleSpinBox,QSpinBox,QCheckBox,QPushButton,QLabel,QHBoxLayout,QVBoxLayout,QProgressBar
from gui.sherloq_app.core.median import MedianEngine,get_features
from gui.sherloq_app.core.jpeg_curve import Cancelled
from gui.sherloq_app.paths import model_path
from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.ui.viewer import ImageViewer
from gui.sherloq_app.ui.jobs import LatestJob

class _Progress(QObject):
    changed = Signal(object,int,str)


def _run(engine,updates,request):
    params,event=request
    try:
        result=engine.compute(params,event.is_set,lambda value,text:updates.changed.emit(event,value,text))
        return params,result
    except Cancelled:return None


class MedianWidget(ToolWidget):
    def __init__(self,image,parent=None):
        super().__init__(parent)
        self.image=image;self.block=64
        self.engine=MedianEngine(image,model_path('median_b64.json'))
        self.cancel_event=Event();self._displayed=None;self._started=False
        self.variance_spin=QSpinBox();self.variance_spin.setRange(0,100);self.variance_spin.setValue(5)
        self.threshold_spin=QDoubleSpinBox();self.threshold_spin.setRange(0,1);self.threshold_spin.setValue(.4);self.threshold_spin.setSingleStep(.01)
        self.showprob_check=QCheckBox('Score map')
        self.filter_check=QCheckBox('Speckle filter');self.filter_check.setChecked(True)
        self.process_button=QPushButton('Process')
        self.avgprob_label=QLabel('Press Process to start')
        self.progress=QProgressBar();self.progress.setRange(0,100)
        self.viewer=ImageViewer(image,image)
        top=QHBoxLayout()
        for widget in [QLabel('Min variance:'),self.variance_spin,QLabel('Threshold:'),self.threshold_spin,self.showprob_check,self.filter_check,self.process_button]:top.addWidget(widget)
        top.addStretch()
        layout=QVBoxLayout(self);layout.addLayout(top);layout.addWidget(self.avgprob_label);layout.addWidget(self.progress);layout.addWidget(self.viewer)
        self.updates=_Progress();self.updates.changed.connect(self._progress)
        self.job=LatestJob(self,partial(_run,self.engine,self.updates),delay=30)
        self.job.result.connect(self.show_result);self.job.failed.connect(self.show_error);self.job.busy.connect(self.set_busy)
        self.process_button.clicked.connect(self.prepare)
        self.variance_spin.valueChanged.connect(self.process);self.threshold_spin.valueChanged.connect(self.process)
        self.showprob_check.toggled.connect(self.process);self.filter_check.toggled.connect(self.process)
        self.set_busy(False)

    def parameters(self):
        return self.variance_spin.value(),self.threshold_spin.value(),self.showprob_check.isChecked(),self.filter_check.isChecked()

    def prepare(self):
        if self.job.is_busy:self.cancel();return
        if self._displayed==self.parameters():return
        self._started=True;self._start()

    def _start(self):
        self.cancel_event.set();self.cancel_event=Event()
        self.job.request((self.parameters(),self.cancel_event))

    def process(self):
        self.threshold_spin.setEnabled(not self.showprob_check.isChecked())
        if self._started:self._start()

    def cancel(self):
        self.cancel_event.set();self.job.invalidate();self.set_busy(False)
        self.avgprob_label.setText('Cancelled. Completed batches retained for Resume.')

    def set_busy(self,busy):
        self.process_button.setText('Cancel' if busy else 'Process')
        self.viewer.set_busy(busy or self._displayed!=self.parameters())
        if busy:self.avgprob_label.setText('Analyzing median-filter traces…')

    def _progress(self,event,value,text):
        if not self.job.closed and event is self.cancel_event and not event.is_set():
            self.progress.setValue(value);self.avgprob_label.setText(text)

    def show_result(self,result):
        if result is None:return
        params,(output,mean)=result
        if params!=self.parameters():return
        self._displayed=params;self.viewer.update_processed(output);self.progress.setValue(100);self.set_busy(False)
        self.avgprob_label.setText(f'Mean model score: {mean:.4f} — {self.job.seconds:.3f} s')
        self.info_message.emit(self.avgprob_label.text())

    def show_error(self,message):
        self.avgprob_label.setText(message);self.set_busy(False)

    def shutdown(self):
        self.cancel_event.set();super().shutdown()
