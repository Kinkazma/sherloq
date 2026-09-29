"""Nonblocking stereogram views; compute disparity only when requested."""
from functools import partial
from threading import Event
from PySide6.QtCore import QObject,Signal
from PySide6.QtWidgets import QRadioButton,QButtonGroup,QPushButton,QLabel,QHBoxLayout,QVBoxLayout,QProgressBar
from gui.sherloq_app.core.stereogram import StereoEngine
from gui.sherloq_app.core.jpeg_curve import Cancelled
from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.ui.viewer import ImageViewer
from gui.sherloq_app.ui.jobs import LatestJob

class _Progress(QObject):
    changed=Signal(object,int,str)

def _run(engine,updates,request):
    mode,event=request
    try:return mode,engine.compute(mode,event.is_set,lambda n,t:updates.changed.emit(event,n,t))
    except Cancelled:return None


class StereoWidget(ToolWidget):
    def __init__(self,image,parent=None):
        super().__init__(parent);self.engine=StereoEngine(image);self.cancel_event=Event();self._displayed=None
        self.pattern_radio=QRadioButton('Pattern');self.silhouette_radio=QRadioButton('Silhouette');self.depth_radio=QRadioButton('Depth');self.shaded_radio=QRadioButton('Shaded')
        self.radios=[self.pattern_radio,self.silhouette_radio,self.depth_radio,self.shaded_radio];self.group=QButtonGroup(self)
        for index,radio in enumerate(self.radios):self.group.addButton(radio,index)
        self.pattern_radio.setChecked(True);self.depth_radio.setToolTip('Relative optical-flow disparity, not a calibrated physical depth.')
        self.process_button=QPushButton('Process');self.status_label=QLabel('Finding stereogram offset…');self.progress=QProgressBar();self.progress.setRange(0,100)
        self.viewer=ImageViewer(image,None,export=True)
        top=QHBoxLayout();top.addWidget(QLabel('Mode:'))
        for radio in self.radios:top.addWidget(radio)
        top.addWidget(self.process_button);top.addStretch()
        layout=QVBoxLayout(self);layout.addLayout(top);layout.addWidget(self.status_label);layout.addWidget(self.progress);layout.addWidget(self.viewer)
        self.updates=_Progress();self.updates.changed.connect(self._progress)
        self.job=LatestJob(self,partial(_run,self.engine,self.updates),delay=30);self.job.result.connect(self.show_result);self.job.failed.connect(self.show_error);self.job.busy.connect(self.set_busy)
        self.group.idClicked.connect(self.choose);self.process_button.clicked.connect(self.process);self.choose()

    def choose(self,*_):
        if self._displayed==self.group.checkedId():
            if self.job.is_busy:self.cancel_event.set();self.job.invalidate()
            self.set_busy(False)
            return
        self.cancel_event.set();self.cancel_event=Event();self.job.request((self.group.checkedId(),self.cancel_event))

    def process(self):
        if self.job.is_busy:
            self.cancel_event.set();self.job.invalidate();self.set_busy(False);self.status_label.setText('Cancelled. Completed offsets and views retained.');return
        self.choose()

    def set_busy(self,busy):
        self.process_button.setText('Cancel' if busy else 'Process');self.viewer.set_busy(busy or self._displayed!=self.group.checkedId())

    def _progress(self,event,n,text):
        if not self.job.closed and event is self.cancel_event and not event.is_set():self.progress.setValue(n);self.status_label.setText(text)

    def show_result(self,result):
        if result is None:return
        mode,output=result
        if mode!=self.group.checkedId():return
        if output is None:
            self.status_label.setText('Unable to detect a stereogram in this image.');self.process_button.setEnabled(False)
            for radio in self.radios:radio.setEnabled(False)
            self.viewer.set_busy(True);return
        self._displayed=mode;self.viewer.update_original(output);self.progress.setValue(100);self.set_busy(False)
        self.status_label.setText(f'Offset: {self.engine.offset} px — {self.job.seconds:.3f} s')

    def show_error(self,error):self.status_label.setText(error);self.set_busy(False)

    def shutdown(self):self.cancel_event.set();super().shutdown()
