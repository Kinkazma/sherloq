from gui.sherloq_app.ui.localization import choice, t
"""Local illuminant colour map with explicit assumptions and retained results."""
from functools import partial
from threading import Event
from PySide6.QtCore import QObject,Signal
from PySide6.QtWidgets import QComboBox,QCheckBox,QPushButton,QLabel,QHBoxLayout,QVBoxLayout,QProgressBar,QFileDialog
from gui.sherloq_app.core.illuminant import IlluminantEngine,export_csv
from gui.sherloq_app.core.jpeg_curve import Cancelled
from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.ui.viewer import ImageViewer
from gui.sherloq_app.ui.jobs import LatestJob

class _Progress(QObject):
    changed=Signal(object,int,str)

def _run(engine,updates,request):
    params,event=request
    try:return params,engine.compute(params,event.is_set,lambda n,t:updates.changed.emit(event,n,t))
    except Cancelled:return None

def _export(request):
    path,settings,result,shape,event=request
    try:export_csv(path,settings,result,shape,event.is_set);return path
    except Cancelled:return None


class IlluminantWidget(ToolWidget):
    def __init__(self,image,parent=None):
        super().__init__(parent);self.image=image;self.engine=IlluminantEngine(image)
        self.cancel_event=Event();self.export_event=Event();self._displayed=None;self.result=None
        self.method_combo=QComboBox();self.method_combo.addItems(['Gray World','Shades of Gray (p=6)','White Patch']);self.method_combo.setCurrentIndex(1)
        self.block_combo=QComboBox();self.block_combo.addItems(['32','64','128','256']);self.block_combo.setCurrentIndex(2)
        self.linear_check=QCheckBox('Linearize sRGB');self.linear_check.setChecked(True)
        self.exclude_check=QCheckBox('Exclude dark/clipped');self.exclude_check.setChecked(True)
        self.exclude_check.setToolTip('Use pixels whose three channels are all between 9 and 249. No sensor RAW or ICC conversion is performed here.')
        self.mode_combo=QComboBox();self.mode_combo.addItems(['Illuminant colour','Difference from global','Valid pixels'])
        self.process_button=QPushButton('Process');self.export_button=QPushButton('Export data (CSV)')
        self.status_label=QLabel('Local colour estimate; coloured surfaces also influence the result.')
        self.legend_label=QLabel();self.progress=QProgressBar();self.progress.setRange(0,100);self.viewer=ImageViewer(image,image)
        first=QHBoxLayout()
        for widget in [QLabel('Method:'),self.method_combo,QLabel('Cell:'),self.block_combo,self.linear_check,self.exclude_check]:first.addWidget(widget)
        first.addStretch();second=QHBoxLayout()
        for widget in [self.mode_combo,self.process_button,self.export_button]:second.addWidget(widget)
        second.addStretch();layout=QVBoxLayout(self);layout.addLayout(first);layout.addLayout(second);layout.addWidget(self.status_label);layout.addWidget(self.legend_label);layout.addWidget(self.progress);layout.addWidget(self.viewer)
        self.updates=_Progress();self.updates.changed.connect(self._progress)
        self.job=LatestJob(self,partial(_run,self.engine,self.updates),delay=60);self.job.result.connect(self.show_result);self.job.failed.connect(self.show_error);self.job.busy.connect(self.set_busy)
        self.export_job=LatestJob(self,_export,delay=0);self.export_job.busy.connect(lambda _:self.set_busy(self.job.is_busy));self.export_job.failed.connect(self.show_error);self.export_job.result.connect(self.exported)
        self.method_combo.currentIndexChanged.connect(self.choose);self.block_combo.currentIndexChanged.connect(self.choose);self.mode_combo.currentIndexChanged.connect(self.choose);self.linear_check.toggled.connect(self.choose);self.exclude_check.toggled.connect(self.choose)
        self.process_button.clicked.connect(self.process);self.export_button.clicked.connect(self.export_data)
        self.choose()

    @property
    def is_busy(self):return self.job.is_busy or self.export_job.is_busy

    def parameters(self):return (int(choice(self.block_combo)),self.method_combo.currentIndex(),self.linear_check.isChecked(),self.exclude_check.isChecked()),self.mode_combo.currentIndex()

    def choose(self):
        self.cancel_event.set();self.cancel_event=Event();self.export_event.set();self.export_job.invalidate()
        self.legend_label.setText(['Estimated RGB colour; dark grey = insufficient valid pixels.','Angular difference: dark purple = 0°, yellow = 90°; grey = unavailable.','Valid-pixel fraction: black = 0%, white = 100%.'][self.mode_combo.currentIndex()])
        self.job.request((self.parameters(),self.cancel_event))

    def process(self):
        if self.is_busy:
            self.cancel_event.set();self.export_event.set();self.job.invalidate();self.export_job.invalidate();self.set_busy(False);self.status_label.setText('Cancelled. Completed cells retained.');return
        if self._displayed!=self.parameters():self.choose()

    def set_busy(self,busy):
        current=self._displayed==self.parameters()
        self.viewer.set_busy(busy or not current);self.export_button.setEnabled(current and not self.is_busy)
        self.process_button.setText('Cancel' if self.is_busy else 'Process')

    def _progress(self,event,n,text):
        if not self.job.closed and event is self.cancel_event and not event.is_set():self.progress.setValue(n)

    def show_result(self,result):
        if result is None:return
        params,(output,data)=result
        if params!=self.parameters():return
        self._displayed=params;self.result=data;self.viewer.update_processed(output);self.progress.setValue(100);self.set_busy(False)
        valid=data[3];self.status_label.setText(f'{int(valid.sum())}/{valid.size} valid cells — {self.job.seconds:.3f} s. Coloured surfaces can affect the estimate.')

    def show_error(self,error):self.status_label.setText(error);self.set_busy(False)

    def export_data(self):
        if not self.export_button.isEnabled():return
        path,_=QFileDialog.getSaveFileName(self,t('Export illuminant data'),'illuminant.csv',t('CSV (*.csv)'))
        if not path:return
        self.export_event=Event();self.export_job.request((path,self._displayed[0],self.result,self.image.shape,self.export_event))

    def exported(self,path):
        if path:self.status_label.setText('Illuminant data exported.');self.set_busy(False)

    def shutdown(self):
        self.cancel_event.set();self.export_event.set();super().shutdown()
