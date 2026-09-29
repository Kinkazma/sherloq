from gui.sherloq_app.ui.localization import t
"""Preview isolated hot/dead-pixel candidates; preserve the source image."""
from functools import partial
from threading import Event
from PySide6.QtCore import QObject,Signal
from PySide6.QtWidgets import QSpinBox,QComboBox,QPushButton,QLabel,QHBoxLayout,QVBoxLayout,QProgressBar,QFileDialog
from gui.sherloq_app.core.defect_pixels import DefectEngine,export_csv
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
    path,settings,image,flags,median,event=request
    try:export_csv(path,settings,image,flags,median,event.is_set);return path
    except Cancelled:return None


class DefectWidget(ToolWidget):
    def __init__(self,image,parent=None):
        super().__init__(parent);self.image=image;self.engine=DefectEngine(image)
        self.cancel_event=Event();self.export_event=Event();self._displayed=None;self.flags=None
        self.radius_combo=QComboBox();self.radius_combo.addItems(['3 × 3','5 × 5'])
        self.threshold_spin=QSpinBox();self.threshold_spin.setRange(1,255);self.threshold_spin.setValue(32)
        self.spread_spin=QSpinBox();self.spread_spin.setRange(0,255);self.spread_spin.setValue(32)
        self.kind_combo=QComboBox();self.kind_combo.addItems(['Hot + dead','Hot','Dead'])
        self.cpu_button=QPushButton('CPU');self.cpu_button.setCheckable(True)
        self.mode_combo=QComboBox();self.mode_combo.addItems(['Candidates overlay','Candidates mask','Correction preview'])
        self.process_button=QPushButton('Process');self.export_button=QPushButton('Export coordinates (CSV)')
        self.status_label=QLabel('Isolated pixel candidates; not a diagnosis of sensor defects.')
        self.legend_label=QLabel('Red = hot; blue = dead; magenta = both. Border pixels are excluded.')
        self.progress=QProgressBar();self.progress.setRange(0,100);self.viewer=ImageViewer(image,image)
        first=QHBoxLayout()
        for widget in [QLabel('Neighbourhood:'),self.radius_combo,QLabel('Min deviation:'),self.threshold_spin,QLabel('Max neighbour range:'),self.spread_spin,self.kind_combo,self.cpu_button]:first.addWidget(widget)
        first.addStretch();second=QHBoxLayout()
        for widget in [self.mode_combo,self.process_button,self.export_button]:second.addWidget(widget)
        second.addStretch();layout=QVBoxLayout(self);layout.addLayout(first);layout.addLayout(second);layout.addWidget(self.status_label);layout.addWidget(self.legend_label);layout.addWidget(self.progress);layout.addWidget(self.viewer)
        self.updates=_Progress();self.updates.changed.connect(self._progress)
        self.job=LatestJob(self,partial(_run,self.engine,self.updates),delay=60);self.job.result.connect(self.show_result);self.job.failed.connect(self.show_error);self.job.busy.connect(self.set_busy)
        self.export_job=LatestJob(self,_export,delay=0);self.export_job.busy.connect(lambda _:self.set_busy(self.job.is_busy));self.export_job.failed.connect(self.show_error);self.export_job.result.connect(self.exported)
        for widget in [self.radius_combo,self.kind_combo,self.mode_combo]:widget.currentIndexChanged.connect(self.choose)
        self.threshold_spin.valueChanged.connect(self.choose);self.spread_spin.valueChanged.connect(self.choose);self.cpu_button.toggled.connect(self.choose)
        self.process_button.clicked.connect(self.process);self.export_button.clicked.connect(self.export_data);self.choose()

    @property
    def is_busy(self):return self.job.is_busy or self.export_job.is_busy

    def parameters(self):return (self.radius_combo.currentIndex()+1,self.threshold_spin.value(),self.spread_spin.value(),self.kind_combo.currentIndex(),self.cpu_button.isChecked()),self.mode_combo.currentIndex()

    def choose(self):
        self.cancel_event.set();self.cancel_event=Event();self.export_event.set();self.export_job.invalidate()
        self.legend_label.setText('Only candidate channels replaced by the local median; original unchanged.' if self.mode_combo.currentIndex()==2 else 'Red = hot; blue = dead; magenta = both. Border pixels are excluded.')
        self.job.request((self.parameters(),self.cancel_event))

    def process(self):
        if self.is_busy:
            self.cancel_event.set();self.export_event.set();self.job.invalidate();self.export_job.invalidate();self.set_busy(False);self.status_label.setText('Cancelled. Completed analysis retained.');return
        if self._displayed!=self.parameters():self.choose()

    def set_busy(self,busy):
        current=self._displayed==self.parameters();self.viewer.set_busy(busy or not current)
        self.export_button.setEnabled(current and not self.is_busy);self.process_button.setText('Cancel' if self.is_busy else 'Process')

    def _progress(self,event,n,text):
        if not self.job.closed and event is self.cancel_event and not event.is_set():self.progress.setValue(n)

    def show_result(self,result):
        if result is None:return
        params,(output,flags,count,median)=result
        if params!=self.parameters():return
        self._displayed=params;self.flags=flags;self.median=median;self.viewer.update_processed(output);self.progress.setValue(100);self.set_busy(False)
        self.status_label.setText(f'{count} candidate pixels — {self.job.seconds:.3f} s. Sensor defects cannot be confirmed from one decoded image.')

    def show_error(self,error):self.status_label.setText(error);self.set_busy(False)

    def export_data(self):
        if not self.export_button.isEnabled():return
        path,_=QFileDialog.getSaveFileName(self,t('Export candidate coordinates'),'pixel-candidates.csv',t('CSV (*.csv)'))
        if not path:return
        settings=self._displayed[0]
        self.export_event=Event();self.export_job.request((path,settings,self.image,self.flags,self.median,self.export_event))

    def exported(self,path):
        if path:self.status_label.setText('Candidate coordinates exported.');self.set_busy(False)

    def shutdown(self):
        self.cancel_event.set();self.export_event.set();super().shutdown()
