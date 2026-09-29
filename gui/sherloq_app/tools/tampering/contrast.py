from gui.sherloq_app.ui.localization import choice
"""Contrast analysis with exact retained maps and cancellable background work."""
from functools import partial
from threading import Event
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QPushButton,QComboBox,QLabel,QHBoxLayout,QVBoxLayout,QProgressBar
from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.ui.viewer import ImageViewer
from gui.sherloq_app.ui.jobs import LatestJob
from gui.sherloq_app.core.contrast import ContrastEngine
from gui.sherloq_app.core.jpeg_curve import Cancelled

class _Progress(QObject):
    changed = Signal(object,int,str)


def _run(engine, updates, request):
    params, event = request
    try:
        output = engine.compute(params,event.is_set,
            lambda value,text:updates.changed.emit(event,value,text))
        return params,output
    except Cancelled:
        return None


class ContrastWidget(ToolWidget):
    def __init__(self,image,parent=None):
        super().__init__(parent)
        self.image = image
        self.engine = ContrastEngine(image)
        self.cancel_event = Event()
        self._requested = self._displayed = None
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(['Histogram Error','Channel Similarity','Joint indicator'])
        self.algo_combo.setCurrentIndex(2)
        self.algo_combo.setToolTip('Product of two heuristic indicators; not a calibrated probability of editing.')
        self.block_combo = QComboBox()
        self.block_combo.addItems(['32','64','128','256'])
        self.block_combo.setCurrentIndex(1)
        self.process_button = QPushButton('Process')
        self.status_label = QLabel('Select a block size and process. Indicators are not proof of contrast editing.')
        self.progress = QProgressBar();self.progress.setRange(0,100)
        self.viewer = ImageViewer(image,image)
        top = QHBoxLayout()
        for widget in [QLabel('Algorithm:'),self.algo_combo,QLabel('Block size:'),self.block_combo,self.process_button]:top.addWidget(widget)
        top.addStretch()
        layout = QVBoxLayout(self)
        layout.addLayout(top);layout.addWidget(self.status_label);layout.addWidget(self.progress);layout.addWidget(self.viewer)
        self.updates = _Progress()
        self.updates.changed.connect(self._progress)
        self.job = LatestJob(self,partial(_run,self.engine,self.updates),delay=0)
        self.job.result.connect(self.show_result);self.job.failed.connect(self.show_error);self.job.busy.connect(self.set_busy)
        self.process_button.clicked.connect(self.process)
        self.algo_combo.currentIndexChanged.connect(self.choose)
        self.block_combo.currentIndexChanged.connect(self.reset)
        self.set_busy(False)

    def parameters(self):
        return int(choice(self.block_combo)),self.algo_combo.currentIndex()

    def _invalidate(self):
        self.cancel_event.set();self.job.invalidate();self._requested=None
        self.set_busy(False)

    def reset(self):
        self._invalidate()
        self.status_label.setText('Block size changed. Process to update.')
        if self.engine.maps.get(self.parameters()[0]) is not None:
            self._start()

    def choose(self):
        was_active = self.job.is_busy
        self._invalidate()
        if was_active or self.engine.maps.get(self.parameters()[0]) is not None:
            self._start()
        else:
            self.status_label.setText('Process to compute the selected indicator.')

    def process(self):
        if self.job.is_busy:
            self._invalidate()
            self.status_label.setText('Cancelled. Completed blocks retained for Resume.')
            return
        if self._displayed == self.parameters():
            return
        self._start()

    def _start(self):
        self.cancel_event.set();self.cancel_event=Event()
        self._requested=self.parameters()
        self.job.request((self._requested,self.cancel_event))

    def set_busy(self,busy):
        self.process_button.setText('Cancel' if busy else 'Process')
        self.viewer.set_busy(busy or self._displayed != self.parameters())
        if busy:self.status_label.setText('Analyzing…')

    def _progress(self,event,value,text):
        if not self.job.closed and event is self.cancel_event and not event.is_set():
            self.progress.setValue(value);self.status_label.setText(text)

    def show_result(self,result):
        if result is None:return
        params,output=result
        if params!=self.parameters():return
        self._displayed=params
        self.viewer.update_processed(output);self.set_busy(False);self.progress.setValue(100)
        self.status_label.setText(f'Contrast analysis = {self.job.seconds:.3f} s — heuristic indicator')
        self.info_message.emit(self.status_label.text())

    def show_error(self,message):
        self.status_label.setText(message);self.set_busy(False)

    def shutdown(self):
        self.cancel_event.set();super().shutdown()
