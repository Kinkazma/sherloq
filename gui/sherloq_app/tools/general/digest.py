from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar
from gui.sherloq_app.ui.table import TableWidget
from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.ui.jobs import LatestJob
from gui.sherloq_app.core.digest import DigestEngine, ballistics


class DigestWidget(ToolWidget):
    def __init__(self, filename, image, parent=None):
        super().__init__(parent)
        self.engine = DigestEngine(filename, image)
        self.table_widget = None
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.progress = QProgressBar()
        self.refresh_button = QPushButton(self.tr('Recalculate'))
        self.cancel_button = QPushButton(self.tr('Cancel'))
        controls = QHBoxLayout()
        controls.addWidget(self.refresh_button)
        controls.addWidget(self.cancel_button)
        controls.addStretch()
        self.main_layout = QVBoxLayout(self)
        self.main_layout.addWidget(self.status)
        self.main_layout.addWidget(self.progress)
        self.main_layout.addLayout(controls)
        self.setMinimumSize(700, 520)
        self.job = LatestJob(self, self.engine.compute, delay=0)
        self.job.result.connect(self._finished)
        self.job.failed.connect(self._failed)
        self.job.busy.connect(self._busy)
        self.refresh_button.clicked.connect(self.recalculate)
        self.cancel_button.clicked.connect(self.engine.cancelled.set)
        self.poll = QTimer(self)
        self.poll.setInterval(100)
        self.poll.timeout.connect(self._progress)
        cancelled = self.engine.cancelled
        self.destroyed.connect(lambda *_: cancelled.set())
        self.recalculate()

    def recalculate(self):
        if self.job.is_busy or self.job.closed:
            return
        if self.table_widget is not None:
            self.main_layout.removeWidget(self.table_widget)
            self.table_widget.deleteLater()
            self.table_widget = None
        self.engine.cancelled.clear()
        self.engine.progress = 0
        self.progress.setValue(0)
        self.status.setText(self.tr('Calculating file and image hashes…'))
        self.job.request(True)

    def _progress(self):
        self.progress.setValue(self.engine.progress)

    def _busy(self, busy):
        self.refresh_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.progress.setVisible(busy)
        if busy:
            self.poll.start()
        else:
            self.poll.stop()

    def _finished(self, result):
        if result is None:
            self.status.setText(self.tr('Analysis cancelled.'))
            return
        rows, _ = result
        self.table_widget = TableWidget(rows, [self.tr('Group'), self.tr('Property'), self.tr('Value')])
        self.main_layout.addWidget(self.table_widget, 1)
        self.status.setText(self.tr('File hashes describe the bytes on disk; image hashes describe the loaded image.'))

    def _failed(self, message):
        self.status.setText(self.tr('Unable to calculate hashes: ') + message)

    def shutdown(self):
        self.engine.cancelled.set()
        self.poll.stop()
        super().shutdown()
