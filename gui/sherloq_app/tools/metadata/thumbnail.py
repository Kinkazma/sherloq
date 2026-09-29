import os
from functools import partial
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar
from gui.sherloq_app.ui.viewer import ImageViewer
from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.ui.process import ProcessJob
from gui.sherloq_app.ui.jobs import LatestJob
from gui.sherloq_app.core.utility import exiftool_exe
from gui.sherloq_app.core.digest import identity
from gui.sherloq_app.core.thumbnail import analyze_thumbnail


class ThumbWidget(ToolWidget):
    def __init__(self, filename, image, parent=None):
        super().__init__(parent)
        self.filename = filename
        self.cancelled = False
        self.viewer = None
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
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
        self.setMinimumSize(740, 500)
        self.process_job = ProcessJob(self)
        self.job = LatestJob(self, partial(analyze_thumbnail, image=image), delay=0)
        self.process_job.result.connect(self.job.request)
        self.process_job.failed.connect(self._failed)
        self.job.result.connect(self._finished)
        self.job.failed.connect(self._failed)
        self.refresh_button.clicked.connect(self.recalculate)
        self.cancel_button.clicked.connect(self.cancel)
        self._busy(False)
        self.recalculate()

    @property
    def is_busy(self):
        return self.process_job.is_busy or self.job.is_busy

    def _busy(self, busy):
        self.progress.setVisible(busy)
        self.refresh_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)

    def recalculate(self):
        if self.is_busy or self.job.closed:
            return
        self.cancelled = False
        if self.viewer is not None:
            self.main_layout.removeWidget(self.viewer)
            self.viewer.deleteLater()
            self.viewer = None
        try:
            self.stamp = identity(os.stat(self.filename))
        except OSError as error:
            self._failed(str(error))
            return
        self._busy(True)
        self.status.setText(self.tr('Extracting and comparing thumbnail…'))
        self.process_job.start(exiftool_exe(), ['-b', '-ThumbnailImage', self.filename])

    def cancel(self):
        self.cancelled = True
        if self.process_job.is_busy:
            self.process_job.cancel()
        else:
            self.status.setText(self.tr('Cancelling…'))

    def _failed(self, error):
        self._busy(False)
        self.status.setText(error)

    def _finished(self, result):
        self._busy(False)
        if self.cancelled:
            self.status.setText(self.tr('Analysis cancelled.'))
            return
        try:
            if identity(os.stat(self.filename)) != self.stamp:
                self.status.setText(self.tr('File changed during analysis. Please recalculate.'))
                return
        except OSError as error:
            self.status.setText(str(error))
            return
        if result is None:
            self.status.setText(self.tr('Thumbnail image not found!'))
            return
        resized, difference = result
        self.viewer = ImageViewer(resized, difference)
        self.main_layout.addWidget(self.viewer, 1)
        self.status.setText(self.tr('Ready'))

    def shutdown(self):
        self.cancelled = True
        self.process_job.shutdown()
        super().shutdown()
