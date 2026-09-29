import os
from pathlib import Path
from PySide6.QtCore import QUrl, QTemporaryDir
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar
from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.ui.process import ProcessJob
from gui.sherloq_app.core.utility import exiftool_exe
from gui.sherloq_app.core.digest import identity


class HeaderWidget(ToolWidget):
    def __init__(self, filename, parent=None):
        super().__init__(parent)
        self.filename = filename
        self.temp_dir = QTemporaryDir()
        self.temp_file = str(Path(self.temp_dir.path()) / 'structure.html')
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
        self.web_view = None
        self.main_layout = QVBoxLayout(self)
        self.main_layout.addWidget(self.status)
        self.main_layout.addWidget(self.progress)
        self.main_layout.addLayout(controls)
        self.setMinimumWidth(900)
        self.job = ProcessJob(self)
        self.job.result.connect(self._finished)
        self.job.failed.connect(self.status.setText)
        self.job.busy.connect(self._busy)
        self.refresh_button.clicked.connect(self.recalculate)
        self.cancel_button.clicked.connect(lambda: self.job.cancel())
        self._busy(False)
        self.recalculate()

    def recalculate(self):
        if self.job.is_busy or self.job.closed:
            return
        if not self.temp_dir.isValid():
            self.status.setText(self.tr('Unable to create a temporary directory.'))
            return
        if self.web_view is not None:
            self.main_layout.removeWidget(self.web_view)
            self.web_view.loadFinished.disconnect(self._loaded)
            self.web_view.stop()
            self.web_view.deleteLater()
            self.web_view = None
        try:
            self.stamp = identity(os.stat(self.filename))
        except OSError as error:
            self.status.setText(str(error))
            return
        self.status.setText(self.tr('Reading file structure…'))
        self.job.start(exiftool_exe(), ['-htmldump0', self.filename], self.temp_file)

    def _busy(self, busy):
        self.progress.setVisible(busy)
        self.refresh_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)

    def _finished(self, _):
        try:
            if identity(os.stat(self.filename)) != self.stamp:
                self.status.setText(self.tr('File changed during analysis. Please recalculate.'))
                return
        except OSError as error:
            self.status.setText(str(error))
            return
        self.web_view = QWebEngineView(self)
        self.web_view.loadFinished.connect(self._loaded)
        self.main_layout.addWidget(self.web_view, 1)
        self.status.setText(self.tr('Displaying file structure…'))
        self.web_view.load(QUrl.fromLocalFile(self.temp_file))

    def _loaded(self, success):
        self.status.setText(self.tr('Ready') if success else self.tr('Unable to display file structure.'))

    def shutdown(self):
        self.job.shutdown()
        if self.web_view is not None:
            self.web_view.stop()
        self.temp_dir.remove()
        super().shutdown()
