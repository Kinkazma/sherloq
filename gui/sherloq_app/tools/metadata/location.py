import os
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar, QApplication
from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.ui.process import ProcessJob
from gui.sherloq_app.ui.browser import BrowserPanel
from gui.sherloq_app.core.utility import exiftool_exe
from gui.sherloq_app.core.digest import identity
from gui.sherloq_app.core.location import parse_location


class LocationWidget(ToolWidget):
    def __init__(self, filename, parent=None):
        super().__init__(parent)
        self.filename = filename
        self.coordinates = None
        self.browser = None
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.refresh_button = QPushButton(self.tr('Recalculate'))
        self.cancel_button = QPushButton(self.tr('Cancel'))
        self.copy_button = QPushButton(self.tr('Copy coordinates'))
        self.copy_button.setEnabled(False)
        controls = QHBoxLayout()
        for button in (self.refresh_button, self.cancel_button, self.copy_button):
            controls.addWidget(button)
        controls.addStretch()
        self.main_layout = QVBoxLayout(self)
        self.main_layout.addWidget(self.status)
        self.main_layout.addWidget(self.progress)
        self.main_layout.addLayout(controls)
        self.setMinimumSize(700, 500)
        self.job = ProcessJob(self, max_output=1024 * 1024)
        self.job.result.connect(self._finished)
        self.job.failed.connect(self.status.setText)
        self.job.busy.connect(self._busy)
        self.refresh_button.clicked.connect(self.recalculate)
        self.cancel_button.clicked.connect(lambda: self.job.cancel())
        self.copy_button.clicked.connect(self.copy_coordinates)
        self._busy(False)
        self.recalculate()

    def _busy(self, busy):
        self.progress.setVisible(busy)
        self.refresh_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)

    def recalculate(self):
        if self.job.is_busy or self.job.closed:
            return
        self.coordinates = None
        self.copy_button.setEnabled(False)
        if self.browser is not None:
            self.main_layout.removeWidget(self.browser)
            self.browser.shutdown()
            self.browser.deleteLater()
            self.browser = None
        try:
            self.stamp = identity(os.stat(self.filename))
        except OSError as error:
            self.status.setText(str(error))
            return
        self.status.setText(self.tr('Reading GPS coordinates…'))
        self.job.start(exiftool_exe(), ['-G', '-n', '-j', '-Composite:GPSLatitude', '-Composite:GPSLongitude', self.filename])

    def _finished(self, data):
        try:
            if identity(os.stat(self.filename)) != self.stamp:
                self.status.setText(self.tr('File changed during analysis. Please recalculate.'))
                return
            result = parse_location(data)
        except (OSError, ValueError) as error:
            self.status.setText(str(error))
            return
        if result is None:
            self.status.setText(self.tr('Geolocation data not found!'))
            return
        lat, lon, url = result
        self.coordinates = (lat, lon)
        self.status.setText(self.tr('Latitude: ') + str(lat) + '°   ' + self.tr('Longitude: ') + str(lon) + '°')
        self.copy_button.setEnabled(True)
        self.browser = BrowserPanel(url, self)
        self.main_layout.addWidget(self.browser, 1)

    def copy_coordinates(self):
        if self.coordinates is not None:
            lat, lon = self.coordinates
            QApplication.clipboard().setText(f'{lat}, {lon}')

    def shutdown(self):
        self.job.shutdown()
        if self.browser is not None:
            self.browser.shutdown()
        super().shutdown()
