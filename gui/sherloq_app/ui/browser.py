from gui.sherloq_app.ui.localization import t
"""One retained WebEngine view per panel, with explicit navigation and exports."""
from pathlib import Path
from PySide6.QtCore import QUrl, QSettings, QTimer
from PySide6.QtGui import QDesktopServices
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog


class BrowserPanel(QWidget):
    def __init__(self, url, parent=None):
        super().__init__(parent)
        self.target = QUrl(url)
        self.started = False
        self.closed = False
        self.downloads = []
        self.web_view = QWebEngineView(self)
        self.status = QLabel(self.tr('Ready'))
        self.status.setWordWrap(True)
        self.retry_button = QPushButton(self.tr('Reload'))
        self.browser_button = QPushButton(self.tr('Open in browser'))
        controls = QHBoxLayout()
        controls.addWidget(self.status, 1)
        controls.addWidget(self.retry_button)
        controls.addWidget(self.browser_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addWidget(self.web_view, 1)
        self.retry_button.clicked.connect(self.reload)
        self.browser_button.clicked.connect(self.open_external)
        self.load_timer = QTimer(self)
        self.load_timer.setSingleShot(True)
        self.load_timer.setInterval(30000)
        self.load_timer.timeout.connect(lambda: self.status.setText(self.tr('This page is taking longer than expected. You can retry or open it in your browser.')))
        self.web_view.loadStarted.connect(self._loading)
        self.web_view.loadProgress.connect(lambda n: self.status.setText(self.tr('Loading… ') + str(n) + '%'))
        self.web_view.loadFinished.connect(self._loaded)

    def showEvent(self, event):
        super().showEvent(event)
        if not self.started and not self.closed:
            self._start()

    def _start(self):
        self.started = True
        # The default profile is shared. Handle only this page's requests.
        self.web_view.page().profile().downloadRequested.connect(self._download)
        self.web_view.load(self.target)

    def load(self, url):
        url = QUrl(url)
        if self.closed or url == self.target:
            return
        self.target = url
        if self.started:
            self.web_view.load(url)

    def reload(self):
        if self.closed:
            return
        if not self.started:
            self._start()
        else:
            self.web_view.reload()

    def open_external(self):
        url = self.web_view.url()
        QDesktopServices.openUrl(url if url.scheme() in ('http', 'https') else self.target)

    def _loading(self):
        if not self.closed:
            self.status.setText(self.tr('Loading…'))
            self.load_timer.start()

    def _loaded(self, success):
        self.load_timer.stop()
        if not self.closed:
            self.status.setText(self.tr('Ready') if success else self.tr('Unable to load this page. Retry or open it in your browser.'))

    def _download(self, request):
        if self.closed or request.page() != self.web_view.page():
            return
        settings = QSettings()
        name = Path(request.downloadFileName() or 'download.bin').name
        folder = Path(settings.value('save_folder') or str(Path.home() / 'Downloads'))
        filename, _ = QFileDialog.getSaveFileName(self, t(self.tr('Save downloaded file')), str(folder / name))
        if not filename or self.closed:
            request.cancel()
            request.deleteLater()
            return
        path = Path(filename)
        request.setDownloadDirectory(str(path.parent))
        request.setDownloadFileName(path.name)
        self.downloads.append(request)
        request.isFinishedChanged.connect(lambda: self._download_finished(request))
        request.accept()
        settings.setValue('save_folder', str(path.parent))
        self.status.setText(self.tr('Saving file…'))

    def _download_finished(self, request):
        if not request.isFinished():
            return
        if request in self.downloads:
            self.downloads.remove(request)
        if not self.closed:
            from PySide6.QtWebEngineCore import QWebEngineDownloadRequest
            done = request.state() == QWebEngineDownloadRequest.DownloadCompleted
            self.status.setText(self.tr('File saved.') if done else self.tr('Download failed or cancelled.'))
        request.deleteLater()

    def shutdown(self):
        self.closed = True
        self.load_timer.stop()
        if self.started:
            self.web_view.stop()
        for request in list(self.downloads):
            request.cancel()
        self.downloads.clear()
