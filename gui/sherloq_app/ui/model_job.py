"""Cancelable model downloads before inference; all disk/network work is off-GUI."""
from threading import Event
from PySide6.QtCore import QObject, Signal, QThreadPool
from .jobs import LatestJob
from ..core.model_store import Store

DOWNLOAD_POOL = QThreadPool()
DOWNLOAD_POOL.setMaxThreadCount(2)
DOWNLOAD_POOL.setStackSize(8 * 1024 * 1024)


class ModelJob(QObject):
    ready = Signal()
    failed = Signal(str)
    progress = Signal(int, str)
    _update = Signal(int, object, object, str)

    def __init__(self, parent, store_factory=Store):
        super().__init__(parent)
        self.store_factory = store_factory
        self.serial = 0
        self.cancel_event = Event()
        self.closed = False
        self.job = LatestJob(self, self.compute, delay=0, pool=DOWNLOAD_POOL)
        self.job.result.connect(lambda _: self.ready.emit())
        self.job.failed.connect(self.failed)
        self._update.connect(self.report)

    def compute(self, request):
        serial, feature, event = request
        self.store_factory().ensure(feature, event.is_set,
            lambda n, total, text: self._update.emit(serial, n, total, text))

    def report(self, serial, count, total, text):
        if serial == self.serial and not self.closed:
            suffix = f' — {count / 1048576:.1f} / {total / 1048576:.1f} MiB' if total else ''
            self.progress.emit(round(100 * count / total) if total else 100, text + suffix)

    def request(self, feature):
        self.cancel()
        self.cancel_event = Event()
        self.job.request((self.serial, feature, self.cancel_event))

    def cancel(self):
        self.cancel_event.set()
        self.serial += 1
        self.job.invalidate()

    def shutdown(self):
        self.closed = True
        self.cancel()
        self.job.shutdown()
