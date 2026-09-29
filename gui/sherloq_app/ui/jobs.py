"""Debounced, serial per-tool jobs with bounded global concurrency.

A slider burst replaces pending work. An already-running native operation may
finish, but an obsolete result can never overwrite the latest requested state.
"""
import logging
import sys
from time import perf_counter
from ..core.macos_activity import user_activity
from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, Slot, Qt

_POOL = QThreadPool()
_POOL.setMaxThreadCount(2)
if sys.platform == "darwin":
    # NumPy's bundled OpenBLAS LAPACK can exhaust Qt's default worker stack even
    # for a 3x3 inversion used by Matplotlib. Reserve a normal 8 MiB stack.
    _POOL.setStackSize(8 * 1024 * 1024)


STAGING_POOL = QThreadPool()
STAGING_POOL.setMaxThreadCount(1)
if sys.platform == 'darwin':STAGING_POOL.setStackSize(8 * 1024 * 1024)


class _Signals(QObject):
    finished = Signal(int, object, object, float)


class _Work(QRunnable):
    def __init__(self, token, compute, params):
        super().__init__()
        self.token, self.compute, self.params = token, compute, params
        self.signals = _Signals()

    def run(self):
        start = perf_counter()
        result = error = None
        try:
            with user_activity():
                result = self.compute(self.params)
        except Exception as exc:
            logging.exception('Interactive analysis failed')
            error = str(exc)
        self.signals.finished.emit(self.token, result, error, perf_counter()-start)


class LatestJob(QObject):
    result = Signal(object)
    busy = Signal(bool)
    failed = Signal(str)

    def __init__(self, parent, compute, delay=80, pool=None):
        super().__init__(parent)
        self.compute = compute
        self.pool = _POOL if pool is None else pool
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(delay)
        self.timer.timeout.connect(self._ready)
        self.token = 0
        self.pending = None
        self.active = None
        self.closed = False
        self.ready = False
        self.seconds = 0
        self.is_busy = False

    def request(self, params):
        if self.closed:
            return
        self.token += 1
        self.pending = params
        self.ready = False
        self.is_busy = True
        self.busy.emit(True)
        self.timer.start()

    def _ready(self):
        self.ready = True
        self._launch()

    def _launch(self):
        if self.closed or self.active is not None or not self.ready or self.pending is None:
            return
        self.active = _Work(self.token, self.compute, self.pending)
        self.pending = None
        self.ready = False
        self.active.signals.finished.connect(self._finished, Qt.QueuedConnection)
        self.pool.start(self.active)

    @Slot(int, object, object, float)
    def _finished(self, token, result, error, seconds):
        self.active = None
        if self.closed:
            return
        if token == self.token:
            self.seconds = seconds
            self.is_busy = False
            self.busy.emit(False)
            if error is None:
                self.result.emit(result)
            else:
                self.failed.emit(error)
        self._launch()

    def invalidate(self):
        """Discard pending/active results without closing this reusable worker."""
        if self.closed:
            return
        self.timer.stop()
        self.pending = None
        self.ready = False
        self.token += 1
        self.is_busy = False
        self.busy.emit(False)

    def shutdown(self):
        self.closed = True
        self.timer.stop()
        self.pending = None
        self.token += 1
