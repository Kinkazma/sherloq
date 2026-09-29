"""Cancellable native commands with bounded captured output, owned by a widget."""
from PySide6.QtCore import QObject, QProcess, QTimer, Signal
from .heavy_jobs import HEAVY_JOBS


class ProcessJob(QObject):
    result = Signal(object)
    failed = Signal(str)
    busy = Signal(bool)
    waiting = Signal(bool)

    def __init__(self, parent, max_output=64 * 1024 * 1024, timeout_ms=120000, heavy=False):
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self._stdout)
        self.process.readyReadStandardError.connect(self._stderr)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(timeout_ms)
        self.timer.timeout.connect(lambda: self.cancel('Analysis timed out.'))
        self.max_output = max_output
        self.output = bytearray()
        self.errors = bytearray()
        self.failure = None
        self.is_busy = False
        self.closed = False
        self.heavy = heavy
        self.queued = False
        self.start_spec = None

    def start(self, program, arguments, output_file=None):
        if self.closed or self.is_busy:
            return
        self.output.clear()
        self.errors.clear()
        self.failure = None
        self.process.setStandardOutputFile(output_file or '')
        self.is_busy = True
        self.busy.emit(True)
        self.start_spec = program, arguments
        if self.heavy:
            self.queued = True
            self.waiting.emit(True)
            HEAVY_JOBS.request(self)
        else:
            self._start_admitted()

    def _start_admitted(self):
        self.queued = False
        self.waiting.emit(False)
        if self.closed or not self.is_busy:
            if self.heavy:
                HEAVY_JOBS.release(self)
            return
        self.process.start(*self.start_spec)
        self.timer.start()

    def _stdout(self):
        data = self.process.readAllStandardOutput()
        if len(self.output) + data.size() > self.max_output:
            self.cancel('Analysis output exceeds the memory limit.')
        else:
            self.output.extend(bytes(data))

    def _stderr(self):
        # Keep the diagnostic tail only; never retain unbounded error output.
        self.errors.extend(bytes(self.process.readAllStandardError()))
        if len(self.errors) > 8192:
            del self.errors[:-8192]

    def cancel(self, message='Analysis cancelled.'):
        if self.is_busy:
            self.failure = message
            if self.queued:
                self.queued = False
                HEAVY_JOBS.release(self)
                self._finished(-1, QProcess.CrashExit)
            else:
                self.process.kill()

    def _error(self, error):
        if error == QProcess.FailedToStart:
            self.failure = self.process.errorString()
            self._finished(-1, QProcess.CrashExit)

    def _finished(self, code, status):
        if not self.is_busy:
            return
        self.timer.stop()
        if self.process.isOpen():
            self._stdout()
            self._stderr()
        self.is_busy = False
        if self.heavy:
            HEAVY_JOBS.release(self)
        if self.closed:
            self.output.clear()
            self.errors.clear()
            return
        error = self.failure
        if error is None and (code != 0 or status != QProcess.NormalExit):
            error = bytes(self.errors).decode('utf-8', errors='replace').strip() or self.process.errorString()
        result = bytes(self.output)
        self.output.clear()
        self.errors.clear()
        self.busy.emit(False)
        if error:
            self.failed.emit(error)
        else:
            self.result.emit(result)

    def shutdown(self):
        self.closed = True
        self.timer.stop()
        if self.process.state() != QProcess.NotRunning:
            self.process.kill()
            self.process.waitForFinished(1000)
        self.output.clear()
        self.errors.clear()
        self.is_busy = False
        self.queued = False
        if self.heavy:
            HEAVY_JOBS.release(self)
