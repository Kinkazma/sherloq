"""One resident TruFor model per application; serialized requests and idle expiry.

All methods run in the GUI thread. Inputs remain owned by the requesting job
until completion/cancellation. Cancelling active inference reaps the worker;
queued requests survive and restart a fresh worker. No Torch import in the GUI.
"""
import atexit
from collections import deque
import json
from pathlib import Path
import sys
import weakref
from .heavy_jobs import HEAVY_JOBS
from PySide6.QtCore import QObject, QCoreApplication, QProcess, QTimer, Signal


class TruForService(QObject):
    worker_filename = 'trufor_worker.py'
    label = 'TruFor'
    def __init__(self, app):
        super().__init__(app)
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self._stdout)
        self.process.readyReadStandardError.connect(self._stderr)
        self.process.started.connect(self._send)
        self.process.finished.connect(self._exit)
        self.process.errorOccurred.connect(self._error)
        self.queue = deque()
        self.active = None
        self.buffer = bytearray()
        self.errors = bytearray()
        self.serial = 0
        self.stopping = False
        self.closed = False
        self.last_memory = {}
        self.admission_pending = False
        self.idle = QTimer(self)
        self.idle.setSingleShot(True)
        self.idle.setInterval(120000)
        self.idle.timeout.connect(self._stop)
        self.deadline = QTimer(self)
        self.deadline.setSingleShot(True)
        self.deadline.setInterval(3600000)
        self.deadline.timeout.connect(lambda: self._abort(f'{self.label} analysis timed out.'))
        app.aboutToQuit.connect(self.shutdown)
        atexit.register(self.shutdown)

    def submit(self, client, request):
        self.serial += 1
        request['id'] = self.serial
        client.request_id = self.serial
        client.waiting.emit(True)
        self.queue.append((weakref.ref(client), request))
        self._pump()

    def _pump(self):
        if self.closed or self.active is not None:
            return
        while self.queue:
            reference, request = self.queue.popleft()
            client = reference()
            if client is not None and not client.closed and client.is_busy:
                self.active = reference, request
                break
        if self.active is None:
            if self.process.state() != QProcess.NotRunning:
                self.idle.start()
            return
        self.admission_pending = True
        HEAVY_JOBS.request(self)

    def _start_admitted(self):
        self.idle.stop()
        self.admission_pending = False
        if self.closed or self.active is None:
            HEAVY_JOBS.release(self)
            return
        client = self.active[0]()
        if client is not None:
            client.waiting.emit(False)
        self.errors.clear()
        self.deadline.start()
        if self.process.state() == QProcess.NotRunning:
            self.buffer.clear()
            worker = Path(__file__).resolve().parents[1]/'core'/self.worker_filename
            self.process.start(getattr(self,'python_executable',sys.executable), ['-u', str(worker), '--serve'])
        else:
            self._send()

    def _send(self):
        if self.active is not None:
            self.process.write((json.dumps(self.active[1])+'\n').encode())

    def _stdout(self):
        self.buffer.extend(bytes(self.process.readAllStandardOutput()))
        if len(self.buffer) > 1024*1024:
            self._abort(f'{self.label} output exceeded its limit.')
            return
        while b'\n' in self.buffer:
            line, _, rest = self.buffer.partition(b'\n')
            self.buffer = bytearray(rest)
            if self.active is None:
                continue
            reference, request = self.active
            client = reference()
            if client is not None:
                client.output.extend(line+b'\n')
                if len(client.output) > 1024*1024:
                    self._abort(f'{self.label} output exceeded its limit.')
                    return
            if line.startswith(b'SHERLOQ '):
                try:
                    event = json.loads(line[8:])
                    if event['id'] != request['id']:
                        raise ValueError(f'Unexpected {self.label} result.')
                    if event['event'] not in ('done', 'error'):
                        raise ValueError(f'Unknown {self.label} response.')
                except (ValueError, KeyError) as exc:
                    self._abort(str(exc))
                    return
                self.last_memory = {k:v for k,v in event.items() if k.startswith('idle_mps_')}
                self._complete(event.get('message') if event['event']=='error' else None)

    def _stderr(self):
        self.errors.extend(bytes(self.process.readAllStandardError()))
        del self.errors[:-8192]

    def _complete(self, error=None):
        self.deadline.stop()
        HEAVY_JOBS.release(self)
        active, self.active = self.active, None
        if active is not None:
            client = active[0]()
            if client is not None and not client.closed:
                client.is_busy = False
                client.busy.emit(False)
                if error:
                    client.failed.emit(error)
                else:
                    client.result.emit(bytes(client.output))
        QTimer.singleShot(0, self._pump)

    def _stop(self):
        self.stopping = True
        self.idle.stop()
        if self.process.state() != QProcess.NotRunning:
            self.process.kill()
            self.process.waitForFinished(1000)
        self.buffer.clear()
        self.stopping = False

    def _abort(self, message):
        self._stop()
        self._complete(message)

    def _exit(self, code, status):
        if not self.stopping and self.active is not None:
            self._stderr()
            self._complete(bytes(self.errors).decode(errors='replace').strip() or f'{self.label} worker stopped.')

    def _error(self, error):
        if error == QProcess.FailedToStart:
            self._complete(self.process.errorString())

    def cancel(self, client):
        self.queue = deque((ref, request) for ref, request in self.queue if ref() is not client)
        if self.active is not None and self.active[0]() is client:
            self._stop()
            self.active = None
            self.deadline.stop()
            self.admission_pending = False
            HEAVY_JOBS.release(self)
        client.is_busy = False
        QTimer.singleShot(0, self._pump)

    def shutdown(self):
        if self.closed:
            return
        self.closed = True
        self.queue.clear()
        self.active = None
        self.deadline.stop()
        self._stop()
        HEAVY_JOBS.release(self)


_SERVICE = None


def service():
    global _SERVICE
    if _SERVICE is None:
        app = QCoreApplication.instance()
        if app is None:
            raise RuntimeError('TruFor needs an application event loop.')
        _SERVICE = TruForService(app)
    return _SERVICE


class TruForCommand(QObject):
    result = Signal(object)
    failed = Signal(str)
    busy = Signal(bool)
    waiting = Signal(bool)

    def __init__(self, parent, service_instance=None):
        super().__init__(parent)
        self.service = service_instance if service_instance is not None else service()
        self.process = self.service.process
        self.output = bytearray()
        self.closed = self.is_busy = False

    def start(self, program, arguments):
        if self.closed or self.is_busy:
            return
        options = dict(zip(arguments[2::2], arguments[3::2]))
        request = dict(input=options['--input'], output_dir=options['--output-dir'], device=options['--device'])
        self.output.clear()
        self.is_busy = True
        self.busy.emit(True)
        self.service.submit(self, request)

    def shutdown(self):
        self.closed = True
        self.service.cancel(self)
        self.output.clear()
