"""Process-isolated comparison with async input staging and resumable checkpoints."""
import json
import os
import signal
import sys
import tempfile
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import numpy as np
from PySide6.QtCore import QObject, Signal
from .jobs import LatestJob
from .process import ProcessJob


def prepare(context):
    context.writing.set()
    try:
        for name, array in [('evidence', context.engine.evidence), ('reference', context.engine.reference)]:
            if context.obsolete.is_set():
                return context
            path = Path(context.folder.name) / (name + '.npy')
            if not path.exists():
                temporary = path.with_suffix('.partial.npy')
                np.save(temporary, array, allow_pickle=False)
                temporary.replace(path)
        return context
    finally:
        context.writing.clear()
        if context.obsolete.is_set():
            context.folder.cleanup()


class ComparisonJob(QObject):
    result = Signal(object)
    failed = Signal(str)
    busy = Signal(bool)

    def __init__(self, parent):
        super().__init__(parent)
        self.context = None
        self.closed = False
        self.is_busy = False
        self.cancelled = False

    def request(self, engine):
        if self.closed or self.is_busy:
            return
        if len(engine.values) == 20:
            self.result.emit({'values': dict(engine.values), 'errors': {}, 'cancelled': False})
            return
        if self.context is None or self.context.engine is not engine:
            self.invalidate()
            context = SimpleNamespace(engine=engine, obsolete=Event(), writing=Event(),
                                      folder=tempfile.TemporaryDirectory(prefix='sherloq-comparison-'), offset=0)
            context.prepare = LatestJob(self, prepare, delay=0)
            context.process = ProcessJob(self, max_output=1024*1024, timeout_ms=3600000,heavy=True)
            context.process.waiting.connect(lambda waiting, c=context: setattr(c.engine, 'progress', (0, 'Waiting for another analysis…')) if waiting else None)
            context.process.timer.timeout.disconnect()
            context.process.timer.timeout.connect(lambda c=context: self._timeout(c))
            context.prepare.result.connect(self._prepared)
            context.prepare.failed.connect(lambda error, c=context: self._failure(c, error))
            context.process.result.connect(lambda _, c=context: self._finish(c))
            context.process.failed.connect(lambda error, c=context: self._failure(c, error))
            context.process.process.readyReadStandardOutput.connect(lambda c=context: self._progress(c))
            self.context = context
        self.is_busy = True
        self.cancelled = False
        self.busy.emit(True)
        self.context.prepare.request(self.context)

    def _prepared(self, context):
        if context is not self.context or self.closed:
            return
        if self.cancelled:
            self._finish(context)
            return
        context.offset = 0
        script = Path(__file__).resolve().parents[1] / 'core/comparison_worker.py'
        context.process.start(sys.executable, [str(script), context.folder.name])

    def _progress(self, context):
        if context is not self.context:
            return
        # ProcessJob's readyRead handler already collected this bounded buffer.
        data = bytes(context.process.output)
        end = data.rfind(b'\n') + 1
        for line in data[context.offset:end].splitlines():
            try:
                context.engine.progress = tuple(json.loads(line)['progress'])
            except (ValueError, KeyError, TypeError):
                continue
        context.offset = end

    def _finish(self, context):
        if context is not self.context or self.closed:
            return
        try:
            checkpoint = Path(context.folder.name) / 'checkpoint.json'
            if not checkpoint.exists() and not self.cancelled:
                raise ValueError('Comparison worker did not produce a result.')
            saved = json.loads(checkpoint.read_text()) if checkpoint.exists() else {'values': {}, 'errors': {}, 'maps': []}
            context.engine.values.update({name: float.fromhex(value) for name, value in saved['values'].items()})
            for name in saved['maps']:
                if name not in context.engine.maps:
                    context.engine.maps[name] = np.asarray(np.load(Path(context.folder.name) / (name + '.npy'), mmap_mode='r'))
            value = {'values': dict(context.engine.values), 'errors': saved['errors'], 'cancelled': self.cancelled}
        except Exception as exc:
            self.is_busy = False
            self.busy.emit(False)
            self.failed.emit(str(exc))
            return
        self.is_busy = False
        self.busy.emit(False)
        self.result.emit(value)

    def _failure(self, context, error):
        if context is not self.context or self.closed:
            return
        if self.cancelled:
            self._finish(context)
            return
        self.is_busy = False
        self.busy.emit(False)
        self.failed.emit(error)

    @staticmethod
    def _kill_group(context):
        pid = context.process.process.processId()
        if pid and os.name == 'posix':
            try:
                # Never signal the application's inherited process group.
                if os.getpgid(pid) == pid:
                    os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _timeout(self, context):
        if context is self.context:
            self._kill_group(context)
            context.process.cancel('Comparison timed out.')

    def cancel(self):
        if not self.is_busy or self.context is None:
            return
        self.cancelled = True
        if self.context.process.is_busy:
            self._kill_group(self.context)
            self.context.process.cancel('Comparison cancelled.')

    def invalidate(self):
        context, self.context = self.context, None
        if context is not None:
            context.obsolete.set()
            self._kill_group(context)
            context.process.shutdown()
            context.process.deleteLater()
            active_preparation = context.writing.is_set()
            context.prepare.shutdown()
            context.prepare.deleteLater()
            if not active_preparation:
                context.folder.cleanup()
            # An active preparer owns the directory until its finally block.
        self.is_busy = False
        self.busy.emit(False)

    def shutdown(self):
        self.closed = True
        self.invalidate()
