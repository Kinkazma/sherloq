"""TruFor process lifecycle with asynchronous staging and mapped result caches."""
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
import json
import sys
import tempfile
import numpy as np
from PySide6.QtCore import QObject, Signal
from .jobs import LatestJob,STAGING_POOL
from .trufor_service import TruForCommand
from .model_job import ModelJob
from gui.sherloq_app.core.jpeg_curve import Cancelled
from gui.sherloq_app.core.image_buffers import all_finite
from gui.sherloq_app.core.memory_resources import require_disk_space


class _Files:
    def __init__(self, image):
        self.image = image
        self.folder = tempfile.TemporaryDirectory(prefix='sherloq-trufor-')
        self.path = Path(self.folder.name)
        self.lock = Lock()
        self.users = 0
        self.closed = False

    @contextmanager
    def borrow(self):
        with self.lock:
            if self.closed:
                raise Cancelled()
            self.users += 1
        try:
            yield self.path
        finally:
            with self.lock:
                self.users -= 1
                if self.closed and self.users == 0:
                    self.folder.cleanup()

    def retire(self):
        with self.lock:
            self.closed = True
            if self.users == 0:
                self.folder.cleanup()


def _prepare(files):
    with files.borrow() as folder:
        path = folder/'image.npy'
        if not path.exists():
            require_disk_space(files.image.nbytes+4096,folder)
            temporary = folder/'image.partial.npy'
            np.save(temporary, files.image, allow_pickle=False)
            temporary.replace(path)
    return files


def _load(request):
    files, device = request
    with files.borrow() as root:
        folder = root/device
        metadata = json.loads((folder/'result.json').read_text())
        if metadata['device'] != device or len(metadata['imgsize']) != 2:
            raise ValueError('Invalid TruFor result metadata.')
        result = {key: np.asarray(value) for key, value in metadata.items()}
        for key in ('map', 'conf', 'np++'):
            array = np.asarray(np.load(folder/(key+'.npy'), mmap_mode='r', allow_pickle=False))
            if array.dtype != np.float32 or array.shape != tuple(metadata['imgsize']) or not all_finite(array):
                raise ValueError('Invalid TruFor result array: '+key)
            result[key] = array
        if not np.isfinite(result['score']).all():
            raise ValueError('Invalid TruFor score.')
    return device, result


class TruForJob(QObject):
    result = Signal(object)
    failed = Signal(str)
    busy = Signal(bool)
    progress = Signal(int, str)

    def __init__(self, parent, image):
        super().__init__(parent)
        self.files = _Files(image)
        self.results = {}
        self.current = None
        self.closed = self.is_busy = False
        self.output_offset = 0
        self.preparer = LatestJob(self, _prepare, delay=0, pool=STAGING_POOL)
        self.loader = LatestJob(self, _load, delay=0)
        self.command = TruForCommand(self)
        self.models = ModelJob(self)
        self.models.ready.connect(self._models_ready)
        self.models.failed.connect(self._failure)
        self.models.progress.connect(self.progress)
        self.preparer.result.connect(self._prepared)
        self.loader.result.connect(self._loaded)
        self.command.waiting.connect(lambda waiting: self.progress.emit(0, 'En attente d’une autre analyse…') if waiting else None)
        self.command.result.connect(self._finished)
        for job in (self.preparer, self.loader, self.command):
            job.failed.connect(self._failure)
        self.command.process.readyReadStandardOutput.connect(self._progress)

    def request(self, device):
        if self.closed or self.is_busy:
            return
        if device not in ('cpu', 'mps'):
            raise ValueError('Unknown TruFor backend.')
        self.current = device
        self.is_busy = True
        self.busy.emit(True)
        if device in self.results:
            self._loaded((device, self.results[device]))
        else:
            self.progress.emit(0, 'Préparation de l’image…')
            self.models.request("trufor")

    def _models_ready(self):
        if not self.closed and self.current is not None:
            self.preparer.request(self.files)

    def _prepared(self, files):
        if self.closed or self.current is None:
            return
        folder = files.path/self.current
        if (folder/'result.json').exists():
            self._finished()
            return
        self.output_offset = 0
        worker = Path(__file__).resolve().parents[1]/'core/trufor_worker.py'
        self.command.start(sys.executable, ['-u', str(worker), '--input', str(files.path/'image.npy'),
            '--output-dir', str(folder), '--device', self.current])

    def _progress(self):
        data = bytes(self.command.output)
        end = data.rfind(b'\n')+1
        for raw in data[self.output_offset:end].splitlines():
            line = raw.decode('utf-8', errors='replace')
            if line.startswith('Loading '):
                self.progress.emit(10, 'Chargement du modèle…')
            elif line.startswith('Analysing '):
                self.progress.emit(20, 'Analyse à la résolution originale…')
            elif line.startswith('Progress '):
                try:
                    done, total = map(int, line[9:].split('/'))
                    self.progress.emit(20+done*65//total, f'Analyse — étape {done}/{total}')
                except (ValueError, ZeroDivisionError):
                    pass
        self.output_offset = end

    def _finished(self, *_):
        if self.closed or self.current is None:
            return
        self.progress.emit(90, 'Lecture des résultats…')
        self.loader.request((self.files, self.current))

    def _loaded(self, value):
        device, data = value
        if self.closed or device != self.current:
            return
        self.results[device] = data
        self.is_busy = False
        self.busy.emit(False)
        self.result.emit(value)

    def _failure(self, message):
        if self.closed:
            return
        self.is_busy = False
        self.busy.emit(False)
        self.failed.emit(message)

    def invalidate(self):
        self.current = None
        self.models.cancel()
        self.preparer.invalidate()
        self.loader.invalidate()
        self.command.shutdown()
        self.command.closed = False
        self.is_busy = False
        self.busy.emit(False)

    def cancel(self):
        self.invalidate()
        self.failed.emit('Analyse annulée. Les résultats achevés sont conservés.')

    def shutdown(self):
        self.closed = True
        self.invalidate()
        self.models.shutdown()
        self.preparer.shutdown()
        self.loader.shutdown()
        self.command.closed = True
        self.files.retire()
        self.results.clear()
