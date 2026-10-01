"""Cancellable Noiseprint processes, asynchronous staging, retained checkpoints."""
import json
import sys
import tempfile
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import numpy as np
from PySide6.QtCore import QObject, Signal
from .jobs import LatestJob
from .process import ProcessJob
from gui.noiseprint.utility.stable_covariance import STATISTICS_POLICY


def prepare(context):
    context.writing.set()
    try:
        if context.obsolete.is_set():return context
        path=Path(context.folder.name)/'image.npy'
        if not path.exists():
            temporary=path.with_suffix('.partial.npy')
            np.save(temporary,context.image,allow_pickle=False)
            temporary.replace(path)
        return context
    finally:
        context.writing.clear()
        if context.obsolete.is_set():context.folder.cleanup()


class SplicingJob(QObject):
    result=Signal(object)
    failed=Signal(str)
    busy=Signal(bool)
    progress=Signal(int,str)

    def __init__(self,parent,image):
        super().__init__(parent)
        self.image=image
        self.context=None
        self.preparer=LatestJob(self,prepare,delay=0)
        self.preparer.result.connect(self._prepared)
        self.preparer.failed.connect(self._failure)
        self.process=ProcessJob(self,max_output=1024*1024,timeout_ms=3600000,heavy=True)
        self.process.waiting.connect(lambda waiting: self.progress.emit(0, 'En attente d’une autre analyse…') if waiting else None)
        self.process.result.connect(self._finish)
        self.process.failed.connect(self._failure)
        self.process.process.readyReadStandardOutput.connect(self._progress)
        self.closed=self.is_busy=self.cancelled=False
        self.offset=0

    def request(self,stage,model,backend="cpu"):
        if self.closed or self.is_busy:return
        if self.context is None or (self.context.model,self.context.backend)!=(model,backend):
            self.invalidate()
            self.context=SimpleNamespace(image=self.image,model=model,stage=stage,backend=backend,
                folder=tempfile.TemporaryDirectory(prefix='sherloq-noiseprint-'),
                writing=Event(),obsolete=Event())
        self.context.stage=stage
        self.is_busy=True;self.cancelled=False;self.busy.emit(True)
        folder=Path(self.context.folder.name)
        policy_matches=True
        if stage=='map':
            try:
                policy_matches=json.loads((folder/'map-statistics.json').read_text()).get('statistics_policy')==STATISTICS_POLICY
            except (OSError,ValueError):
                policy_matches=False
        if policy_matches and (folder/(stage+'-display.npy')).exists() and (folder/'result.json').exists():
            self._finish();return
        self.preparer.request(self.context)

    def _prepared(self,context):
        if self.closed or context is not self.context:return
        if self.cancelled:
            self._failure('Cancelled. Completed stages are retained.');return
        self.offset=0
        script=Path(__file__).resolve().parents[1]/'core/splicing_worker.py'
        self.process.start(sys.executable,[str(script),context.folder.name,context.stage,str(context.model),context.backend])

    def _progress(self):
        if self.closed or self.context is None:return
        data=bytes(self.process.output);end=data.rfind(b'\n')+1
        for line in data[self.offset:end].splitlines():
            try:
                value,text=json.loads(line)['progress'];self.progress.emit(value,text)
            except (ValueError,KeyError,TypeError):pass
        self.offset=end

    def _finish(self,*args):
        if self.closed or self.context is None:return
        folder=Path(self.context.folder.name)
        try:
            result=json.loads((folder/'result.json').read_text())
            result['noise']=np.asarray(np.load(folder/'noise.npy',mmap_mode='r',allow_pickle=False))
            result['noise_display']=np.asarray(np.load(folder/'noise-display.npy',mmap_mode='r',allow_pickle=False))
            if (folder/'map-display.npy').exists():
                result['map']=np.asarray(np.load(folder/'map-display.npy',mmap_mode='r',allow_pickle=False))
                result['raw_map']=np.asarray(np.load(folder/'map.npy',mmap_mode='r',allow_pickle=False))
        except Exception as exc:
            self._failure(str(exc));return
        self.is_busy=False;self.busy.emit(False);self.result.emit(result)

    def _failure(self,message):
        if self.closed:return
        self.is_busy=False;self.busy.emit(False)
        self.failed.emit('Cancelled. Completed tiles and stages are retained.' if self.cancelled else message)

    def cancel(self):
        if not self.is_busy:return
        self.cancelled=True
        if self.process.is_busy:self.process.cancel()

    def invalidate(self):
        context,self.context=self.context,None
        self.preparer.invalidate()
        # Disconnect completion delivery while stopping the old process.
        self.process.shutdown()
        self.process.closed=False
        if context is not None:
            context.obsolete.set()
            if not context.writing.is_set():context.folder.cleanup()
        self.is_busy=False
        self.busy.emit(False)

    def shutdown(self):
        self.closed=True
        self.invalidate()
        self.preparer.shutdown()
        self.process.closed=True
