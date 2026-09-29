"""Asynchronous staging, resident inference, validated cached array results."""
import hashlib,json
import shutil
from pathlib import Path
import numpy as np
from PySide6.QtCore import QObject,Signal
from .trufor_job import _Files,_prepare
from .research_service import ResearchCommand
from .jobs import LatestJob,STAGING_POOL
from ..core.interactive import ArrayCache

def prepare(files):
    _prepare(files)
    if getattr(files,'source_filename',None):
        with files.borrow() as root:
            target=root/'source.image'
            if not target.exists():
                source=Path(files.source_filename);before=source.stat()
                if before.st_size>512*1024**2:raise ValueError('Source file exceeds 512 MiB.')
                partial=root/'source.partial';shutil.copyfile(source,partial);after=source.stat()
                if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):raise ValueError('Source changed during copying. Reload the image.')
                partial.replace(target)
    return files

def read_result(request):
    files,key=request
    with files.borrow() as root:
        folder=root/key;metadata=json.loads((folder/'result.json').read_text());result={'metadata':metadata}
        for name,info in metadata['arrays'].items():
            if not name.isidentifier():raise ValueError('Invalid output array name.')
            array=np.load(folder/f'{name}.npy',allow_pickle=False)
            if list(array.shape)!=info['shape'] or str(array.dtype)!=info['dtype'] or not np.isfinite(array).all():raise ValueError('Invalid analysis array: '+name)
            result[name]=array
        return key,result

class ResearchJob(QObject):
    result=Signal(object);failed=Signal(str);busy=Signal(bool);progress=Signal(int,str)
    def __init__(self,parent,image,method,filename=None):
        super().__init__(parent);self.files=_Files(image);self.cache=ArrayCache(384);self.closed=self.is_busy=False;self.current=None;self.offset=0
        self.files.source_filename=filename
        self.preparer=LatestJob(self,prepare,delay=0,pool=STAGING_POOL);self.loader=LatestJob(self,read_result,delay=0);self.command=ResearchCommand(self,method)
        self.preparer.result.connect(self._prepared);self.loader.result.connect(self._loaded);self.command.result.connect(self._finished)
        for job in (self.preparer,self.loader,self.command):job.failed.connect(self._failure)
        self.command.waiting.connect(lambda w:self.progress.emit(0,'En attente d’une autre analyse…') if w else None)
        self.command.process.readyReadStandardOutput.connect(self._progress)
    def request(self,params,device):
        if self.closed or self.is_busy:return
        key=hashlib.sha256(json.dumps([params,device],sort_keys=True).encode()).hexdigest();self.current=(key,params,device);self.is_busy=True;self.busy.emit(True)
        cached=self.cache.get(key)
        if cached is not None:self._loaded((key,cached));return
        self.preparer.request(self.files)
    def _prepared(self,files):
        if self.closed or self.current is None:return
        key,params,device=self.current
        if (files.path/key/'result.json').exists():self._finished();return
        self.offset=0;self.command.start(dict(input=str(files.path/'image.npy'),source=str(files.path/'source.image') if self.files.source_filename else None,output_dir=str(files.path/key),params=params,device=device))
    def _finished(self,*_):
        if not self.closed and self.current is not None:self.loader.request((self.files,self.current[0]))
    def _loaded(self,value):
        key,result=value
        if self.closed or self.current is None or key!=self.current[0]:return
        self.cache.put(key,result);self.is_busy=False;self.busy.emit(False);self.result.emit(result)
    def _progress(self):
        raw=bytes(self.command.output);end=raw.rfind(b'\n')+1
        for line in raw[self.offset:end].decode(errors='replace').splitlines():
            if line.startswith('Progress '):
                try:
                    d,n=map(int,line[9:].split('/'));self.progress.emit(10+80*d//n,f'Analyse {d}/{n}')
                except (ValueError,ZeroDivisionError):pass
            elif line.startswith('Loading'):self.progress.emit(5,'Chargement du modèle…')
        self.offset=end
    def _failure(self,error):
        if self.closed:return
        self.is_busy=False;self.busy.emit(False);self.failed.emit(error)
    def cancel(self):
        self.current=None;self.preparer.invalidate();self.loader.invalidate();self.command.cancel();self.is_busy=False;self.busy.emit(False)
    def shutdown(self):
        if self.closed:return
        self.closed=True;self.cancel();self.preparer.shutdown();self.loader.shutdown();self.command.shutdown();self.files.retire();self.cache.clear()
