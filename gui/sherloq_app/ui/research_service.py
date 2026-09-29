"""Resident analysis workers with the same cancellation/admission contract as TruFor."""
from PySide6.QtCore import QCoreApplication
from .trufor_service import TruForService,TruForCommand

class ResearchService(TruForService):
    worker_filename='research_worker.py'
    label='Forensic analysis'

class CloneService(ResearchService):
    worker_filename='clone_detector_worker.py'
    label='Détecteurs de clones'
    @property
    def python_executable(self):
        from pathlib import Path
        return str(Path(__file__).resolve().parents[4]/'integration/clone_detectors/.venv-validation/bin/python')

SERVICES={}
def service(method):
    if method not in SERVICES:SERVICES[method]=(CloneService if method=='clone_detectors' else ResearchService)(QCoreApplication.instance())
    return SERVICES[method]

class ResearchCommand(TruForCommand):
    def __init__(self,parent,method):
        super().__init__(parent,service(method));self.method=method
    def start(self,request):
        if self.closed or self.is_busy:return
        self.output.clear();self.is_busy=True;self.busy.emit(True)
        self.service.submit(self,dict(request,method=self.method))
    def cancel(self):
        self.service.cancel(self);self.busy.emit(False)
