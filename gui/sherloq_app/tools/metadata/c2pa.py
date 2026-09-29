from gui.sherloq_app.ui.localization import t
"""C2PA read-only provenance panel; full report stays in memory."""
import json,tempfile
from pathlib import Path
from PySide6.QtCore import QProcessEnvironment,Qt
from PySide6.QtWidgets import QLabel,QPushButton,QHBoxLayout,QVBoxLayout,QFileDialog,QPlainTextEdit
from gui.sherloq_app.core import c2pa
from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.ui.jobs import LatestJob
from gui.sherloq_app.ui.process import ProcessJob

class C2paWidget(ToolWidget):
    def __init__(self,filename,parent=None):
        super().__init__(parent);self.filename=filename;self.trust_file=None;self.result=None;self.metadata=None;self.active=False;self.closed=False;self.directory=tempfile.TemporaryDirectory(prefix='sherloq-c2pa-')
        self.run=QPushButton('Valider');self.trust=QPushButton('Liste de confiance…');self.clear=QPushButton('Sans liste');self.save=QPushButton('Exporter JSON');self.save.setEnabled(False)
        self.summary=QLabel('Lecture hors ligne. Intégrité, signature et confiance seront vérifiées séparément.');self.summary.setWordWrap(True);self.summary.setTextFormat(Qt.PlainText)
        self.context=QLabel('Confiance : aucune liste configurée. Révocation en ligne non consultée.');self.context.setWordWrap(True);self.context.setTextFormat(Qt.PlainText)
        self.details=QPlainTextEdit();self.details.setReadOnly(True);layout=QVBoxLayout(self);row=QHBoxLayout()
        for c in (self.run,self.trust,self.clear,self.save):row.addWidget(c)
        row.addStretch();layout.addLayout(row);layout.addWidget(self.summary);layout.addWidget(self.context);layout.addWidget(self.details)
        self.prepare_job=LatestJob(self,c2pa.prepare,delay=0);self.parse_job=LatestJob(self,c2pa.parse,delay=0);self.export_job=LatestJob(self,c2pa.export,delay=0);self.process=ProcessJob(self,max_output=32*1024*1024,timeout_ms=60000)
        env=QProcessEnvironment.systemEnvironment()
        for name in env.keys():
            if name.startswith(('C2PA','C2PATOOL')):env.remove(name)
        self.process.process.setProcessEnvironment(env)
        self.prepare_job.result.connect(self.launch);self.prepare_job.failed.connect(self.failed);self.process.result.connect(lambda data:self.parse_job.request((data,'',self.metadata)));self.process.failed.connect(self.process_failed);self.parse_job.result.connect(self.complete);self.parse_job.failed.connect(self.failed);self.export_job.failed.connect(self.summary.setText);self.export_job.result.connect(lambda p:self.summary.setText(f'Rapport exporté : {p}'))
        self.run.clicked.connect(self.start);self.trust.clicked.connect(self.choose_trust);self.clear.clicked.connect(self.clear_trust);self.save.clicked.connect(self.export_data)
    def state(self):
        self.run.setText('Annuler' if self.active else 'Valider');self.trust.setEnabled(not self.active);self.clear.setEnabled(not self.active);self.save.setEnabled(not self.active and self.result is not None)
    def start(self):
        if self.active:
            self.prepare_job.invalidate();self.parse_job.invalidate();self.process.cancel();self.active=False;self.state();self.summary.setText('Annulé.');return
        if self.process.is_busy:return
        self.active=True;self.state();self.summary.setText('Validation hors ligne…');self.prepare_job.request((self.filename,self.trust_file,self.directory.name,self.directory))
    def launch(self,value):
        if not self.active:return
        args,self.metadata=value;self.process.start(str(c2pa.BINARY),args)
    def process_failed(self,error):
        if not self.active:return
        if self.process.failure:self.failed(error)
        else:self.parse_job.request((b'',error,self.metadata))
    def failed(self,error):self.active=False;self.state();self.summary.setText(error)
    def complete(self,result):
        if not self.active:return
        self.active=False;self.result=result;self.state();self.details.setPlainText(json.dumps(result,ensure_ascii=False,indent=2))
        state=result['manifest']
        if state!='present':self.summary.setText(dict(absent='Aucun manifeste C2PA trouvé.',remote_unavailable='Manifeste distant : non consulté en mode hors ligne.',read_error='Le fichier n’a pas pu être validé.',unknown='État du manifeste indéterminé.')[state]);return
        labels=dict(valid='validée',invalid='invalide',unknown='indéterminée',not_configured='non configurée',trusted='reconnue par la liste locale',untrusted='non reconnue par la liste locale')
        self.summary.setText(f"Intégrité : {labels[result['integrity']]} · Signature : {labels[result['signature']]} · Confiance : {labels[result['trust']]}.\nCes états ne constituent pas un verdict d’authenticité de la scène.")
    def choose_trust(self):
        path,_=QFileDialog.getOpenFileName(self,t('Liste locale de certificats de confiance'),'',t('Certificats (*.pem *.crt);;Tous les fichiers (*)'))
        if path:self.trust_file=path;self.result=None;self.state();self.context.setText(f'Liste locale : {path}. Révocation en ligne non consultée.');self.start()
    def clear_trust(self):
        self.trust_file=None;self.result=None;self.state();self.context.setText('Confiance : aucune liste configurée. Révocation en ligne non consultée.');self.start()
    def export_data(self):
        if not self.save.isEnabled():return
        path,_=QFileDialog.getSaveFileName(self,t('Exporter la validation C2PA'),'c2pa.json',t('JSON (*.json)'))
        if path:self.export_job.request((str(Path(path).with_suffix('.json')),self.result))
    def shutdown(self):
        if self.closed:return
        self.closed=True;self.active=False;self.process.shutdown();super().shutdown();directory=self.directory;self.directory=None
        if self.prepare_job.active is not None:self.prepare_job.active.signals.finished.connect(lambda *args:directory.cleanup())
        else:directory.cleanup()
