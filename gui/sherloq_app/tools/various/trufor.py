from gui.sherloq_app.ui.localization import t
"""Official TruFor integration: retained results, asynchronous views and exports.

See gui/TruFor_main/test_docker/LICENSE.txt for the model's license.
"""
from pathlib import Path
from threading import Event
import platform
import sys
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QComboBox,QProgressBar,QFileDialog
from gui.sherloq_app.paths import TRUFOR_DIR
from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.ui.viewer import ImageViewer
from gui.sherloq_app.ui.jobs import LatestJob
from gui.sherloq_app.ui.trufor_job import TruForJob
from gui.sherloq_app.core.trufor import TruForRenderer,export_npz
from gui.sherloq_app.core.jpeg_curve import Cancelled


def _render(request):
    device,renderer,choice=request
    return device,choice,renderer.render(choice)


def _export(request):
    result,destination,event=request
    try:
        return export_npz(result,destination,event.is_set)
    except Cancelled:
        return None


class TruForWidget(ToolWidget):
    ready=Signal()

    def __init__(self,filename,image,parent=None):
        super().__init__(parent)
        self.filename=filename;self.image=image;self.result=None
        self.renderers={};self.displayed=None;self.cancelled=False;self.output=''
        self.export_event=Event()
        self.cpu_button=QPushButton('CPU');self.cpu_button.setCheckable(True)
        self.cpu_button.setChecked(sys.platform!='darwin' or platform.machine()!='arm64')
        self.process_button=QPushButton('Analyser avec TruFor')
        self.cancel_button=QPushButton('Annuler')
        self.map_combo=QComboBox();self.map_combo.addItems(['Localisation des anomalies','Confiance du modèle','Noiseprint++'])
        self.export_button=QPushButton('Exporter les données (.npz)')
        self.output_label=QLabel('Analyse locale, à la résolution originale. Aucun envoi de votre image.');self.output_label.setWordWrap(True)
        self.map_legend=QLabel('')
        self.progress=QProgressBar();self.progress.setRange(0,100)
        self.viewer=ImageViewer(image,image,export=True)
        row=QHBoxLayout()
        for widget in [self.cpu_button,self.process_button,self.cancel_button]:row.addWidget(widget)
        row.addStretch();maps=QHBoxLayout();maps.addWidget(self.map_combo);maps.addWidget(self.export_button)
        layout=QVBoxLayout(self);layout.addLayout(row);layout.addWidget(self.output_label);layout.addWidget(self.progress);layout.addLayout(maps);layout.addWidget(self.map_legend);layout.addWidget(self.viewer)
        self.job=TruForJob(self,image)
        self.process=self.job.command.process
        self.render_job=LatestJob(self,_render,delay=0)
        self.export_job=LatestJob(self,_export,delay=0)
        self.job.result.connect(self.loaded);self.job.progress.connect(self.show_progress)
        self.render_job.result.connect(self.rendered);self.export_job.result.connect(self.exported)
        for job in [self.job,self.render_job,self.export_job]:
            job.busy.connect(self.set_busy);job.failed.connect(self.show_error)
        self.process_button.clicked.connect(self.trufor_process)
        self.cancel_button.clicked.connect(self.cancel)
        self.cpu_button.toggled.connect(self.change_device)
        self.map_combo.currentIndexChanged.connect(self.show_map)
        self.export_button.clicked.connect(self.export_data)
        self.model_available=True
        self.set_busy()
        if not (TRUFOR_DIR/'test_docker/weights/trufor-state.pt').is_file():
            self.output_label.setText('Le modèle TruFor sera téléchargé au lancement de l’analyse.')

    @property
    def device(self):return 'cpu' if self.cpu_button.isChecked() else 'mps'

    @property
    def is_busy(self):return self.job.is_busy or self.render_job.is_busy or self.export_job.is_busy

    def set_busy(self,*_):
        analyzing=self.job.is_busy
        self.process_button.setEnabled(self.model_available and not analyzing and not self.render_job.is_busy)
        self.cancel_button.setEnabled(self.is_busy)
        self.map_combo.setEnabled(not analyzing and self.result is not None)
        self.export_button.setEnabled(not analyzing and not self.export_job.is_busy and self.result is not None)
        self.viewer.set_busy(self.result is None or analyzing or self.displayed!=(self.device,self.map_combo.currentIndex()))
        self.progress.setVisible(self.is_busy)

    def trufor_process(self):
        if self.job.is_busy:return
        if self.result is not None and self.displayed==(self.device,self.map_combo.currentIndex()):return
        self.cancelled=False
        self.output_label.setText('Préparation de l’analyse…')
        self.job.request(self.device)

    def change_device(self,*_):
        self.job.invalidate();self.render_job.invalidate()
        self.result=None;self.displayed=None
        self.viewer.update_processed(self.image);self.map_legend.setText('')
        self.set_busy()
        self.output_label.setText(('CPU' if self.device=='cpu' else 'GPU')+' sélectionné. Cliquez sur Analyser.')
        if self.device in self.job.results:self.job.request(self.device)

    def loaded(self,value):
        device,result=value
        if device!=self.device:return
        self.result=result
        if device not in self.renderers:self.renderers[device]=TruForRenderer(result)
        self.show_map()

    def show_map(self,*_):
        if self.result is None:return
        choice=self.map_combo.currentIndex()
        if self.displayed==(self.device,choice):
            self.set_busy();return
        self.progress.setValue(95)
        self.render_job.request((self.device,self.renderers[self.device],choice))

    def rendered(self,value):
        device,choice,output=value
        if device!=self.device or choice!=self.map_combo.currentIndex() or self.result is None:return
        self.viewer.update_processed(output);self.displayed=(device,choice)
        legends=['Localisation : rouge = anomalies élevées ; bleu = faibles.',
                 'Confiance : blanc = élevée ; noir = faible.',
                 'Noiseprint++ : empreinte de bruit apprise ; contraste ajusté pour la visualisation.']
        self.map_legend.setText(legends[choice]);self.progress.setValue(100);self.set_busy()
        score=float(self.result['score']);seconds=float(self.result['seconds'])
        self.output_label.setText(f"Score TruFor : {score:.4f} / 1 — {seconds:.1f} s — {'CPU' if device=='cpu' else 'GPU'}. Résultats conservés.")
        self.ready.emit()

    def show_progress(self,value,text):
        self.progress.setValue(value);self.output_label.setText(text)

    def show_error(self,message):
        self.output=message
        self.output_label.setText(message.strip().splitlines()[-1] if message.strip() else 'L’analyse n’a pas abouti.')
        self.set_busy()

    def export_data(self):
        if self.result is None or self.export_job.is_busy:return
        filename,_=QFileDialog.getSaveFileName(self,t('Exporter TruFor'),'',t('TruFor data (*.npz)'))
        if not filename:return
        if not filename.lower().endswith('.npz'):filename+='.npz'
        self.export_event=Event()
        self.export_job.request((self.result,filename,self.export_event))
        self.output_label.setText('Export des données…')

    def exported(self,destination):
        self.set_busy()
        self.output_label.setText('Export terminé : '+destination if destination else 'Export annulé.')

    def cancel(self):
        self.cancelled=True;self.job.cancel();self.render_job.invalidate()
        self.export_event.set();self.export_job.invalidate()
        self.set_busy();self.output_label.setText('Opération annulée. Les résultats achevés sont conservés.')

    def shutdown(self):
        self.export_event.set();self.job.shutdown();super().shutdown()
