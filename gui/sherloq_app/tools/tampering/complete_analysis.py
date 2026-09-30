"""Full automatic clone search and ELA with cached, selectable display layers."""
from functools import partial
from pathlib import Path
from threading import Event
import hashlib
import numpy as np
from PySide6.QtCore import QTimer,Qt,QSignalBlocker
from PySide6.QtWidgets import QComboBox,QDoubleSpinBox,QSpinBox,QLabel,QHBoxLayout,QFileDialog,QCheckBox,QTabBar
from .automatic_clones import AutomaticClonesWidget
from ...core.complete_analysis import SOURCES, ELA_SOURCE, SIFT_SOURCE, prepare, render, CompletePreparation
from ...core.cloning2 import Cloning2Engine
from .cloning2 import analyze as analyze_clones, Progress
from ...core.automatic_clones import visible
from ...core.ela_biomes import ElaBiomeEngine,DEFAULT_BLOCK,DEFAULT_THRESHOLD,DEFAULT_MINIMUM_CELLS
from ...ui.ela_biomes import analyze,energy_controls,histogram_controls,apply_automatic
from ...ui.jobs import LatestJob
from ...ui.ela_profiles import ElaProfiles,compute_profile
from ...ui.ela_settings import ElaSettings
from ...ui.localization import choice,t
from ...ui.extensions import ExtensionOutline,BLUE


class CompleteAnalysisWidget(AutomaticClonesWidget):
    sources=SOURCES

    def __init__(self,image,filename=None,parent=None,*,autostart=True):
        self.ela_cancel=Event();self.ela_base=None;self.ela_key=None
        super().__init__(image,parent,autostart=False)
        self.ela_engine=ElaBiomeEngine(image,filename)
        self.sift_engine=Cloning2Engine(image);self.sift_updates=Progress()
        self.sift_job=LatestJob(self,partial(analyze_clones,self.sift_engine,self.sift_updates),delay=0)
        self.sift_job.result.connect(lambda result:self.complete('sift',result))
        self.sift_job.failed.connect(lambda error:self.failed('sift',error))
        self.sift_updates.changed.connect(lambda event,value,message:self.progress('sift',value)
                                         if event is self.cancel and not event.is_set() else None)
        self.ela_job=LatestJob(self,partial(analyze,self.ela_engine),delay=120)
        self.ela_job.result.connect(self.ela_complete)
        self.ela_job.failed.connect(lambda error:self.failed('ela',error))
        # Reuse the proven automatic-clone controls, lifecycle and check state.
        self.preparation=CompletePreparation();self.prepare.compute=self.preparation;self.draw.compute=render
        self.ela_block=QComboBox();self.ela_block.addItems(['16','32','64','96']);self.ela_block.setCurrentText(str(DEFAULT_BLOCK))
        self.ela_threshold=QDoubleSpinBox();self.ela_threshold.setRange(1,20);self.ela_threshold.setSingleStep(.25);self.ela_threshold.setValue(DEFAULT_THRESHOLD)
        self.ela_minimum=QSpinBox();self.ela_minimum.setRange(1,1000);self.ela_minimum.setValue(DEFAULT_MINIMUM_CELLS)
        self.ela_ghost=QCheckBox('JPEG Ghosts');self.ela_ghost.setChecked(True)
        self.ela_grids=QCheckBox('All JPEG grids')
        self.ela_background=QCheckBox('Energy disparities');self.ela_background.setChecked(True)
        self.ela_background.setToolTip('Map low and high residual energy relative to the central histogram levels.')
        self.ela_mode=QComboBox()
        self.ela_mode.addItems(['Biomes on image','Biomes on ELA','ELA only','Low ELA energy','High ELA energy'])
        self.ela_mode.setToolTip('ELA display in Overlay and ELA tabs. Individual clone tabs keep the original image.')
        self.ela_energy_visible=QCheckBox('Energy biomes');self.ela_energy_visible.setChecked(True)
        self.ela_legacy_visible=QCheckBox('Cell biomes (legacy)');self.ela_legacy_visible.setChecked(True)
        self.ela_energy_visible.toggled.connect(self.tab_changed);self.ela_legacy_visible.toggled.connect(self.tab_changed)
        energy_row,(self.ela_shadow_threshold,self.ela_highlight_threshold)=energy_controls()
        histogram_row,(self.ela_histogram_low,self.ela_histogram_high)=histogram_controls()
        self.ela_auto_thresholds=QCheckBox('Auto thresholds');self.ela_auto_thresholds.setChecked(True)
        self.ela_auto_profile=QComboBox();self.ela_auto_profile.addItem('Conservative','standard')
        self.ela_settings=ElaSettings(self.ela_mode,self.ela_auto_profile,self.ela_auto_thresholds,histogram_row,energy_row,
            [('ELA cell (px)',self.ela_block),('ELA threshold',self.ela_threshold),('ELA cells / biome',self.ela_minimum)],
            (self.ela_ghost,self.ela_grids,self.ela_background),(self.ela_energy_visible,self.ela_legacy_visible))
        self.layout().insertWidget(3,self.ela_settings)
        self.ela_auto_profile.currentIndexChanged.connect(self.ela_profile_changed)
        self.ela_auto_thresholds.toggled.connect(self.ela_changed)
        for slider in (self.ela_histogram_low,self.ela_histogram_high,self.ela_shadow_threshold,self.ela_highlight_threshold):
            slider.sliderPressed.connect(self.take_manual_ela_control)
        self.ela_histogram_low.valueChanged.connect(self.manual_ela_bounds)
        self.ela_histogram_high.valueChanged.connect(self.manual_ela_bounds)
        self.ela_shadow_threshold.valueChanged.connect(self.manual_ela_deviation)
        self.ela_highlight_threshold.valueChanged.connect(self.manual_ela_deviation)
        self.source_checks={}
        for index,source in enumerate(self.sources,1):
            check=QCheckBox();check.setChecked(True)
            check.setAccessibleName(source)
            check.setToolTip('Show or hide this source without recalculating')
            self.source_checks[source]=check
            self.tabs.setTabButton(index,QTabBar.LeftSide,check)
            check.toggled.connect(self.sources_changed)
        self.overlay_check=QCheckBox();self.overlay_check.setTristate(True)
        self.overlay_check.setCheckState(Qt.Checked)
        self.overlay_check.setAccessibleName('All sources')
        self.overlay_check.setToolTip('Show or hide all sources')
        self.tabs.setTabButton(0,QTabBar.LeftSide,self.overlay_check)
        self.overlay_check.clicked.connect(self.toggle_all_sources)
        self.ela_mode.currentIndexChanged.connect(self.tab_changed)
        self.ela_ghost.toggled.connect(self.ela_changed);self.ela_grids.toggled.connect(self.ela_changed)
        self.ela_background.toggled.connect(self.ela_changed)
        self.ela_block.currentIndexChanged.connect(self.ela_changed)
        self.ela_threshold.valueChanged.connect(self.prepare_biomes)
        self.ela_minimum.valueChanged.connect(self.prepare_biomes)
        self.blue_outline=ExtensionOutline(self.tabs);self.blue_outline.color=BLUE
        self.blue_outline.update()
        self.profiles=ElaProfiles(self,self.ela_auto_profile,self.ela_auto_thresholds,(self.ela_histogram_low,self.ela_histogram_high,self.ela_shadow_threshold,self.ela_highlight_threshold),self.ela_changed,self.ela_settings.toolbar)
        self.profiles.editStarted.connect(self.take_manual_ela_control)
        self.profiles.apply()
        self.tabs.currentChanged.connect(self.settings_visibility)
        self.settings_visibility()
        if autostart:QTimer.singleShot(0,self.initial_detection)

    def settings_visibility(self,*_):
        self.ela_settings.setVisible(self.source()==ELA_SOURCE)
        self.clone_intervals.setVisible(self.source()!=ELA_SOURCE)

    def enabled_sources(self):
        return tuple(s for s in self.sources if not hasattr(self,'source_checks') or self.source_checks[s].isChecked())

    def sources_changed(self,*_):
        enabled=self.enabled_sources()
        with QSignalBlocker(self.overlay_check):
            self.overlay_check.setCheckState(Qt.Checked if len(enabled)==len(self.sources)
                                             else Qt.PartiallyChecked if enabled else Qt.Unchecked)
        self.tab_changed()

    def toggle_all_sources(self,enabled):
        for check in self.source_checks.values():
            with QSignalBlocker(check):check.setChecked(enabled)
        self.sources_changed()

    def effective_ela_mode(self):
        if (not hasattr(self,'ela_mode') or ELA_SOURCE not in self.enabled_sources()
                or self.source() not in (None,ELA_SOURCE)):return 0
        return self.ela_mode.currentIndex()

    def display_entries(self):
        if self.effective_ela_mode()==2:return ()
        enabled=self.enabled_sources()
        mode=self.effective_ela_mode()
        kind='low' if mode==3 else 'high' if mode==4 else None
        return tuple(e for e in super().display_entries() if e['source'] in enabled
                     and (e['source']!=ELA_SOURCE or (self.ela_energy_visible.isChecked() if 'kind' in e else self.ela_legacy_visible.isChecked()))
                     and (kind is None or e['source']!=ELA_SOURCE or e.get('kind')==kind))

    def current_ela_preview(self):
        if (self.ela_base is not None and self.ela_key==self.ela_params()
                and self.states.get('ela')=='Complete'):return self.ela_base['ela']
        return None

    def redraw(self):
        if self.closed:return
        # This path only draws cached results; source and view changes never
        # schedule detection, JPEG recompression or biome segmentation.
        shown=visible(self.display_entries(),None,self.model.hidden,self.focused)
        self.viewer.view.biomes=shown
        mode=self.effective_ela_mode()
        self.viewer.view.show_regions=self.show_zones.isChecked() and mode!=2
        self.viewer.view.viewport().update()
        excluded=tuple(r for i,r in enumerate(self.viewer.view.snapshot())
                       if i in self.zone_model.disabled and r!=self.envelope)
        preview=self.current_ela_preview() if hasattr(self,'ela_block') else None
        self.draw.request((self.image,shown,excluded,preview,mode))

    def toggle_zones(self,*_):
        self.redraw()

    def initial_states(self):
        return {**super().initial_states(),'sift':'Waiting','ela':'Waiting'}

    def state_labels(self):
        return {**super().state_labels(),'sift':t(SIFT_SOURCE),'ela':ELA_SOURCE}

    def cancel_work(self):
        super().cancel_work()
        self.ela_cancel.set()
        if hasattr(self,'ela_job'):self.ela_job.invalidate()
        if hasattr(self,'sift_job'):self.sift_job.invalidate()

    def ela_params(self):return int(choice(self.ela_block)),0,self.ela_ghost.isChecked(),self.ela_grids.isChecked(),self.ela_background.isChecked(),self.ela_histogram_low.value()/1000,self.ela_histogram_high.value()/1000,self.ela_auto_thresholds.isChecked(),compute_profile(self.ela_auto_profile)

    def start(self):
        super().start()
        if not self.closed and self.states.get('sift')=='Running':
            p=list(self.submitted['patchmatch_parameters']);p[0]=SIFT_SOURCE;p[3]=10.;p[4]=.725
            p[6:9]=['Affine',5.,10];p[11]=False;p[15]=False
            zones=tuple(r for i,r in enumerate(self.submitted['zones']) if i not in self.submitted['disabled_zones'])
            self.submitted['sift_parameters']=tuple(p)
            self.sift_job.request((tuple(p),zones,False,self.cancel))
        if not self.closed and self.states.get('ela')=='Running':self.queue_ela()

    def ela_profile_changed(self,*_):
        self.profiles.apply()
    def take_manual_ela_control(self):
        was_auto=self.ela_auto_thresholds.isChecked()
        self.profiles.manual()
        if was_auto:
            if self.ela_job.is_busy:self.ela_changed()
            elif self.ela_base is not None:self.ela_key=self.ela_params()
    def manual_ela_bounds(self,*_):
        self.profiles.manual()
        self.ela_changed()
    def manual_ela_deviation(self,*_):
        self.profiles.manual()
        if self.ela_base is not None and not self.ela_job.is_busy:
            self.ela_base={**self.ela_base,'metadata':{**self.ela_base['metadata'],'energy':{**self.ela_base['metadata'].get('energy',{}),'automatic':None}}}
            self.ela_key=self.ela_params();self.prepare_biomes()
        else:self.ela_changed()
    def ela_changed(self,*_):
        if self.closed or not self.viewer.view.snapshot():return
        self.queue_ela()

    def queue_ela(self):
        self.ela_cancel.set();self.ela_cancel=Event()
        self.ela_base=None;self.ela_key=None;self.results.pop('ela',None)
        self.states['ela']='Running';self.errors.pop('ela',None)
        self.ela_job.request((self.ela_params(),self.ela_cancel))
        self.prepare_biomes();self.update_status()

    def ela_complete(self,answer):
        params,base=answer
        if self.closed or base is None or params!=self.ela_params():return
        if self.ela_auto_thresholds.isChecked():apply_automatic(base,(self.ela_histogram_low,self.ela_histogram_high,self.ela_shadow_threshold,self.ela_highlight_threshold))
        self.ela_base=base;self.ela_key=self.ela_params();self.states['ela']='Complete'
        self.prepare_biomes();self.update_status()

    def prepare_biomes(self,*_):
        if self.closed or not hasattr(self,'ela_block'):return
        self.draw.invalidate();self.save.setEnabled(False)
        zones=self.viewer.view.snapshot()
        active=tuple(r for i,r in enumerate(zones) if i not in self.zone_model.disabled)
        excluded=tuple(r for i,r in enumerate(zones) if i in self.zone_model.disabled and r!=self.envelope)
        base=self.ela_base if self.ela_key==self.ela_params() and self.states.get('ela')=='Complete' else None
        self.prepare.request((self.results.get('patchmatch'),self.results.get('forgeryscope'),base,
            self.minimum.value(),self.maximum.value(),self.overlap.value()/100,
            self.ela_threshold.value(),self.ela_minimum.value(),active,excluded,self.image.shape,
            (self.ela_shadow_threshold.value()/10,self.ela_highlight_threshold.value()/10),self.results.get('sift')))

    def biomes_ready(self,answer):
        biomes,ela=answer
        if ela is not None:self.results['ela']=ela
        else:self.results.pop('ela',None)
        super().biomes_ready(biomes)

    def export_results(self):
        if not self.save.isEnabled():return
        path,_=QFileDialog.getSaveFileName(self,t('Export complete analysis'),'complete-analysis.npz',t('NumPy (*.npz)'))
        if path:
            snapshot=dict(version=1,method='complete_automatic_analysis',configuration=self.submitted,
                ela_profile=self.profiles.snapshot(),
                ela_settings=dict(block=self.ela_params()[0],threshold=self.ela_threshold.value(),minimum_cells=self.ela_minimum.value(),quality='auto',jpeg_ghosts=self.ela_ghost.isChecked(),all_jpeg_grids=self.ela_grids.isChecked(),residual_levels=self.ela_background.isChecked(),energy_disparities=self.ela_background.isChecked(),auto_thresholds=self.ela_auto_thresholds.isChecked(),auto_profile=self.ela_auto_profile.currentData(),histogram_quantiles=list(self.ela_params()[5:7]),shadow_threshold=self.ela_shadow_threshold.value()/10,highlight_threshold=self.ela_highlight_threshold.value()/10),
                results=dict(self.results),states=dict(self.states),errors=dict(self.errors),biomes=self.biomes,
                display=dict(background='ela' if self.effective_ela_mode() and self.current_ela_preview() is not None else 'original',
                    energy_biomes=self.ela_energy_visible.isChecked(),legacy_biomes=self.ela_legacy_visible.isChecked(),
                    ela_view=('image','ela_biomes','ela','energy_low','energy_high')[self.ela_mode.currentIndex()],
                    enabled_sources=self.enabled_sources(),source=self.source(),forgeryscope_branch=self.forge_branch.currentData(),hidden=sorted(self.model.hidden),focused=self.focused,
                    minimum_length_px=self.minimum.value(),maximum_length_px=self.maximum.value(),maximum_overlap=self.overlap.value()/100),
                image_shape=self.image.shape,decoded_bgr8_sha256=hashlib.sha256(memoryview(np.ascontiguousarray(self.image))).hexdigest())
            self.export_job.request((str(Path(path).with_suffix('.npz')),snapshot))
