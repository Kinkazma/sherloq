"""Interactive resampling analysis with retained maps and cancellable CPU work."""
from functools import partial
from threading import Event
import numpy as np
import cv2 as cv
from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtWidgets import (QWidget,QDoubleSpinBox,QVBoxLayout,QHBoxLayout,QSlider,
    QLabel,QCheckBox,QPushButton,QButtonGroup,QGridLayout,QSizePolicy,QScrollArea,QProgressBar)
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from gui.sherloq_app.ui.plot_canvas import FigureCanvas
from matplotlib.backend_bases import MouseButton
from matplotlib.figure import Figure
from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.ui.jobs import LatestJob
from gui.sherloq_app.core.resampling import ResamplingEngine,normalize_gray,check
from gui.sherloq_app.core.jpeg_curve import Cancelled


class _Progress(QObject):
    changed=Signal(object,int,str)


def rectangles(points,shape,minimum,default=False):
    h,w=shape
    if len(points)%2:
        raise ValueError('Select the second corner of the region, or remove the last point.')
    if not points:
        return ((0,0,w,h),) if default else ()
    result=[]
    for first,second in zip(points[::2],points[1::2]):
        x0,x1=sorted((first[0],second[0]));y0,y1=sorted((first[1],second[1]))
        if not (0<=x0<=x1<w and 0<=y0<=y1<h):raise ValueError('Selected region is outside the image.')
        if x1-x0+1<minimum or y1-y0+1<minimum:raise ValueError(f'Select a region at least {minimum} × {minimum} pixels.')
        result.append((x0,y0,x1+1,y1+1))
    return tuple(result)


def _run(engine,updates,request):
    snapshot,event=request
    stage,size,regions,fourier_regions,params=snapshot
    try:
        if stage=='probability':
            for index,a in enumerate(regions):
                for b in regions[index+1:]:
                    if max(a[0],b[0])<min(a[2],b[2]) and max(a[1],b[1])<min(a[3],b[3]):
                        raise ValueError('Probability regions must not overlap.')
        maps=[];composite_key=(size,regions);composite=engine.composites.get(composite_key);build_composite=composite is None
        if build_composite:composite=engine.gray.copy()
        border=size//2
        for index,rect in enumerate(regions):
            check(event.is_set)
            update=lambda done,total:updates.changed.emit(event,int(100*(index+done/total)/max(1,len(regions))),f'Region {index+1}/{len(regions)} — iteration {done}/{total}')
            result=engine.probability(rect,size,event.is_set,update)
            maps.append(result);x0,y0,x1,y1=rect
            if build_composite:composite[y0+border:y1-border,x0+border:x1-border]=result
        if build_composite:engine.composites.put(composite_key,composite)
        output=[]
        if stage=='fourier':
            sources=[(array,('probability',rect,size),rect) for array,rect in zip(maps,regions)]
            sources += [(composite[y0:y1,x0:x1],('region',regions,size,rect),rect) for rect in fourier_regions for x0,y0,x1,y1 in [rect]]
            if not sources:raise ValueError('Calculate a probability map or select a Fourier region first.')
            for index,(array,key,rect) in enumerate(sources):
                check(event.is_set)
                result=engine.fourier(array,key,params,event.is_set)
                output.append((array,result,rect));updates.changed.emit(event,100*(index+1)//len(sources),f'Fourier map {index+1}/{len(sources)}')
        return snapshot,maps,composite,output
    except Cancelled:
        return None


class ResamplingWidget(ToolWidget):
    def __init__(self,filename,image,parent=None):
        super().__init__(parent)
        self.filename=filename;self.image_original=image
        gray=cv.imread(filename,cv.IMREAD_GRAYSCALE)
        if gray is None:gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        self.imagegray_nomalized_copy=normalize_gray(gray)
        self.imagegray=self.imagegray_nomalized_copy
        self.imagegray_copy_for_probabilitymaps=self.imagegray
        self.userfourplot=self.imagegray
        self.engine=ResamplingEngine(self.imagegray)
        self.selected_points_probability=[];self.selected_points_fourier=[]
        self.probability_maps=[];self.fourier_maps=[];self.original_sizes_widgets={}
        self._probability_key=self._fourier_key=None;self._closed=False
        self.cancel_event=Event();self.updates=_Progress()
        self.updates.changed.connect(self._progress)
        self.job=LatestJob(self,partial(_run,self.engine,self.updates),delay=0)
        self.job.result.connect(self.show_result);self.job.failed.connect(self.show_error);self.job.busy.connect(self.set_busy)
        self.calculate_probability_button=QPushButton('Calculate probability')
        self.calculate_fourier_button=QPushButton('Calculate Fourier')
        self.cancel_button=QPushButton('Cancel');self.cancel_button.clicked.connect(self.cancel)
        self.status_label=QLabel('Select two corners with left clicks; right click removes the last corner. Without a probability region, the entire image is analyzed.')
        self.status_label.setWordWrap(True);self.progress=QProgressBar()
        self.filter_3x3_Check=QCheckBox('3×3 probability filter');self.filter_5x5_Check=QCheckBox('5×5 probability filter')
        self.probability_check=QCheckBox('Probability windows');self.fourier_check=QCheckBox('Fourier windows')
        self.hanning_check=QCheckBox('Hanning');self.rotationally_invariant_window_check=QCheckBox('R. I. W.')
        self.simple_highpass_check=QCheckBox('Highpass 1');self.complex_highpass_check=QCheckBox('Highpass 2')
        self._groups=[]
        for first,second in [(self.filter_3x3_Check,self.filter_5x5_Check),(self.probability_check,self.fourier_check),(self.hanning_check,self.rotationally_invariant_window_check),(self.simple_highpass_check,self.complex_highpass_check)]:
            group=QButtonGroup(self);group.setExclusive(True);group.addButton(first);group.addButton(second);first.setChecked(True);self._groups.append(group)
        self.upsample_check=QCheckBox('Upsample');self.upsample_check.setChecked(True)
        self.center_four_check=QCheckBox('Take center of Fourier')
        self.rescale_check=QCheckBox('Rescale spectrum');self.rescale_check.setChecked(True)
        self.gamma_spin=QDoubleSpinBox();self.gamma_spin.setRange(0,5);self.gamma_spin.setSingleStep(.1);self.gamma_spin.setValue(4)
        top=QGridLayout()
        for index,widget in enumerate([self.probability_check,self.fourier_check,self.filter_3x3_Check,self.filter_5x5_Check,self.calculate_probability_button,self.calculate_fourier_button,self.cancel_button]):top.addWidget(widget,index//4,index%4)
        options=QHBoxLayout()
        for widget in [self.hanning_check,self.rotationally_invariant_window_check,self.upsample_check,self.center_four_check,self.simple_highpass_check,self.complex_highpass_check,QLabel('Gamma'),self.gamma_spin,self.rescale_check]:options.addWidget(widget)
        self.canvas=FigureCanvas(Figure());self.axes=self.canvas.figure.subplots()
        self.probability_image_canvas_object=self.axes.imshow(self.imagegray,cmap='gray',vmin=0,vmax=1)
        self.canvas.mpl_connect('button_press_event',self.click_on_canvas)
        self.toolbar_prob=NavigationToolbar(self.canvas,self)
        self.figure_four=Figure(figsize=(10,15));self.canvas_fourier_maps=FigureCanvas(self.figure_four)
        self.toolbar_four=NavigationToolbar(self.canvas_fourier_maps,self)
        self.zoom_slider=QSlider(Qt.Horizontal);self.zoom_slider.setRange(100,200);self.zoom_slider.setValue(100);self.zoom_slider.valueChanged.connect(self.slider_zoom)
        self.scroll_area=QScrollArea(self);self.scroll_widget=QWidget();scroll=QVBoxLayout(self.scroll_widget)
        for widget in [self.toolbar_prob,self.canvas,self.toolbar_four,self.canvas_fourier_maps]:scroll.addWidget(widget)
        for canvas in [self.canvas,self.canvas_fourier_maps]:
            canvas.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed);self.original_sizes_widgets[canvas]=canvas.sizeHint()
        self.scroll_area.setWidget(self.scroll_widget);self.scroll_area.setWidgetResizable(True)
        layout=QVBoxLayout(self);layout.addLayout(top);layout.addWidget(self.status_label);layout.addWidget(self.progress);layout.addLayout(options);layout.addWidget(self.zoom_slider);layout.addWidget(self.scroll_area)
        self.calculate_probability_button.clicked.connect(self.calculate_probability_maps)
        self.calculate_fourier_button.clicked.connect(self.calculate_fourier_maps)
        for control in [self.filter_3x3_Check,self.filter_5x5_Check]:control.toggled.connect(self.invalidate_probability)
        for control in [self.hanning_check,self.rotationally_invariant_window_check,self.upsample_check,self.center_four_check,self.simple_highpass_check,self.complex_highpass_check,self.rescale_check]:control.toggled.connect(self.invalidate_fourier)
        self.gamma_spin.valueChanged.connect(self.invalidate_fourier)
        self.set_busy(False)

    def slider_zoom(self):
        factor=self.zoom_slider.value()/100
        for canvas,size in self.original_sizes_widgets.items():canvas.setFixedSize(round(size.width()*factor),round(size.height()*factor))

    def snapshot(self,stage):
        size=5 if self.filter_5x5_Check.isChecked() else 3
        regions=rectangles(self.selected_points_probability,self.engine.gray.shape,size,default=True)
        if stage=='fourier' and self._probability_key!=(size,regions):regions=()
        extra=rectangles(self.selected_points_fourier,self.engine.gray.shape,2) if stage=='fourier' else ()
        params=('hanning' if self.hanning_check.isChecked() else 'radial',self.upsample_check.isChecked(),self.center_four_check.isChecked(),'simple' if self.simple_highpass_check.isChecked() else 'radial',self.gamma_spin.value(),self.rescale_check.isChecked())
        return stage,size,regions,extra,params

    def calculate_probability_maps(self):self.start('probability')
    def calculate_fourier_maps(self):self.start('fourier')

    def start(self,stage):
        try:snapshot=self.snapshot(stage)
        except ValueError as exc:self.show_error(str(exc));return
        _,size,regions,extra,params=snapshot
        if stage=='probability' and self._probability_key==(size,regions):return
        if stage=='fourier' and self._fourier_key==snapshot:return
        self.cancel_event.set();self.cancel_event=Event();self.job.request((snapshot,self.cancel_event))

    def cancel(self):
        self.cancel_event.set();self.job.invalidate();self.status_label.setText('Cancelled. Completed maps and iterations retained.');self.set_busy(False)

    def invalidate_probability(self,*_):
        self.cancel();self._probability_key=None;self._fourier_key=None;self.probability_maps=[];self.fourier_maps=[]
        self.imagegray_copy_for_probabilitymaps=self.engine.gray
        self.draw_selection();self.set_busy(False);self.status_label.setText('Region or filter changed. Calculate probability to update.')

    def invalidate_fourier(self,*_):
        self.cancel();self._fourier_key=None;self.set_busy(False);self.status_label.setText('Fourier options changed. Calculate Fourier to update.')

    def set_busy(self,busy):
        self.cancel_button.setEnabled(busy)
        self.calculate_probability_button.setEnabled(not busy);self.calculate_fourier_button.setEnabled(not busy)
        self.toolbar_prob.setEnabled(not busy);self.toolbar_four.setEnabled(not busy)
        self.toolbar_prob._actions['save_figure'].setEnabled(not busy and self._probability_key is not None)
        self.toolbar_four._actions['save_figure'].setEnabled(not busy and self._fourier_key is not None)
        if busy:self.status_label.setText('Analyzing…')

    def _progress(self,event,value,text):
        if not self._closed and event is self.cancel_event and not event.is_set():self.progress.setValue(value);self.status_label.setText(text)

    def show_result(self,result):
        if result is None:return
        snapshot,maps,composite,output=result
        stage,size,regions,extra,params=snapshot
        self.probability_maps=maps;self.imagegray_copy_for_probabilitymaps=composite
        if regions:self._probability_key=(size,regions)
        self.probability_image_canvas_object.set_data(composite);self.canvas.draw_idle()
        if stage=='fourier':
            self.fourier_maps=[row[1] for row in output];self._fourier_key=snapshot
            existing=getattr(self,'axes_fourier_maps',None)
            if existing is None or len(existing)!=len(output):
                self.figure_four.clear();self.axes_fourier_maps=self.figure_four.subplots(len(output),2,squeeze=False)
            for axes,(source,spectrum,rect) in zip(self.axes_fourier_maps,output):
                for ax,array in zip(axes,(source,spectrum)):
                    if ax.images and ax.images[0].get_array().shape==array.shape:ax.images[0].set_data(array)
                    else:
                        ax.clear();ax.imshow(array,cmap='gray',vmin=0,vmax=1)
                    ax.axis('off')
            self.figure_four.subplots_adjust(left=.05,right=.95,top=.95,bottom=.05,wspace=.05,hspace=.05);self.canvas_fourier_maps.draw_idle()
        self.progress.setValue(100);self.set_busy(False);self.status_label.setText(f'Completed in {self.job.seconds:.3f} s. Results retained.')

    def show_error(self,message):self.status_label.setText(message);self.set_busy(False)

    def click_on_canvas(self,event):
        if self.toolbar_prob.mode or event.inaxes is not self.axes:return
        points=self.selected_points_probability if self.probability_check.isChecked() else self.selected_points_fourier
        if event.button==MouseButton.LEFT and event.xdata is not None and event.ydata is not None:
            x,y=int(event.xdata),int(event.ydata)
            if not(0<=x<self.engine.gray.shape[1] and 0<=y<self.engine.gray.shape[0]):return
            points.append((x,y))
        elif event.button==MouseButton.RIGHT and points:points.pop()
        else:return
        if self.probability_check.isChecked():self.invalidate_probability()
        else:self.invalidate_fourier()
        self.draw_selection()

    def draw_selection(self):
        canvas=self.imagegray_copy_for_probabilitymaps.copy()
        points=self.selected_points_probability if self.probability_check.isChecked() else self.selected_points_fourier
        for index,point in enumerate(points):
            cv.circle(canvas,point,3,1,-1)
            if index%2:cv.rectangle(canvas,points[index-1],point,0,2)
        self.probability_image_canvas_object.set_data(canvas);self.canvas.draw_idle()

    def shutdown(self):
        self._closed=True;self.cancel_event.set();super().shutdown()
