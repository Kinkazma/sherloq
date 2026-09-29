from gui.sherloq_app.ui.localization import t
import pyqtgraph.opengl as gl
from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QVector3D, QFont
from PySide6.QtWidgets import QPushButton, QFileDialog
from gui.sherloq_app.core.interactive import PlotEngine
from gui.sherloq_app.ui.jobs import LatestJob
from gui.sherloq_app.ui.gestures import native_position
from gui.sherloq_app.ui.plot2d import FastPlot2D

import numpy as np
from PySide6.QtWidgets import (
    QHBoxLayout,
    QSpinBox,
    QCheckBox,
    QWidget,
    QComboBox,
    QLabel,
    QVBoxLayout,
    QDoubleSpinBox,
    QGridLayout,
    QTabWidget,
    QStackedWidget,
)
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from gui.sherloq_app.ui.plot_canvas import FigureCanvas

from gui.sherloq_app.ui.tools import ToolWidget


class GPUPlotView(gl.GLViewWidget):
    """Retained GPU points with native Qt labels above the OpenGL surface."""
    axis_names = ('Hue', 'Saturation', 'Value')

    def __init__(self):
        super().__init__()
        self.labels = []
        for _ in range(15):
            label = QLabel(self)
            label.setAttribute(Qt.WA_TransparentForMouseEvents)
            label.setStyleSheet('color: rgb(35, 35, 35); background: transparent;')
            label.setFont(QFont('Helvetica Neue', 11))
            self.labels.append(label)

    def zoom_at(self,factor,pos):
        if factor<=0:return
        def plane_point():
            matrix=self.projectionMatrix(self.getViewport(),self.getViewport())*self.viewMatrix()
            depth=matrix.map(self.opts['center']).z()
            inverse,valid=matrix.inverted()
            if not valid:return None
            return inverse.map(QVector3D(2*pos.x()/max(1,self.width())-1,1-2*pos.y()/max(1,self.height()),depth))
        before=plane_point()
        self.opts['distance']=min(1e6,max(1e-4,self.opts['distance']/factor))
        after=plane_point()
        if before is not None and after is not None:self.opts['center']+=before-after
        self.update()

    def wheelEvent(self,event):
        if event.modifiers() & Qt.ControlModifier:return super().wheelEvent(event)
        delta=event.angleDelta().y() or event.angleDelta().x() or event.pixelDelta().y()
        self.zoom_at(.999**(-delta),event.position());event.accept()

    def event(self,event):
        if event.type()==QEvent.NativeGesture and event.gestureType()==Qt.ZoomNativeGesture:
            self.zoom_at(1.+event.value(),native_position(self,event));event.accept();return True
        return super().event(event)

    def paintGL(self):
        super().paintGL()
        matrix = self.projectionMatrix(self.getViewport(), self.getViewport()) * self.viewMatrix()
        index = 0
        for axis, name in enumerate(self.axis_names):
            for distance, text in ((0.25, '0.25'), (0.5, '0.5'), (0.75, '0.75'), (1, '1'), (1.16, name)):
                xyz = [0., 0., 0.]
                xyz[axis] = distance
                ndc = matrix.map(QVector3D(*xyz))
                label = self.labels[index]
                index += 1
                label.setText(t(text))
                label.adjustSize()
                label.move(round((ndc.x()+1)*self.width()/2+4),
                           round((1-ndc.y())*self.height()/2-label.height()-4))



class PlotsWidget(ToolWidget):
    def __init__(self, image, parent=None):
        super(PlotsWidget, self).__init__(parent)

        choices = ["Red", "Green", "Blue", "Hue", "Saturation", "Value"]
        self.xaxis_combo = QComboBox()
        self.xaxis_combo.addItems(choices)
        self.xaxis_combo.setCurrentIndex(3)
        self.yaxis_combo = QComboBox()
        self.yaxis_combo.addItems(choices)
        self.yaxis_combo.setCurrentIndex(4)
        self.zaxis_combo = QComboBox()
        self.zaxis_combo.addItems(choices)
        self.zaxis_combo.setCurrentIndex(5)
        self.sampling_spin = QSpinBox()
        levels = int(np.log2(min(image.shape[:-1])))
        self.sampling_spin.setRange(0, levels)
        self.sampling_spin.setSpecialValueText(self.tr("Off"))
        # self.sampling_spin.setSuffix(self.tr(' level(s)'))
        self.sampling_spin.setValue(1)
        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 10)
        self.size_spin.setValue(1)
        self.size_spin.setSuffix(self.tr(" pt"))
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0, 1)
        self.alpha_spin.setDecimals(2)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setValue(1)
        self.colors_check = QCheckBox(self.tr("Show colors"))
        self.grid_check = QCheckBox(self.tr("Show grid"))
        self.norm_check = QCheckBox(self.tr("Normalized"))
        self.total_label = QLabel()

        self.engine = PlotEngine(image)
        self.data = None
        self.data_scale = None
        self.artist2 = None
        self._requested_key = None
        self._prepared = None
        self._position_keys = {}
        self._color_keys = {}
        self._classic_style = None
        self.job = LatestJob(self, self.engine.prepare_plot, delay=40)
        self.job.result.connect(self.receive_colors)
        self.job.failed.connect(self.show_error)

        figure2 = Figure()
        plot2_canvas = FigureCanvas(figure2)
        self.axes2 = plot2_canvas.figure.subplots()
        toolbar2 = NavigationToolbar(plot2_canvas, self)
        self.toolbar2 = toolbar2
        classic_widget = QWidget()
        classic_layout = QVBoxLayout(classic_widget)
        classic_layout.setContentsMargins(0, 0, 0, 0)
        classic_layout.addWidget(plot2_canvas)
        classic_layout.addWidget(toolbar2)
        self.fast2 = FastPlot2D()
        self.renderer_combo = QComboBox()
        self.renderer_combo.addItems([self.tr("Fast view"), self.tr("Classic (Matplotlib)")])
        self.renderer_combo.setToolTip(self.tr("Both views use all selected points. Classic preserves the original renderer and vector exports; large colored plots can be slow."))
        self.plot2_stack = QStackedWidget()
        self.plot2_stack.addWidget(self.fast2)
        self.plot2_stack.addWidget(classic_widget)
        self.export2 = QPushButton(self.tr("Save plot…"))
        self.export2.clicked.connect(self.export_2d)
        reset2 = QPushButton(self.tr("Reset view"));reset2.clicked.connect(self.fast2.view.reset_view)
        zoom_in = QPushButton('+');zoom_in.setToolTip(self.tr("Zoom in"));zoom_in.clicked.connect(lambda: self.fast2.view.zoom(.8))
        zoom_out = QPushButton('−');zoom_out.setToolTip(self.tr("Zoom out"));zoom_out.clicked.connect(lambda: self.fast2.view.zoom(1.25))
        self.fast_toolbar = QWidget()
        buttons = QHBoxLayout(self.fast_toolbar)
        buttons.setContentsMargins(0, 0, 0, 0)
        for widget in (reset2, zoom_in, zoom_out, self.export2):
            buttons.addWidget(widget)
        buttons.addWidget(QLabel(self.tr("Drag to pan · wheel to zoom · Home to reset")))
        buttons.addStretch()
        plot2_layout = QVBoxLayout()
        plot2_layout.addWidget(self.plot2_stack, 1)
        plot2_layout.addWidget(self.fast_toolbar)
        renderer_layout = QHBoxLayout()
        renderer_layout.addWidget(QLabel(self.tr("Rendering:")))
        renderer_layout.addWidget(self.renderer_combo)
        renderer_layout.addStretch()
        plot2_layout.addLayout(renderer_layout)
        plot2_widget = QWidget()
        plot2_widget.setLayout(plot2_layout)

        self.gl_view = GPUPlotView()
        self.gl_view.setBackgroundColor('w')
        self.reset_camera()
        self.scatter3 = gl.GLScatterPlotItem(pos=np.empty((0, 3), np.float32),
                                            color=(0.12, 0.47, 0.71, 1), size=2,
                                            pxMode=True, glOptions='opaque')
        self.gl_view.addItem(self.scatter3)
        self.grid3 = gl.GLGridItem()
        self.grid3.setSize(1, 1, 1)
        self.grid3.setSpacing(0.25, 0.25, 0.25)
        self.grid3.translate(0.5, 0.5, 0)
        self.grid3.setColor((120, 120, 120, 110))
        self.gl_view.addItem(self.grid3)
        for end in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            line = gl.GLLinePlotItem(pos=np.array([(0, 0, 0), end], np.float32),
                                     color=(0.2, 0.2, 0.2, 1), width=1, antialias=True)
            self.gl_view.addItem(line)
        reset = QPushButton(self.tr("Reset view"))
        reset.clicked.connect(self.reset_camera)
        export = QPushButton(self.tr("Save plot…"))
        export.clicked.connect(self.export_plot)
        self.export3 = export
        toolbar3 = QHBoxLayout()
        toolbar3.addWidget(reset)
        toolbar3.addWidget(export)
        toolbar3.addWidget(QLabel(self.tr("Drag to rotate · wheel to zoom · Ctrl-drag to pan")))
        toolbar3.addStretch()
        plot3_layout = QVBoxLayout()
        plot3_layout.addWidget(self.gl_view, 1)
        plot3_layout.addLayout(toolbar3)
        plot3_widget = QWidget()
        plot3_widget.setLayout(plot3_layout)

        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(plot2_widget, "2D Plot")
        self.tab_widget.addTab(plot3_widget, "3D Plot")
        figure2.set_tight_layout(True)

        self.xaxis_combo.currentIndexChanged.connect(self.redraw)
        self.yaxis_combo.currentIndexChanged.connect(self.redraw)
        self.zaxis_combo.currentIndexChanged.connect(self.redraw)
        self.sampling_spin.valueChanged.connect(self.redraw)
        self.size_spin.valueChanged.connect(self.redraw)
        self.alpha_spin.valueChanged.connect(self.redraw)
        self.colors_check.stateChanged.connect(self.redraw)
        self.grid_check.stateChanged.connect(self.redraw)
        self.norm_check.stateChanged.connect(self.redraw)
        self.tab_widget.currentChanged.connect(self.redraw)
        self.renderer_combo.currentIndexChanged.connect(self.change_renderer)

        params_layout = QGridLayout()
        params_layout.addWidget(QLabel(self.tr("X axis:")), 0, 0)
        params_layout.addWidget(self.xaxis_combo, 0, 1)
        params_layout.addWidget(QLabel(self.tr("Y axis:")), 1, 0)
        params_layout.addWidget(self.yaxis_combo, 1, 1)
        params_layout.addWidget(QLabel(self.tr("Z axis:")), 2, 0)
        params_layout.addWidget(self.zaxis_combo, 2, 1)
        params_layout.addWidget(QLabel(self.tr("Subsampling:")), 0, 2)
        params_layout.addWidget(self.sampling_spin, 0, 3)
        params_layout.addWidget(QLabel(self.tr("Point size:")), 1, 2)
        params_layout.addWidget(self.size_spin, 1, 3)
        params_layout.addWidget(QLabel(self.tr("Point alpha:")), 2, 2)
        params_layout.addWidget(self.alpha_spin, 2, 3)
        params_layout.addWidget(self.colors_check, 0, 4)
        params_layout.addWidget(self.grid_check, 1, 4)
        params_layout.addWidget(self.total_label, 2, 4)
        bottom_layout = QHBoxLayout()
        bottom_layout.addLayout(params_layout)
        bottom_layout.addStretch()

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tab_widget)
        main_layout.addLayout(bottom_layout)
        self.setLayout(main_layout)
        self.job.busy.connect(self.set_busy)
        self.redraw()

    def reset_camera(self):
        self.gl_view.opts['center'] = QVector3D(0.5, 0.5, 0.5)
        self.gl_view.setCameraPosition(distance=3.5, elevation=25, azimuth=45)

    def export_plot(self):
        if self.job.is_busy:
            return
        filename, _ = QFileDialog.getSaveFileName(self, t(self.tr("Save 3D plot")), "plot.png", t("PNG (*.png)"))
        if filename:
            if not filename.lower().endswith('.png'):
                filename += '.png'
            if not self.gl_view.grab().save(filename):
                self.info_message.emit(self.tr("Could not save plot"))

    def export_2d(self):
        if self.job.is_busy:
            return
        filename, _ = QFileDialog.getSaveFileName(self, t(self.tr("Save 2D plot")), "plot.png", t("PNG (*.png)"))
        if filename:
            if not filename.lower().endswith('.png'):
                filename += '.png'
            if not self.fast2.grab().save(filename):
                self.info_message.emit(self.tr("Could not save plot"))

    def set_busy(self, busy):
        self.export2.setEnabled(not busy)
        self.export3.setEnabled(not busy)
        self.toolbar2._actions['save_figure'].setEnabled(not busy)
        if busy:
            self.total_label.setText(self.tr("Preparing points…"))

    def change_renderer(self):
        classic = self.renderer_combo.currentIndex() == 1
        if classic:
            x0, x1, y0, y1 = self.fast2.view.limits
            self.axes2.set_xlim(x0, x1);self.axes2.set_ylim(y0, y1)
        else:
            self.fast2.view.set_range((*self.axes2.get_xlim(), *self.axes2.get_ylim()))
        self.plot2_stack.setCurrentIndex(int(classic))
        self.fast_toolbar.setVisible(not classic)
        self.redraw()

    def receive_colors(self, prepared):
        self._prepared = prepared
        self.data = prepared['data']
        self.data_scale = prepared['params'][0]
        self._show_prepared()

    def show_error(self, error):
        self._requested_key = None
        self.total_label.setText(self.tr("Error: ") + error)

    def redraw(self):
        scale = self.sampling_spin.value()
        three_d = self.tab_widget.currentIndex() == 1
        self.zaxis_combo.setEnabled(three_d)
        kind = '3d' if three_d else 'classic' if self.renderer_combo.currentIndex() else '2d'
        x, y = self.xaxis_combo.currentIndex(), self.yaxis_combo.currentIndex()
        z = self.zaxis_combo.currentIndex() if three_d else -1
        key = (scale, x, y, z, self.colors_check.isChecked(), self.alpha_spin.value(), kind)
        if key != self._requested_key:
            self._requested_key = key
            self.job.request(key)
        elif not self.job.is_busy and self._prepared is not None:
            self._show_prepared()

    def _show_prepared(self):
        prepared = self._prepared
        scale, x, y, z, colored, alpha, kind = prepared['params']
        size = self.size_spin.value()
        if kind in ('2d', '3d'):
            scatter = self.scatter3 if kind == '3d' else self.fast2.view.scatter
            changes = {'size': size*2.0}
            if prepared['position_key'] != self._position_keys.get(kind):
                changes['pos'] = prepared['positions']
                self._position_keys[kind] = prepared['position_key']
            if prepared['color_key'] != self._color_keys.get(kind):
                changes['color'] = prepared['colors']
                self._color_keys[kind] = prepared['color_key']
            if kind == '3d':
                self.scatter3.setGLOptions('opaque' if alpha == 1 else 'translucent')
                self.gl_view.setBackgroundColor((190, 190, 190) if colored else 'w')
                self.grid3.setVisible(self.grid_check.isChecked())
                self.gl_view.axis_names = tuple(combo.currentText() for combo in
                                                (self.xaxis_combo, self.yaxis_combo, self.zaxis_combo))
            else:
                self.fast2.view.setBackgroundColor((128, 128, 128) if colored else 'w')
                self.fast2.view.grid.setVisible(self.grid_check.isChecked())
                self.fast2.set_axes(self.xaxis_combo.currentText(), self.yaxis_combo.currentText())
            scatter.setData(**changes)
        else:
            if self.artist2 is None:
                self.artist2 = self.axes2.scatter([], [], marker='.')
            if prepared['position_key'] != self._position_keys.get(kind):
                self.artist2.set_offsets(prepared['positions'])
                self._position_keys[kind] = prepared['position_key']
            style = prepared['color_key'], size
            if style != self._classic_style:
                self.artist2.set_facecolors(self.data[:, :3] if colored else 'C0')
                self.artist2.set_edgecolors('face')
                self.artist2.set_sizes([size**2])
                self.artist2.set_alpha(alpha)
                self._classic_style = style
            self.axes2.set_facecolor([.5]*3 if colored else [1.]*3)
            self.axes2.set_xlabel(self.xaxis_combo.currentText())
            self.axes2.set_ylabel(self.yaxis_combo.currentText())
            self.axes2.grid(self.grid_check.isChecked(), which='both')
            self.axes2.figure.canvas.draw_idle()
        self.total_label.setText(f"[{len(self.data):,} points]")
