"""Retained 2D point view: orthographic OpenGL points and native Qt axes."""
import numpy as np
from .localization import t
from .gestures import native_position
import pyqtgraph.opengl as gl
from OpenGL import GL
from PySide6.QtCore import Qt, Signal, QRectF, QEvent
from PySide6.QtGui import QMatrix4x4, QPainter, QColor, QPen
from PySide6.QtWidgets import QWidget, QGridLayout


_POINT_OPTIONS = {
    GL.GL_DEPTH_TEST: False,
    GL.GL_BLEND: True,
    GL.GL_CULL_FACE: False,
    'glBlendFuncSeparate': (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA,
                            GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA),
}


class PlanarView(gl.GLViewWidget):
    rangeChanged = Signal()

    def __init__(self):
        super().__init__()
        self.limits = (0., 1., 0., 1.)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName('RGB/HSV 2D plot')
        self.setBackgroundColor('w')
        self.scatter = gl.GLScatterPlotItem(pos=np.empty((0, 3), np.float32),
                                           color=(.12, .47, .71, 1), size=2,
                                           pxMode=True, glOptions=_POINT_OPTIONS)
        self.addItem(self.scatter)
        self.grid = gl.GLLinePlotItem(pos=np.empty((0, 3), np.float32),
                                     color=(.4, .4, .4, .4), width=1,
                                     mode='lines', glOptions=_POINT_OPTIONS)
        self.grid.setDepthValue(1)
        self.addItem(self.grid)
        self.grid.setVisible(False)
        self._update_grid()

    def projectionMatrix(self, region, viewport):
        matrix = QMatrix4x4()
        x0, x1, y0, y1 = self.limits
        matrix.ortho(x0, x1, y0, y1, -1., 1.)
        return matrix

    def viewMatrix(self):
        return QMatrix4x4()

    def set_range(self, limits):
        x0, x1, y0, y1 = map(float, limits)
        if not all(np.isfinite((x0, x1, y0, y1))) or x1 <= x0 or y1 <= y0:
            return
        self.limits = x0, x1, y0, y1
        self._update_grid()
        self.rangeChanged.emit()
        self.update()

    def _update_grid(self):
        x0, x1, y0, y1 = self.limits
        points = []
        for value in np.linspace(x0, x1, 5):
            points.extend([(value, y0, 0), (value, y1, 0)])
        for value in np.linspace(y0, y1, 5):
            points.extend([(x0, value, 0), (x1, value, 0)])
        self.grid.setData(pos=np.asarray(points, np.float32))

    def reset_view(self):
        self.set_range((0, 1, 0, 1))

    def zoom(self, factor, anchor=None):
        x0, x1, y0, y1 = self.limits
        cx, cy = anchor if anchor is not None else ((x0+x1)/2, (y0+y1)/2)
        # Float32 points cannot support arbitrarily small visible intervals.
        if min((x1-x0)*factor, (y1-y0)*factor) < 1e-6:
            return
        self.set_range((cx+(x0-cx)*factor, cx+(x1-cx)*factor,
                        cy+(y0-cy)*factor, cy+(y1-cy)*factor))

    def mouseMoveEvent(self, event):
        pos = event.position()
        previous = getattr(self, 'mousePos', pos)
        self.mousePos = pos
        if event.buttons() & (Qt.LeftButton | Qt.MiddleButton):
            x0, x1, y0, y1 = self.limits
            dx = (pos.x()-previous.x())*(x1-x0)/max(self.width(), 1)
            dy = (pos.y()-previous.y())*(y1-y0)/max(self.height(), 1)
            self.set_range((x0-dx, x1-dx, y0+dy, y1+dy))
            event.accept()

    def wheelEvent(self, event):
        delta = event.angleDelta().y() or event.angleDelta().x()
        if not delta:
            delta = event.pixelDelta().y()
        x0, x1, y0, y1 = self.limits
        pos = event.position()
        anchor = (x0+pos.x()/max(self.width(), 1)*(x1-x0),
                  y1-pos.y()/max(self.height(), 1)*(y1-y0))
        self.zoom(0.999 ** delta, anchor)
        event.accept()

    def event(self,event):
        if event.type()==QEvent.NativeGesture and event.gestureType()==Qt.ZoomNativeGesture:
            factor=1.+event.value()
            if factor>0:
                x0,x1,y0,y1=self.limits;pos=native_position(self,event)
                anchor=(x0+pos.x()/max(1,self.width())*(x1-x0),y1-pos.y()/max(1,self.height())*(y1-y0))
                self.zoom(1./factor,anchor)
            event.accept();return True
        return super().event(event)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Home:
            self.reset_view()
        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom(.8)
        elif key == Qt.Key_Minus:
            self.zoom(1.25)
        else:
            super().keyPressEvent(event)


class PlotAxis(QWidget):
    def __init__(self, view, vertical=False):
        super().__init__()
        self.view, self.vertical = view, vertical
        self.title = ''
        if vertical:
            self.setFixedWidth(84)
        else:
            self.setFixedHeight(58)
        view.rangeChanged.connect(self.update)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('white'))
        painter.setPen(QPen(QColor('#222222'), 1))
        x0, x1, y0, y1 = self.view.limits
        if self.vertical:
            width, height = self.width(), self.height()
            painter.drawLine(width-1, 0, width-1, height)
            for fraction, value in zip(np.linspace(0, 1, 5), np.linspace(y1, y0, 5)):
                y = round(fraction*(height-1))
                painter.drawLine(width-5, y, width-1, y)
                painter.drawText(QRectF(20, min(max(y-9, 0), height-18), width-28, 18),
                                 Qt.AlignRight | Qt.AlignVCenter, f'{value:.7g}')
            painter.translate(14, height/2)
            painter.rotate(-90)
            painter.drawText(QRectF(-height/2, -12, height, 24), Qt.AlignCenter, t(self.title))
        else:
            width = self.width()
            painter.drawLine(0, 0, width, 0)
            for fraction, value in zip(np.linspace(0, 1, 5), np.linspace(x0, x1, 5)):
                x = round(fraction*(width-1));text = f'{value:.7g}'
                text_width = painter.fontMetrics().horizontalAdvance(text)+4
                painter.drawLine(x, 0, x, 5)
                left = min(max(x-text_width/2, 0), width-text_width)
                painter.drawText(QRectF(left, 7, text_width, 20), Qt.AlignCenter, text)
            painter.drawText(QRectF(0, 30, width, 24), Qt.AlignCenter, t(self.title))


class FastPlot2D(QWidget):
    def __init__(self):
        super().__init__()
        self.view = PlanarView()
        self.xaxis = PlotAxis(self.view)
        self.yaxis = PlotAxis(self.view, vertical=True)
        layout = QGridLayout(self)
        layout.setContentsMargins(4, 12, 12, 4)
        layout.setSpacing(0)
        layout.addWidget(self.yaxis, 0, 0)
        layout.addWidget(self.view, 0, 1)
        layout.addWidget(self.xaxis, 1, 1)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(0, 1)

    def set_axes(self, x, y):
        self.xaxis.title, self.yaxis.title = x, y
        self.xaxis.update();self.yaxis.update()
