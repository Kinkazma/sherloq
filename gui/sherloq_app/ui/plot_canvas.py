"""Qt canvas whose deferred draw may safely outlive a closed tool window."""
from matplotlib.backends.backend_qt5agg import FigureCanvas as _FigureCanvas
from shiboken6 import isValid
from PySide6.QtCore import QEvent,Qt
import numpy as np
from .gestures import native_position


class FigureCanvas(_FigureCanvas):
    def zoom_at(self,factor,position):
        if factor<=0 or not np.isfinite(factor):return
        x,y=self.mouseEventCoords(position)
        for axes in reversed(self.figure.axes):
            if axes.name=='3d' or hasattr(axes,'_colorbar') or not axes.bbox.contains(x,y):continue
            ax,ay=axes.transAxes.inverted().transform((x,y))
            bounds=[]
            for axis,limits,ratio in ((axes.xaxis,axes.get_xlim(),ax),(axes.yaxis,axes.get_ylim(),ay)):
                transform=axis.get_transform();lo,hi=transform.transform(np.asarray(limits))
                anchor=lo+ratio*(hi-lo)
                values=transform.inverted().transform(anchor+(np.array([lo,hi])-anchor)/factor)
                if not np.isfinite(values).all():return
                bounds.append(values)
            toolbar=getattr(self,'toolbar',None)
            if toolbar is not None:toolbar.push_current()
            axes.set_xlim(bounds[0]);axes.set_ylim(bounds[1]);self.draw_idle()
            if toolbar is not None:toolbar.push_current()
            return

    def wheelEvent(self,event):
        delta=event.pixelDelta().y()/40 if not event.pixelDelta().isNull() else event.angleDelta().y()/120
        if delta:self.zoom_at(2**(.2*delta),event.position());event.accept()
        else:super().wheelEvent(event)

    def event(self,event):
        if event.type()==QEvent.NativeGesture and event.gestureType()==Qt.ZoomNativeGesture:
            self.zoom_at(1.+event.value(),native_position(self,event));event.accept();return True
        return super().event(event)

    def _draw_idle(self):
        # Matplotlib uses a zero-delay singleShot bound to a Python callable.
        # Qt can delete the canvas before that callable runs after panel close.
        if not isValid(self):
            self._draw_pending = False
            return
        super()._draw_idle()
