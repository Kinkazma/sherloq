"""Image-coordinate ROI drawing: rectangle, auto-closing lasso and polygon."""
import numpy as np
from PySide6.QtCore import Qt,QPointF,Signal
from PySide6.QtWidgets import QGraphicsView
from .localization import t
from PySide6.QtGui import QPainterPath,QPen,QColor
from .viewer import DynamicView


class SelectionView(DynamicView):
    regionsChanged = Signal(object)
    interactionStarted = Signal()

    def __init__(self,image,parent=None):
        self.regions=[];self.draft=[];self.mode='Pan';self.dragging=False;self.show_regions=True;self.enabled_regions=None;self.focus_regions=set();self.region_names=[];self._space_pan=False
        super().__init__(image,parent)

    def set_mode(self,mode):
        self._space_pan=False;self.mode=mode;self.draft=[];self.dragging=False
        self.setFocus(Qt.OtherFocusReason)
        self.setDragMode(QGraphicsView.ScrollHandDrag if mode=='Pan' else QGraphicsView.NoDrag)
        self.viewport().setCursor(Qt.OpenHandCursor if mode=='Pan' else Qt.CrossCursor)
        self.viewport().update()

    def clear_regions(self):
        self.regions=[];self.draft=[];self.dragging=False
        self.regionsChanged.emit([]);self.viewport().update()

    def undo_region(self):
        if self.draft:self.draft=[]
        elif self.regions:self.regions.pop();self.regionsChanged.emit(self.snapshot())
        self.viewport().update()

    def snapshot(self):
        return tuple(tuple((float(x),float(y)) for x,y in region) for region in self.regions)

    def point(self,event):
        p=self.mapToScene(event.position().toPoint());r=self.scene.sceneRect()
        return (min(r.right()-1,max(0.,p.x())),min(r.bottom()-1,max(0.,p.y())))

    def finish(self):
        if len(self.draft)>=3:
            p=np.asarray(self.draft)
            area=abs(np.dot(p[:,0],np.roll(p[:,1],1))-np.dot(p[:,1],np.roll(p[:,0],1)))/2
            if area>=1:
                self.regions.append(tuple(self.draft));self.regionsChanged.emit(self.snapshot())
        self.draft=[];self.dragging=False;self.viewport().update()

    def mousePressEvent(self,event):
        self.interactionStarted.emit()
        if self.mode=='Pan' or self._space_pan:return super().mousePressEvent(event)
        if event.button()==Qt.RightButton:
            if self.mode=='Polygon':self.finish()
            event.accept();return
        if event.button()==Qt.LeftButton:
            p=self.point(event)
            if self.mode=='Polygon':self.draft.append(p)
            else:self.draft=[p];self.dragging=True
            self.viewport().update();event.accept();return
        super().mousePressEvent(event)

    def mouseMoveEvent(self,event):
        if self.mode=='Pan' or self._space_pan:return super().mouseMoveEvent(event)
        if self.dragging:
            p=self.point(event)
            if self.mode=='Rectangle':
                a=self.draft[0];self.draft=[a,(p[0],a[1]),p,(a[0],p[1])]
            elif not self.draft or np.linalg.norm(np.subtract(p,self.draft[-1]))>=.5:self.draft.append(p)
            self.viewport().update()
        event.accept()

    def mouseReleaseEvent(self,event):
        if self.mode=='Pan' or self._space_pan:return super().mouseReleaseEvent(event)
        if event.button()==Qt.LeftButton and self.dragging:
            self.mouseMoveEvent(event);self.finish()
        event.accept()

    def mouseDoubleClickEvent(self,event):
        if self.mode=='Pan':super().mouseDoubleClickEvent(event)
        else:event.accept()

    def keyPressEvent(self,event):
        if event.key()==Qt.Key_Space and self.mode!='Pan':
            if not event.isAutoRepeat() and not self.dragging:
                self._space_pan=True;self.setDragMode(QGraphicsView.ScrollHandDrag);self.viewport().setCursor(Qt.OpenHandCursor)
            event.accept();return
        if event.key()==Qt.Key_Escape:
            self.draft=[];self.dragging=False;self.viewport().update();event.accept()
        else:super().keyPressEvent(event)

    def restore_selection_mode(self):
        self._space_pan=False;self.mouse_pressed=False
        self.setDragMode(QGraphicsView.ScrollHandDrag if self.mode=='Pan' else QGraphicsView.NoDrag)
        self.viewport().setCursor(Qt.OpenHandCursor if self.mode=='Pan' else Qt.CrossCursor)

    def keyReleaseEvent(self,event):
        if event.key()==Qt.Key_Space and (self._space_pan or self.mode!='Pan'):
            if not event.isAutoRepeat():self.restore_selection_mode()
            event.accept();return
        super().keyReleaseEvent(event)

    def focusOutEvent(self,event):
        if self._space_pan:self.restore_selection_mode()
        super().focusOutEvent(event)

    def set_regions(self,regions):
        self.regions=list(regions);self.draft=[];self.dragging=False
        self.regionsChanged.emit(self.snapshot());self.viewport().update()

    def drawForeground(self,painter,rect):
        super().drawForeground(painter,rect)
        for index,points in enumerate(self.regions+[self.draft]):
            if not points or (index<len(self.regions) and not self.show_regions):continue
            if index<len(self.regions):
                if self.enabled_regions is not None and index not in self.enabled_regions:continue
                if self.focus_regions and index not in self.focus_regions:continue
            path=QPainterPath(QPointF(*points[0]))
            for p in points[1:]:path.lineTo(QPointF(*p))
            if len(points)>2:path.closeSubpath()
            envelope=index<len(self.region_names) and self.region_names[index] in ('Ensemble','Enclosing zone')
            pen=QPen(QColor('#ff8e47' if envelope else '#ffe060'),2);pen.setCosmetic(True);pen.setStyle(Qt.DashLine if index==len(self.regions) or envelope else Qt.SolidLine)
            painter.setPen(pen);painter.setBrush(Qt.NoBrush if envelope else QColor(255,224,96,18));painter.drawPath(path)
            label_pos=painter.worldTransform().map(QPointF(*points[2] if envelope else points[0]))
            label=self.region_names[index] if index<len(self.region_names) else f'Zone {index+1}'
            label=t(label)
            painter.save();painter.resetTransform();painter.drawText(label_pos+(QPointF(-painter.fontMetrics().horizontalAdvance(label)-4,-6) if envelope else QPointF(4,-4)),label);painter.restore()
