"""Fixed percentile ranges with more pointer travel near the histogram tails."""
import math
from PySide6.QtCore import Qt,QRectF,QPointF
from PySide6.QtGui import QPainter,QColor,QPen,QRegion
from PySide6.QtWidgets import QSlider,QStyle,QStyleOptionSlider

class PercentileSlider(QSlider):
    STEPS=10000
    CURVE=2.0
    def __init__(self,upper=False):
        super().__init__(Qt.Horizontal)
        self.upper=upper
        self.setProperty('histogramPercentile',True)
        super().setRange(0,self.STEPS)
        self.setSingleStep(10)
    def value(self):
        position=super().value()/self.STEPS
        tail=1-position if self.upper else position
        units=round(500*math.expm1(self.CURVE*tail)/math.expm1(self.CURVE))
        return 1000-units if self.upper else units
    def setValue(self,value):
        value=max(500 if self.upper else 0,min(1000 if self.upper else 500,int(value)))
        super().setValue(self.position_for(value))
    def position_for(self,value):
        units=1000-value if self.upper else value
        position=math.log1p(units/500*math.expm1(self.CURVE))/self.CURVE
        if self.upper:position=1-position
        return round(position*self.STEPS)
    def minimum(self):return 500 if self.upper else 0
    def maximum(self):return 1000 if self.upper else 500
    def paintEvent(self,event):
        super().paintEvent(event)
        option=QStyleOptionSlider();self.initStyleOption(option)
        style=self.style();handle=style.subControlRect(QStyle.CC_Slider,option,QStyle.SC_SliderHandle,self)
        groove=style.subControlRect(QStyle.CC_Slider,option,QStyle.SC_SliderGroove,self)
        def center(position):
            option.sliderPosition=position
            return style.subControlRect(QStyle.CC_Slider,option,QStyle.SC_SliderHandle,self).center().x()
        mark=center(self.position_for(900 if self.upper else 100))
        edge=center(0 if self.upper else self.STEPS)
        painter=QPainter(self)
        painter.setClipRegion(QRegion(self.rect()).subtracted(QRegion(handle.adjusted(-1,-1,1,1))))
        y=groove.center().y()
        painter.fillRect(QRectF(min(mark,edge),y-2,abs(edge-mark),4),QColor(210,55,65,70))
        painter.setPen(QPen(QColor(185,55,65,180),1))
        painter.drawLine(QPointF(mark,y-5),QPointF(mark,y+5))
        painter.end()
    def keyPressEvent(self,event):
        steps={Qt.Key_Left:-1,Qt.Key_Down:-1,Qt.Key_Right:1,Qt.Key_Up:1,Qt.Key_PageDown:-10,Qt.Key_PageUp:10}
        if event.key() in steps:self.setValue(self.value()+steps[event.key()]);event.accept()
        elif event.key()==Qt.Key_Home:self.setValue(self.minimum());event.accept()
        elif event.key()==Qt.Key_End:self.setValue(self.maximum());event.accept()
        else:super().keyPressEvent(event)
