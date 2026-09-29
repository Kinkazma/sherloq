"""Bounded presentation of large arrays; analysis/export keep every pixel."""
import math
import cv2 as cv
import numpy as np
from PySide6.QtCore import QRectF
from PySide6.QtWidgets import QGraphicsItem
from gui.sherloq_app.core.utility import mat2img
from gui.sherloq_app.core.memory_resources import TemporaryArrays,MiB
from gui.sherloq_app.core.cache_budget import ArrayCache


class TiledImageItem(QGraphicsItem):
    def __init__(self,image):
        super().__init__()
        self.setFlag(QGraphicsItem.ItemUsesExtendedStyleOption)
        self.cache=ArrayCache(32)
        self.set_image(image)

    def set_image(self,image):
        if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
            raise ValueError('Expected a BGR uint8 display image')
        self.prepareGeometryChange();self.image=image
        self.cache.clear()
        self.store=TemporaryArrays(32*MiB);self.store.watch(image)
        self.update()

    def boundingRect(self):
        return QRectF(0,0,self.image.shape[1],self.image.shape[0])

    @property
    def cache_bytes(self):return self.cache.bytes

    @property
    def cache_limit(self):return self.cache.limit

    def tile(self,level,x,y,span):
        key=(level,x,y)
        cached=self.cache.get(key)
        if cached is not None:return cached
        h,w=self.image.shape[:2];step=1<<level
        roi=self.image[y:min(h,y+span),x:min(w,x+span)]
        # OpenCV reads the original stride without a full-image conversion.
        # Area reduction affects display only; 100% uses exact source pixels.
        if step>1:
            size=((roi.shape[1]+step-1)//step,(roi.shape[0]+step-1)//step)
            roi=cv.resize(roi,size,interpolation=cv.INTER_AREA)
        tile=mat2img(np.ascontiguousarray(roi)).copy()
        self.cache.put(key,tile)
        self.store.checkpoint()
        return tile

    def paint(self,painter,option,widget=None):
        visible=option.exposedRect.intersected(self.boundingRect())
        if visible.isEmpty():return
        transform=painter.worldTransform()
        scale=max(math.hypot(transform.m11(),transform.m12()),1e-12)
        level=max(0,min(30,int(math.floor(-math.log2(scale)))))
        span=min(2048,256*(1<<level));h,w=self.image.shape[:2]
        x0=max(0,int(visible.left())//span*span);y0=max(0,int(visible.top())//span*span)
        for y in range(y0,min(h,math.ceil(visible.bottom())),span):
            for x in range(x0,min(w,math.ceil(visible.right())),span):
                tile=self.tile(level,x,y,span)
                target=QRectF(x,y,min(span,w-x),min(span,h-y))
                painter.drawImage(target,tile)
