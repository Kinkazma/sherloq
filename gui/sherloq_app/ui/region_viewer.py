"""Retain the full raster and patch only changed regions, with lazy BGR export.

The same single-pixmap rendering path is kept at fractional zooms: separate
cropped scene items can change Qt's pixel sampling at their boundaries.
"""
import numpy as np
from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QPixmap, QPainter
from .viewer import ImageViewer
from ..core.utility import mat2img


class RegionViewer(ImageViewer):
    def __init__(self, image, parent=None):
        self.region = None
        self.bounds = None
        self._composed = None
        super().__init__(image, image, parent=parent)
        self._item = self.view.scene.items()[0]
        self._original_pixmap = self._item.pixmap()
        self._processed_pixmap = QPixmap(self._original_pixmap)

    @property
    def processed(self):
        if self.region is None:
            return self.original
        if self._composed is None:
            self._composed = np.copy(self.original)
            x1, y1, x2, y2 = self.bounds
            self._composed[y1:y2, x1:x2] = self.region
        return self._composed

    @processed.setter
    def processed(self, _):
        # ImageViewer initializes this attribute; source is already installed.
        self._composed = None

    def update_region(self, result):
        # Release the scene's implicit sharing reference before editing the
        # retained pixmap. No events are processed until it is installed again.
        self._item.setPixmap(QPixmap())
        painter = QPainter(self._processed_pixmap)
        try:
            if self.region is not None:
                x1, y1, x2, y2 = self.bounds
                rect = QRect(x1, y1, x2-x1, y2-y1)
                painter.drawPixmap(rect, self._original_pixmap, rect)
            bounds, region = result
            if region is not None:
                painter.drawImage(bounds[0], bounds[1], mat2img(region))
        finally:
            painter.end()
        self.bounds, self.region = result
        self._composed = None
        self.toggle_mode(self.original_radio.isChecked())

    def toggle_mode(self, original):
        if hasattr(self, '_item'):
            self._item.setPixmap(self._original_pixmap if original else self._processed_pixmap)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            (self.process_radio if self.original_radio.isChecked() else self.original_radio).setChecked(True)
        else:
            super().keyPressEvent(event)
