from gui.sherloq_app.ui.localization import t
import math
import os

import cv2 as cv
import numpy as np
from PySide6.QtCore import QSettings, Qt, Signal, QRect, QRectF, QPointF, QEvent
from PySide6.QtGui import QPainter, QTransform, QPixmap, QImage, QInputDevice
from PySide6.QtWidgets import (
    QLabel,
    QRadioButton,
    QToolButton,
    QFileDialog,
    QMessageBox,
    QWidget,
    QGraphicsScene,
    QGraphicsView,
    QVBoxLayout,
    QHBoxLayout,
)

from gui.sherloq_app.core.utility import mat2img, modify_font
from gui.sherloq_app.ui.icons import themed_icon
from gui.sherloq_app.ui.gestures import native_position


class DynamicView(QGraphicsView):
    viewChanged = Signal(QRect, float, int, int)

    def __init__(self, image, parent=None):
        super(DynamicView, self).__init__(parent)
        self.scene = QGraphicsScene(self)
        self.scene.setBackgroundBrush(Qt.darkGray)
        self.setScene(self.scene)
        self.set_image(image)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.ZOOM_STEP = 0.2
        self.mouse_pressed = False
        self.next_fit = False
        self.fit_scale = 0
        self.zoom_fit()

    def set_image(self, image):
        from gui.sherloq_app.ui.tiled_image import TiledImageItem
        current=getattr(self,'image_item',None)
        if isinstance(image,np.ndarray) and (isinstance(image,np.memmap)
                or image.shape[0]*image.shape[1]>16_000_000 or max(image.shape[:2])>16384):
            if isinstance(current,TiledImageItem):current.set_image(image)
            else:
                if current is not None:self.scene.removeItem(current)
                self.image_item=TiledImageItem(image);self.scene.addItem(self.image_item)
            self.scene.setSceneRect(self.image_item.boundingRect())
            return
        if type(image) is QPixmap:
            pixmap = image
        elif type(image) is QImage:
            pixmap = QPixmap.fromImage(image)
        elif isinstance(image,np.ndarray):
            pixmap = QPixmap.fromImage(mat2img(image))
        else:
            raise TypeError(
                self.tr(f"DynamicView.set_image: Unsupported type: {type(image)}")
            )
        if current is None or isinstance(current,TiledImageItem):
            if current is not None:self.scene.removeItem(current)
            self.image_item = self.scene.addPixmap(pixmap)
        else:
            self.image_item.setPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect()))

    def zoom_full(self):
        self.set_scaling(1)
        self.next_fit = True
        self.notify_change()

    def zoom_fit(self):
        self.setSceneRect(self.scene.sceneRect())
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.fit_scale = self.transform().m11()
        if self.fit_scale > 1:
            self.fit_scale = 1
            self.zoom_full()
        else:
            self.next_fit = False
            self.notify_change()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mouse_pressed = True
        QGraphicsView.mousePressEvent(self, event)

    def mouseMoveEvent(self, event):
        QGraphicsView.mouseMoveEvent(self, event)
        if self.mouse_pressed:
            self.notify_change()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mouse_pressed = False
        QGraphicsView.mouseReleaseEvent(self, event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.next_fit:
                self.zoom_fit()
            else:
                self.zoom_full()
        QGraphicsView.mouseDoubleClickEvent(self, event)

    def zoom_at(self, factor, position):
        """Keep the image coordinate under the pointer fixed, including borders."""
        old = self.transform().m11()
        if old <= 0 or not math.isfinite(factor) or factor <= 0:
            return
        scaling = min(4., max(self.fit_scale, old * factor))
        if scaling == old:
            return
        if scaling <= self.fit_scale:
            self.zoom_fit()
            return
        position = QPointF(position)
        before = self.viewportTransform().inverted()[0].map(position)
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        # Padding permits cursor anchoring even when one image dimension is
        # smaller than the viewport. The scene itself retains the image bounds.
        rect = self.scene.sceneRect()
        px, py = self.viewport().width()/scaling, self.viewport().height()/scaling
        self.setSceneRect(rect.adjusted(-px, -py, px, py))
        self.set_scaling(scaling)
        after = self.viewportTransform().inverted()[0].map(position)
        self.horizontalScrollBar().setValue(round(self.horizontalScrollBar().value()+(before.x()-after.x())*scaling))
        self.verticalScrollBar().setValue(round(self.verticalScrollBar().value()+(before.y()-after.y())*scaling))
        self.next_fit = scaling > self.fit_scale
        self.notify_change()

    def wheelEvent(self, event):
        touchpad = (event.pointingDevice().type() == QInputDevice.DeviceType.TouchPad
                    or not event.pixelDelta().isNull() or event.phase() != Qt.NoScrollPhase)
        if touchpad:
            delta = event.pixelDelta() if not event.pixelDelta().isNull() else event.angleDelta() / 3
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self.notify_change()
            event.accept()
            return
        delta = event.angleDelta().y() / 120
        if delta:
            self.zoom_at(2 ** (self.ZOOM_STEP * delta), event.position())
            event.accept()
        else:
            super().wheelEvent(event)

    def viewportEvent(self, event):
        if event.type() == QEvent.NativeGesture and event.gestureType() == Qt.ZoomNativeGesture:
            self.zoom_at(1. + event.value(), native_position(self.viewport(), event))
            event.accept()
            return True
        return super().viewportEvent(event)

    def resizeEvent(self, event):
        # FIXME: Se la finestra viene massimizzata, il valore di fit_scale non si aggiorna
        if self.transform().m11() <= self.fit_scale:
            self.zoom_fit()
        else:
            self.notify_change()
        QGraphicsView.resizeEvent(self, event)

    def change_zoom(self, direction):
        self.zoom_at(2 ** (self.ZOOM_STEP * direction), self.viewport().rect().center())

    def set_scaling(self, scaling):
        transform = QTransform()
        transform.scale(scaling, scaling)
        self.setTransform(transform)

    def change_view(self, _, new_scaling, new_horiz, new_vert):
        old_factor = self.transform().m11()
        old_horiz = self.horizontalScrollBar().value()
        old_vert = self.verticalScrollBar().value()
        if new_scaling != old_factor or new_horiz != old_horiz or new_vert != old_vert:
            if new_scaling > self.fit_scale:
                rect = self.scene.sceneRect()
                px, py = self.viewport().width()/new_scaling, self.viewport().height()/new_scaling
                self.setSceneRect(rect.adjusted(-px, -py, px, py))
            else:
                self.setSceneRect(self.scene.sceneRect())
            self.set_scaling(new_scaling)
            self.horizontalScrollBar().setValue(new_horiz)
            self.verticalScrollBar().setValue(new_vert)
            self.notify_change()

    def notify_change(self):
        scene_rect = self.get_rect()
        horiz_scroll = self.horizontalScrollBar().value()
        vert_scroll = self.verticalScrollBar().value()
        zoom_factor = self.transform().m11()
        self.viewChanged.emit(scene_rect, zoom_factor, horiz_scroll, vert_scroll)

    def get_rect(self):
        top_left = self.mapToScene(0, 0).toPoint()
        if top_left.x() < 0:
            top_left.setX(0)
        if top_left.y() < 0:
            top_left.setY(0)
        view_size = self.viewport().size()
        bottom_right = self.mapToScene(view_size.width(), view_size.height()).toPoint()
        image_size = self.scene.sceneRect().toRect()
        if bottom_right.x() >= image_size.width():
            bottom_right.setX(image_size.width() - 1)
        if bottom_right.y() >= image_size.height():
            bottom_right.setY(image_size.height() - 1)
        top_left.setX(min(image_size.width()-1,max(0,top_left.x())))
        top_left.setY(min(image_size.height()-1,max(0,top_left.y())))
        bottom_right.setX(min(image_size.width()-1,max(0,bottom_right.x())))
        bottom_right.setY(min(image_size.height()-1,max(0,bottom_right.y())))
        return QRect(top_left, bottom_right)


class ImageViewer(QWidget):
    viewChanged = Signal(QRect, float, int, int)

    def __init__(self, original, processed, title=None, parent=None, export=False, view_class=DynamicView):
        super(ImageViewer, self).__init__(parent)
        if original is None and processed is None:
            raise ValueError(self.tr("ImageViewer.__init__: Empty image received"))
        if original is None and processed is not None:
            original = processed
        self.original = original
        self.processed = processed
        self.analysis_busy = False
        from gui.sherloq_app.ui.image_io import ImageExportJob
        self.export_job = ImageExportJob(self)
        self.export_job.busy.connect(self._export_busy)
        self.export_job.failed.connect(self._export_error)
        self.export_job.result.connect(self._export_done)
        if self.original is not None and self.processed is None:
            self.view = view_class(self.original)
        else:
            self.view = view_class(self.processed)

        # view_label = QLabel(self.tr('View:'))
        self.original_radio = QRadioButton(self.tr("Original"))
        self.original_radio.setToolTip(
            self.tr("Show the original image for comparison (press SPACE to toggle)")
        )
        self.process_radio = QRadioButton(self.tr("Processed"))
        self.process_radio.setToolTip(
            self.tr("Show result of the current processing (press SPACE to toggle)")
        )
        self.zoom_label = QLabel()
        full_button = QToolButton()
        full_button.setText(self.tr("100%"))
        fit_button = QToolButton()
        fit_button.setText(self.tr("Fit"))
        height, width, _ = self.original.shape
        size_label = self.size_label = QLabel(self.tr(f"[{height}x{width} px]"))
        export_button = self.export_button = QToolButton()
        export_button.setToolTip(self.tr("Export processed image"))
        # export_button.setText(self.tr('Export...'))
        export_button.setIcon(themed_icon("export.svg"))

        tool_layout = QHBoxLayout()
        tool_layout.addWidget(QLabel(self.tr("Zoom:")))
        tool_layout.addWidget(self.zoom_label)
        # tool_layout.addWidget(full_button)
        # tool_layout.addWidget(fit_button)
        tool_layout.addStretch()
        if processed is not None:
            # tool_layout.addWidget(view_label)
            tool_layout.addWidget(self.original_radio)
            tool_layout.addWidget(self.process_radio)
            tool_layout.addStretch()
        tool_layout.addWidget(size_label)
        if export or processed is not None:
            tool_layout.addWidget(export_button)
        if processed is not None:
            self.original_radio.setChecked(False)
            self.process_radio.setChecked(True)
            # DynamicView already contains this image; do not convert it twice.

        vert_layout = QVBoxLayout()
        if title is not None:
            self.title_label = QLabel(title)
            modify_font(self.title_label, bold=True)
            self.title_label.setAlignment(Qt.AlignCenter)
            vert_layout.addWidget(self.title_label)
        else:
            self.title_label = None
        vert_layout.addWidget(self.view)
        vert_layout.addLayout(tool_layout)
        self.setLayout(vert_layout)

        self.original_radio.toggled.connect(self.toggle_mode)
        fit_button.clicked.connect(self.view.zoom_fit)
        full_button.clicked.connect(self.view.zoom_full)
        export_button.clicked.connect(lambda: self.export_image(asynchronous=True))
        self.view.viewChanged.connect(self.forward_changed)

        # view_label.setVisible(processed is not None)
        # self.original_radio.setVisible(processed is not None)
        # self.process_radio.setVisible(processed is not None)
        # export_button.setVisible(processed is not None)
        # if processed is not None:
        #
        # self.adjustSize()

    def set_busy(self, busy):
        self.analysis_busy = busy
        self.export_button.setEnabled(not busy and not self.export_job.is_busy)

    def _export_busy(self, busy):
        self.export_button.setEnabled(not busy and not self.analysis_busy)
        self.export_button.setToolTip('Exporting image…' if busy else 'Export processed image')

    def _export_done(self, filename):
        if filename:
            QSettings().setValue('save_folder', os.path.dirname(os.path.abspath(filename)))
            self.export_button.setToolTip('Image exported')

    def _export_error(self, message):
        QMessageBox.warning(self, t(self.tr('Export failed')), t(message))

    def closeEvent(self, event):
        self.export_job.shutdown()
        super().closeEvent(event)

    def update_processed(self, image):
        if self.processed is None:
            return
        self.processed = image
        self.toggle_mode(self.original_radio.isChecked())

    def update_original(self, image):
        self.original = image
        self.toggle_mode(True)

    def changeView(self, rect, scaling, horizontal, vertical):
        self.view.change_view(rect, scaling, horizontal, vertical)

    def forward_changed(self, rect, scaling, horizontal, vertical):
        self.zoom_label.setText(f"{scaling * 100:.2f}%")
        modify_font(self.zoom_label, scaling == 1)
        self.viewChanged.emit(rect, scaling, horizontal, vertical)

    def get_rect(self):
        return self.view.get_rect()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and self.processed is not None:
            if self.original_radio.isChecked():
                self.process_radio.setChecked(True)
            else:
                self.original_radio.setChecked(True)
        QWidget.keyPressEvent(self, event)

    def toggle_mode(self, toggled):
        image = self.original if toggled else self.processed
        if image is None:
            return
        height, width = image.shape[:2]
        old_rect = self.view.sceneRect()
        self.view.set_image(image)
        self.size_label.setText(self.tr(f"[{height}x{width} px]"))
        # A plot may have different dimensions from its source photograph.
        # Fit only on a size change; preserve navigation for subsequent results.
        if old_rect.width() != width or old_rect.height() != height:
            self.view.zoom_fit()

    def export_image(self, *, asynchronous=False):
        """The GUI button writes asynchronously; scripts can use the sync API."""
        if not self.export_button.isEnabled():
            return
        settings = QSettings()
        filename = QFileDialog.getSaveFileName(
            self,
            t(self.tr("Export image...")),
            settings.value("save_folder"),
            t(self.tr("Images (*.png *.jpg);;PNG files (*.png);;JPG files (*.jpg)")),
        )[0]
        if not filename:
            return
        if not os.path.splitext(filename)[1]:
            filename += ".png"
        image = self.processed if self.processed is not None else self.original
        if asynchronous:
            self.export_job.save(filename,image)
        else:
            from gui.sherloq_app.core.image_io import save_image
            try:
                self._export_done(save_image(filename,image))
            except Exception as error:
                self._export_error(str(error))

    def set_title(self, title):
        if self.title_label is not None:
            self.title_label.setText(title)
