from gui.sherloq_app.ui.localization import t
import os
import sys
import shutil
import rawpy
from time import time

import cv2 as cv
import numpy as np
from PySide6.QtCore import QSettings, QFileInfo, Signal, Qt, QMimeDatabase
from PySide6.QtGui import QImage, QFontDatabase, QColor, QBrush
from PySide6.QtWidgets import (
    QLabel,
    QTreeWidgetItem,
    QFileDialog,
    QMessageBox,
    QWidget,
    QSlider,
    QSpinBox,
    QHBoxLayout,
)

from gui.sherloq_app.paths import BUTTERAUGLI_DIR, PYEXIFTOOL_DIR, SSIMULACRA_DIR


def mat2img(cvmat):
    height, width, channels = cvmat.shape
    if channels != 3 or cvmat.dtype != np.uint8:
        raise ValueError('Display expects an 8-bit BGR image.')
    if not cvmat.flags.c_contiguous:
        # QImage must own the temporary copy after this function returns.
        contiguous = np.ascontiguousarray(cvmat)
        return QImage(contiguous.data, width, height, 3 * width, QImage.Format_BGR888).copy()
    return QImage(cvmat.data, width, height, 3 * width, QImage.Format_BGR888)


def color_by_value(item, value, splits):
    if value < splits[0]:
        hue = 0
    elif value < splits[1]:
        hue = 30
    elif value < splits[2]:
        hue = 60
    else:
        hue = 90
    item.setBackground(QBrush(QColor.fromHsv(hue, 96, 255)))


def modify_font(obj, bold=False, italic=False, underline=False, mono=False):
    if obj is None:
        return
    if mono:
        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
    else:
        if type(obj) is QTreeWidgetItem:
            font = obj.font(0)
        else:
            font = obj.font()
    font.setBold(bold)
    font.setItalic(italic)
    font.setUnderline(underline)
    if type(obj) is QTreeWidgetItem:
        obj.setFont(0, font)
    else:
        obj.setFont(font)


def pad_image(image, bsize, reflect=False):
    rows, cols = image.shape[:2]
    top = left = 0
    bottom = bsize - rows % bsize
    right = bsize - cols % bsize
    border = cv.BORDER_CONSTANT if not reflect else cv.BORDER_REFLECT_101
    padded = cv.copyMakeBorder(image, top, bottom, left, right, border)
    return padded


def shift_image(image, bsize):
    rows, cols = image.shape[:2]
    shifted = np.zeros_like(image)
    shifted[: rows - bsize, : cols - bsize] = image[bsize:, bsize:]
    return shifted


def human_size(total, binary=False, suffix="B"):
    units = ["", "K", "M", "G", "T", "P", "E", "Z", "Y"]
    if binary:
        units = [unit + "i" for unit in units]
        factor = 1024.0
    else:
        factor = 1000.0
    for unit in units:
        if abs(total) < factor:
            return f"{total:3.1f} {unit}{suffix}"
        total /= factor
    return f"{total:.1f} {units[-1]}{suffix}"


def create_lut(low, high):
    if low >= 0:
        p1 = (+low, 0)
    else:
        p1 = (0, -low)
    if high >= 0:
        p2 = (255 - high, 255)
    else:
        p2 = (255, 255 + high)
    if p1[0] == p2[0]:
        return np.full(256, 255, np.uint8)
    lut = [
        (x * (p1[1] - p2[1]) + p1[0] * p2[1] - p1[1] * p2[0]) / (p1[0] - p2[0])
        for x in range(256)
    ]
    return np.clip(np.array(lut), 0, 255).astype(np.uint8)


def compute_hist(image, normalize=False):
    from .histogram import exact_channel_histogram
    hist = exact_channel_histogram(image)
    return hist / image.size if normalize else hist


def auto_lut(image, centile):
    hist = compute_hist(image, normalize=True)
    if centile == 0:
        nonzero = np.nonzero(hist)[0]
        low = nonzero[0]
        high = 255 - nonzero[-1]
    else:
        low_sum = high_sum = 0
        low = 0
        high = 255
        for i, h in enumerate(hist):
            low_sum += h
            if low_sum >= centile:
                low = i
                break
        for i, h in enumerate(np.flip(hist)):
            high_sum += h
            if high_sum >= centile:
                high = i
                break
    return create_lut(low, high)


def elapsed_time(start, ms=True):
    elapsed = time() - start
    if ms:
        return f"{int(np.round(elapsed * 1000))} ms"
    return f"{elapsed:.2f} sec"


def signed_value(value):
    return f"{'+' if value > 0 else ''}{value}"


def equalize_img(image):
    return cv.merge([cv.equalizeHist(c) for c in cv.split(image)])


def norm_img(image):
    return cv.merge([norm_mat(c) for c in cv.split(image)])


def clip_value(value, minv=None, maxv=None):
    if minv is not None:
        value = max(value, minv)
    if maxv is not None:
        value = min(value, maxv)
    return value


def bgr_to_gray3(image):
    return cv.cvtColor(cv.cvtColor(image, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)


def gray_to_bgr(image):
    return cv.cvtColor(image, cv.COLOR_GRAY2BGR)


def load_image(parent, filename=None):
    """Synchronous compatibility API; main-window opening uses ImageLoadJob."""
    from gui.sherloq_app.ui.image_io import choose_image
    from gui.sherloq_app.core.image_io import decode_image
    if filename is None:
        filename = choose_image(parent)
    if not filename:
        return [None] * 3
    try:
        filename, basename, image, metadata = decode_image(filename)
    except Exception as error:
        QMessageBox.critical(parent, t(parent.tr('Error')), t(str(error)))
        return [None] * 3
    if metadata['frames'] > 1:
        QMessageBox.warning(parent, t(parent.tr('Warning')), t(parent.tr('Animated GIF: importing first frame')))
    QSettings().setValue('load_folder', QFileInfo(filename).absolutePath())
    return filename, basename, image


def desaturate(image):
    return cv.cvtColor(cv.cvtColor(image, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)


def norm_mat(matrix, to_bgr=False):
    norm = cv.normalize(matrix, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    if not to_bgr:
        return norm
    return cv.cvtColor(norm, cv.COLOR_GRAY2BGR)


def exiftool_exe():
    if sys.platform == "darwin":
        private_exiftool = PYEXIFTOOL_DIR.parents[2] / "native" / "exiftool" / "exiftool"
        return str(private_exiftool) if private_exiftool.is_file() else (shutil.which("exiftool") or str(PYEXIFTOOL_DIR / "exiftool" / "linux" / "exiftool"))
    if sys.platform.startswith("linux"):
        return str(PYEXIFTOOL_DIR / "exiftool" / "linux" / "exiftool")
    if sys.platform.startswith("win32"):
        return str(PYEXIFTOOL_DIR / "exiftool" / "windows" / "exiftool(-k).exe")
    return None


def butter_exe():
    if sys.platform == "darwin":
        return str(BUTTERAUGLI_DIR / "macos" / "butteraugli")
    if sys.platform.startswith("linux"):
        return str(BUTTERAUGLI_DIR / "linux" / "butteraugli")
    if sys.platform.startswith("win32"):
        return None
    return None


def ssimul_exe():
    if sys.platform == "darwin":
        return str(SSIMULACRA_DIR / "macos" / "ssimulacra")
    if sys.platform.startswith("linux"):
        return str(SSIMULACRA_DIR / "linux" / "ssimulacra")
    if sys.platform.startswith("win32"):
        return None
    return None


class ParamSlider(QWidget):
    valueChanged = Signal(int)

    def __init__(
        self,
        interval,
        ticks=10,
        reset=0,
        suffix=None,
        label=None,
        bold=False,
        special=None,
        parent=None,
    ):
        super(ParamSlider, self).__init__(parent)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(interval[0], interval[1])
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval((interval[1] - interval[0] + 1) / ticks)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.slider.setValue(reset)
        self.slider.mouseDoubleClickEvent = self.doubleClicked

        self.spin = QSpinBox()
        self.spin.setRange(interval[0], interval[1])
        self.spin.setValue(reset)
        self.spin.setSuffix(suffix)
        self.spin.setFixedWidth(55)
        self.spin.setSpecialValueText(special)

        self.reset = reset
        self.slider.valueChanged.connect(self.spin.setValue)
        self.spin.valueChanged.connect(self.slider.setValue)
        self.slider.valueChanged.connect(self.valueChanged)

        layout = QHBoxLayout()
        if label is not None:
            lab = QLabel(label)
            modify_font(lab, bold=bold)
            layout.addWidget(lab)
        layout.addWidget(self.slider)
        layout.addWidget(self.spin)
        self.setLayout(layout)

    def doubleClicked(self, _):
        self.reset_value()

    def reset_value(self):
        self.slider.setValue(self.reset)
        self.spin.setValue(self.reset)

    def value(self):
        return self.slider.value()

    def setValue(self, value):
        self.spin.setValue(value)
        self.slider.setValue(value)
        self.valueChanged.emit(value)

    def sync(self):
        self.spin.setValue(self.slider.value())
        self.slider.setValue(self.spin.value())
        self.valueChanged.emit(self.slider.value())
