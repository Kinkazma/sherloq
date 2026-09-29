from gui.sherloq_app.ui.localization import t
import os

import cv2 as cv
import numpy as np
from PySide6.QtCore import Qt, QTimer
from gui.sherloq_app.ui.jobs import LatestJob
from gui.sherloq_app.ui.comparison_job import ComparisonJob
from gui.sherloq_app.core import comparison as computation
from PySide6.QtWidgets import (
    QAbstractItemView,
    QTableWidgetItem,
    QTableWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QLabel,
    QRadioButton,
    QProgressBar,
)

from gui.sherloq_app.ui.icons import themed_icon
from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.core.utility import (
    norm_mat,
    equalize_img,
    modify_font,
    load_image,
    desaturate,
    butter_exe,
    ssimul_exe,
)
from gui.sherloq_app.ui.viewer import ImageViewer


class ComparisonWidget(ToolWidget):
    def __init__(self, filename, image, parent=None):
        super(ComparisonWidget, self).__init__(parent)

        load_button = QPushButton(self.tr("Load reference image..."))
        self.comp_label = QLabel(self.tr("Comparison:"))
        self.normal_radio = QRadioButton(self.tr("Normal"))
        self.normal_radio.setToolTip(self.tr("Show reference (raw pixels)"))
        self.normal_radio.setChecked(True)
        self.difference_radio = QRadioButton(self.tr("Difference"))
        self.difference_radio.setToolTip(self.tr("Show evidence/reference difference"))
        self.ssim_radio = QRadioButton(self.tr("SSIM Map"))
        self.ssim_radio.setToolTip(self.tr("Structure similarity quality map"))
        self.butter_radio = QRadioButton(self.tr("Butteraugli"))
        self.butter_radio.setToolTip(self.tr("Butteraugli spatial changes heatmap"))
        self.gray_check = QCheckBox(self.tr("Grayscale"))
        self.gray_check.setToolTip(self.tr("Show desaturated output"))
        self.equalize_check = QCheckBox(self.tr("Equalized"))
        self.equalize_check.setToolTip(self.tr("Apply histogram equalization"))
        self.last_radio = self.normal_radio
        self.metric_button = QPushButton(self.tr("Compute metrics"))
        self.metric_button.setToolTip(self.tr("Image quality assessment metrics"))

        self.evidence = image
        self.reference = self.difference = self.ssim_map = self.butter_map = None
        basename = os.path.basename(filename)
        self.evidence_viewer = ImageViewer(
            self.evidence, None, self.tr(f"Evidence: {basename}")
        )
        self.reference_viewer = ImageViewer(
            np.full_like(self.evidence, 127), None, self.tr("Reference")
        )

        self.table_widget = QTableWidget(20, 3)
        self.table_widget.setHorizontalHeaderLabels(
            [self.tr("Metric"), self.tr("Value"), self.tr("Best")]
        )
        self.table_widget.setItem(0, 0, QTableWidgetItem(self.tr("RMSE")))
        self.table_widget.setItem(0, 2, QTableWidgetItem(themed_icon("low.svg"), "(0)"))
        self.table_widget.item(0, 0).setToolTip(
            self.tr(
                "Root Mean Square Error (RMSE) is commonly used to compare \n"
                "the difference between the reference and evidence images \n"
                "by directly computing the variation in pixel values. \n"
                "The combined image is close to the reference image when \n"
                "RMSE value is zero. RMSE is a good indicator of the spectral \n"
                "quality of the reference image."
            )
        )
        self.table_widget.setItem(1, 0, QTableWidgetItem(self.tr("SAM")))
        self.table_widget.setItem(1, 2, QTableWidgetItem(themed_icon("low.svg"), "(0)"))
        self.table_widget.item(1, 0).setToolTip(
            self.tr(
                "It computes the spectral angle between the pixel, vector of the \n"
                "evidence image and reference image. It is worked out in either \n"
                "degrees or radians. It is performed on a pixel-by-pixel base. \n"
                "A SAM equal to zero denotes the absence of spectral distortion."
            )
        )
        self.table_widget.setItem(2, 0, QTableWidgetItem(self.tr("ERGAS")))
        self.table_widget.setItem(2, 2, QTableWidgetItem(themed_icon("low.svg"), "(0)"))
        self.table_widget.item(2, 0).setToolTip(
            self.tr(
                "It is used to compute the quality of reference image in terms \n"
                "of normalized average error of each band of the reference image. \n"
                "Increase in the value of ERGAS indicates distortion in the \n"
                "reference image, lower value of ERGAS indicates that it is \n"
                "similar to the reference image."
            )
        )
        self.table_widget.setItem(3, 0, QTableWidgetItem(self.tr("MB")))
        self.table_widget.setItem(3, 2, QTableWidgetItem(themed_icon("low.svg"), "(0)"))
        self.table_widget.item(3, 0).setToolTip(
            self.tr(
                "Mean Bias is the difference between the mean of the evidence \n"
                "image and reference image. The ideal value is zero and indicates \n"
                "that the evidence and reference images are similar. Mean value \n"
                "refers to the grey level of pixels in an image."
            )
        )
        self.table_widget.setItem(4, 0, QTableWidgetItem(self.tr("PFE")))
        self.table_widget.setItem(4, 2, QTableWidgetItem(themed_icon("low.svg"), "(0)"))
        self.table_widget.item(4, 0).setToolTip(
            self.tr(
                "It computes the norm of the difference between the corresponding \n"
                "pixels of the reference and fused image to the norm of the reference \n"
                "image. When the calculated value is zero, it indicates that both the \n"
                "reference and fused images are similar and value will be increased \n"
                "when the merged image is not similar to the reference image."
            )
        )
        self.table_widget.setItem(5, 0, QTableWidgetItem(self.tr("PSNR")))
        self.table_widget.setItem(
            5, 2, QTableWidgetItem(themed_icon("high.svg"), "(+" + "\u221e" + ")")
        )
        self.table_widget.item(5, 0).setToolTip(
            self.tr(
                "It is widely used metric it is computed by the number of gray levels \n"
                "in the image divided by the corresponding pixels in the evidence and \n"
                "the reference images. When the value is high, both images are similar."
            )
        )
        # self.table_widget.setItem(6, 0, QTableWidgetItem(self.tr('PSNR-B')))
        # self.table_widget.setItem(6, 2, QTableWidgetItem(QIcon('icons/high.svg'), '(+' + u'\u221e' + ')'))
        # self.table_widget.item(6, 0).setToolTip(self.tr('PSNR with Blocking Effect Factor.'))
        self.table_widget.setItem(6, 0, QTableWidgetItem(self.tr("SSIM")))
        self.table_widget.setItem(
            6, 2, QTableWidgetItem(themed_icon("high.svg"), "(1)")
        )
        self.table_widget.item(6, 0).setToolTip(
            self.tr(
                "SSIM is used to compare the local patterns of pixel intensities between \n"
                "reference and fused images. The range varies between -1 to 1. \n"
                "The value 1 indicates the reference and fused images are similar."
            )
        )
        self.table_widget.setItem(7, 0, QTableWidgetItem(self.tr("MS-SSIM")))
        self.table_widget.setItem(
            7, 2, QTableWidgetItem(themed_icon("high.svg"), "(1)")
        )
        self.table_widget.item(7, 0).setToolTip(self.tr("Multiscale version of SSIM."))
        self.table_widget.setItem(8, 0, QTableWidgetItem(self.tr("RASE")))
        self.table_widget.setItem(8, 2, QTableWidgetItem(themed_icon("low.svg"), "(0)"))
        self.table_widget.item(8, 0).setToolTip(
            self.tr("Relative average spectral error")
        )
        self.table_widget.setItem(9, 0, QTableWidgetItem(self.tr("SCC")))
        self.table_widget.setItem(
            9, 2, QTableWidgetItem(themed_icon("high.svg"), "(1)")
        )
        self.table_widget.item(9, 0).setToolTip(
            self.tr("Spatial Correlation Coefficient")
        )
        self.table_widget.setItem(10, 0, QTableWidgetItem(self.tr("UQI")))
        self.table_widget.setItem(
            10, 2, QTableWidgetItem(themed_icon("high.svg"), "(1)")
        )
        self.table_widget.item(10, 0).setToolTip(
            self.tr("Universal Image Quality Index")
        )
        self.table_widget.setItem(11, 0, QTableWidgetItem(self.tr("VIF-P")))
        self.table_widget.setItem(
            11, 2, QTableWidgetItem(themed_icon("high.svg"), "(1)")
        )
        self.table_widget.item(11, 0).setToolTip(
            self.tr("Pixel-based Visual Information Fidelity")
        )
        self.table_widget.setItem(12, 0, QTableWidgetItem(self.tr("SSIMulacra")))
        self.table_widget.setItem(
            12, 2, QTableWidgetItem(themed_icon("low.svg"), "(0)")
        )
        self.table_widget.item(12, 0).setToolTip(
            self.tr(
                "Structural SIMilarity Unveiling Local And Compression Related Artifacts"
            )
        )
        self.table_widget.setItem(13, 0, QTableWidgetItem(self.tr("Butteraugli")))
        self.table_widget.setItem(
            13, 2, QTableWidgetItem(themed_icon("low.svg"), "(0)")
        )
        self.table_widget.item(13, 0).setToolTip(self.tr("Estimate psychovisual error"))
        self.table_widget.setItem(14, 0, QTableWidgetItem(self.tr("Correlation")))
        self.table_widget.setItem(
            14, 2, QTableWidgetItem(themed_icon("high.svg"), "(1)")
        )
        self.table_widget.item(14, 0).setToolTip(self.tr("Histogram correlation"))
        self.table_widget.setItem(15, 0, QTableWidgetItem(self.tr("Chi-Square")))
        self.table_widget.setItem(
            15, 2, QTableWidgetItem(themed_icon("low.svg"), "(0)")
        )
        self.table_widget.item(15, 0).setToolTip(self.tr("Histogram Chi-Square"))
        self.table_widget.setItem(16, 0, QTableWidgetItem(self.tr("Chi-Square 2")))
        self.table_widget.setItem(
            16, 2, QTableWidgetItem(themed_icon("low.svg"), "(0)")
        )
        self.table_widget.item(16, 0).setToolTip(self.tr("Alternative Chi-Square"))
        self.table_widget.setItem(17, 0, QTableWidgetItem(self.tr("Intersection")))
        self.table_widget.setItem(
            17, 2, QTableWidgetItem(themed_icon("high.svg"), "(+" + "\u221e" + ")")
        )
        self.table_widget.item(17, 0).setToolTip(self.tr("Histogram intersection"))
        self.table_widget.setItem(18, 0, QTableWidgetItem(self.tr("Hellinger")))
        self.table_widget.setItem(
            18, 2, QTableWidgetItem(themed_icon("low.svg"), "(0)")
        )
        self.table_widget.item(18, 0).setToolTip(
            self.tr("Histogram Hellinger distance")
        )
        self.table_widget.setItem(19, 0, QTableWidgetItem(self.tr("Divergence")))
        self.table_widget.setItem(
            19, 2, QTableWidgetItem(themed_icon("low.svg"), "(0)")
        )
        self.table_widget.item(19, 0).setToolTip(self.tr("Kullback-Leibler divergence"))

        for i in range(self.table_widget.rowCount()):
            modify_font(self.table_widget.item(i, 0), bold=True)
        self.table_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_widget.resizeColumnsToContents()
        self.table_widget.setMinimumWidth(min(480, self.table_widget.horizontalHeader().length()
                                                + self.table_widget.verticalHeader().width() + 30))
        self.table_widget.setMaximumWidth(250)
        self.table_widget.setAlternatingRowColors(True)
        self.stopped = False
        self.engine = None
        self.job = ComparisonJob(self)
        self.job.result.connect(self._metrics_ready)
        self.job.failed.connect(self._failed)
        self.display_job = LatestJob(self, lambda p: p[0].display(*p[1:]), delay=40)
        self.display_job.result.connect(self.reference_viewer.update_original)
        self.display_job.failed.connect(self._failed)
        self.status = QLabel()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 20)
        self.progress_bar.hide()
        self.cancel_button = QPushButton(self.tr("Cancel"))
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)
        self.progress_timer = QTimer(self)
        self.progress_timer.setInterval(100)
        self.progress_timer.timeout.connect(self._progress)


        self.comp_label.setEnabled(False)
        self.normal_radio.setEnabled(False)
        self.difference_radio.setEnabled(False)
        self.ssim_radio.setEnabled(False)
        self.butter_radio.setEnabled(False)
        self.gray_check.setEnabled(False)
        self.equalize_check.setEnabled(False)
        self.metric_button.setEnabled(False)
        self.table_widget.setEnabled(False)

        load_button.clicked.connect(self.load)
        self.normal_radio.clicked.connect(self.change)
        self.difference_radio.clicked.connect(self.change)
        self.butter_radio.clicked.connect(self.change)
        self.gray_check.stateChanged.connect(self.change)
        self.equalize_check.stateChanged.connect(self.change)
        self.ssim_radio.clicked.connect(self.change)
        self.evidence_viewer.viewChanged.connect(self.reference_viewer.changeView)
        self.reference_viewer.viewChanged.connect(self.evidence_viewer.changeView)
        self.metric_button.clicked.connect(self.metrics)

        top_layout = QHBoxLayout()
        top_layout.addWidget(load_button)
        top_layout.addStretch()
        top_layout.addWidget(self.comp_label)
        top_layout.addWidget(self.normal_radio)
        top_layout.addWidget(self.difference_radio)
        top_layout.addWidget(self.ssim_radio)
        top_layout.addWidget(self.butter_radio)
        top_layout.addWidget(self.gray_check)
        top_layout.addWidget(self.equalize_check)

        metric_layout = QVBoxLayout()
        index_label = QLabel(self.tr("Image Quality Assessment"))
        index_label.setAlignment(Qt.AlignCenter)
        modify_font(index_label, bold=True)
        metric_layout.addWidget(index_label)
        metric_layout.addWidget(self.table_widget)
        metric_layout.addWidget(self.metric_button)
        metric_layout.addWidget(self.progress_bar)
        metric_layout.addWidget(self.cancel_button)

        center_layout = QHBoxLayout()
        center_layout.addWidget(self.evidence_viewer, 1)
        center_layout.addWidget(self.reference_viewer, 1)
        center_layout.addLayout(metric_layout)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.status)
        main_layout.addLayout(center_layout, 1)
        self.setLayout(main_layout)

    def load(self):
        filename, basename, reference = load_image(self)
        if filename is None:
            return
        if reference.shape != self.evidence.shape:
            QMessageBox.critical(
                self,
                t(self.tr("Error")),
                t(self.tr("Evidence and reference must have the same size!")),
            )
            return
        if self.engine is not None:
            self.engine.cancelled.set()
        self.job.invalidate()
        self.display_job.invalidate()
        self.progress_timer.stop()
        self.cancel_button.setEnabled(False)
        self.progress_bar.hide()
        self.status.clear()
        self.reference = reference
        self.engine = computation.ComparisonEngine(self.evidence, reference)
        self.ssim_map = self.butter_map = self.difference = None
        self.reference_viewer.set_title(self.tr(f"Reference: {basename}"))

        self.comp_label.setEnabled(True)
        self.normal_radio.setEnabled(True)
        self.difference_radio.setEnabled(True)
        self.ssim_radio.setEnabled(False)
        self.butter_radio.setEnabled(False)
        self.gray_check.setEnabled(True)
        self.equalize_check.setEnabled(True)
        self.metric_button.setEnabled(True)
        for i in range(self.table_widget.rowCount()):
            self.table_widget.setItem(i, 1, QTableWidgetItem())
        self.normal_radio.setChecked(True)
        self.table_widget.setEnabled(False)
        self.change()

    @property
    def is_busy(self):
        return self.job.is_busy or self.display_job.is_busy

    def _ensure_engine(self):
        if self.reference is None:
            return False
        if self.engine is None or self.engine.reference is not self.reference:
            if self.engine is not None:
                self.engine.cancelled.set()
            self.engine = computation.ComparisonEngine(self.evidence, self.reference)
        return True

    def change(self):
        if not self._ensure_engine():
            return
        if self.normal_radio.isChecked():
            mode, gray, equalized = 'normal', False, False
            self.last_radio = self.normal_radio
        elif self.difference_radio.isChecked():
            mode, gray, equalized = 'difference', True, True
            self.last_radio = self.difference_radio
        elif self.ssim_radio.isChecked():
            mode, gray, equalized = 'ssim', False, True
            self.last_radio = self.ssim_radio
        elif self.butter_radio.isChecked():
            mode, gray, equalized = 'butter', True, False
            self.last_radio = self.butter_radio
        else:
            self.last_radio.setChecked(True)
            return
        self.gray_check.setEnabled(gray)
        self.equalize_check.setEnabled(equalized)
        self.display_job.request((self.engine, mode,
                                  equalized and self.equalize_check.isChecked(),
                                  gray and self.gray_check.isChecked()))

    def metrics(self):
        if self.job.is_busy or not self._ensure_engine():
            return
        self.stopped = False
        self.engine.cancelled.clear()
        self.metric_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.progress_timer.start()
        self.job.request(self.engine)

    def _progress(self):
        if self.engine is not None:
            value, name = self.engine.progress
            self.progress_bar.setValue(value)
            self.status.setText(self.tr("Computing metrics: ") + name)

    def _metrics_ready(self, result):
        self.progress_timer.stop()
        self.cancel_button.setEnabled(False)
        self.progress_bar.hide()
        self.table_widget.setEnabled(True)
        values, errors = result['values'], result['errors']
        precision = (2, 4, 2, 4, 2, 2, 4, 4, 2, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2)
        for row, name in enumerate(computation.METRICS):
            if name in values:
                value = values[name]
                text = ('+∞ dB' if np.isposinf(value) else f'{value:.2f} dB') if name == 'psnr' else f'{value:.{precision[row]}f}'
            else:
                text = self.tr("Not available") if name in errors else ''
            item = QTableWidgetItem(text)
            if name in errors:
                item.setToolTip(errors[name])
            self.table_widget.setItem(row, 1, item)
        self.table_widget.resizeColumnsToContents()
        self.table_widget.setMinimumWidth(min(480, self.table_widget.horizontalHeader().length()
                                                + self.table_widget.verticalHeader().width() + 30))
        self.ssim_map = self.engine.maps.get('ssim')
        self.butter_map = self.engine.maps.get('butter')
        self.ssim_radio.setEnabled(self.ssim_map is not None)
        self.butter_radio.setEnabled(self.butter_map is not None)
        self.metric_button.setEnabled(result['cancelled'] or bool(errors))
        self.status.setText(self.tr("Cancelled; computed metrics retained.") if result['cancelled'] else
                            self.tr("Some metrics are unavailable; see their tooltips.") if errors else '')
        if self.ssim_radio.isChecked() or self.butter_radio.isChecked():
            self.change()

    def _failed(self, message):
        self.progress_timer.stop()
        self.progress_bar.hide()
        self.cancel_button.setEnabled(False)
        self.metric_button.setEnabled(self.reference is not None)
        self.status.setText(message)

    def cancel(self):
        self.stopped = True
        if self.engine is not None:
            self.engine.cancelled.set()
        self.job.cancel()
        self.cancel_button.setEnabled(False)
        self.status.setText(self.tr("Cancelling…"))
        self.progress_timer.stop()

    def shutdown(self):
        if self.engine is not None:
            self.engine.cancelled.set()
        self.progress_timer.stop()
        self.job.shutdown()
        super().shutdown()

    rmse = staticmethod(computation.rmse)
    mb = staticmethod(computation.mb)
    pfe = staticmethod(computation.pfe)
    psnr = staticmethod(computation.psnr)
    ssim = staticmethod(computation.ssim)
    corr = staticmethod(computation.corr)
    butter = staticmethod(computation.butter)
    ssimul = staticmethod(computation.ssimul)
