from threading import Event
from functools import partial
import cv2 as cv
import numpy as np
from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import (QLabel, QVBoxLayout, QGridLayout, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QPushButton, QProgressBar, QScrollArea, QWidget)
from gui.sherloq_app.ui.plot_canvas import FigureCanvas
from matplotlib.figure import Figure
from gui.sherloq_app.core.jpeg import DCT_SIZE, loss_curve
from gui.sherloq_app.core.jpeg_quality import QualityEngine, Cancelled
from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.ui.jobs import LatestJob
from gui.sherloq_app.core.utility import modify_font, clip_value


class _Progress(QObject):
    changed = Signal(int, str)


def _run_quality(engine, updates, cancel):
    try:
        return engine.compute(cancel.is_set, updates.changed.emit)
    except Cancelled:
        return None


class QualityWidget(ToolWidget):
    def __init__(self, filename, image, parent=None):
        super().__init__(parent)
        self.engine = QualityEngine(filename, image)
        self.cancel_event = Event()
        self.result = None
        self.status = QLabel("Computing JPEG quality…")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.process_button = QPushButton("Cancel")
        self.process_button.clicked.connect(self.process)
        self.canvas = FigureCanvas(Figure())
        self.main_layout = QVBoxLayout(self)
        self.main_layout.addWidget(self.status)
        self.main_layout.addWidget(self.progress)
        self.main_layout.addWidget(self.process_button)
        self.main_layout.addWidget(self.canvas, 1)
        self.table_area = QScrollArea()
        self.table_area.setWidgetResizable(True)
        self.table_area.setMinimumHeight(310)
        self.main_layout.addWidget(self.table_area)
        self.updates = _Progress()
        self.updates.changed.connect(self._progress)
        self.job = LatestJob(self, partial(_run_quality, self.engine, self.updates), delay=0)
        self.job.result.connect(self._result)
        self.job.failed.connect(self._error)
        self.job.request(self.cancel_event)

    def _progress(self, value, text):
        if not self.job.closed and not self.cancel_event.is_set():
            self.progress.setValue(value)
            self.status.setText(text)

    def process(self):
        if self.job.is_busy:
            self.cancel_event.set()
            self.process_button.setEnabled(False)
            self.status.setText("Cancelling after current recompressions…")
        elif self.result is None:
            self.cancel_event = Event()
            self.process_button.setText("Cancel")
            self.job.request(self.cancel_event)

    def _error(self, message):
        self.status.setText(f"JPEG quality error: {message}")
        self.process_button.setText("Retry")
        self.process_button.setEnabled(True)

    def _result(self, result):
        self.process_button.setEnabled(True)
        if result is None:
            self.status.setText("Cancelled. Completed quality levels are retained for retry.")
            self.process_button.setText("Resume")
            return
        self.result = result
        self.process_button.setVisible(False)
        self.progress.setValue(100)
        self.status.setText("JPEG quality analysis complete")
        figure = self.canvas.figure
        figure.clear()
        axes = figure.subplots()
        x = np.arange(1, 101)
        y = result['curve']
        minimum = result['minimum']
        axes.plot(x, y * 100, label="compression loss")
        axes.fill_between(x, y * 100, alpha=0.2)
        axes.axvline(minimum, linestyle=":", color="k", label=f"min error (q = {minimum})")
        axes.set_xticks(np.append(axes.get_xticks(), 1))
        axes.set_xlim([1, 100]); axes.set_ylim([0, 100])
        axes.set_xlabel(self.tr("JPEG quality (%)"))
        axes.set_ylabel(self.tr("average error (%)"))
        axes.grid(True, which="both")
        axes.legend(loc="upper center")
        figure.set_tight_layout(True)
        self.canvas.draw_idle()
        container = QWidget()
        layout = QGridLayout(container)
        quantization = result['quantization']
        if quantization is not None:
            tables, components = quantization
            for col, (key, matrix) in enumerate(tables.items()):
                role = 'Luminance' if key == components[0] else 'Chrominance' if len(components) == 3 and key in components[1:] else 'Component'
                level = (1 - (np.mean(matrix.ravel()[1:]) - 1) / 254) * 100
                suffix = f" (level = {level:.2f}%)" if np.max(matrix) <= 255 else " (16-bit entries)"
                label = QLabel(f"{role} quantization table {key}{suffix}")
                label.setAlignment(Qt.AlignCenter)
                layout.addWidget(label, 0, col)
                table = self.create_table(matrix)
                table.setMinimumWidth(300)
                table.setFixedHeight(sum(table.rowHeight(i) for i in range(8)) + table.horizontalHeader().height() + 12)
                layout.addWidget(table, 1, col)
            estimate = result['estimate']
            if estimate is None:
                text = "JPEG tables available; conventional quality estimate unavailable for these components."
            else:
                quality, deviation, _ = estimate
                detail = "standard tables" if deviation == 0 else f"nonstandard tables; deviation {deviation:.4f}"
                text = f"[JPEG] Estimated last saved quality: {quality}% ({detail})"
            note = QLabel(text)
            note.setWordWrap(True)
            layout.addWidget(note, 2, 0, 1, max(1, len(tables)))
        else:
            if result['metadata_error']:
                text = f"Quantization tables unavailable: {result['metadata_error']}"
            elif result['model_error']:
                text = f"Learned quality estimate unavailable: {result['model_error']}"
            else:
                quality = result['prediction']
                text = f"[Non-JPEG format] Model estimate of previous JPEG quality: {quality:.1f}%"
                text += " — heuristic, not proof of JPEG compression history."
            note = QLabel(text)
            note.setWordWrap(True)
            layout.addWidget(note, 0, 0)
        self.status.setText(text)
        self.status.setWordWrap(True)
        self.table_area.setWidget(container)
        self.info_message.emit(self.status.text())

    def shutdown(self):
        self.cancel_event.set()
        super().shutdown()

    @staticmethod
    def get_features(image):
        return loss_curve(image).reshape(1, 100)

    @staticmethod
    def create_table(matrix):
        table_widget = QTableWidget(DCT_SIZE, DCT_SIZE)
        hsv = np.array([[[0, 192, 255]]])
        maximum = clip_value(np.max(matrix) - 1, minv=1)
        for i in range(DCT_SIZE):
            for j in range(DCT_SIZE):
                value = matrix[i, j]
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                hsv[0, 0, 0] = 64 - 64 * ((value - 1) / maximum)
                rgb = cv.cvtColor(hsv.astype(np.uint8), cv.COLOR_HSV2RGB)
                item.setBackground(
                    QBrush(QColor(rgb[0, 0, 0], rgb[0, 0, 1], rgb[0, 0, 2]))
                )
                table_widget.setItem(i, j, item)
            table_widget.setColumnWidth(i, 32)
        table_widget.resizeRowsToContents()
        table_widget.setStyleSheet("color: black;")
        table_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        modify_font(table_widget, mono=True)
        return table_widget
