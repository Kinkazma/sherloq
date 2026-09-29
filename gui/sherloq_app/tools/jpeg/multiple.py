from gui.sherloq_app.ui.localization import t
"""Historical recompression curve plus an experimental aligned-JPEG detector."""
import csv
from threading import Event
from gui.sherloq_app.ui.plot_canvas import FigureCanvas
from matplotlib.figure import Figure
from functools import partial
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                              QProgressBar, QFileDialog, QTabWidget, QWidget)
from gui.sherloq_app.core.jpeg_curve import RecompressionCurve, Cancelled
from gui.sherloq_app.ui.jobs import LatestJob
from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.ui.double_jpeg import DoubleJpegPanel


class _Progress(QObject):
    changed = Signal(int, str)


def _run_curve(engine, updates, cancel):
    # Worker owns no QWidget: closing/reopening cannot collect GUI objects on
    # a computation thread. The progress signal object outlives active work.
    try:
        return engine.compute(cancel.is_set, updates.changed.emit)
    except Cancelled:
        return None


class MultipleWidget(ToolWidget):
    def __init__(self, image, parent=None, filename=None):
        super().__init__(parent)
        self.engine = RecompressionCurve(image, qualities=range(101))
        self.cancel_event = Event()
        self.values = None
        note = QLabel("Historical recompression curve. This curve alone does not determine how many JPEG compressions occurred.")
        note.setWordWrap(True)
        self.status_label = QLabel("Computing recompression losses…")
        self.progress = QProgressBar()
        self.progress.setRange(0, 101)
        self.process_button = QPushButton("Cancel")
        self.process_button.clicked.connect(self.process)
        self.export_button = QPushButton("Export CSV")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_csv)
        self.image_button = QPushButton("Export PNG")
        self.image_button.setEnabled(False)
        self.image_button.clicked.connect(self.export_png)
        self.figure = Figure(tight_layout=True)
        self.chart_view = FigureCanvas(self.figure)
        self.chart_view.setMinimumSize(600, 400)
        self.axes = self.figure.subplots()
        self.axes.set_title("Mean absolute error after JPEG recompression")
        self.axes.set_xlabel("JPEG quality (0 uses the codec's minimum quality)")
        self.axes.set_ylabel("Mean absolute pixel error (0–255)")
        self.axes.set_xlim(0, 100)
        self.axes.set_ylim(0, 1)
        self.axes.grid(True)
        self.line, = self.axes.plot([], [])
        controls = QHBoxLayout()
        for button in (self.process_button, self.export_button, self.image_button):
            controls.addWidget(button)
        controls.addStretch()
        outer = QVBoxLayout(self)
        self.tabs = QTabWidget()
        curve = QWidget()
        self.detector = DoubleJpegPanel(filename, self)
        self.tabs.addTab(curve, "Recompression curve")
        self.tabs.addTab(self.detector, "Double JPEG detection")
        from gui.sherloq_app.ui.extensions import mark_tab
        mark_tab(self.tabs,1)
        outer.addWidget(self.tabs)
        layout = QVBoxLayout(curve)
        layout.addWidget(note)
        layout.addWidget(self.chart_view, 1)
        layout.addWidget(self.progress)
        layout.addWidget(self.status_label)
        layout.addLayout(controls)
        self.updates = _Progress()
        self.updates.changed.connect(self._progress)
        self.job = LatestJob(self, partial(_run_curve, self.engine, self.updates), delay=0)
        self.job.result.connect(self.show_result)
        self.job.failed.connect(self.show_error)
        self.job.request(self.cancel_event)

    def _progress(self, value, text):
        if not self.job.closed and not self.cancel_event.is_set():
            self.progress.setValue(value)
            self.status_label.setText(text)

    def process(self):
        if self.job.is_busy:
            self.cancel_event.set()
            self.process_button.setEnabled(False)
            self.status_label.setText("Cancelling after active recompressions…")
        elif self.values is None:
            self.cancel_event = Event()
            self.process_button.setText("Cancel")
            self.job.request(self.cancel_event)

    def show_result(self, values):
        self.process_button.setEnabled(True)
        if values is None:
            self.process_button.setText("Resume")
            self.status_label.setText("Cancelled. Completed qualities are retained.")
            return
        self.values = values
        self.line.set_data(self.engine.qualities, values)
        self.axes.set_ylim(0, max(1., float(values.max()) * 1.05))
        self.chart_view.draw_idle()
        self.progress.setValue(101)
        self.status_label.setText(f"101 quality levels — {self.job.seconds:.3f} s. Loss values are raw pixel differences, not percentages.")
        self.export_button.setEnabled(True)
        self.image_button.setEnabled(True)
        self.process_button.hide()
        self.info_message.emit(self.status_label.text())

    def show_error(self, message):
        self.process_button.setEnabled(True)
        self.process_button.setText("Retry")
        self.status_label.setText(message)

    def export_csv(self):
        if self.values is None or self.job.is_busy:
            return
        path, _ = QFileDialog.getSaveFileName(self, t("Export Recompression Curve"), "recompression.csv", t("CSV (*.csv)"))
        if path:
            try:
                with open(path, 'w', newline='', encoding='utf-8') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['jpeg_quality', 'mean_absolute_pixel_error_0_255'])
                    writer.writerows(zip(self.engine.qualities, map(float, self.values)))
            except OSError as exc:
                self.status_label.setText(f"Export failed: {exc}")

    def export_png(self):
        if self.values is None or self.job.is_busy:
            return
        path, _ = QFileDialog.getSaveFileName(self, t("Export Recompression Plot"), "recompression.png", t("PNG (*.png)"))
        if path and not self.chart_view.grab().save(path, 'PNG'):
            self.status_label.setText("PNG export failed")

    def shutdown(self):
        self.cancel_event.set()
        self.detector.shutdown()
        super().shutdown()
