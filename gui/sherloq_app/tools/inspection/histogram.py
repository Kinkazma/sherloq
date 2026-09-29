import numpy as np
from PySide6.QtGui import QColor, QBrush
from PySide6.QtCore import QTimer
from gui.sherloq_app.ui.jobs import LatestJob
from gui.sherloq_app.core.histogram import analyze_histogram
from PySide6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QGridLayout,
    QSplitter,
    QLabel,
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QRadioButton,
)
from gui.sherloq_app.ui.plot_canvas import FigureCanvas
from matplotlib.figure import Figure

from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.core.utility import modify_font, ParamSlider


class HistWidget(ToolWidget):
    def __init__(self, image, parent=None):
        super(ToolWidget, self).__init__(parent)

        self.value_radio = QRadioButton(self.tr("Value"))
        self.value_radio.setChecked(True)
        self.last_radio = self.value_radio
        self.red_radio = QRadioButton(self.tr("Red"))
        self.green_radio = QRadioButton(self.tr("Green"))
        self.blue_radio = QRadioButton(self.tr("Blue"))
        self.rgb_radio = QRadioButton(self.tr("RGB"))
        self.smooth_check = QCheckBox(self.tr("Smooth line"))
        self.smooth_check.setToolTip(self.tr("Interpolated values plot"))
        self.log_check = QCheckBox(self.tr("Log scale"))
        self.log_check.setToolTip(self.tr("Y-axes logarithmic scale"))
        self.grid_check = QCheckBox(self.tr("Show grid"))
        self.grid_check.setToolTip(self.tr("Display XY main grid lines"))
        self.marker_check = QCheckBox(self.tr("Show markers"))
        self.marker_check.setToolTip(
            self.tr("Show plot markers for min(--), avg(-), max(-.)")
        )
        self.start_slider = ParamSlider([0, 255], 8, 0, bold=True)
        self.end_slider = ParamSlider([0, 255], 8, 255, bold=True)

        self.hist = None
        self.unique_colors = self.unique_ratio = None
        self.status = QLabel(self.tr("Computing histograms and unique colours…"))
        self.redraw_timer = QTimer(self)
        self.redraw_timer.setSingleShot(True)
        self.redraw_timer.setInterval(40)
        self.redraw_timer.timeout.connect(self._redraw)
        self.job = LatestJob(self, lambda _: analyze_histogram(image), delay=0)
        self.job.result.connect(self._computed)
        self.job.failed.connect(self.status.setText)

        self.value_radio.clicked.connect(self.redraw)
        self.red_radio.clicked.connect(self.redraw)
        self.green_radio.clicked.connect(self.redraw)
        self.blue_radio.clicked.connect(self.redraw)
        self.rgb_radio.clicked.connect(self.redraw)
        self.smooth_check.stateChanged.connect(self.redraw)
        self.log_check.stateChanged.connect(self.redraw)
        self.grid_check.stateChanged.connect(self.redraw)
        self.marker_check.stateChanged.connect(self.redraw)
        self.start_slider.valueChanged.connect(self.redraw)
        self.end_slider.valueChanged.connect(self.redraw)

        self.table_widget = QTableWidget(13, 2)
        self.table_widget.setHorizontalHeaderLabels(
            [self.tr("Property"), self.tr("Value")]
        )
        self.table_widget.setItem(0, 0, QTableWidgetItem(self.tr("Least frequent")))
        self.table_widget.item(0, 0).setToolTip(self.tr("Value that appears less"))
        self.table_widget.setItem(1, 0, QTableWidgetItem(self.tr("Most frequent")))
        self.table_widget.item(1, 0).setToolTip(self.tr("Value that appears more"))
        self.table_widget.setItem(2, 0, QTableWidgetItem(self.tr("Average level")))
        self.table_widget.item(2, 0).setToolTip(self.tr("Histogram mean value"))
        self.table_widget.setItem(3, 0, QTableWidgetItem(self.tr("Median level")))
        self.table_widget.item(3, 0).setToolTip(self.tr("Histogram median value"))
        self.table_widget.setItem(4, 0, QTableWidgetItem(self.tr("Deviation")))
        self.table_widget.item(4, 0).setToolTip(self.tr("Histogram standard deviation"))
        self.table_widget.setItem(5, 0, QTableWidgetItem(self.tr("Pixel count")))
        self.table_widget.item(5, 0).setToolTip(
            self.tr("Total values in current range")
        )
        self.table_widget.setItem(6, 0, QTableWidgetItem(self.tr("Percentile")))
        self.table_widget.item(6, 0).setToolTip(self.tr("Percentage of total pixels"))
        self.table_widget.setItem(7, 0, QTableWidgetItem(self.tr("Nonzero range")))
        self.table_widget.item(7, 0).setToolTip(
            self.tr("Minimal range without empty bins")
        )
        self.table_widget.setItem(8, 0, QTableWidgetItem(self.tr("Empty bins")))
        self.table_widget.item(8, 0).setToolTip(self.tr("Number of missing values"))
        self.table_widget.setItem(9, 0, QTableWidgetItem(self.tr("Unique colors")))
        self.table_widget.item(9, 0).setToolTip(self.tr("Unique RGB color count"))
        self.table_widget.setItem(10, 0, QTableWidgetItem(self.tr("Unique ratio")))
        self.table_widget.item(10, 0).setToolTip(
            self.tr("Unique colors vs total pixels")
        )
        self.table_widget.setItem(11, 0, QTableWidgetItem(self.tr("Smoothness")))
        self.table_widget.item(11, 0).setToolTip(
            self.tr("Estimated correlation among bin values")
        )
        self.table_widget.setItem(12, 0, QTableWidgetItem(self.tr("Fullness")))
        self.table_widget.item(12, 0).setToolTip(self.tr("Area covered vs total size"))
        for i in range(self.table_widget.rowCount()):
            modify_font(self.table_widget.item(i, 0), bold=True)
        self.table_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_widget.setAlternatingRowColors(True)
        self.table_widget.setMinimumWidth(200)
        self.table_widget.resizeColumnsToContents()

        figure = Figure()
        plot_canvas = FigureCanvas(figure)
        self.axes = plot_canvas.figure.subplots()
        figure.set_tight_layout(True)

        range_layout = QGridLayout()
        range_layout.addWidget(QLabel(self.tr("Start:")), 0, 0)
        range_layout.addWidget(self.start_slider, 0, 1)
        range_layout.addWidget(QLabel(self.tr("End:")), 1, 0)
        range_layout.addWidget(self.end_slider, 1, 1)

        right_frame = QFrame()
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.table_widget)
        right_layout.addLayout(range_layout)
        right_frame.setLayout(right_layout)

        center_split = QSplitter()
        center_split.addWidget(plot_canvas)
        center_split.addWidget(right_frame)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(QLabel(self.tr("Channel:")))
        bottom_layout.addWidget(self.value_radio)
        bottom_layout.addWidget(self.red_radio)
        bottom_layout.addWidget(self.green_radio)
        bottom_layout.addWidget(self.blue_radio)
        bottom_layout.addWidget(self.rgb_radio)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.smooth_check)
        bottom_layout.addWidget(self.log_check)
        bottom_layout.addWidget(self.grid_check)
        bottom_layout.addWidget(self.marker_check)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.status)
        main_layout.addWidget(center_split, 1)
        main_layout.addLayout(bottom_layout)
        self.setLayout(main_layout)
        self.job.request(True)

    def _computed(self, result):
        self.hist, self.unique_colors, self.unique_ratio = result
        self.status.setText("")
        self._redraw()

    def redraw(self):
        if self.hist is not None:
            self.redraw_timer.start()

    def shutdown(self):
        self.redraw_timer.stop()
        super().shutdown()

    def _redraw(self):
        if self.hist is None:
            return
        x = np.arange(256)
        alpha = 0.25
        rgb = self.rgb_radio.isChecked()
        red = self.red_radio.isChecked()
        green = self.green_radio.isChecked()
        blue = self.blue_radio.isChecked()
        value = self.value_radio.isChecked()
        smoothness = self.smooth_check.isChecked()
        grid = self.grid_check.isChecked()
        log = self.log_check.isChecked()
        try:
            self.axes.clear()
        except RecursionError:
            return
        y = None
        step = None if smoothness else "mid"
        if value:
            y = self.hist[3]
            if smoothness:
                self.axes.plot(x, y, "k")
            else:
                self.axes.step(x, y, "k", where="mid")
            self.axes.fill_between(x, y, alpha=alpha, facecolor="k", step=step)
        else:
            if red or rgb:
                y = self.hist[0]
                if smoothness:
                    self.axes.plot(x, y, "r")
                else:
                    self.axes.step(x, y, "r", where="mid")
                self.axes.fill_between(x, y, alpha=alpha, facecolor="r", step=step)
            if green or rgb:
                y = self.hist[1]
                if smoothness:
                    self.axes.plot(x, y, "g")
                else:
                    self.axes.step(x, y, "g", where="mid")
                self.axes.fill_between(x, y, alpha=alpha, facecolor="g", step=step)
            if blue or rgb:
                y = self.hist[2]
                if smoothness:
                    self.axes.plot(x, y, "b")
                else:
                    self.axes.step(x, y, "b", where="mid")
                self.axes.fill_between(x, y, alpha=alpha, facecolor="b", step=step)
        if log:
            self.axes.set_yscale("log")
            self.axes.set_ylim(bottom=1)
        else:
            self.axes.set_yscale("linear")
            self.axes.set_ylim(bottom=0)
        self.axes.set_xlim([0, 255])
        self.axes.set_xlabel(self.tr("intensity value"))
        self.axes.set_ylabel(self.tr("pixel count"))
        self.axes.set_xticks([0, 64, 128, 192, 255])
        self.axes.grid(grid, which="both")

        if rgb:
            self.table_widget.setEnabled(False)
            self.marker_check.setEnabled(False)
            self.start_slider.setEnabled(False)
            self.end_slider.setEnabled(False)
            for i in range(self.table_widget.rowCount()):
                if self.table_widget.item(i, 1) is not None:
                    self.table_widget.item(i, 1).setText("")
                    self.table_widget.item(i, 1).setBackground(QBrush(QColor("white")))
        else:
            self.table_widget.setEnabled(True)
            self.marker_check.setEnabled(True)
            self.start_slider.setEnabled(True)
            self.end_slider.setEnabled(True)
            start = self.start_slider.value()
            end = self.end_slider.value()
            start, end = sorted((start, end))
            total = np.sum(y)
            x = x[start : end + 1]
            y = y[start : end + 1]
            count = np.sum(y)
            if count != 0:
                argmin = np.argmin(y) + start
                argmax = np.argmax(y) + start
                mean = np.round(np.sum(x * y) / count, 2)
                stddev = np.round(np.sqrt(np.sum(((x - mean) ** 2) * y) / count), 2)
                median = np.argmax(np.cumsum(y) > count / 2) + start
                percent = np.round(count / total * 100, 2)
                empty = len(x) - np.count_nonzero(y)
                nonzero = [np.nonzero(y)[0][0] + start, np.nonzero(y)[0][-1] + start]
                fullness = np.round(count / (len(y) * np.max(y)) * 100, 2)
                y = y / np.max(y)
                sweep = len(y)
                smoothness = 0
                if sweep >= 5:
                    for i in range(2, sweep - 2):
                        yl = 2 * y[i - 1] - y[i - 2]
                        yr = 2 * y[i + 1] - y[i + 2]
                        smoothness += abs(y[i] - (yl + yr) / 2)
                    smoothness = np.round((1 - (smoothness / (sweep - 2))) * 100, 2)
                if self.marker_check.isChecked():
                    self.axes.axvline(argmin, linestyle="--", color="m")
                    self.axes.axvline(mean, linestyle="-", color="m")
                    self.axes.axvline(argmax, linestyle="-.", color="m")
                    self.axes.axvline(median, linestyle=":", color="m")
            else:
                argmin = (
                    argmax
                ) = (
                    mean
                ) = (
                    stddev
                ) = median = percent = smoothness = fullness = 0
                empty = len(x)
                nonzero = []

            self.table_widget.setItem(0, 1, QTableWidgetItem(str(argmin)))
            self.table_widget.setItem(1, 1, QTableWidgetItem(str(argmax)))
            self.table_widget.setItem(2, 1, QTableWidgetItem(str(mean)))
            self.table_widget.setItem(3, 1, QTableWidgetItem(str(median)))
            self.table_widget.setItem(4, 1, QTableWidgetItem(str(stddev)))
            self.table_widget.setItem(5, 1, QTableWidgetItem(str(count)))
            self.table_widget.setItem(6, 1, QTableWidgetItem(str(percent) + "%"))
            self.table_widget.setItem(7, 1, QTableWidgetItem(str(nonzero)))
            self.table_widget.setItem(8, 1, QTableWidgetItem(str(empty)))
            self.table_widget.setItem(9, 1, QTableWidgetItem(str(self.unique_colors)))
            self.table_widget.setItem(
                10, 1, QTableWidgetItem(str(self.unique_ratio) + "%")
            )
            # color_by_value(
            #     self.table_widget.item(10, 1), self.unique_ratio, [25, 50, 75]
            # )
            self.table_widget.setItem(11, 1, QTableWidgetItem(str(smoothness) + "%"))
            # color_by_value(self.table_widget.item(11, 1), smoothness, [80, 90, 95])
            self.table_widget.setItem(12, 1, QTableWidgetItem(str(fullness) + "%"))
            # color_by_value(self.table_widget.item(12, 1), fullness, [5, 10, 20])
            self.table_widget.resizeColumnsToContents()
            if start != 0 or end != 255:
                self.axes.axvline(start, linestyle=":", color="k")
                self.axes.axvline(end, linestyle=":", color="k")
                _, top = self.axes.get_ylim()
                self.axes.fill_between(
                    np.arange(start, end + 1), top, facecolor="y", alpha=alpha * 2
                )
        self.axes.figure.canvas.draw_idle()
