from gui.sherloq_app.core.pca import PcaEngine
from gui.sherloq_app.ui.jobs import LatestJob
from PySide6.QtWidgets import (
    QAbstractItemView,
    QTableWidgetItem,
    QTableWidget,
    QVBoxLayout,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QCheckBox,
)

from gui.sherloq_app.ui.tools import ToolWidget
from gui.sherloq_app.core.utility import modify_font
from gui.sherloq_app.ui.viewer import ImageViewer


class PcaWidget(ToolWidget):
    def __init__(self, image, parent=None):
        super(PcaWidget, self).__init__(parent)

        self.component_combo = QComboBox()
        self.component_combo.addItems([self.tr(f"#{i + 1}") for i in range(3)])
        self.distance_radio = QRadioButton(self.tr("Distance"))
        self.distance_radio.setToolTip(
            self.tr("Distance from the closest point on selected component")
        )
        self.project_radio = QRadioButton(self.tr("Projection"))
        self.project_radio.setToolTip(
            self.tr("Projection onto the selected principal component")
        )
        self.crossprod_radio = QRadioButton(self.tr("Cross product"))
        self.crossprod_radio.setToolTip(
            self.tr("Cross product between input and selected component")
        )
        self.distance_radio.setChecked(True)
        self.last_radio = self.distance_radio
        self.invert_check = QCheckBox(self.tr("Invert"))
        self.invert_check.setToolTip(self.tr("Output bitwise complement"))
        self.equalize_check = QCheckBox(self.tr("Equalize"))
        self.equalize_check.setToolTip(self.tr("Apply histogram equalization"))

        table_widget = self.table_widget = QTableWidget(4, 5)
        table_widget.setHorizontalHeaderLabels(
            [self.tr("Element"), self.tr("Red"), self.tr("Green"), self.tr("Blue"), self.tr("Eigenvalue")]
        )
        for row, title in enumerate([self.tr("Mean vector"), self.tr("Eigenvector 1"),
                                     self.tr("Eigenvector 2"), self.tr("Eigenvector 3")]):
            table_widget.setItem(row, 0, QTableWidgetItem(title))
            modify_font(table_widget.item(row, 0), bold=True)
        self._model_displayed = False
        table_widget.resizeColumnsToContents()
        table_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        table_widget.setMaximumHeight(190)

        self.viewer = ImageViewer(image, image, None)
        self.engine = PcaEngine(image)
        self.job = LatestJob(self, self.engine.compute, delay=0)
        self.status_label = QLabel()
        self.job.result.connect(self.show_result)
        self.job.busy.connect(self.set_busy)
        self.job.failed.connect(self.show_error)
        self._requested = None
        self.process()

        self.component_combo.currentIndexChanged.connect(self.process)
        self.distance_radio.clicked.connect(self.process)
        self.project_radio.clicked.connect(self.process)
        self.crossprod_radio.clicked.connect(self.process)
        self.invert_check.stateChanged.connect(self.process)
        self.equalize_check.stateChanged.connect(self.process)

        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel(self.tr("Component:")))
        top_layout.addWidget(self.component_combo)
        top_layout.addWidget(QLabel(self.tr("Mode:")))
        top_layout.addWidget(self.distance_radio)
        top_layout.addWidget(self.project_radio)
        top_layout.addWidget(self.crossprod_radio)
        top_layout.addWidget(self.invert_check)
        top_layout.addWidget(self.equalize_check)
        top_layout.addWidget(self.status_label)
        top_layout.addStretch()
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(table_widget)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.viewer)
        main_layout.addLayout(bottom_layout)
        self.setLayout(main_layout)

    def process(self):
        mode = next((name for name in ('distance', 'project', 'crossprod')
                     if getattr(self, name+'_radio').isChecked()), None)
        if mode is None:
            self.last_radio.setChecked(True)
            return
        self.last_radio = getattr(self, mode+'_radio')
        params = (self.component_combo.currentIndex(), mode,
                  self.invert_check.isChecked(), self.equalize_check.isChecked())
        if params != self._requested:
            self._requested = params
            self.job.request(params)

    def set_busy(self, busy):
        self.viewer.set_busy(busy)
        if busy:
            self.status_label.setText(self.tr("Calculating…"))

    def show_result(self, result):
        image, (mean, vectors, values) = result
        self.viewer.update_processed(image)
        if not self._model_displayed:
            table = self.table_widget
            for row, vector in enumerate((mean[0], *vectors)):
                for column, value in enumerate(vector[::-1], 1):
                    table.setItem(row, column, QTableWidgetItem(str(value)))
                if row:
                    table.setItem(row, 4, QTableWidgetItem(str(values[row-1, 0])))
            table.resizeColumnsToContents()
            self._model_displayed = True
        self.status_label.setText(f"{self.job.seconds:.3f} s")

    def show_error(self, error):
        self._requested = None
        self.status_label.setText(error)
