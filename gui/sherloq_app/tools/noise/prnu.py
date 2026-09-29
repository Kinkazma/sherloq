from gui.sherloq_app.ui.localization import t
import os
from pathlib import Path

from PySide6.QtCore import QThread, Qt, Signal, QTimer, Slot
from gui.sherloq_app.ui.heavy_jobs import HEAVY_JOBS
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
import numpy as np
import h5py

from gui.sherloq_app.ui.tools import ToolWidget

from threading import Event
from PySide6.QtCore import QStandardPaths, QCoreApplication
from gui.sherloq_app.core.prnu import (
    PrnuEngine, Cancelled, validate_database, NCC_THRESHOLD,
    parse_camera_label, scan_dataset, extract_residual, load_image_gray, ncc,
)

# Keep a cancelled thread alive until its native computation returns. QWidget
# closure never destroys a running QThread or blocks the GUI on wait().
_ACTIVE_WORKERS = set()
_QUIT_HOOK_INSTALLED = False


def _finish_workers_at_exit():
    workers = list(_ACTIVE_WORKERS)
    for worker in workers:
        worker.request_cancel()
    for worker in workers:
        worker.wait()



class PrnuWorker(QThread):
    progress = Signal(int, str)
    results = Signal(list)
    error = Signal(str)
    cancelled = Signal()

    def __init__(self, engine, path, folder=None, label=None):
        super().__init__()
        self.engine, self.path, self.folder, self.label = engine, path, folder, label
        self.cancel_event = Event()
        self.admission_pending = False
        self.finished.connect(self._release_lane)

    def start(self, priority=QThread.InheritPriority):
        self.priority_requested = priority
        self.admission_pending = True
        self.progress.emit(0, 'Waiting for another analysis…')
        HEAVY_JOBS.request(self)

    def _start_admitted(self):
        self.admission_pending = False
        super().start(self.priority_requested)

    @Slot()
    def _release_lane(self):
        HEAVY_JOBS.release(self)

    def _cancel_queued(self):
        self.cancelled.emit()
        self.finished.emit()

    def request_cancel(self):
        self.cancel_event.set()
        if self.admission_pending:
            self.admission_pending = False
            HEAVY_JOBS.release(self)
            QTimer.singleShot(0, self._cancel_queued)

    def run(self):
        try:
            results = self.engine.identify(self.path, self.folder, self.label,
                                           self.progress.emit, self.cancel_event.is_set)
            self.results.emit(results)
        except Cancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.error.emit(str(exc))


class PrnuWidget(ToolWidget):
    def __init__(self, filename: str, image: np.ndarray, parent=None):
        super().__init__(parent)
        global _QUIT_HOOK_INSTALLED
        if not _QUIT_HOOK_INSTALLED:
            QCoreApplication.instance().aboutToQuit.connect(_finish_workers_at_exit)
            _QUIT_HOOK_INSTALLED = True
        self.filename = filename
        self.image = image
        self.hdf5_path = os.path.join(
            QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation),
            "fingerprints", "prnu.h5")
        self.engine = PrnuEngine(image, filename)
        self._closed = False
        self._busy = False
        self.dataset_dir = None
        self.worker = None
        self._on_start()
        self._check_hdf5()

    def _on_start(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        hdf5_layout = QHBoxLayout()
        self.hdf5_label = QLabel(self.hdf5_path)
        self.hdf5_label.setToolTip(self.hdf5_path)
        self.hdf5_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        browse_h5_button = self.browse_h5_button = QPushButton("Browse")
        browse_h5_button.setFixedWidth(90)
        browse_h5_button.clicked.connect(self._browse_hdf5)
        hdf5_layout.addWidget(QLabel("Fingerprints (.h5):"))
        hdf5_layout.addWidget(self.hdf5_label)
        hdf5_layout.addWidget(browse_h5_button)
        self.new_database_button = QPushButton("New database")
        self.new_database_button.clicked.connect(self._new_database)
        hdf5_layout.addWidget(self.new_database_button)

        self.stack = QStackedWidget()
        self.stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.status_label = QLabel("Ready.")

        # index 0 (h5 found)
        found_widget = QWidget()
        found_layout = QVBoxLayout(found_widget)
        found_layout.setContentsMargins(0, 0, 0, 0)
        self.found_label = QLabel()
        self.found_label.setWordWrap(True)
        self.found_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.found_label.setStyleSheet("color: #1a7f1a; font-weight: bold;")
        self.identify_button = QPushButton("Identify Camera Source")
        self.identify_button.setFixedHeight(34)
        self.identify_button.clicked.connect(self._identify)

        found_layout.addWidget(self.found_label)
        found_layout.addWidget(self.identify_button)

        # index 1 (there is no h5)
        not_found_widget = QWidget()
        nf_layout = QVBoxLayout(not_found_widget)
        nf_layout.setContentsMargins(0, 0, 0, 0)
        default_label = QLabel(
            "No fingerprints database found.\n"
            "Select your camera image dataset folder to generate fingerprints, then identification runs automatically."
        )
        default_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        default_label.setStyleSheet("color: #b85c00;")

        dataset_layout = QHBoxLayout()
        self.dataset_label = QLabel("No folder selected")
        self.dataset_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        browse_dataset_button = self.browse_dataset_button = QPushButton("Select Dataset")
        browse_dataset_button.setFixedWidth(130)
        browse_dataset_button.clicked.connect(self._browse_dataset)
        dataset_layout.addWidget(QLabel("Dataset folder:"))
        dataset_layout.addWidget(self.dataset_label)
        dataset_layout.addWidget(browse_dataset_button)

        gen_fp_layout = QHBoxLayout()
        self.gen_fp_button = QPushButton("Generate Fingerprints And Identify")
        self.gen_fp_button.setFixedHeight(34)
        self.gen_fp_button.setEnabled(False)
        self.gen_fp_button.clicked.connect(self._identify)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setFixedHeight(34)
        self.cancel_button.setFixedWidth(100)
        self.cancel_button.setEnabled(False)
        self.cancel_button.setStyleSheet("""
            QPushButton          { color: #aaaaaa; font-weight: bold;
                                   background-color: #e8e8e8;
                                   border: 1px solid #cccccc; border-radius: 4px; }
            QPushButton:enabled  { color: #c0392b; background-color: #fdecea;
                                   border: 1px solid #e74c3c; }
            QPushButton:enabled:hover   { background-color: #f5b7b1; }
            QPushButton:enabled:pressed { background-color: #e74c3c; color: white; }
        """)
        self.cancel_button.clicked.connect(self._cancel)
        gen_fp_layout.addWidget(self.gen_fp_button)


        nf_layout.addWidget(default_label)
        nf_layout.addLayout(dataset_layout)
        self.camera_label = QLineEdit()
        self.camera_label.setPlaceholderText("Single-camera folder: label (e.g. Sony RX10M4)")
        self.camera_label.setToolTip("Fill only if ALL JPEGs in this folder and its subfolders come from the same physical camera. Leave empty for historical filename grouping (camera_name_number.jpg).")
        nf_layout.addWidget(self.camera_label)
        nf_layout.addLayout(gen_fp_layout)

        self.stack.addWidget(found_widget)
        self.stack.addWidget(not_found_widget)

        # result of identification
        res_box = QGroupBox("Identification Results")
        res_layout = QVBoxLayout(res_box)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["Rank", "Camera", "NCC Score", "Score gap"]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        res_layout.addWidget(self.table)

        self.verdict = QLabel("")
        self.verdict.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vf = QFont()
        vf.setPointSize(10)
        vf.setBold(True)
        self.verdict.setFont(vf)

        main_layout.addLayout(hdf5_layout)
        main_layout.addWidget(self.hline())
        main_layout.addWidget(self.stack)
        main_layout.addWidget(self.cancel_button)
        note = QLabel("Experimental Wiener/NCC method. Scores and the 0.005 threshold are not calibrated probabilities. Use independent test photos with matching orientation, size and framing.")
        note.setWordWrap(True)
        main_layout.addWidget(note)
        main_layout.addWidget(self.progress)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(res_box)
        main_layout.addWidget(self.verdict)

    def _new_database(self):
        path, _ = QFileDialog.getSaveFileName(self, t("Create a New Training Snapshot"),
                                             os.path.dirname(self.hdf5_path), t("HDF5 Files (*.h5)"))
        if not path:
            return
        if not path.lower().endswith('.h5'):
            path += '.h5'
        if os.path.exists(path):
            QMessageBox.warning(self, t("Choose a new file"), t("Existing databases are preserved. Choose a different filename."))
            return
        self.hdf5_path = path
        self.hdf5_label.setText(path)
        self.hdf5_label.setToolTip(path)
        self.table.setRowCount(0)
        self.verdict.setText("")
        self._check_hdf5()

    def _browse_hdf5(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            t("Select Fingerprints File"),
            os.path.dirname(self.hdf5_path),
            t("HDF5 Files (*.h5)"),
        )

        if path:
            self.hdf5_path = path
            self.hdf5_label.setText(path)
            self.hdf5_label.setToolTip(path)
            self._check_hdf5()

    def _check_hdf5(self):
        if os.path.exists(self.hdf5_path):
            try:
                with h5py.File(self.hdf5_path, "r") as f:
                    n = len(validate_database(f))
                    snapshot = "Training snapshot" if f.attrs.get('schema') else "Legacy database: completeness and training membership unverified"
                self.found_label.setText(
                    f"{n} camera fingerprint(s) ready — {snapshot}\n{self.hdf5_path}"
                )
            except Exception as exc:
                self.found_label.setText(f"Invalid database: {exc}")
                self.identify_button.setEnabled(False)
                self.stack.setCurrentIndex(0)
                return
            self.identify_button.setEnabled(True)
            self.stack.setCurrentIndex(0)
            self.status_label.setText("Fingerprints loaded. Press Identify to run.")
        else:
            self.stack.setCurrentIndex(1)
            self.status_label.setText("Choose a training folder for the new snapshot.")

    def _identify(self):
        if self._busy or self._closed:
            return
        if self.image is None:
            QMessageBox.warning(self, t("No Image"), t("No Image loaded in Sherloq"))
            return
        if not os.path.exists(self.hdf5_path) and not self.dataset_dir:
            QMessageBox.warning(
                self,
                t("Dataset required"),
                t("No fingerprints file and no dataset folder selected"),
            )
            return

        self._set_busy(True)
        self.table.setRowCount(0)
        self.verdict.setText("")
        self.progress.setValue(0)
        self.info_message.emit("PRNU identification is running ...")

        worker = PrnuWorker(self.engine, self.hdf5_path, self.dataset_dir,
                            self.camera_label.text().strip() or None)
        self.worker = worker
        _ACTIVE_WORKERS.add(worker)
        worker.finished.connect(self._thread_finished)
        worker.finished.connect(lambda: _ACTIVE_WORKERS.discard(worker))
        worker.finished.connect(worker.deleteLater)
        worker.progress.connect(self._on_progress)
        worker.results.connect(self._on_finished)
        worker.error.connect(self._on_error)
        worker.cancelled.connect(self._on_cancelled)
        worker.start()

    def _set_busy(self, busy: bool):
        """Toggle buttons correctly for running vs idle state"""
        self._busy = busy
        self.browse_h5_button.setEnabled(not busy)
        self.new_database_button.setEnabled(not busy)
        self.browse_dataset_button.setEnabled(not busy)
        self.camera_label.setEnabled(not busy)
        self.identify_button.setEnabled(not busy)
        self.gen_fp_button.setEnabled(not busy and bool(self.dataset_dir))
        self.cancel_button.setEnabled(busy)

    def _browse_dataset(self):
        folder = QFileDialog.getExistingDirectory(
            self, t("Select Camera Image Dataset Folder"), os.path.expanduser("~")
        )
        if folder:
            self.dataset_dir = folder
            self.dataset_label.setText(folder)
            self.dataset_label.setToolTip(folder)
            self.gen_fp_button.setEnabled(True)

    def _cancel(self):
        """Request cancellation, worker stops at next image boundary."""
        if self.worker is not None:
            self.cancel_button.setEnabled(False)
            self.status_label.setText("Cancelling... finishing current image...")
            self.worker.request_cancel()

    def _on_progress(self, pct, msg):
        if self._closed:
            return
        self.progress.setValue(pct)
        self.status_label.setText(msg)

    def _on_finished(self, res):
        if self._closed:
            return
        self.progress.setValue(100)
        if self.stack.currentIndex() == 1 and os.path.exists(self.hdf5_path):
            self._check_hdf5()

        if not res:
            self.status_label.setText("No results.")
            return

        top = res[0][1]
        sec = res[1][1] if len(res) > 1 else 0.0
        margin = top - sec

        self.table.setRowCount(len(res))
        for row, (cam, score) in enumerate(res):
            conf = f"{margin:.5f}" if row == 0 and len(res) > 1 else ""
            items = [
                QTableWidgetItem(str(row + 1)),
                QTableWidgetItem(cam),
                QTableWidgetItem(f"{score:.5f}"),
                QTableWidgetItem(conf),
            ]
            for item in items:
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if row == 0:
                for item in items:
                    item.setBackground(QColor(225, 235, 250))
            for col, item in enumerate(items):
                self.table.setItem(row, col, item)

        if top < NCC_THRESHOLD:
            self.verdict.setText("No candidate exceeds the experimental threshold")
            self.verdict.setStyleSheet("color: #8e44ad; font-weight: bold;")
            self.status_label.setText(
                "The source camera may not be in the fingerprint database."
            )
            self.info_message.emit("PRNU: score below the experimental threshold.")
            return

        self.verdict.setText(f"Highest NCC candidate:  {res[0][0]} (uncalibrated)")
        self.verdict.setStyleSheet(
            "color: #1a7f1a;" if margin > 0.01 else "color: #c47a00;"
        )
        self.status_label.setText(f"Matched {len(res)} camera(s)")
        self.info_message.emit(f"PRNU: predicted camera -> {res[0][0]}")

    def _thread_finished(self):
        self.worker = None
        self._set_busy(False)

    def _on_cancelled(self):
        if self._closed:
            return
        self.progress.setValue(0)
        self.table.setRowCount(0)
        self.verdict.setText("")
        self._check_hdf5()
        self.status_label.setText("Cancelled. Existing databases are preserved.")

    def shutdown(self):
        self._closed = True
        if self.worker is not None:
            self.worker.request_cancel()
            for signal in (self.worker.progress, self.worker.results,
                           self.worker.error, self.worker.cancelled):
                signal.disconnect()
            self.worker.finished.disconnect(self._thread_finished)
            self.worker = None
        super().shutdown()

    def _on_error(self, msg):
        if self._closed:
            return
        self.progress.setValue(0)
        self.status_label.setText(f"Error: {msg}")
        QMessageBox.critical(self, t("PRNU Error"), t(msg))
        self.info_message.emit("PRNU identification failed.")

    @staticmethod
    def hline():
        """
        Create and return a horizontal separator line widget.
        """
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line
