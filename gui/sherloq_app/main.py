from gui.sherloq_app.ui.localization import t
import os
import sys

from PySide6.QtCore import Qt, QSettings, Signal, QTimer
from PySide6.QtGui import QKeySequence, QAction, QPainter, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMdiArea,
    QMdiSubWindow,
    QDockWidget,
    QMessageBox,
    QPushButton,
    QLabel,
    QProgressBar,
)

from gui.sherloq_app.tools.colors.pca import PcaWidget
from gui.sherloq_app.tools.colors.plots import PlotsWidget
from gui.sherloq_app.tools.colors.space import SpaceWidget
from gui.sherloq_app.tools.colors.stats import StatsWidget
from gui.sherloq_app.tools.detail.echo import EchoWidget
from gui.sherloq_app.tools.detail.frequency import FrequencyWidget
from gui.sherloq_app.tools.detail.gradient import GradientWidget
from gui.sherloq_app.tools.detail.wavelets import WaveletWidget
from gui.sherloq_app.tools.general.digest import DigestWidget
from gui.sherloq_app.tools.general.editor import EditorWidget
from gui.sherloq_app.tools.general.original import OriginalWidget
from gui.sherloq_app.tools.general.reverse import ReverseWidget
from gui.sherloq_app.tools.inspection.adjust import AdjustWidget
from gui.sherloq_app.tools.inspection.comparison import ComparisonWidget
from gui.sherloq_app.tools.inspection.histogram import HistWidget
from gui.sherloq_app.tools.inspection.magnifier import MagnifierWidget
from gui.sherloq_app.tools.jpeg.ela import ElaWidget
from gui.sherloq_app.tools.jpeg.ghostmmaps import GhostmapWidget
from gui.sherloq_app.tools.jpeg.multiple import MultipleWidget
from gui.sherloq_app.tools.jpeg.quality import QualityWidget
from gui.sherloq_app.tools.metadata.exif import ExifWidget
from gui.sherloq_app.tools.metadata.header import HeaderWidget
from gui.sherloq_app.tools.metadata.location import LocationWidget
from gui.sherloq_app.tools.metadata.thumbnail import ThumbWidget
from gui.sherloq_app.tools.noise.minmax import MinMaxWidget
from gui.sherloq_app.tools.noise.noise import NoiseWidget
from gui.sherloq_app.tools.noise.noise_estimmation import NoiseWaveletBlockingWidget
from gui.sherloq_app.tools.noise.planes import PlanesWidget
from gui.sherloq_app.tools.noise.prnu import PrnuWidget
from gui.sherloq_app.tools.tampering.cloning import CloningWidget
from gui.sherloq_app.tools.tampering.contrast import ContrastWidget
from gui.sherloq_app.tools.tampering.resampling import ResamplingWidget
from gui.sherloq_app.tools.various.median import MedianWidget
from gui.sherloq_app.tools.various.illuminant import IlluminantWidget
from gui.sherloq_app.tools.various.defect_pixels import DefectWidget
from gui.sherloq_app.tools.various.stereogram import StereoWidget
from gui.sherloq_app.tools.various.trufor import TruForWidget
from gui.sherloq_app.ui.icons import themed_icon, application_icon
from gui.sherloq_app.ui.tools import ToolTree
from gui.sherloq_app.core.utility import modify_font, load_image

class ImageWorkspace(QMdiArea):
    imageDropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)

    @staticmethod
    def dropped_file(mime_data):
        urls = mime_data.urls()
        if len(urls) != 1 or not urls[0].isLocalFile():
            return None
        filename = urls[0].toLocalFile()
        return filename if os.path.isfile(filename) else None

    def dragEnterEvent(self, event):
        if self.dropped_file(event.mimeData()) is not None:
            event.setDropAction(Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        self.dragEnterEvent(event)

    def dropEvent(self, event):
        filename = self.dropped_file(event.mimeData())
        if filename is None:
            event.ignore()
            return
        event.setDropAction(Qt.CopyAction)
        event.accept()
        # Complete the native drag before loading or closing analysis windows.
        QTimer.singleShot(0, lambda: self.imageDropped.emit(filename))

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.subWindowList():
            painter = QPainter(self.viewport())
            painter.setPen(self.palette().color(QPalette.Text))
            font = painter.font()
            font.setPointSize(15)
            painter.setFont(font)
            painter.drawText(
                self.viewport().rect(), Qt.AlignCenter,
                t("Glissez une image ici\nou utilisez « Load image… » (⌘O)"),
            )


class AnalysisSubWindow(QMdiSubWindow):
    def closeEvent(self, event):
        widget = self.widget()
        if hasattr(widget, "shutdown"):
            widget.shutdown()
        super().closeEvent(event)


class MainWindow(QMainWindow):
    max_recent = 5
    image_loaded = Signal(str)

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        QApplication.setApplicationName("Sherloq")
        QApplication.setOrganizationName("Guido Bartoli")
        QApplication.setOrganizationDomain("http://www.guidobartoli.com")
        from gui.sherloq_app.ui.localization import install
        self.language_manager = install()
        QApplication.setApplicationVersion(ToolTree().version)
        icon = application_icon()
        if icon is not None:
            QApplication.setWindowIcon(icon)
        self.setWindowTitle(
            f"{QApplication.applicationName()} {QApplication.applicationVersion()}"
        )
        self.mdi_area = ImageWorkspace()
        self.mdi_area.imageDropped.connect(self.open_image)
        self.setCentralWidget(self.mdi_area)
        self.filename = None
        self.image = None
        from gui.sherloq_app.ui.image_io import ImageLoadJob
        self.load_job = ImageLoadJob(self)
        self.load_job.result.connect(self._image_ready)
        self.load_job.failed.connect(self._load_error)
        self.load_job.busy.connect(self._load_busy)
        self.load_progress = QProgressBar();self.load_progress.setRange(0,0);self.load_progress.setMaximumWidth(110)
        self.cancel_load_button = QPushButton('Cancel loading');self.cancel_load_button.clicked.connect(self.cancel_loading)
        self.image_details = QLabel()
        self.statusBar().addPermanentWidget(self.image_details)
        self.statusBar().addPermanentWidget(self.load_progress)
        self.statusBar().addPermanentWidget(self.cancel_load_button)
        self._load_busy(False)
        modify_font(self.statusBar(), bold=True)

        tree_dock = QDockWidget(self.tr("TOOLS"), self)
        tree_dock.setObjectName("tree_dock")
        tree_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, tree_dock)
        self.tree_widget = ToolTree()
        self.tree_widget.setObjectName("tree_widget")
        self.tree_widget.itemDoubleClicked.connect(self.open_tool)
        tree_dock.setWidget(self.tree_widget)

        tools_action = tree_dock.toggleViewAction()
        tools_action.setText(self.tr("Show tools"))
        tools_action.setToolTip(self.tr("Toggle toolset visibility"))
        tools_action.setShortcut(QKeySequence(Qt.Key_Tab))
        tools_action.setObjectName("tools_action")
        tools_action.setIcon(themed_icon("tools.svg"))

        help_action = QAction(self.tr("Show help"), self)
        help_action.setToolTip(self.tr("Toggle online help"))
        help_action.setShortcut(QKeySequence.HelpContents)
        help_action.setObjectName("help_action")
        help_action.setIcon(themed_icon("help.svg"))
        help_action.setCheckable(True)
        help_action.setEnabled(False)

        load_action = QAction(self.tr("&Load image..."), self)
        load_action.setToolTip(self.tr("Load an image to analyze"))
        load_action.setShortcut(QKeySequence.Open)
        load_action.triggered.connect(self.load_file)
        load_action.setObjectName("load_action")
        load_action.setIcon(themed_icon("load.svg"))

        quit_action = QAction(self.tr("&Quit"), self)
        quit_action.setToolTip(self.tr("Exit from Sherloq"))
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        quit_action.setObjectName("quit_action")
        quit_action.setIcon(themed_icon("quit.svg"))

        tabbed_action = QAction(self.tr("&Tabbed"), self)
        tabbed_action.setToolTip(self.tr("Toggle tabbed view for window area"))
        tabbed_action.setShortcut(QKeySequence(Qt.Key_F10))
        tabbed_action.setCheckable(True)
        tabbed_action.triggered.connect(self.toggle_view)
        tabbed_action.setObjectName("tabbed_action")
        tabbed_action.setIcon(themed_icon("tabbed.svg"))

        prev_action = QAction(self.tr("&Previous"), self)
        prev_action.setToolTip(self.tr("Select the previous tool window"))
        prev_action.setShortcut(QKeySequence.PreviousChild)
        prev_action.triggered.connect(self.mdi_area.activatePreviousSubWindow)
        prev_action.setObjectName("prev_action")
        prev_action.setIcon(themed_icon("previous.svg"))

        next_action = QAction(self.tr("&Next"), self)
        next_action.setToolTip(self.tr("Select the next tool window"))
        next_action.setShortcut(QKeySequence.NextChild)
        next_action.triggered.connect(self.mdi_area.activateNextSubWindow)
        next_action.setObjectName("next_action")
        next_action.setIcon(themed_icon("next.svg"))

        tile_action = QAction(self.tr("&Tile"), self)
        tile_action.setToolTip(self.tr("Arrange windows into non-overlapping views"))
        tile_action.setShortcut(QKeySequence(Qt.Key_F11))
        tile_action.triggered.connect(self.mdi_area.tileSubWindows)
        tile_action.setObjectName("tile_action")
        tile_action.setIcon(themed_icon("tile.svg"))

        cascade_action = QAction(self.tr("&Cascade"), self)
        cascade_action.setToolTip(self.tr("Arrange windows into overlapping views"))
        cascade_action.setShortcut(QKeySequence(Qt.Key_F12))
        cascade_action.triggered.connect(self.mdi_area.cascadeSubWindows)
        cascade_action.setObjectName("cascade_action")
        cascade_action.setIcon(themed_icon("cascade.svg"))

        close_action = QAction(self.tr("Close &All"), self)
        close_action.setToolTip(self.tr("Close all open tool windows"))
        close_action.setShortcut(QKeySequence(Qt.CTRL | Qt.SHIFT | Qt.Key_W))
        close_action.triggered.connect(self.mdi_area.closeAllSubWindows)
        close_action.setObjectName("close_action")
        close_action.setIcon(themed_icon("close.svg"))

        self.full_action = QAction(self.tr("Full screen"), self)
        self.full_action.setToolTip(self.tr("Switch to full screen mode"))
        self.full_action.setShortcut(QKeySequence.FullScreen)
        self.full_action.triggered.connect(self.change_view)
        self.full_action.setObjectName("full_action")
        self.full_action.setIcon(themed_icon("full.svg"))

        self.normal_action = QAction(self.tr("Normal view"), self)
        self.normal_action.setToolTip(self.tr("Back to normal view mode"))
        self.normal_action.setShortcut(QKeySequence(Qt.CTRL | Qt.Key_F12))
        self.normal_action.triggered.connect(self.change_view)
        self.normal_action.setObjectName("normal_action")
        self.normal_action.setIcon(themed_icon("normal.svg"))

        about_action = QAction(self.tr("&About..."), self)
        about_action.setToolTip(self.tr("Information about this program"))
        about_action.triggered.connect(self.show_about)
        about_action.setObjectName("about_action")
        about_action.setIcon(themed_icon("sherloq_alpha.png"))

        about_qt_action = QAction(self.tr("About &Qt"), self)
        about_qt_action.setToolTip(self.tr("Information about the Qt Framework"))
        about_qt_action.triggered.connect(QApplication.aboutQt)
        about_qt_action.setIcon(themed_icon("Qt.svg"))

        file_menu = self.menuBar().addMenu(self.tr("&File"))
        file_menu.addAction(load_action)
        file_menu.addSeparator()
        self.recent_actions = [None] * self.max_recent
        for i in range(len(self.recent_actions)):
            self.recent_actions[i] = QAction(self)
            self.recent_actions[i].setVisible(False)
            self.recent_actions[i].triggered.connect(self.open_recent)
            file_menu.addAction(self.recent_actions[i])
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

        view_menu = self.menuBar().addMenu(self.tr("&View"))
        view_menu.addAction(tools_action)
        view_menu.addAction(help_action)
        view_menu.addSeparator()
        view_menu.addAction(self.full_action)
        view_menu.addAction(self.normal_action)

        window_menu = self.menuBar().addMenu(self.tr("&Window"))
        window_menu.addAction(prev_action)
        window_menu.addAction(next_action)
        window_menu.addSeparator()
        window_menu.addAction(tile_action)
        window_menu.addAction(cascade_action)
        window_menu.addAction(tabbed_action)
        window_menu.addSeparator()
        window_menu.addAction(close_action)

        help_menu = self.menuBar().addMenu(self.tr("&Help"))
        help_menu.addAction(help_action)
        help_menu.addSeparator()
        help_menu.addAction(about_action)
        help_menu.addAction(about_qt_action)
        self.language_manager.add_menu(self.menuBar())

        main_toolbar = self.addToolBar(self.tr("&Toolbar"))
        main_toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        main_toolbar.addAction(load_action)
        main_toolbar.addSeparator()
        main_toolbar.addAction(tools_action)
        main_toolbar.addAction(help_action)
        main_toolbar.addSeparator()
        main_toolbar.addAction(prev_action)
        main_toolbar.addAction(next_action)
        main_toolbar.addSeparator()
        main_toolbar.addAction(tile_action)
        main_toolbar.addAction(cascade_action)
        main_toolbar.addAction(tabbed_action)
        main_toolbar.addAction(close_action)
        # main_toolbar.addSeparator()
        # main_toolbar.addAction(self.normal_action)
        # main_toolbar.addAction(self.full_action)
        main_toolbar.setAllowedAreas(Qt.TopToolBarArea | Qt.BottomToolBarArea)
        main_toolbar.setObjectName("main_toolbar")

        settings = QSettings()
        settings.beginGroup("main_window")
        self.restoreGeometry(settings.value("geometry"))
        self.restoreState(settings.value("state"))
        self.recent_files = settings.value("recent_files")
        if self.recent_files is None:
            self.recent_files = []
        elif not isinstance(self.recent_files, list):
            self.recent_files = [self.recent_files]
        self.update_recent()
        settings.endGroup()

        prev_action.setEnabled(False)
        next_action.setEnabled(False)
        tile_action.setEnabled(False)
        cascade_action.setEnabled(False)
        close_action.setEnabled(False)
        tabbed_action.setEnabled(False)
        self.tree_widget.setEnabled(False)
        if self.width() < 900 or self.height() < 600:
            self.resize(1280, 850)
        self.showMaximized()
        self.normal_action.setEnabled(False)
        self.show_message(self.tr("Ready"))

    def change_view(self):
        if self.isFullScreen():
            self.showNormal()
            self.showMaximized()
            self.full_action.setEnabled(True)
            self.normal_action.setEnabled(False)
        else:
            self.showFullScreen()
            self.full_action.setEnabled(False)
            self.normal_action.setEnabled(True)

    def closeEvent(self, event):
        self.load_job.shutdown()
        self.mdi_area.closeAllSubWindows()
        settings = QSettings()
        settings.beginGroup("main_window")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("state", self.saveState())
        settings.setValue("recent_files", self.recent_files)
        settings.endGroup()
        super(MainWindow, self).closeEvent(event)

    def update_recent(self):
        if not self.recent_files:
            return
        self.recent_files = [f for f in self.recent_files if os.path.isfile(f)]
        for i in range(len(self.recent_actions)):
            if i < len(self.recent_files):
                text = f"&{i + 1} {os.path.basename(self.recent_files[i])}"
                self.recent_actions[i].setText(text)
                self.recent_actions[i].setData(self.recent_files[i])
                self.recent_actions[i].setVisible(True)
            else:
                self.recent_actions[i].setVisible(False)

    def open_recent(self):
        action = self.sender()
        if action:
            self.open_image(action.data())

    def initialize(self, filename, basename, image):
        self.load_job.invalidate()
        self.image_details.clear()
        self.filename = filename
        self.image = image
        self.findChild(ToolTree, "tree_widget").setEnabled(True)
        self.findChild(QAction, "prev_action").setEnabled(True)
        self.findChild(QAction, "next_action").setEnabled(True)
        self.findChild(QAction, "tile_action").setEnabled(True)
        self.findChild(QAction, "cascade_action").setEnabled(True)
        self.findChild(QAction, "close_action").setEnabled(True)
        self.findChild(QAction, "tabbed_action").setEnabled(True)
        self.setWindowTitle(
            f"({basename}) - {QApplication.applicationName()} {QApplication.applicationVersion()}"
        )
        if filename not in self.recent_files:
            self.recent_files.insert(0, filename)
            if len(self.recent_files) > self.max_recent:
                self.recent_files = self.recent_files[: self.max_recent]
            self.update_recent()
        self.show_message(self.tr(f'Image "{basename}" successfully loaded'))

        # FIXME: disable_bold della chiusura viene chiamato DOPO open_tool e nell'albero la voce NON diventa neretto
        self.mdi_area.closeAllSubWindows()
        self.open_tool(self.tree_widget.tool_item(0, 0), None)

    def load_file(self):
        self.open_image()

    def open_image(self, filename=None):
        from gui.sherloq_app.ui.image_io import choose_image
        if filename is None:
            filename = choose_image(self)
        if not filename:
            return
        self.show_message(f'Loading {os.path.basename(filename)}…')
        self.load_job.load(filename)

    def _load_busy(self, busy):
        self.load_progress.setVisible(busy)
        self.cancel_load_button.setVisible(busy)

    def cancel_loading(self):
        self.load_job.invalidate()
        self.show_message('Loading cancelled. Previous image retained.')

    def _load_error(self, message):
        self.show_message(f'Unable to load image: {message}')

    def _image_ready(self, result):
        if result is None:
            return
        filename, basename, image, metadata = result
        self.initialize(filename, basename, image)
        QSettings().setValue('load_folder', os.path.dirname(os.path.abspath(filename)))
        labels = []
        if metadata.get('format') == 'RAW':labels.append('RAW → 8-bit analysis')
        if metadata['source_bits'] and metadata['source_bits'] > 8:
            labels.append(f"{metadata['source_bits']}-bit source → 8-bit analysis")
        if metadata['icc']:labels.append('ICC not applied')
        if metadata['alpha']:labels.append('alpha not analysed')
        if metadata['frames'] > 1:labels.append('first frame')
        self.image_details.setText(' · '.join(labels))
        self.image_details.setToolTip('Analysis uses decoded 8-bit pixels. The source file remains unchanged. Colour profiles are not applied to this analysis copy.')
        self.image_loaded.emit(filename)

    def open_tool(self, item, _):
        if not item.data(0, Qt.UserRole):
            return
        group = item.data(0, Qt.UserRole + 1)
        tool = item.data(0, Qt.UserRole + 2)
        for sub_window in self.mdi_area.subWindowList():
            if sub_window.property('toolGroup') == group and sub_window.property('toolIndex') == tool:
                sub_window.setFocus()
                return

        if group == 0:
            if tool == 0:
                tool_widget = OriginalWidget(self.image)
            elif tool == 1:
                tool_widget = DigestWidget(self.filename, self.image)
            elif tool == 2:
                tool_widget = EditorWidget()
            elif tool == 3:
                tool_widget = ReverseWidget()
            else:
                return
        elif group == 1:
            if tool == 0:
                tool_widget = HeaderWidget(self.filename)
            elif tool == 1:
                tool_widget = ExifWidget(self.filename)
            elif tool == 2:
                tool_widget = ThumbWidget(self.filename, self.image)
            elif tool == 3:
                tool_widget = LocationWidget(self.filename)
            elif tool == 4:
                from gui.sherloq_app.tools.metadata.c2pa import C2paWidget
                tool_widget = C2paWidget(self.filename)
            else:
                return
        elif group == 2:
            if tool == 0:
                tool_widget = MagnifierWidget(self.image)
            elif tool == 1:
                tool_widget = HistWidget(self.image)
            elif tool == 2:
                tool_widget = AdjustWidget(self.image)
            elif tool == 3:
                tool_widget = ComparisonWidget(self.filename, self.image)
            else:
                return
        elif group == 3:
            if tool == 0:
                tool_widget = GradientWidget(self.image)
            elif tool == 1:
                tool_widget = EchoWidget(self.image)
            elif tool == 2:
                tool_widget = WaveletWidget(self.image)
            elif tool == 3:
                tool_widget = FrequencyWidget(self.image)
            else:
                return
        elif group == 4:
            if tool == 0:
                tool_widget = PlotsWidget(self.image)
            elif tool == 1:
                tool_widget = SpaceWidget(self.image)
            elif tool == 2:
                tool_widget = PcaWidget(self.image)
            elif tool == 3:
                tool_widget = StatsWidget(self.image)
            else:
                return
        elif group == 5:
            if tool == 0:
                tool_widget = NoiseWidget(self.image)
            elif tool == 1:
                tool_widget = MinMaxWidget(self.image)
            elif tool == 2:
                tool_widget = PlanesWidget(self.image)
            elif tool == 3:
                tool_widget = NoiseWaveletBlockingWidget(self.filename, self.image)
            elif tool == 4:
                tool_widget = PrnuWidget(self.filename, self.image)
            elif tool == 5:
                from gui.sherloq_app.tools.noise.noisesniffer import NoisesnifferWidget
                tool_widget = NoisesnifferWidget(self.image)
            else:
                return
        elif group == 6:
            if tool == 0:
                tool_widget = QualityWidget(self.filename, self.image)
            elif tool == 1:
                tool_widget = ElaWidget(self.image, filename=self.filename)
            elif tool == 2:
                tool_widget = MultipleWidget(self.image, filename=self.filename)
            elif tool == 3:
                tool_widget = GhostmapWidget(self.filename, self.image)
            elif tool == 4:
                from gui.sherloq_app.tools.jpeg.zero import ZeroWidget
                tool_widget = ZeroWidget(self.image)
            else:
                return
        elif group == 7:
            if tool == 0:
                tool_widget = ContrastWidget(self.image)
            elif tool == 1:
                tool_widget = CloningWidget(self.image)
            elif tool == 2:
                from gui.sherloq_app.tools.tampering.splicing import SplicingWidget
                tool_widget = SplicingWidget(self.image)
            elif tool == 3:
                tool_widget = ResamplingWidget(self.filename, self.image)
            elif tool == 4:
                from gui.sherloq_app.tools.tampering.cloning2 import Cloning2Widget
                tool_widget = Cloning2Widget(self.image)
            elif tool == 5:
                from gui.sherloq_app.tools.tampering.adaptive_cfa import AdaptiveCFAWidget
                tool_widget = AdaptiveCFAWidget(self.image)
            elif tool == 6:
                from gui.sherloq_app.tools.tampering.clone_detectors import CloneDetectorsWidget
                tool_widget = CloneDetectorsWidget(self.image)
            elif tool == 7:
                from gui.sherloq_app.tools.tampering.automatic_clones import AutomaticClonesWidget
                tool_widget = AutomaticClonesWidget(self.image)
            elif tool == 8:
                from gui.sherloq_app.tools.tampering.complete_analysis import CompleteAnalysisWidget
                tool_widget = CompleteAnalysisWidget(self.image, self.filename)
            else:
                return
        elif group == 8:
            if tool == 0:
                tool_widget = TruForWidget(self.filename, self.image)
            elif tool == 1:
                from gui.sherloq_app.tools.various.catnet import CatNetWidget
                tool_widget = CatNetWidget(self.filename,self.image)
            elif tool == 2:
                from gui.sherloq_app.tools.various.safire import SafireWidget
                tool_widget = SafireWidget(self.image)
            elif tool == 3:
                from gui.sherloq_app.tools.various.focal import FocalWidget
                tool_widget = FocalWidget(self.image)
            elif tool == 4:
                from gui.sherloq_app.tools.various.adaifl import AdaIFLWidget
                tool_widget = AdaIFLWidget(self.image)
            else:
                return
        elif group == 9:
            if tool == 0:
                tool_widget = MedianWidget(self.image)
            elif tool == 1:
                tool_widget = IlluminantWidget(self.image)
            elif tool == 2:
                tool_widget = DefectWidget(self.image)
            elif tool == 3:
                tool_widget = StereoWidget(self.image)
            else:
                return
        else:
            return
        tool_widget.info_message.connect(self.show_message)

        sub_window = AnalysisSubWindow()
        sub_window.setWidget(tool_widget)
        sub_window.setProperty('toolGroup', group)
        sub_window.setProperty('toolIndex', tool)
        sub_window.setWindowTitle(item.text(0))
        sub_window.setObjectName(item.text(0))
        sub_window.setAttribute(Qt.WA_DeleteOnClose)
        sub_window.setWindowIcon(themed_icon(f"{group}.svg"))
        self.mdi_area.addSubWindow(sub_window)
        sub_window.showMaximized()
        sub_window.destroyed.connect(lambda _=None, key=(group,tool): self.tree_widget.set_bold(key, enabled=False))
        self.tree_widget.set_bold((group,tool), enabled=True)

    def disable_bold(self, item):
        self.tree_widget.set_bold(item.windowTitle(), enabled=False)

    def toggle_view(self, tabbed):
        if tabbed:
            self.mdi_area.setViewMode(QMdiArea.TabbedView)
            self.mdi_area.setTabsClosable(True)
            self.mdi_area.setTabsMovable(True)
        else:
            self.mdi_area.setViewMode(QMdiArea.SubWindowView)
        self.findChild(QAction, "tile_action").setEnabled(not tabbed)
        self.findChild(QAction, "cascade_action").setEnabled(not tabbed)

    def show_about(self):
        message = f"<h2>{QApplication.applicationName()} {QApplication.applicationVersion()}</h2>"
        message += f"<h3>{t('A digital image forensic toolkit')}</h3>"
        message += f'<p>{t("Author")}: <a href="{QApplication.organizationDomain()}">{QApplication.organizationName()}</a></p>'
        message += f'<p>Source: <a href="https://github.com/GuidoBartoli/sherloq">{t("GitHub repository")}</a></p>'
        message += f'<p>{t("License")}: <a href="https://www.gnu.org/licenses/gpl-3.0.html">GNU GPLv3</a></p>'
        message += f'<p>{t("Libraries")}: <a href="https://opencv.org/">OpenCV</a> <a href="https://exiftool.org/">ExifTool</a> <a href="https://www.tensorflow.org/">TensorFlow</a></p>'
        QMessageBox.about(self, t(self.tr("About")), t(message))

    def show_message(self, message):
        self.statusBar().showMessage(message, 10000)


def main():
    application = QApplication(sys.argv)
    mainwindow = MainWindow()
    sys.exit(application.exec())


if __name__ == "__main__":
    main()
