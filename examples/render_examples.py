"""Render real SHERLOQ tool panels; run with the native installation's Python."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2 as cv
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PySide6.QtGui import QFont
from gui.sherloq_app.ui.jobs import LatestJob
from gui.sherloq_app.ui.viewer import ImageViewer
from gui.sherloq_app.tools.inspection.histogram import HistWidget
from gui.sherloq_app.tools.detail.frequency import FrequencyWidget
from gui.sherloq_app.tools.noise.noise import NoiseWidget
from gui.sherloq_app.tools.noise.planes import PlanesWidget

app = QApplication([])
app.setOrganizationName("SHERLOQExampleRendering")
app.setApplicationName("Documentation")
app.setStyle("Fusion")
app.setPalette(app.style().standardPalette())
app.setFont(QFont("Arial", 11))
OUT = ROOT / "screenshots/fork"


def settle(panel):
    deadline = time.monotonic() + 90
    stable = None
    while time.monotonic() < deadline:
        app.processEvents()
        if any(j.is_busy or j.timer.isActive() for j in panel.findChildren(LatestJob)):
            stable = None
        elif stable is None:
            stable = time.monotonic()
        elif time.monotonic() - stable > 0.5:
            return
        time.sleep(0.01)
    raise TimeoutError("Tool did not finish")


def capture(kind, title, path, name, configure=None, original=False):
    image = cv.imread(str(ROOT / path))
    if image is None:
        raise ValueError(path)
    frame = QWidget()
    layout = QVBoxLayout(frame)
    label = QLabel(title)
    label.setFont(QFont("Arial", 17, QFont.Weight.Bold))
    layout.addWidget(label)
    body = QHBoxLayout()
    layout.addLayout(body, 1)
    if original:
        preview = ImageViewer(image, None, "Source image")
        body.addWidget(preview, 1)
    tool = kind(image)
    errors = []
    for job in tool.findChildren(LatestJob):
        job.failed.connect(errors.append)
    body.addWidget(tool, 2 if original else 1)
    layout.addWidget(QLabel("SHERLOQ — actual tool panel rendered from the repository source"))
    frame.resize(1400, 900)
    frame.show()
    if configure:
        configure(tool)
    settle(frame)
    if errors:
        raise RuntimeError(errors)
    for viewer in frame.findChildren(ImageViewer):
        viewer.view.zoom_fit()
    app.processEvents()
    if not frame.grab().save(str(OUT / name)):
        raise RuntimeError("Could not save " + name)
    print(name, flush=True)
    tool.shutdown()
    frame.close()
    frame.deleteLater()
    app.processEvents()


capture(HistWidget, "Channel Histogram · fluorescence microscopy",
        "examples/bbbc039-00809.png", "histogram-microscopy.png", original=True)
capture(FrequencyWidget, "Frequency Split · original texture",
        "screenshots/fork/texture-original.png", "frequency-texture.png")
capture(NoiseWidget, "Noise Separation · fluorescence microscopy",
        "examples/bbbc039-00809.png", "noise-microscopy.png",
        configure=lambda tool: tool.levels_spin.setValue(0), original=True)
capture(PlanesWidget, "Bit Planes Values · original texture",
        "screenshots/fork/texture-original.png", "bit-planes-texture.png",
        configure=lambda tool: tool.plane_spin.setValue(5), original=True)
