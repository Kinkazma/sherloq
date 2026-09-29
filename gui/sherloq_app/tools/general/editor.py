from PySide6.QtWidgets import QVBoxLayout
from gui.sherloq_app.ui.browser import BrowserPanel
from gui.sherloq_app.ui.tools import ToolWidget


class EditorWidget(ToolWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.browser = BrowserPanel('https://hexed.it/', self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.browser)

    def shutdown(self):
        self.browser.shutdown()
        super().shutdown()
