"""Subtle test landmarks, painted over native controls without changing labels."""
from PySide6.QtCore import Qt,QEvent,QRectF,QObject,Signal,QSettings
from PySide6.QtGui import QColor,QPainter,QPen,QAction
from PySide6.QtWidgets import QWidget,QStyledItemDelegate,QStyleOptionViewItem,QStyle,QApplication,QAbstractItemView,QComboBox

EXTENSION_ROLE=int(Qt.UserRole)+32
COLOR=QColor(205,60,70,145)
GREEN=QColor(30,157,84,160)
BLUE=QColor(35,115,225,175)

class HighlightSettings(QObject):
    changed=Signal(bool)
    KEY='interface/highlight_new_features'
    def __init__(self,parent=None,settings=None):
        super().__init__(parent)
        self.settings=QSettings() if settings is None else settings
        self.enabled=self.settings.value(self.KEY,False,type=bool)
    def set_enabled(self,enabled):
        enabled=bool(enabled)
        if enabled==self.enabled:return
        self.enabled=enabled
        self.settings.setValue(self.KEY,enabled)
        self.changed.emit(enabled)

def highlight_settings():
    app=QApplication.instance()
    if not hasattr(app,'_extension_highlights'):
        app._extension_highlights=HighlightSettings(app)
    return app._extension_highlights

def highlight_action(parent):
    action=QAction('Highlight new features',parent);action.setCheckable(True)
    settings=highlight_settings();action.setChecked(settings.enabled)
    action.toggled.connect(settings.set_enabled)
    settings.changed.connect(action.setChecked)
    return action

def outline(painter,rect,color=COLOR):
    if not highlight_settings().enabled:return
    painter.save();painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(color,.8));painter.setBrush(Qt.NoBrush)
    painter.drawRoundedRect(QRectF(rect).adjusted(1,1,-1,-1),3,3);painter.restore()

class ExtensionDelegate(QStyledItemDelegate):
    def __init__(self,parent=None):
        super().__init__(parent)
        highlight_settings().changed.connect(self.refresh)
    def refresh(self,*_):
        target=self.parent()
        if isinstance(target,QAbstractItemView):target.viewport().update()
        elif isinstance(target,QComboBox):
            target.update();target.view().viewport().update()
    def paint(self,painter,option,index):
        super().paint(painter,option,index)
        if not index.data(EXTENSION_ROLE):return
        opt=QStyleOptionViewItem(option);self.initStyleOption(opt,index)
        style=opt.widget.style() if opt.widget else None
        rect=style.subElementRect(QStyle.SE_ItemViewItemText,opt,opt.widget) if style else opt.rect
        rect.setWidth(min(rect.width(),opt.fontMetrics.horizontalAdvance(opt.text)+18))
        outline(painter,rect,{'green':GREEN,'blue':BLUE}.get(index.data(EXTENSION_ROLE),COLOR))

class ExtensionOutline(QWidget):
    """Mouse-transparent overlay; keep Cocoa's native widget/focus behaviour."""
    def __init__(self,target,tab=None):
        super().__init__(target);self.target=target;self.tab=tab;self.color=COLOR
        self.setAttribute(Qt.WA_TransparentForMouseEvents);self.setAttribute(Qt.WA_NoSystemBackground)
        self.setFocusPolicy(Qt.NoFocus);target.installEventFilter(self)
        highlight_settings().changed.connect(self.refresh)
        self.setGeometry(target.rect());self.show();self.raise_()
    def refresh(self,*_):
        # Repaint only: no layout, value changes or analysis invalidation.
        self.update();self.target.update()
    def eventFilter(self,watched,event):
        if event.type() in (QEvent.Resize,QEvent.Show,QEvent.LayoutRequest):
            self.setGeometry(self.target.rect());self.raise_();self.update()
        return False
    def paintEvent(self,event):
        painter=QPainter(self)
        outline(painter,self.rect() if self.tab is None else self.target.tabRect(self.tab),self.color)

def mark_combo(combo,new_items,green_items=()):
    names=set(new_items)
    for i in range(combo.count()):combo.setItemData(i,'green' if combo.itemText(i) in green_items else combo.itemText(i) in names,EXTENSION_ROLE)
    combo.setItemDelegate(ExtensionDelegate(combo));overlay=ExtensionOutline(combo)
    def update(*_):
        overlay.color=GREEN if combo.currentData(EXTENSION_ROLE)=='green' else COLOR
        overlay.setVisible(bool(combo.currentData(EXTENSION_ROLE)));overlay.update()
    combo.currentIndexChanged.connect(update);update()
    # Explicit Python ownership as well as Qt parent ownership.
    combo.extension_outline=overlay

def mark_tab(tabs,index):
    overlay=ExtensionOutline(tabs.tabBar(),index)
    tabs.currentChanged.connect(lambda *_:overlay.update())
    tabs.extension_outline=overlay
