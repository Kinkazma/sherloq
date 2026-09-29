"""Check states are persistent exclusions; row selection is temporary focus."""
from PySide6.QtCore import Qt,Signal
from PySide6.QtWidgets import QListView,QAbstractItemView,QStyleOptionViewItem,QStyle


class CheckList(QListView):
    focusCleared=Signal()
    def __init__(self,parent=None):
        super().__init__(parent);self.setSelectionMode(QAbstractItemView.SingleSelection)
    def mousePressEvent(self,event):
        index=self.indexAt(event.position().toPoint())
        if event.button()==Qt.LeftButton:
            if not index.isValid():
                self.clearSelection();self.setCurrentIndex(index);self.focusCleared.emit();event.accept();return
            option=QStyleOptionViewItem();self.itemDelegate().initStyleOption(option,index)
            option.rect=self.visualRect(index);option.widget=self
            check=self.style().subElementRect(QStyle.SE_ItemViewItemCheckIndicator,option,self)
            if check.contains(event.position().toPoint()):
                value=Qt.Unchecked if index.data(Qt.CheckStateRole)==Qt.Checked else Qt.Checked
                self.model().setData(index,value,Qt.CheckStateRole);event.accept();return
        super().mousePressEvent(event)
