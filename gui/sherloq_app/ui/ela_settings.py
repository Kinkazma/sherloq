"""Compact ELA controls shared by standalone and combined analysis panels."""
from PySide6.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QGridLayout,QTabWidget,QToolButton,QLabel,QSizePolicy
from PySide6.QtCore import Qt


class ElaSettings(QWidget):
    def __init__(self,mode,profile,automatic,histogram_row,energy_row,cells,options,layers,actions=()):
        super().__init__()
        layout=QVBoxLayout(self);layout.setContentsMargins(0,0,0,0);layout.setSpacing(4)
        self.toolbar=QHBoxLayout();self.toolbar.setSpacing(8)
        self.toolbar.addWidget(mode);self.toolbar.addWidget(profile)
        self.toggle=QToolButton();self.toggle.setText('Settings');self.toggle.setCheckable(True);self.toggle.setChecked(True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon);self.toggle.setArrowType(Qt.DownArrow)
        self.toolbar.addWidget(self.toggle);self.toolbar.addStretch()
        for action in actions:self.toolbar.addWidget(action)
        layout.addLayout(self.toolbar)
        self.pages=QTabWidget();layout.addWidget(self.pages)
        energy=QWidget();grid=QGridLayout(energy);grid.setContentsMargins(8,4,8,4);grid.setVerticalSpacing(4)
        # Each row contains two independent label/slider/value groups.
        for row_index,row in enumerate((histogram_row,energy_row)):
            column=0
            while row.count():
                item=row.takeAt(0);widget=item.widget()
                if widget is not None:
                    grid.addWidget(widget,row_index,column);column+=1
        for column in (1,4):grid.setColumnStretch(column,1)
        bottom=QHBoxLayout();bottom.addWidget(automatic)
        for control in layers:bottom.addWidget(control)
        bottom.addStretch();grid.addLayout(bottom,2,0,1,6)
        self.pages.addTab(energy,'Energy')
        legacy=QWidget();legacy_layout=QVBoxLayout(legacy);legacy_layout.setContentsMargins(8,4,8,4)
        row=QHBoxLayout()
        for title,control in cells:row.addWidget(QLabel(title));row.addWidget(control)
        row.addStretch();legacy_layout.addLayout(row)
        row=QHBoxLayout()
        for control in options:row.addWidget(control)
        row.addStretch();legacy_layout.addLayout(row)
        self.pages.addTab(legacy,'Cells / Ghosts')
        self.toggle.toggled.connect(self.set_expanded)
        self.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Maximum)
    def set_expanded(self,expanded):
        self.pages.setVisible(expanded)
        self.toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
