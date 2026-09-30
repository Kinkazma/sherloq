"""Global display preference: all three outline colors, live repaint, no layout change."""
import os,sys,tempfile,json
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtCore import Qt,QSettings
from PySide6.QtWidgets import QApplication,QWidget,QVBoxLayout,QComboBox,QTabWidget,QTreeWidget,QTreeWidgetItem,QMenuBar,QPushButton
from PySide6.QtTest import QTest
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'source'))
from gui.sherloq_app.ui.extensions import (HighlightSettings,highlight_settings,highlight_action,
    ExtensionOutline,ExtensionDelegate,EXTENSION_ROLE,GREEN,BLUE,mark_combo,mark_tab)
from gui.sherloq_app.ui.localization import install
app=QApplication([])
with tempfile.TemporaryDirectory() as tmp:
 settings=QSettings(str(Path(tmp)/'preferences.ini'),QSettings.IniFormat)
 app._extension_highlights=HighlightSettings(app,settings)
 pref=highlight_settings();assert not pref.enabled
 win=QWidget();layout=QVBoxLayout(win);menu=QMenuBar();view=menu.addMenu('&View');action=highlight_action(win);view.addAction(action);layout.addWidget(menu)
 tree=QTreeWidget();tree.setHeaderHidden(True);tree.setItemDelegate(ExtensionDelegate(tree));layout.addWidget(tree)
 for label,color in [('red',True),('green','green'),('blue','blue')]:
  item=QTreeWidgetItem([label]);item.setData(0,EXTENSION_ROLE,color);tree.addTopLevelItem(item)
 combo=QComboBox();combo.addItems(['Existing','New red','New green']);mark_combo(combo,['New red'],['New green']);combo.setCurrentIndex(1);layout.addWidget(combo)
 tabs=QTabWidget();tabs.addTab(QWidget(),'Existing');tabs.addTab(QWidget(),'New');mark_tab(tabs,1);layout.addWidget(tabs)
 buttons=[]
 for color in (GREEN,BLUE):
  button=QPushButton('New control');mark=ExtensionOutline(button);mark.color=color;layout.addWidget(button);buttons.append(button)
 win.resize(650,500);win.show();QTest.qWait(50)
 controls=[tree,combo,tabs,*buttons]
 def captures():return [w.grab().toImage() for w in controls]
 def geometry():return [w.geometry() for w in controls]
 def events():app.processEvents();QTest.qWait(50)
 before=captures();sizes=geometry();changes=[]
 combo.currentIndexChanged.connect(lambda i:changes.append(i))
 action.trigger();events();assert pref.enabled and action.isChecked()
 enabled=captures();assert all(a!=b for a,b in zip(before,enabled))
 assert sizes==geometry() and not changes
 action.trigger();events()
 assert all(a==b for a,b in zip(before,captures())) and sizes==geometry() and not changes
 action.trigger();events()
 # Popup items have the same shared policy as their selected value.
 combo.showPopup();events();popup_on=combo.view().grab().toImage()
 action.trigger();events();popup_off=combo.view().grab().toImage();assert popup_on!=popup_off
 combo.hidePopup();events();assert not pref.enabled and not action.isChecked()
 assert sizes==geometry() and not changes
 # New panels, selected algorithms and saved preferences obey the same switch.
 combo.setCurrentIndex(2);events();green_off=combo.grab().toImage()
 action.trigger();events();assert combo.grab().toImage()!=green_off
 assert HighlightSettings(settings=settings).enabled
 late=QComboBox();late.addItems(['New']);mark_combo(late,['New']);layout.addWidget(late);events()
 late_on=late.grab().toImage();action.trigger();events();assert late.grab().toImage()!=late_on
 assert not HighlightSettings(settings=settings).enabled
 # Live translation is also applied to the new View menu action.
 manager=install();manager.set_mode('fr',persist=False);manager.refresh();events()
 assert action.text()=='Surbrillance des nouveautés',action.text()
 manager.set_mode('en',persist=False);manager.refresh();events();assert action.text()=='Highlight new features'
 manager.shutdown();win.close();events()
report=dict(passed=True,default_off=True,tree_combo_popup_tabs_controls=True,three_colors=True,
 immediate_repaint=True,geometry_unchanged=True,no_value_signals=True,persistent_choice=True,live_translation=True)
Path(__file__).with_name('results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
