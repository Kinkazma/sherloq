"""Canonical IDs, category, localization and persisted favorite after the rename."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ['SHERLOQ_LANGUAGE']='en'
from pathlib import Path
import sys,tempfile,json
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'source'))
from PySide6.QtCore import QSettings,Qt
from PySide6.QtWidgets import QApplication
from gui.sherloq_app.ui.tools import ToolTree
from gui.sherloq_app.ui.extensions import EXTENSION_ROLE
from gui.sherloq_app.ui.localization import install
app=QApplication([]);manager=install()
with tempfile.TemporaryDirectory() as folder:
 settings=QSettings(str(Path(folder)/'prefs.ini'),QSettings.IniFormat)
 settings.setValue(ToolTree.FAVORITES_SETTING,'["7:6"]')
 tree=ToolTree(settings=settings);manager.refresh();item=tree.tool_item(7,6)
 assert item.text(0)=='AI Clone Detection',item.text(0)
 parent=item.parent();assert parent is tree.tool_item(8,4).parent()
 assert parent.child(parent.indexOfChild(item)-1) is tree.tool_item(8,4)
 assert item.data(0,Qt.UserRole+1)==7 and item.data(0,Qt.UserRole+2)==6
 assert item.data(0,EXTENSION_ROLE)=='green'
 assert tree.tool_item(7,8).parent() is tree.tool_item(2,0).parent()
 assert tree.tool_item(7,5).parent() is tree.tool_item(7,1).parent()
 assert tree._favorite_items[(7,6)].text(0)=='AI Clone Detection'
 manager.activate('fr');app.processEvents();manager.refresh()
 assert item.text(0)=='Détection de clones par IA',item.text(0)
 assert tree._favorite_items[(7,6)].text(0)==item.text(0)
 assert item.data(0,Qt.UserRole+1)==7 and item.data(0,Qt.UserRole+2)==6
 tree.close();manager.activate('en')
 second=ToolTree(settings=settings);manager.refresh()
 assert second._favorite_items[(7,6)].text(0)=='AI Clone Detection'
 assert len(second._tool_items)==50
 second.close()
manager.closed=True;manager.watch.stop();manager.refresh_timer.stop()
record=dict(passed=True,canonical_id=[7,6],green_marker=True,ai_category_after_adaifl=True,french=True,persisted_favorite=True,menu_entries=50)
Path(__file__).with_name('menu-results.json').write_text(json.dumps(record,indent=2)+'\n');print(record)
