"""Shared named snapshots of the four ELA energy controls, persisted locally."""
import json
import uuid
from PySide6.QtCore import QObject,Signal,QSettings,QCoreApplication,QSignalBlocker
from PySide6.QtWidgets import QToolButton,QMenu,QInputDialog,QMessageBox
from PySide6.QtGui import QAction
from .localization import t,LITERAL_ROLE

BUILTINS=(('Aggressive','sensitive'),('Sensitive','conservative'),('Conservative','standard'),('Manual','manual'))
STANDARD=(10,990,50,50)  # Slider units: 0.1 percentile point and 0.1 deviation.
RANGES=((0,500),(500,1000),(0,200),(0,200))

def compute_profile(combo):
    value=combo.currentData()
    return value if value in ('sensitive','conservative') else 'sensitive'

class ProfileStore(QObject):
    changed=Signal()
    KEY='ela/energy_profiles_v1'
    def __init__(self,settings=None,parent=None):
        super().__init__(parent)
        self.settings=QSettings() if settings is None else settings
        self.items=[]
        try:
            data=json.loads(self.settings.value(self.KEY,'[]'))
            if not isinstance(data,list):return
            for item in data:
                try:
                    name=self.valid_name(item['name']);values=self.valid_values(item['values']);key=item['id']
                    if not isinstance(key,str) or not key.startswith('user:') or len(key)!=37:continue
                    uuid.UUID(hex=key[5:])
                    if self.get(key):continue
                    self.items.append(dict(id=key,name=name,values=values))
                except (ValueError,KeyError,TypeError):continue
        except (ValueError,TypeError):pass
    def get(self,key):return next((p for p in self.items if p['id']==key),None)
    def valid_name(self,name,exclude=None):
        if not isinstance(name,str):raise ValueError('Enter a profile name.')
        name=name.strip()
        if not name or len(name)>80 or any(ord(c)<32 for c in name):raise ValueError('Use a name from 1 to 80 characters.')
        reserved={label.casefold() for label,_ in BUILTINS}|{'sensible','prudent','agressif','conservateur','manuel','standard'}
        if name.casefold() in reserved or any(p['name'].casefold()==name.casefold() and p['id']!=exclude for p in self.items):
            raise ValueError('This profile name is already used.')
        return name
    @staticmethod
    def valid_values(values):
        if not isinstance(values,(tuple,list)) or len(values)!=4:raise ValueError('Invalid ELA profile values.')
        if any(type(v) is not int or not lo<=v<=hi for v,(lo,hi) in zip(values,RANGES)):raise ValueError('Invalid ELA profile values.')
        return list(values)
    def write(self,items):
        self.settings.setValue(self.KEY,json.dumps(items,ensure_ascii=False));self.settings.sync()
        if self.settings.status()!=QSettings.NoError:raise OSError('Could not save ELA profiles.')
        self.items=items;self.changed.emit()
    def add(self,name,values):
        record=dict(id='user:'+uuid.uuid4().hex,name=self.valid_name(name),values=self.valid_values(values))
        self.write([*self.items,record]);return record['id']
    def rename(self,key,name):
        if self.get(key) is None:raise ValueError('Select a saved profile.')
        name=self.valid_name(name,key)
        self.write([{**p,'name':name} if p['id']==key else p for p in self.items])
    def remove(self,key):
        if self.get(key) is None:raise ValueError('Select a saved profile.')
        self.write([p for p in self.items if p['id']!=key])

def profile_store():
    app=QCoreApplication.instance()
    if not hasattr(app,'_ela_profile_store'):app._ela_profile_store=ProfileStore(parent=app)
    return app._ela_profile_store

class ElaProfiles(QObject):
    def __init__(self,owner,combo,automatic,sliders,changed,row):
        super().__init__(owner)
        self.owner=owner;self.combo=combo;self.automatic=automatic;self.sliders=sliders;self.recompute=changed
        self.store=profile_store()
        self.menu_button=QToolButton();self.menu_button.setText('Profiles…')
        self.menu_button.setPopupMode(QToolButton.InstantPopup)
        self.menu=QMenu(self.menu_button);self.menu_button.setMenu(self.menu)
        self.save=QAction('Save profile…',self.menu);self.rename=QAction('Rename profile…',self.menu);self.delete=QAction('Delete profile',self.menu)
        for action in (self.save,self.rename,self.delete):self.menu.addAction(action)
        row.insertWidget(2,self.menu_button)
        self.save.setToolTip('Save the current histogram bounds and shadow/highlight deviations.')
        self.save.triggered.connect(self.save_current);self.rename.triggered.connect(self.rename_current);self.delete.triggered.connect(self.delete_current)
        self.store.changed.connect(self.refresh);self.refresh()
    def refresh(self):
        selected=self.combo.currentData() or 'standard'
        with QSignalBlocker(self.combo):
            self.combo.clear()
            for label,key in BUILTINS:self.combo.addItem(t(label),key)
            for item in self.store.items:
                self.combo.addItem(item['name'],item['id'])
                # A user name is literal, even if it matches a translation key.
                self.combo.setItemData(self.combo.count()-1,True,LITERAL_ROLE)
            index=self.combo.findData(selected)
            self.combo.setCurrentIndex(index if index>=0 else self.combo.findData('manual'))
        self.buttons()
    def buttons(self):
        saved=self.store.get(self.combo.currentData()) is not None
        self.rename.setEnabled(saved);self.delete.setEnabled(saved)
        self.automatic.setEnabled(self.combo.currentData() in ('sensitive','conservative'))
    def apply(self):
        key=self.combo.currentData();self.buttons()
        adaptive=key in ('sensitive','conservative')
        with QSignalBlocker(self.automatic):self.automatic.setChecked(adaptive)
        item=self.store.get(key)
        values=STANDARD if key=='standard' else item['values'] if item else None
        if values is not None:
            for slider,value in zip(self.sliders,values):
                with QSignalBlocker(slider):slider.setValue(value)
                label=slider.property('valueLabel')
                if label is not None:label.setText(f'{value/10:.1f}'+(' %' if slider.property('histogramPercentile') else ''))
        self.recompute()
    def manual(self):
        # Changing a snapshot never overwrites the saved profile or applies a new one.
        with QSignalBlocker(self.combo),QSignalBlocker(self.automatic):
            self.combo.setCurrentIndex(self.combo.findData('manual'))
            self.automatic.setChecked(False)
        self.buttons()
    def snapshot(self):
        return dict(id=self.combo.currentData(),name=self.combo.currentText(),values=[s.value() for s in self.sliders],adaptive=self.automatic.isChecked())
    def error(self,error):QMessageBox.warning(self.owner,t('ELA profiles'),t(str(error)))
    def save_current(self):
        name,ok=QInputDialog.getText(self.owner,t('Save ELA profile'),t('Profile name:'))
        if not ok:return
        try:
            key=self.store.add(name,[s.value() for s in self.sliders]);self.combo.setCurrentIndex(self.combo.findData(key))
        except (ValueError,OSError) as e:self.error(e)
    def rename_current(self):
        item=self.store.get(self.combo.currentData())
        if item is None:return
        name,ok=QInputDialog.getText(self.owner,t('Rename ELA profile'),t('Profile name:'),text=item['name'])
        if ok:
            try:self.store.rename(item['id'],name)
            except (ValueError,OSError) as e:self.error(e)
    def delete_current(self):
        key=self.combo.currentData()
        if self.store.get(key) is None:return
        try:self.store.remove(key)
        except (ValueError,OSError) as e:self.error(e)
