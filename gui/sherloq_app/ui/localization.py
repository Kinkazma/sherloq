"""Live presentation-only localization. Analysis identifiers and data stay canonical.

No widgets/models are recreated. Combo source strings live in a private item role;
consumer code uses choice() for algorithm dispatch, independently of visible text.
User data cells, EXIF trees, browser content and file paths are never translated.
"""
import json
import os
from pathlib import Path
import re
import sys
from functools import wraps, lru_cache

from PySide6.QtCore import QObject, Signal, QEvent, QTimer, QLocale, QSettings, QSignalBlocker, QProcess, QTranslator, QLibraryInfo, Qt
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QAbstractButton, QGroupBox,
    QComboBox, QTabBar, QMenu, QToolBar, QDockWidget, QListWidget, QTreeWidget, QAbstractItemView, QLineEdit, QAbstractSpinBox, QStatusBar, QTableWidget)
from shiboken6 import isValid

SOURCE_ROLE = Qt.UserRole + 1403
LITERAL_ROLE = Qt.UserRole + 1404
CATALOG = json.loads((Path(__file__).with_name('locales')/'fr.json').read_text())
REVERSE = {v:k for k,v in CATALOG.items() if k != v}
_INSTANCE = None


def _templates(catalog):
    rows=[]
    for source,target in catalog.items():
        if not re.search(r'\{\d+\}',source):continue
        parts=re.split(r'(\{\d+\})',source)
        pattern=''.join('(.+?)' if re.fullmatch(r'\{\d+\}',part) else re.escape(part) for part in parts)
        slots=[p for p in parts if re.fullmatch(r'\{\d+\}',p)]
        rows.append((re.compile('^'+pattern+'$', re.DOTALL),slots,target,source.startswith(('PatchMatch Zernike + PatchMatch SIFT:', 'Forgeryscope microscopy:', 'ELA biomes:', 'ELA · biomes:'))))
    return rows

TEMPLATES={'fr':_templates(CATALOG),'en':_templates(REVERSE)}


def language_for(preferences):
    for name in preferences:
        code=str(name).replace('_','-').split('-')[0].lower()
        if code in ('en','fr'):return code
    return 'en'


def t(text, language=None):
    """Translate UI text only; unknown text and captured user values stay intact."""
    if not isinstance(text,str) or not text:return text
    language=language or (_INSTANCE.language if _INSTANCE is not None else 'en')
    return _translate(text,language)


@lru_cache(maxsize=4096)
def _translate(text,language):
    if not any(c.isalpha() for c in text):return text
    catalog=CATALOG if language=='fr' else REVERSE
    if text in catalog:return catalog[text]
    if re.search(r'<[a-zA-Z][^>]*>',text):
        return re.sub(r'(^|>)([^<>]+)(?=<|$)',lambda m:m[1]+t(m[2],language),text)
    if text.strip()!=text:
        translated=t(text.strip(),language)
        if translated!=text.strip():return text[:len(text)-len(text.lstrip())]+translated+text[len(text.rstrip()):]
    for separator in ('\n',' | '):
        if separator in text:return separator.join(t(part,language) for part in text.split(separator))
    for pattern,slots,target,status in TEMPLATES[language]:
        match=pattern.fullmatch(text)
        if match:
            for slot,value in zip(slots,match.groups()):target=target.replace(slot,t(value,language) if status else value)
            return target
    # Composite progress labels are made of independent, known phrases.
    for separator in (' · ',):
        if separator in text:return separator.join(t(part,language) for part in text.split(separator))
    return text


def choice(combo):
    """Return the original item identifier, never its translated display label."""
    record=combo.currentData(SOURCE_ROLE)
    current=combo.currentText()
    return record[0] if record and len(record)==2 and record[1]==current else current


def localized_data(function):
    """Translate only display/tooltip roles of application-owned legend models."""
    @wraps(function)
    def data(self,index,role=Qt.DisplayRole):
        value=function(self,index,role)
        return t(value) if role in (Qt.DisplayRole,Qt.ToolTipRole) else value
    return data


class LanguageManager(QObject):
    changed=Signal(str)

    def __init__(self,app):
        super().__init__(app)
        self.app=app;self.applying=False;self.closed=False;self.language='en';self.mode='system'
        self.system_language=language_for(QLocale.system().uiLanguages())
        override=os.environ.get('SHERLOQ_LANGUAGE')
        self.mode=override if override in ('en','fr','system') else str(QSettings().value('interface/language','system'))
        if self.mode not in ('system','en','fr'):self.mode='system'
        self.qt_translator=QTranslator(self)
        self.refresh_timer=QTimer(self);self.refresh_timer.setSingleShot(True);self.refresh_timer.timeout.connect(self.refresh)
        self.probe=QProcess(self);self.probe.finished.connect(self.system_result)
        self.probe.errorOccurred.connect(lambda _:None)
        self.watch=QTimer(self);self.watch.setInterval(5000);self.watch.timeout.connect(self.probe_system)
        self.language=self.system_language if self.mode=='system' else self.mode
        self.set_qt_language()
        app.installEventFilter(self)
        self.watch.start();QTimer.singleShot(0,self.probe_system)
        app.aboutToQuit.connect(self.shutdown)

    def set_qt_language(self):
        self.app.removeTranslator(self.qt_translator)
        if self.language=='fr' and self.qt_translator.load('qtbase_fr',QLibraryInfo.path(QLibraryInfo.TranslationsPath)):
            self.app.installTranslator(self.qt_translator)

    def set_mode(self,mode,persist=True):
        if mode not in ('system','en','fr'):raise ValueError('Unsupported interface language')
        self.mode=mode
        if persist:QSettings().setValue('interface/language',mode)
        self.activate(self.system_language if mode=='system' else mode)
        if mode=='system':self.probe_system()

    def activate(self,language):
        different=language!=self.language
        self.language=language
        if different:self.set_qt_language()
        self.refresh()
        if different:self.changed.emit(language)

    def probe_system(self):
        if self.closed or self.mode!='system':return
        if sys.platform=='darwin':
            if self.probe.state()==QProcess.NotRunning:
                # Target only the language preference; never dump all preferences.
                self.probe.start('/usr/bin/defaults',['read','-g','AppleLanguages'])
        else:self.set_system_preferences(QLocale.system().uiLanguages())

    def system_result(self,code,*_):
        raw=bytes(self.probe.readAllStandardOutput()).decode(errors='replace')
        self.probe.readAllStandardError()
        if code==0:
            languages=re.findall(r'(?m)^\s*"?([a-zA-Z]{2,3}(?:[-_][a-zA-Z0-9]+)*)"?\s*,?\s*$',raw)
            if languages:self.set_system_preferences(languages)

    def set_system_preferences(self,languages):
        self.system_language=language_for(languages)
        if self.mode=='system' and self.language!=self.system_language:self.activate(self.system_language)

    def scalar(self,obj,name,getter,setter):
        value=getter()
        if not isinstance(value,str):return
        records=getattr(obj,'_language_sources',None)
        if records is None:records={};obj._language_sources=records
        old=records.get(name)
        source=old[0] if old and old[1]==value else value
        rendered=t(source,self.language);records[name]=(source,rendered)
        if rendered!=value:
            if isinstance(obj,QAction):setter(rendered)
            else:
                with QSignalBlocker(obj):setter(rendered)

    def item(self,item,column=0):
        for role in (Qt.DisplayRole,Qt.ToolTipRole):
            store=SOURCE_ROLE+(0 if role==Qt.DisplayRole else 1)
            current=item.data(column,role) or '';record=item.data(column,store)
            source=record[0] if record and record[1]==current else current
            rendered=t(source,self.language)
            item.setData(column,store,[source,rendered])
            if rendered!=current:item.setData(column,role,rendered)
        for i in range(item.childCount()):self.item(item.child(i))

    def widget(self,w):
        if not isValid(w):return
        for name in ('windowTitle','toolTip','statusTip','whatsThis'):
            self.scalar(w,name,getattr(w,name),getattr(w,'set'+name[0].upper()+name[1:]))
        if isinstance(w,(QLabel,QAbstractButton)):
            self.scalar(w,'text',w.text,w.setText)
        if isinstance(w,(QGroupBox,QMenu,QToolBar,QDockWidget)):
            if isinstance(w,(QGroupBox,QMenu)):self.scalar(w,'title',w.title,w.setTitle)
        if isinstance(w,QLineEdit):self.scalar(w,'placeholderText',w.placeholderText,w.setPlaceholderText)
        if isinstance(w,QAbstractSpinBox):
            self.scalar(w,'specialValueText',w.specialValueText,w.setSpecialValueText)
            for name in ('prefix','suffix'):
                if hasattr(w,name):self.scalar(w,name,getattr(w,name),getattr(w,'set'+name.title()))
        if isinstance(w,QStatusBar):self.scalar(w,'message',w.currentMessage,lambda v:w.showMessage(v,10000))
        if isinstance(w,QTableWidget):
            # Translate headers only; analysis/file data cells remain verbatim.
            for i in range(w.columnCount()):
                item=w.horizontalHeaderItem(i)
                if item is not None:
                    record=item.data(SOURCE_ROLE);current=item.text()
                    source=record[0] if record and record[1]==current else current
                    rendered=t(source,self.language);item.setData(SOURCE_ROLE,[source,rendered])
                    if rendered!=current:item.setText(rendered)
        if isinstance(w,QComboBox):
            with QSignalBlocker(w),QSignalBlocker(w.model()):
                for i in range(w.count()):
                    if w.itemData(i,LITERAL_ROLE):continue
                    current=w.itemText(i);record=w.itemData(i,SOURCE_ROLE)
                    source=record[0] if record and record[1]==current else current
                    rendered=t(source,self.language)
                    w.setItemData(i,[source,rendered],SOURCE_ROLE)
                    if rendered!=current:w.setItemText(i,rendered)
        if isinstance(w,QTabBar):
            with QSignalBlocker(w):
                for i in range(w.count()):self.scalar(w,'tab'+str(i),lambda i=i:w.tabText(i),lambda v,i=i:w.setTabText(i,v))
        if isinstance(w,QListWidget):
            with QSignalBlocker(w),QSignalBlocker(w.model()):
                for i in range(w.count()):
                    item=w.item(i);current=item.text();record=item.data(SOURCE_ROLE)
                    source=record[0] if record and record[1]==current else current
                    rendered=t(source,self.language);item.setData(SOURCE_ROLE,[source,rendered])
                    if rendered!=current:item.setText(rendered)
        if isinstance(w,QTreeWidget) and w.property('localizeToolTree'):
            with QSignalBlocker(w),QSignalBlocker(w.model()):
                for i in range(w.topLevelItemCount()):self.item(w.topLevelItem(i))
        for action in w.actions():
            if action.property('localizationInvariant'):continue
            self.scalar(action,'text',action.text,action.setText)
            self.scalar(action,'toolTip',action.toolTip,action.setToolTip)
        # Existing Matplotlib artists are updated in place, without analysis calls.
        if hasattr(w,'figure') and hasattr(w,'draw_idle'):
            changed=False
            for ax in w.figure.axes:
                texts=[ax.title,ax.xaxis.label,ax.yaxis.label]
                if hasattr(ax,'zaxis'):texts.append(ax.zaxis.label)
                for artist in texts:
                    current=artist.get_text();old=getattr(artist,'_language_source',None)
                    source=old[0] if old and old[1]==current else current;rendered=t(source,self.language)
                    artist._language_source=(source,rendered)
                    if rendered!=current:artist.set_text(rendered);changed=True
            if changed:w.draw_idle()

    def refresh(self):
        if self.applying:return
        self.applying=True
        try:
            for w in self.app.allWidgets():
                self.widget(w)
                if isinstance(w,QAbstractItemView):w.viewport().update()
                if w.__class__.__name__ in ('ImageWorkspace','SelectionView','AutomaticView','PlotAxis'):w.update()
        finally:self.applying=False

    def eventFilter(self,obj,event):
        if self.closed or self.applying:return False
        kind=event.type()
        if kind in (QEvent.LocaleChange,QEvent.ApplicationActivate):
            self.probe_system()
            if not self.refresh_timer.isActive():self.refresh_timer.start(0)
        elif isinstance(obj,QWidget) and kind in (QEvent.Show,QEvent.Paint):
            self.applying=True
            try:self.widget(obj)
            finally:self.applying=False
        return False

    def add_menu(self,menu_bar):
        menu=menu_bar.addMenu('Language');group=QActionGroup(menu);group.setExclusive(True)
        self.language_actions={}
        for mode,title in [('system','System language'),('fr','Français'),('en','English')]:
            action=QAction(title,menu);action.setCheckable(True);action.setChecked(mode==self.mode)
            if mode!='system':action.setProperty('localizationInvariant',True)
            group.addAction(action);menu.addAction(action);self.language_actions[mode]=action
            action.triggered.connect(lambda _,m=mode:self.set_mode(m))
        return menu

    def shutdown(self):
        self.closed=True;self.app.removeEventFilter(self)
        self.watch.stop();self.refresh_timer.stop()
        if self.probe.state()!=QProcess.NotRunning:self.probe.kill();self.probe.waitForFinished(500)


def install():
    global _INSTANCE
    if _INSTANCE is None or not isValid(_INSTANCE):_INSTANCE=LanguageManager(QApplication.instance())
    return _INSTANCE
