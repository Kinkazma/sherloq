from gui.sherloq_app.ui.localization import t
"""GUI file selection and cancellable image I/O workers."""
import os,sys
from pathlib import Path
from threading import Event
from PySide6.QtCore import QMimeDatabase,QSettings,QThreadPool
from PySide6.QtWidgets import QFileDialog
from .jobs import LatestJob
from ..core.image_io import RAW_EXTENSIONS,decode_image,decode_image_isolated,save_image
from ..core.jpeg_curve import Cancelled

_LOAD_POOL=QThreadPool();_LOAD_POOL.setMaxThreadCount(1)
if sys.platform=='darwin':_LOAD_POOL.setStackSize(8*1024*1024)


def choose_image(parent):
    patterns=[];db=QMimeDatabase()
    for mime in ['image/jpeg','image/png','image/tiff','image/gif','image/bmp','image/webp','image/x-portable-pixmap','image/x-portable-graymap','image/x-portable-bitmap']:
        patterns.extend(db.mimeTypeForName(mime).globPatterns())
    patterns.extend('*.'+ext for ext in sorted(RAW_EXTENSIONS))
    selected,_=QFileDialog.getOpenFileName(parent,t(parent.tr('Open image')),QSettings().value('load_folder',os.path.expanduser('~')),t('Supported formats ('+' '.join(patterns)+');;All files (*)'))
    return selected or None


def _decode(request):
    filename,event=request
    try:
        decoder=decode_image_isolated if Path(filename).suffix.lower().lstrip('.') in RAW_EXTENSIONS else decode_image
        return decoder(filename,event.is_set)
    except Cancelled:return None


def _save(request):
    filename,image,event=request
    try:return save_image(filename,image,event.is_set)
    except Cancelled:return None


class ImageLoadJob(LatestJob):
    def __init__(self,parent):
        super().__init__(parent,_decode,delay=0,pool=_LOAD_POOL);self.event=Event()
    def load(self,filename):
        self.event.set();self.event=Event();self.request((filename,self.event))
    def invalidate(self):self.event.set();super().invalidate()
    def shutdown(self):self.event.set();super().shutdown()


class ImageExportJob(LatestJob):
    def __init__(self,parent):
        super().__init__(parent,_save,delay=0);self.event=Event()
    def save(self,filename,image):
        self.event.set();self.event=Event()
        # Snapshot before returning to the UI; later tool edits cannot change it.
        self.request((filename,image.copy(),self.event))
    def invalidate(self):self.event.set();super().invalidate()
    def shutdown(self):self.event.set();super().shutdown()
