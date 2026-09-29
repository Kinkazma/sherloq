import os;os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
import sys,json,gc
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from PySide6.QtWidgets import QApplication,QStyleOptionGraphicsItem
from PySide6.QtGui import QImage,QPainter
from PySide6.QtCore import QRectF,QPointF
from gui.sherloq_app.core.memory_resources import TemporaryArrays,MiB,resident_memory
from gui.sherloq_app.core.utility import mat2img
from gui.sherloq_app.ui.tiled_image import TiledImageItem
from gui.sherloq_app.ui.viewer import DynamicView,ImageViewer
app=QApplication([]);rng=np.random.default_rng(55);store=TemporaryArrays(32*MiB)
im=store.array((545,567,3),np.uint8);im[:]=rng.integers(0,256,im.shape,dtype=np.uint8)
item=TiledImageItem(im);out=QImage(567,545,QImage.Format_BGR888);out.fill(0)
option=QStyleOptionGraphicsItem();option.exposedRect=item.boundingRect();p=QPainter(out);item.paint(p,option);p.end()
def pixels(q):return np.frombuffer(q.constBits(),np.uint8).reshape(q.height(),q.bytesPerLine())[:,:q.width()*3].reshape(q.height(),q.width(),3)
assert np.array_equal(pixels(out),im)
view=DynamicView(im);assert isinstance(view.image_item,TiledImageItem);assert view.scene.sceneRect()==QRectF(0,0,567,545)
view.set_image(np.zeros((32,64,3),np.uint8));assert not isinstance(view.image_item,TiledImageItem)
view.set_image(im);assert isinstance(view.image_item,TiledImageItem) and len(view.scene.items())==1
view.resize(700,500);view.show();app.processEvents();view.zoom_fit()
pos=QPointF(140,220);before=view.viewportTransform().inverted()[0].map(pos);view.zoom_at(1.7,pos);app.processEvents()
assert (view.viewportTransform().map(before)-pos).manhattanLength()<2
viewer=ImageViewer(np.asarray(im),np.asarray(im));viewer.update_processed(im);assert isinstance(viewer.view.image_item,TiledImageItem)
# Beyond QPixmap texture dimensions: paint just a visible exact source window.
large=store.array((25000,40000,3),np.uint8);large[24000:24512,39000:39512]=[31,77,193]
big=TiledImageItem(large);option.exposedRect=QRectF(39000,24000,512,512)
out=QImage(512,512,QImage.Format_BGR888);out.fill(0);p=QPainter(out);p.translate(-39000,-24000);big.paint(p,option);p.end()
assert np.all(pixels(out)==[31,77,193]);assert big.cache_bytes<4*MiB
# Reduced display tiles leave source bytes and full-resolution coordinates intact.
a=np.asarray(im).copy();small=item.tile(2,0,0,1024);assert small.width()==142 and small.height()==137
assert np.array_equal(im,a) and item.cache_bytes<=item.cache_limit
r=dict(passed=True,mapped_output_display=True,exact_100_percent=True,billion_pixel_visible_window=True,full_resolution_coordinates=True,cache_bytes=big.cache_bytes)
Path(__file__).with_name('viewer-results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
viewer.close();view.close();app.processEvents()
