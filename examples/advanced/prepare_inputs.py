from pathlib import Path
import cv2 as cv
import numpy as np
from skimage import data
import json
OUT=Path(__file__).resolve().parent
original=cv.cvtColor(data.coffee(),cv.COLOR_RGB2BGR)
cv.imwrite(str(OUT/'coffee-original.png'),original)
# Controlled duplication on a real CC0 photograph: copy the spoon and its edge.
base=cv.imdecode(cv.imencode('.jpg',original,[cv.IMWRITE_JPEG_QUALITY,65])[1],1)
copy=base.copy();copy[260:370,482:576]=base[232:342,320:414]
cv.imwrite(str(OUT/'coffee-copy.png'),copy)
cv.imwrite(str(OUT/'coffee-copy.jpg'),copy,[cv.IMWRITE_JPEG_QUALITY,95])
mask=np.zeros(base.shape[:2],np.uint8);mask[260:370,482:576]=255
cv.imwrite(str(OUT/'coffee-copy-mask.png'),mask)
cv.imwrite(str(OUT/'coffee-double.jpg'),base,[cv.IMWRITE_JPEG_QUALITY,95])
defects=original.copy();points=[(240,60),(270,65),(310,70),(335,90),(355,110),(240,120),(270,140),(295,155),(325,130),(290,95)]
for i,(x,y) in enumerate(points):defects[y,x]=255 if i%2==0 else 0
cv.imwrite(str(OUT/'coffee-pixels.png'),defects)
(OUT/'input-operations.json').write_text(json.dumps({'source':'skimage.data.coffee','photographer':'Rachel Michetti','license':'CC0','source_url':'https://scikit-image.org/docs/stable/api/skimage.data#skimage.data.coffee','coffee-copy':{'base_jpeg_quality':65,'source_xywh':[320,232,94,110],'destination_xywh':[482,260,94,110],'final_jpeg_quality':95},'coffee-double.jpg':{'jpeg_qualities':[65,95],'aligned':True},'coffee-pixels.png':{'coordinates_xy':points,'values_alternating':[255,0]}},indent=2)+'\n')
