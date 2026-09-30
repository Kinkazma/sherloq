"""Private real-image validation; image path is supplied, never exported."""
import sys,time,json
from pathlib import Path
import cv2 as cv
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core.cloning2 import Cloning2Engine,PANELS_TEXT,render
im=cv.imread(sys.argv[1]);assert im is not None
cv.setNumThreads(6)
p=(PANELS_TEXT,6000,float((im.shape[0]**2+im.shape[1]**2)**.5),10.,.725,50.,'Affine',5.,10,8,8,False,2.,False,True,False,(),())
engine=Cloning2Engine(im);start=time.perf_counter()
r=engine.analyze(p,progress=lambda n,s:print(n,s,flush=True) if n in (0,10,82,91,100) else None)
record=dict(passed=True,seconds=time.perf_counter()-start,shape=im.shape,points=len(r['points']),
            pairs=len(r['pairs']),biomes=len(r['groups']),zones=len(r['regions']),backend=r['backend'],
            text_boxes=r['preprocessing']['text_boxes'],rejected=len(r['rejected_biomes']))
assert len(r['regions'])==17
out,visible,_=render(im,r,(10.,p[2],2,(),False,False,False,True,(),True))
cv.imwrite(str(Path(__file__).with_name('board-overlay.jpg')),out)
Path(__file__).with_name('board-results.json').write_text(json.dumps(record,indent=2)+'\n');print(record)
