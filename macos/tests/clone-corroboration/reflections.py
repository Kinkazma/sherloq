import sys,time,json
import numpy as np,cv2 as cv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'source'))
records=[]
from gui.sherloq_app.core.cloning2 import Cloning2Engine,PANELS_TEXT
cv.setNumThreads(4)
rng=np.random.default_rng(552)
a=cv.GaussianBlur(rng.integers(0,256,(130,140,3),np.uint8),(3,3),0)
p=(PANELS_TEXT,1000,700.,10.,.725,50.,'Affine',3.,8,8,8,True,2.,True,True,False,(),())
for kind,b in [('mirror',a[:,::-1]),('vertical',a[::-1]),('rotate180',a[::-1,::-1]),('normal',a),('intact',cv.GaussianBlur(rng.integers(0,256,a.shape,np.uint8),(3,3),0))]:
 im=np.full((170,340,3),255,np.uint8);im[20:150,10:150]=a;im[20:150,190:330]=b
 zones=(((10,20),(149,20),(149,149),(10,149)),((190,20),(329,20),(329,149),(190,149)))
 start=time.perf_counter();e=Cloning2Engine(im);r=e.analyze(p,zones,True)
 print(kind,len(r['groups']),[len(g) for g in r['groups']], [round(float(np.linalg.det(np.array(m['matrix'])[:2,:2])),2) for m in r['models']],time.perf_counter()-start,flush=True)
 assert bool(r['groups'])==(kind!='intact')
 if kind in ('mirror','vertical'):
  assert all(np.linalg.det(np.array(m['matrix'])[:2,:2])<0 for m in r['models'])
 if kind=='rotate180':assert all(np.linalg.det(np.array(m['matrix'])[:2,:2])>0 for m in r['models'])
 counts=e.counts.copy();assert e.analyze(p,zones,True) is r and e.counts==counts
 isolated=e.analyze(p,zones,False)
 for pair in isolated['pairs']:
  xs=isolated['points'][pair[:2].astype(int),0]
  assert (xs[0]<170)==(xs[1]<170), 'Reflection must not connect separate Search zones'
 records.append(dict(kind=kind,groups=len(r['groups']),matches=[len(g) for g in r['groups']]))
Path(__file__).with_name('reflections-results.json').write_text(json.dumps(dict(passed=True,tests=records),indent=2)+'\n')
