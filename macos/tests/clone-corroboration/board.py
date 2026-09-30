import sys,time,json
from pathlib import Path
import cv2 as cv,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core.cloning2 import Cloning2Engine,PANELS_TEXT,render
from gui.sherloq_app.core.automatic_clones import entries
from gui.sherloq_app.core.clone_corroboration import render_heat
im=cv.imread(sys.argv[1]);cv.setNumThreads(6)
p=(PANELS_TEXT,6000,float(np.hypot(*im.shape[:2])),10.,.725,50.,'Affine',5.,10,8,8,True,2.,False,True,False,(),())
e=Cloning2Engine(im);start=time.perf_counter();r=e.analyze(p)
print('seconds',time.perf_counter()-start,'groups',len(r['groups']),'mirror',sum(m.get('variant')=='reflection' for m in r['models']),flush=True)
regions=r['regions'];found=[]
for g,m in zip(r['groups'],r['models']):
 a=r['points'][m['source_point_indices'],:2];b=r['points'][m['destination_point_indices'],:2]
 def zone(xy):
  counts=[sum(cv.pointPolygonTest(np.array(z,np.float32),tuple(map(float,x)),False)>=0 for x in xy) for z in regions[:-1]]
  return int(np.argmax(counts))+1
 found.append(dict(a=zone(a),b=zone(b),count=len(g),variant=m.get('variant','normal'),det=float(np.linalg.det(np.array(m['matrix'])[:2,:2]))))
print(json.dumps(found),flush=True)
out,_,_=render(im,r,(10.,p[2],2,(),False,False,False,True,(),False))
folder=Path(__file__).parent;folder.mkdir(exist_ok=True)
cv.imwrite(str(folder/'board-mirror.jpg'),out)
(folder/'board-results.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,groups=found),indent=2))
np.savez_compressed(folder/'board-evidence-private.npz',points=r['points'],pairs=r['pairs'],**{f'group{i}':g for i,g in enumerate(r['groups'])})
assert any({x['a'],x['b']}=={1,3} and x['variant']=='reflection' for x in found),found
