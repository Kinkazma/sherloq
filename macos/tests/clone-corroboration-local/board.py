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

from gui.sherloq_app.core.automatic_clones import entries,render as render_auto
from gui.sherloq_app.core.clone_relations import annotate
from gui.sherloq_app.core.clone_corroboration import context_counts
from gui.sherloq_app.core.auto_zones import enclosing
folder=Path(__file__).parent
rows=entries(sift=r,high=p[2]);rows=annotate(rows,r['regions'],enclosing(r['regions']),im.shape)
counts={key:sum(e['relation']==key for e in rows) for key in ('within','between','unassigned')}
assert all(len(v)==len(r['pairs']) for v in (r['pair_search_regions'],))
(folder/'display-entries-private.json').write_text(json.dumps(rows,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist(),indent=2)+'\n')
for relation in ('within','between','all'):
 selected=tuple(e for e in rows if relation=='all' or e['relation']==relation)
 for mode in ('biomes','overlay','heat'):
  cv.imwrite(str(folder/f'board-{relation}-{mode}.jpg'),render_auto((im,selected,(),mode,.7)))
report=dict(seconds=time.perf_counter()-start,relations=counts,groups=found,
            contexts=len({e['search_context'] for e in rows}),heat_eligible=sum(e['heat_eligible'] for e in rows),local_preview_only=True,no_count_weighting=True,visual_attenuation_only=True)
(folder/'board-results.json').write_text(json.dumps(report,indent=2)+'\n')
assert counts['within'] and counts['between']
assert any({x['a'],x['b']}=={1,3} and x['variant']=='reflection' for x in found)
assert any(x['count']==313 and x['variant']=='normal' for x in found)
print('RELATIONS',counts,'CONTEXTS',report['contexts'],flush=True)
