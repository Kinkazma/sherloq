"""Compare cached private real evidence without rerunning feature matching."""
import sys,json,time
from pathlib import Path
import numpy as np,cv2 as cv
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core.copy_subbiomes import refine
from gui.sherloq_app.core.cloning2 import render,palette
z=np.load(Path(__file__).with_name('before.npz'));r=json.loads(str(z['metadata']))
r['points']=z['points'];r['pairs']=z['pairs'];r['groups']=tuple(z[f'group{i}'] for i in range(len(r['models'])))
r['colors'],r['bases']=palette(r['groups'],r['pairs'],.725)
start=time.perf_counter();g,m,c,b,parts=refine(r['points'],r['pairs'],r['groups'],r['models'],r['colors'],r['bases'],50.)
largest=int(np.argmax([len(g) for g in r['groups']]))
children=[i for i,p in enumerate(parts) if p['parent']==largest]
assert len(children)==1 and np.array_equal(g[children[0]],r['groups'][largest])
assert b[children[0]]==r['bases'][largest]
assert np.array_equal(np.sort(np.concatenate(g)),np.sort(np.concatenate(r['groups'])))
record=dict(passed=True,seconds=time.perf_counter()-start,before=len(r['groups']),after=len(g),
            parents_split=len({p['parent'] for p in parts if p['parts']>1}),large_matches=len(g[children[0]]),
            large_geometry_and_colour_exact=True,all_retained_correspondences_preserved=True)
im=cv.imread(sys.argv[1]);r.update(groups=g,models=m,colors=c,bases=b,biome_partitions=parts)
out,visible,_=render(im,r,(10.,4000.,2,(),False,False,False,True,(),False))
cv.imwrite(str(Path(__file__).with_name('after.jpg')),out)
Path(__file__).with_name('results.json').write_text(json.dumps(record,indent=2)+'\n');print(record)
