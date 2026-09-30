"""Serial/parallel preserve exact retained links from each overlapping search."""
import sys,json
from pathlib import Path
import cv2 as cv,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core.cloning2 import Cloning2Engine
rng=np.random.default_rng(935);im=rng.integers(30,220,(112,200,3),np.uint8)
im[34:74,120:160]=im[24:64,25:65]
zones=(((5,5),(194,5),(194,106),(5,106)),((10,10),(189,10),(189,101),(10,101)))
p=('PatchMatch Zernike',1200,230.,5.,.3,25.,'Similarity',3.,6,4,4,False,2.,True,True,False,(),())
a=Cloning2Engine(im).analyze((*p,1),zones)
b=Cloning2Engine(im).analyze((*p,4),zones)
assert np.array_equal(a['points'],b['points']) and np.array_equal(a['pairs'],b['pairs'])
assert np.array_equal(a['pair_search_regions'],b['pair_search_regions'])
assert set(a['pair_search_regions'])=={0,1}
sets=[]
for region in (0,1):
 ids=a['pairs'][a['pair_search_regions']==region,:2].astype(int)
 coords=a['points'][ids,:2];swap=(coords[:,0,0]>coords[:,1,0])|((coords[:,0,0]==coords[:,1,0])&(coords[:,0,1]>coords[:,1,1]))
 coords[swap]=coords[swap,::-1];sets.append(set(map(tuple,coords.reshape(-1,4))))
shared=len(sets[0]&sets[1]);assert shared>0,'Fixture must detect the same links in independent overlapping searches'
for group in a['groups']:assert len(np.unique(a['pair_search_regions'][group]))==1
report=dict(passed=True,serial_parallel_exact=True,overlapping_searches_preserved=True,shared_links_kept_per_search=shared,pairs=len(a['pairs']))
Path(__file__).with_name('dense-origins-results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
