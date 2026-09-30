"""Split disconnected geometric survivors; preserve arbitrarily large coherent groups."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core.copy_subbiomes import refine
from gui.sherloq_app.core.cloning2 import palette

def exercise(coords,expected):
    n=len(coords);points=np.zeros((n*2,7));points[:n,:2]=coords;points[n:,:2]=coords+(1500,0)
    pairs=np.column_stack((np.arange(n),np.arange(n)+n,np.full(n,.1),np.full(n,1500.)))
    group=np.arange(n);groups=(group,)
    model=dict(matrix=[[1,0,1500],[0,1,0],[0,0,1]],source_point_indices=list(range(n)),
               destination_point_indices=list(range(n,n*2)),inliers=n,model='Affine',
               distinct_centres=[n,n],median_error_px=0.,maximum_error_px=0.)
    colors,bases=palette(groups,pairs,.725)
    g,m,c,b,p=refine(points,pairs,groups,(model,),colors,bases,50.)
    assert len(g)==expected and np.array_equal(np.sort(np.concatenate(g)),group)
    assert all(np.array_equal(mm['matrix'],model['matrix']) for mm in m)
    assert model['inliers']==n
    if expected==1:
        assert np.array_equal(g[0],group) and m[0] is model
        assert b==bases and np.array_equal(c,colors)
    else:
        assert all(mm['parent_inliers']==n for mm in m)
        assert [mm['inliers'] for mm in m]==[len(gg) for gg in g]
    return {'points':n,'parts':len(g),'raw_evidence_preserved':True}

# Two tiny islands once connected by rejected outliers must no longer share a hull.
small=np.array([(x,y) for x in (0,5,10) for y in (0,5,10)],float)
split=exercise(np.vstack((small,small+(0,180))),2)
# Large coherent area, with the same local spacing: no maximum diameter rule.
large=exercise(np.array([(x,y) for x in range(0,801,25) for y in range(0,501,25)],float),1)
compact=exercise(small,1)
print(json.dumps(dict(passed=True,split=split,large=large,compact=compact)))
