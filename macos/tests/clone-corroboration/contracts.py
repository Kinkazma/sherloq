"""Agreement is per source; near-identical pairs do not erase smaller evidence."""
import sys,json,tempfile
from pathlib import Path
import cv2 as cv,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core.automatic_clones import SOURCES,_entry,point_entries,export
from gui.sherloq_app.core.clone_corroboration import votes,unique_envelopes,render_heat
from gui.sherloq_app.core.cloning2 import palette

def square(x,y,size):return [[x,y],[x+size,y],[x+size,y+size],[x,y+size]]
a=_entry(SOURCES[1],[square(10,10,20),square(60,10,20)],10,{})
b=_entry(SOURCES[2],a['polygons'][::-1],12,{})
repeated=dict(a,id='duplicate')
c=_entry(SOURCES[3],[square(15,15,5),square(65,15,5)],8,{})
ela=dict(a,id='ela',source='ELA biomes')
shape=(100,100,3);rows=(a,b,repeated,c,ela)
count=votes(shape,rows)
assert count[12,12]==2 and count[17,17]==3 and count[0,0]==0
assert votes(shape,(repeated,a)).max()==1
exclude=(square(10,10,5),)
assert votes(shape,rows,exclude)[12,12]==0
assert np.array_equal(votes(shape,rows),votes(shape,rows[::-1]))
fused=unique_envelopes((a,b,repeated,c))
assert len(fused)==2 and len(fused[0]['member_ids'])==3
assert set(fused[0]['corroborating_sources'])==set(SOURCES[1:3])
# One common end is insufficient: the relation to the other end matters.
d=_entry(SOURCES[0],[a['polygons'][0],square(60,60,20)],10,{})
assert len(unique_envelopes((a,d)))==2
# AI stays independently visible even with the exact same paired envelopes.
ai=_entry(SOURCES[0],a['polygons'],10,{})
assert len(unique_envelopes((a,ai)))==2
image=np.full(shape,120,np.uint8)
assert np.array_equal(render_heat(image,rows,(),'overlay',0),image)
assert np.array_equal(render_heat(image,rows,(),'overlay',.4)[0,0],image[0,0])
assert not render_heat(image,rows,(),'heat',.4)[0,0].any()
# Old dense methods: split disconnected survivors, never by total hull area.
small=np.array([(x,y) for x in (0,5,10) for y in (0,5,10)],float)
for coords,expected,global_model in [(np.vstack((small,small+(0,180))),2,False),
 (np.array([(x,y) for x in range(0,801,25) for y in range(0,501,25)],float),1,False),
 (np.vstack((small,small+(0,180))),1,True)]:
 n=len(coords);points=np.zeros((n*2,7));points[:n,:2]=coords;points[n:,:2]=coords+(1500,0)
 pairs=np.column_stack((np.arange(n),np.arange(n)+n,np.full(n,.1),np.full(n,1500.)))
 result=dict(points=points,pairs=pairs,groups=(np.arange(n),),models=({'source_panels':[1,3]} if global_model else {},),
             group_algorithms=(SOURCES[1],),params=(None,)*5+(50.,))
 entries=point_entries(result,None,10,4000,.8,split=True)
 assert len(entries)==expected and sum(e['count'] for e in entries)==n
 assert len(result['groups'])==1
with tempfile.TemporaryDirectory() as folder:
 path=export((str(Path(folder)/'votes.npz'),dict(image_shape=shape,corroboration=dict(entries=rows,excluded=exclude))))
 with np.load(path,allow_pickle=False) as data:
  assert np.array_equal(data['root_corroboration_counts'],votes(shape,rows,exclude))
print(json.dumps(dict(passed=True,one_vote_per_method=True,ela_excluded=True,paired_dedup=True,
                     nested_evidence_preserved=True,large_connected_and_global_models_preserved=True,npz_exact=True)))
