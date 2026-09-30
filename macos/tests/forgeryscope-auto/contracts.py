"""Exercise all public branches without relying on a model's predictions."""
import json,sys
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'source'))
from gui.sherloq_app.core.clone_models import runtime
runtime()
from gui.sherloq_app.core.forgeryscope_auto import predict
import sherloq_clone_models.forgeryscope as api
import sherloq_clone_models.forgeryscope.matcher.lightglue as matching
import sherloq_clone_models.forgeryscope.matcher.lane as lane
from sherloq_clone_models.forgeryscope.matcher.lane import BlotLane,BlotMatch

im=np.zeros((100,500,3),np.uint8)
panels=[(label,1.,10+i*95,10,80+i*95,80) for i,label in enumerate(['Microscopy']*3+['Blots']*2)]
class Embed:
 def __init__(self,name):self.name=name
 def get_embedding_batch(self,crops):return self
 def cpu(self):return self
 def numpy(self):return np.array([[names.index(self.name)]])
names=['micro_overlap_embedder','wblot_overlap_embedder','wblot_duplicate_embedder','wblot_lane_embedder']
loaded=dict(panel=SimpleNamespace(extract_panels=lambda image:panels),lane=None,micro=None,blot=None,
 embeds={name:Embed(name) for name in names})
def fake_pairs(vectors,threshold):
 name=names[int(vectors[0,0])]
 return [(0,1,.9),(0,2,.9),(1,2,.9)] if name=='micro_overlap_embedder' else [(0,1,.86)] if name=='wblot_overlap_embedder' else [(0,1,.95)]
seen=[]
def fake_matches(rgb,ps,crops,frame,*args,**kwargs):
 assert kwargs['test_transforms_micro'] and not kwargs['test_transforms_blot']
 for _,row in frame.iterrows():
  i,j=int(row.idx1),int(row.idx2);seen.append((i,j,float(row.score)))
  record=dict(panel_id0=i,panel_id1=j,panel_label=row.label)
  for n,p in enumerate((ps[i],ps[j])):
   x0,y0,x1,y1=map(int,p[-4:]);m=np.zeros(rgb.shape[:2],np.uint8);m[y0:y1,x0:x1]=1
   record['mask'+str(n)]=m;record['poly_coords'+str(n)]=[[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
  record['match_result']=dict(inliers=10,mean_match_score=.9) if row.label=='Microscopy' else dict(inliers=0,mean_match_score=0.,fallback='bbox')
  yield record
with patch.object(api.Embedder,'find_similar_pairs',side_effect=fake_pairs),patch.object(matching,'create_duplicate_masks',side_effect=fake_matches),patch.object(lane,'find_lanes_in_blot_panels') as lanes:
 r=predict(im,loaded)
 assert len(seen)==4 and (3,4,.95) in seen # duplicate/overlap union once, best score
 assert not lanes.called and r['metadata']['lane_search'] is False
 assert len(r['metadata']['merged_groups'])==2 # microscopy triangle + blot pair
 assert r['branch_blots'].any() and r['branch_microscopy'].any() and not r['branch_lanes'].any()
 assert np.array_equal(r['candidates'],r['branch_blots']) and np.array_equal(r['geometric'],r['branch_microscopy'])
 assert np.array_equal(r['mask'],r['branch_blots']|r['branch_microscopy'])
 assert all(m['accepted'] for m in r['metadata']['comparisons'])
 json.dumps(r['metadata'])

# No blot candidates triggers lane matching, including a single panel.
single=[('Blots',1.,10,10,180,80)]
loaded['panel'].extract_panels=lambda image:single
ls=[BlotLane(0,[0,0,20,20],im,0,[10,10,180,80]),BlotLane(0,[80,0,100,20],im,1,[10,10,180,80])]
match=BlotMatch(0,1,.9,[0,0,20,20],[80,0,100,20],[10,10,30,30],[90,10,110,30],0,0)
with patch.object(lane,'find_lanes_in_blot_panels',return_value=dict(lanes=ls,best_matches=[match])) as called:
 r=predict(im,loaded)
 assert called.call_count==1 and r['branch_lanes'].any() and r['metadata']['lane_matches']==1
 assert r['mask'][60,150] # upstream >50% matched lanes => whole panel
 assert not r['geometric'].any() and np.array_equal(r['mask'],r['candidates'])
 r=predict(im,loaded,excluded_boxes=[(10,10,20,20)])
 assert len(r['metadata']['excluded_panels'])==1 and not r['mask'].any() and called.call_count==1

# Weak microscopy matches are rejected, never silently converted to suggestions.
loaded['panel'].extract_panels=lambda image:panels[:2]
def weak(*args,**kwargs):
 for rec in fake_matches(*args,**kwargs):
  rec['match_result']['inliers']=7;yield rec
with patch.object(api.Embedder,'find_similar_pairs',return_value=[(0,1,.9)]),patch.object(matching,'create_duplicate_masks',side_effect=weak):
 r=predict(im,loaded);assert not r['mask'].any() and not r['candidates'].any()
 assert r['metadata']['comparisons'][0]['accepted'] is False

from gui.sherloq_app.core.clone_detectors import analyze
from gui.sherloq_app.core import forgeryscope_adapter
calls=[]
def per_zone(crop,loaded,profile,**kwargs):
 calls.append(crop.shape)
 mask=np.ones(crop.shape[:2],np.uint8)
 return dict(map=mask.astype(np.float32),mask=mask,candidates=mask,geometric=mask,
  branch_microscopy=mask,branch_blots=mask,branch_lanes=mask,metadata=dict(status='ok'))
class Models:
 def get(self,*args):return dict(weights={},device='cpu'),False
zones=(((0,0),(19,0),(19,19),(0,19)),((40,0),(59,0),(59,19),(40,19)))
with patch.object(forgeryscope_adapter,'predict',side_effect=per_zone):
 out=analyze(im,dict(variant='Forgeryscope Auto',regions=zones),'cpu',Models())
 assert calls==[(20,20,3),(20,20,3)]
 assert out['branch_lanes'].sum()==800 and not out['branch_lanes'][:,20:40].any()
 try:analyze(im,dict(variant='Forgeryscope Auto',regions=zones,compare=True),'cpu',Models())
 except ValueError:pass
 else:raise AssertionError('Auto must not invent a panel type for manual comparison')

report=dict(passed=True,best_pair_fusion=True,clique_merge=True,blot_fallback_distinct=True,
 independent_zones=True,branch_arrays_propagated=True,manual_comparison_not_misclassified=True,
 lane_fallback=True,whole_panel_lane_rule=True,excluded_panels=True,microscopy_rejection=True)
print(report);(Path(__file__).with_name('contracts.json')).write_text(json.dumps(report,indent=2))
