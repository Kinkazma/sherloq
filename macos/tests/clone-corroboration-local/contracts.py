"""Unapproved local revision: integer counts, visual attenuation only, envelope votes retained."""
import sys,json,tempfile
from pathlib import Path
import numpy as np,cv2 as cv
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core.automatic_clones import SOURCES,_entry,point_entries,render,export
from gui.sherloq_app.core.clone_relations import annotate,polygon_key
from gui.sherloq_app.core.clone_corroboration import context_counts,CachedRenderer

def rect(x,y,w,h):return ((x,y),(x+w,y),(x+w,y+h),(x,y+h))
panels=(rect(10,10,180,180),rect(230,10,180,180));envelope=rect(10,10,400,180)
regions=(*panels,envelope);shape=(210,430,3)
def entry(polygons,context):
 return _entry(SOURCES[3],polygons,10,dict(search_context='roi:'+polygon_key(np.asarray(context)),search_region=context))
large=entry((rect(15,15,165,165),rect(235,15,165,165)),envelope)
small=entry((rect(30,40,8,8),rect(80,90,8,8)),panels[0])
extra=entry(small['polygons'],envelope)
# Same geometry/source, different actual search origin, no identifier collision.
assert extra['id']!=small['id']
a,b,c=annotate((large,small,extra),regions,envelope,shape)
assert a['relation']=='between' and b['relation']==c['relation']=='within'
assert all(e['heat_eligible'] for e in (a,b,c))
assert a['biome_fill_alpha'][0]<b['biome_fill_alpha'][0]<.25
assert not any('area_weights' in e for e in (a,b,c))
assert context_counts(shape,(a,b,c))[42,32]==2
assert np.array_equal(context_counts(shape,(a,b,c)),context_counts(shape,tuple(dict(e,biome_fill_alpha=[1.,1.],heat_eligible=False) for e in (a,b,c))))
# A second overlapping local search counts; the enclosing search also contributes by its own context.
other=entry(small['polygons'],rect(20,20,130,130))
d=annotate((other,),regions,envelope,shape)[0]
assert d['heat_eligible'] and d['search_context']!=b['search_context']
assert np.array_equal(context_counts(shape,(b,d)),context_counts(shape,(b,))*2)
assert np.array_equal(context_counts(shape,(b,b)),context_counts(shape,(b,)))
ela=dict(b,source='ELA biomes');assert np.array_equal(context_counts(shape,(b,ela)),context_counts(shape,(b,)))
# IA is retained in its own inter-panel layer despite whole-board analysis.
ai=dict(a,source=SOURCES[0]);ai=annotate((ai,),regions,envelope,shape)[0]
assert ai['heat_eligible'] and context_counts(shape,(ai,)).max()==1
# A standalone image remains analyzable: its only search region is not excluded.
whole=entry((rect(15,15,40,40),rect(100,100,40,40)),panels[0])
whole=annotate((whole,),(panels[0],),panels[0],shape)[0]
assert whole['heat_eligible'] and context_counts(shape,(whole,)).max()==1
# Visual opacity depends on area, never the numerical count; small boundaries drawn last.
image=np.full(shape,120,np.uint8)
assert np.array_equal(render((image,(a,b),())),render((image,(b,a),())))
cache=CachedRenderer(render);cache((image,(a,b),(),'overlay',.45));n=cache.builds
cache((image,(a,b),(),'overlay',.25));cache((image,(a,b),(),'heat',.25));assert cache.builds==n
# Mixed retained group: divide relation and origin without deleting raw links.
q=np.array([[0,0],[2,0],[2,2],[0,2]],float)
p0=np.vstack((q+(25,25),q+(25,60),q+(25,100)));p1=np.vstack((q+(70,25),q+(260,60),q+(70,100)))
points=np.zeros((24,7));points[:12,:2]=p0;points[12:,:2]=p1
pairs=np.column_stack((np.arange(12),np.arange(12)+12,np.full(12,.1),np.linalg.norm(p0-p1,axis=1)))
r=dict(points=points,pairs=pairs,groups=(np.arange(12),),models=(),regions=regions,
 pair_search_regions=np.array([0]*4+[2]*8),group_algorithms=(SOURCES[1],),params=(None,)*5+(50.,))
rows=point_entries(r,None,0,500,.8)
assert len(rows)==3 and sum(e['count'] for e in rows)==12 and len(r['groups'])==1
out=annotate(rows,regions,envelope,shape)
assert sum(e['relation']=='within' for e in out)==2 and sum(e['relation']=='between' for e in out)==1
assert sum(e['heat_eligible'] for e in out)==3
with tempfile.TemporaryDirectory() as folder:
 path=export((str(Path(folder)/'local.npz'),dict(image_shape=shape,corroboration=dict(entries=(a,b,c),excluded=()))))
 with np.load(path) as z:assert np.array_equal(z['root_corroboration_context_counts'],context_counts(shape,(a,b,c)))
report=dict(passed=True,local_only=True,no_size_weighting=True,distinct_search_contexts=True,
 intra_inter_separate=True,enclosing_retained=True,visual_attenuation_only=True,ia_retained=True,standalone_retained=True,
 raw_links_preserved=True,opacity_cache=True,exports_exact=True)
Path(__file__).with_name('contracts-results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
