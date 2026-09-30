"""Compare real Auto inference against the public README orchestration.

Model loading/device adaptations are shared; this qualifies orchestration,
not identity of MPS arithmetic against CUDA or competition reproducibility.
"""
import os,sys,json,time,socket
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'source'))
os.environ.update(YOLO_CONFIG_DIR=str(R/'integration/clone_detectors/yolo-config'),YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',NO_ALBUMENTATIONS_UPDATE='1',PYTORCH_MPS_FAST_MATH='0')
def deny(*args,**kwargs):raise RuntimeError('Unexpected network access')
socket.socket.connect=deny;socket.create_connection=deny
import cv2 as cv,numpy as np,torch,pandas as pd
torch.set_num_threads(8)
from gui.sherloq_app.core.clone_detectors import Models,analyze
models=Models()
im=cv.imread(sys.argv[1]);assert im is not None
start=time.perf_counter();torch.manual_seed(22);cv.setRNGSeed(22)
r=analyze(im,dict(variant='Forgeryscope Auto'),'mps',models,lambda n,d:print(n,d,flush=True))
elapsed=time.perf_counter()-start
np.savez_compressed(Path(__file__).with_name('native-maps.npz'),**{k:v for k,v in r.items() if k!='metadata'})
Path(__file__).with_name('native-metadata.json').write_text(json.dumps(r['metadata'],indent=2))
overlay=im.copy();selected=r['mask'].astype(bool);overlay[selected]=(overlay[selected]*.6+np.array([0,210,255])*.4).astype(np.uint8)
cv.imwrite(str(Path(__file__).with_name('native-overlay.jpg')),overlay)
print('AUTO COMPLETE',elapsed,int(r['mask'].sum()),flush=True)

# Literal public sequence, all pairs in one call, independent of auto wrapper.
from sherloq_clone_models.forgeryscope import Embedder,PanelExtractor
from sherloq_clone_models.forgeryscope.matcher.geometry import get_intersections
from sherloq_clone_models.forgeryscope.matcher.lightglue import create_duplicate_masks,merge_masks_by_max_cliques
from sherloq_clone_models.forgeryscope.matcher.lane import find_lanes_in_blot_panels,create_lane_match_masks
loaded=models.value;rgb=np.ascontiguousarray(im[:,:,::-1]);torch.manual_seed(22);cv.setRNGSeed(22)
panels=loaded['panel'].extract_panels(rgb);crops=PanelExtractor.crop_panels(rgb,panels)
blots=[i for i,p in enumerate(panels) if p[0]=='Blots'];micro=[i for i,p in enumerate(panels) if p[0]=='Microscopy']
def pairs(ids,name,threshold,label):
 if not ids:return []
 vectors=loaded['embeds'][name].get_embedding_batch([crops[i] for i in ids]).cpu()
 return [(label,float(score),ids[i],ids[j]) for i,j,score in Embedder.find_similar_pairs(vectors,threshold=threshold)]
best={}
for label,score,i,j in pairs(blots,'wblot_overlap_embedder',.85,'Blots')+pairs(blots,'wblot_duplicate_embedder',.84,'Blots'):
 key=tuple(sorted((i,j)));best[key]=max(best.get(key,-1),score)
bp=[('Blots',score,*key) for key,score in best.items()]
similar=bp+pairs(micro,'micro_overlap_embedder',.58,'Microscopy')
intersections=get_intersections(panels,margin=10)
similar=[row for row in similar if (row[2],row[3]) not in intersections and (row[3],row[2]) not in intersections]
frame=pd.DataFrame(similar,columns=['label','score','idx1','idx2'])
infos=[];masks=[]
for info in create_duplicate_masks(rgb,panels,crops,frame,loaded['micro'],loaded['blot'],to_bbox_micro=False,to_bbox_blot=True,fallback_for_wblot=True,test_transforms_blot=False,test_transforms_micro=True):
 match=info['match_result']
 if info['panel_label']!='Blots' and (match['inliers']<8 or match['mean_match_score']<.73):continue
 infos.append(info);masks.append((info['mask0']|info['mask1']).astype(np.uint8))
merged,_=merge_masks_by_max_cliques(masks,infos,verbose=False)
if blots and not bp:
 lr=find_lanes_in_blot_panels(panels,blots,crops,loaded['lane'],loaded['embeds']['wblot_lane_embedder'],similarity_threshold=.65,overlap_threshold=5)
 if lr:merged+=create_lane_match_masks(im.shape,lr['best_matches'],lanes=lr['lanes'])
reference=np.zeros(im.shape[:2],np.uint8)
for m in merged:reference|=m
assert np.array_equal(reference,r['mask']),'Public pipeline mismatch'
empty=analyze(np.zeros((192,256,3),np.uint8),dict(variant='Forgeryscope Auto'),'mps',models)
assert not empty['mask'].any() and empty['metadata']['status']=='no_panels'
report=dict(passed=True,public_reference_mask_exact=True,uniform_empty=True,seconds=elapsed,
 shape=list(im.shape),mask_pixels=int(r['mask'].sum()),candidate_pixels=int(r['candidates'].sum()),
 branches={k:int(r['branch_'+k].sum()) for k in ('microscopy','blots','lanes')},
 panels=len(r['metadata']['zones'][0]['panels']),pairs=len(similar),device=r['metadata']['device'])
Path(__file__).with_name('native-results.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
models.clear()
