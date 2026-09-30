"""Orchestration of Forgeryscope's published README pipeline (models-v1).

No competition-only ensemble or segmentation fallback is implied. Keep
embedding-only blot evidence identifiable while retaining upstream decisions.
"""
import numpy as np


def _polygon(box):
    x0,y0,x1,y1=map(float,box[-4:])
    return [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]


def predict(image,loaded,progress=lambda *args:None,excluded_boxes=()):
    import pandas as pd
    from sherloq_clone_models.forgeryscope import Embedder,PanelExtractor
    from sherloq_clone_models.forgeryscope.matcher.geometry import get_intersections
    from sherloq_clone_models.forgeryscope.matcher.lightglue import create_duplicate_masks,merge_masks_by_max_cliques
    from sherloq_clone_models.forgeryscope.matcher.lane import find_lanes_in_blot_panels,create_lane_match_masks
    rgb=np.ascontiguousarray(image[:,:,::-1])
    progress(0,100)
    detected=loaded['panel'].extract_panels(rgb)
    panels=[];rejected=[]
    for panel in detected:
        if panel[0] not in ('Blots','Microscopy'):continue
        if any(min(panel[4],x1)>max(panel[2],x0) and min(panel[5],y1)>max(panel[3],y0)
               for x0,y0,x1,y1 in excluded_boxes):rejected.append(tuple(panel))
        else:panels.append(tuple(panel))
    crops=PanelExtractor.crop_panels(rgb,panels)
    branches={name:np.zeros(image.shape[:2],np.uint8) for name in ('microscopy','blots','lanes')}
    mask=np.zeros(image.shape[:2],np.uint8);candidates=np.zeros_like(mask);geometric=np.zeros_like(mask)
    meta=dict(profile='Forgeryscope Auto',public_simplified=True,panels=panels,
        excluded_panels=rejected,comparisons=[],embedding_candidates=[],merged_groups=[],
        lane_search=False,lane_matches=0,status='no_panels',
        thresholds=dict(microscopy=.58,blot_duplicate=.84,blot_overlap=.85,lanes=.65,inliers=8,match_score=.73),
        mask_semantics='published pipeline: microscopy geometry, blot and lane similarity evidence',
        competition_ensemble=False,segmentation_fallback=False)
    intersections=set(get_intersections(panels,margin=10))
    ids={label:[i for i,p in enumerate(panels) if p[0]==label] for label in ('Blots','Microscopy')}
    def pairs(label,name,threshold):
        indices=ids[label]
        if len(indices)<2:return []
        vectors=np.concatenate([loaded['embeds'][name].get_embedding_batch(
            [crops[i] for i in indices[start:start+8]]).cpu().numpy()
            for start in range(0,len(indices),8)])
        return [(label,float(score),indices[i],indices[j])
                for i,j,score in Embedder.find_similar_pairs(vectors,threshold=threshold)]
    best={}
    for label,score,i,j in (pairs('Blots','wblot_overlap_embedder',.85)+
                            pairs('Blots','wblot_duplicate_embedder',.84)):
        key=tuple(sorted((i,j)));best[key]=max(score,best.get(key,-float('inf')))
    # Upstream lane condition deliberately uses the pre-intersection blot list.
    blot_pairs=[('Blots',score,*key) for key,score in best.items()]
    similar=blot_pairs+pairs('Microscopy','micro_overlap_embedder',.58)
    similar=[row for row in similar if (row[2],row[3]) not in intersections and (row[3],row[2]) not in intersections]
    meta['embedding_candidates']=[dict(label=l,score=s,panel0=i,panel1=j) for l,s,i,j in similar]
    progress(20,100)
    accepted_masks=[];accepted_info=[]
    frame=pd.DataFrame(similar,columns=['label','score','idx1','idx2'])
    for index in range(len(frame)):
        progress(20+int(65*index/max(1,len(frame))),100)
        records=create_duplicate_masks(rgb,panels,crops,frame.iloc[index:index+1],loaded['micro'],loaded['blot'],
            to_bbox_micro=False,to_bbox_blot=True,fallback_for_wblot=True,
            test_transforms_blot=False,test_transforms_micro=True)
        for info in records:
            match=info['match_result'];label=info['panel_label']
            supported=('fallback' not in match and match['inliers']>=8 and match['mean_match_score']>=.73)
            accepted=label=='Blots' or supported
            branch='microscopy' if label=='Microscopy' else 'blots'
            union=(info['mask0']|info['mask1']).astype(np.uint8)
            if accepted:
                accepted_masks.append(union);accepted_info.append(info);branches[branch]|=union
                if supported:geometric|=union
                else:candidates|=union
            meta['comparisons'].append(dict(panel0=int(info['panel_id0']),panel1=int(info['panel_id1']),
                supported=bool(supported),accepted=bool(accepted),branch=branch,
                evidence='geometry' if supported else 'embedding',inliers=int(match['inliers']),
                score=float(match['mean_match_score']),polygon0=np.asarray(info['poly_coords0']).tolist(),
                polygon1=np.asarray(info['poly_coords1']).tolist(),fallback=match.get('fallback')))
    merged,groups=merge_masks_by_max_cliques(accepted_masks,accepted_info,verbose=False)
    for item in merged:mask|=item.astype(np.uint8)
    for group in groups:
        meta['merged_groups'].append(dict(panel_ids=[int(i) for i in group['panel_ids']],
            n_pairs=int(group['n_pairs']),total_inliers=int(group['total_inliers']),
            avg_match_score=float(group['avg_match_score']),
            polygons=[np.asarray(p).tolist() for p in group['all_poly_coords']]))
    progress(85,100)
    if ids['Blots'] and not blot_pairs:
        meta['lane_search']=True
        lane_result=find_lanes_in_blot_panels(panels=panels,blot_panels_ids=ids['Blots'],crops_list=crops,
            segmentator=loaded['lane'],blot_duplicate_detector=loaded['embeds']['wblot_lane_embedder'],
            similarity_threshold=.65,overlap_threshold=5)
        if lane_result:
            lanes=lane_result['lanes'];matches=lane_result['best_matches']
            meta['lane_matches']=len(matches)
            for item in create_lane_match_masks(image.shape,matches,lanes=lanes):
                branches['lanes']|=item.astype(np.uint8)
            candidates|=branches['lanes'];mask|=branches['lanes']
            # Carry the same >50% whole-panel expansion into automatic biomes.
            from collections import Counter
            totals=Counter(l.panel_idx for l in lanes)
            matched=Counter(p for m in matches for p in (m.panel_idx1,m.panel_idx2))
            whole={p for p,n in totals.items() if matched[p]>.5*n}
            panel_boxes={}
            for lane in lanes:panel_boxes.setdefault(lane.panel_idx,lane.panel_bbox)
            def display_box(panel,box):
                return panel_boxes[panel][-4:] if panel in whole and panel_boxes.get(panel) is not None else box
            # Export lane evidence independently of geometric panel matches.
            meta['lane_pairs']=[dict(panel0=int(m.panel_idx1),panel1=int(m.panel_idx2),
                score=float(m.similarity),polygon0=_polygon(m.bbox1_absolute),
                polygon1=_polygon(m.bbox2_absolute),evidence='embedding',
                display_polygon0=_polygon(display_box(m.panel_idx1,m.bbox1_absolute)),
                display_polygon1=_polygon(display_box(m.panel_idx2,m.bbox2_absolute))) for m in matches]
    for x0,y0,x1,y1 in excluded_boxes:
        for item in (mask,candidates,geometric,*branches.values()):item[y0:y1,x0:x1]=0
    meta['status']='ok' if mask.any() else 'empty' if panels else 'no_panels'
    progress(100,100)
    return dict(mask=mask,map=mask.astype(np.float32),candidates=candidates,geometric=geometric,
                **{'branch_'+name:value for name,value in branches.items()},metadata=meta)
