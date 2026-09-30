"""Add corroborated mirrors while retaining the selected normal/extended base."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
import numpy as np
from .cloning import check
from .copy_detail import detail_image, corroborated_groups, DETAIL_POLICY
from .copy_geometry import project


def coalesce_global(extra, groups, models, guides, geometry, cancel):
    """Refit fragments linking the same two panels, within the same Search ROI.

    Internal panel searches remain separate. Ambiguous/overlapping guide
    memberships and mixed-panel groups are left alone. RANSAC retains actual
    correspondences; it does not turn a panel rectangle into a detection mask.
    """
    if not models or not guides or extra['compare']:
        return groups, models
    from .cloning2 import memberships
    from .copy_geometry import verify
    members = memberships(extra['points'], extra['_image_shape'], guides)
    labels = np.where(members.sum(1)==1, members.argmax(1), -1)
    partitions = {}; unchanged = []
    for i, group in enumerate(groups):
        check(cancel)
        endpoints = extra['pairs'][group,:2].astype(int)
        panel_pairs = np.sort(labels[endpoints],axis=1)
        if (panel_pairs>=0).all() and (panel_pairs[:,0]!=panel_pairs[:,1]).all() and np.all(panel_pairs==panel_pairs[0]):
            owners = extra['_pair_regions'][group]
            if np.all(owners==owners[0]):
                key = (int(owners[0]), *map(int,panel_pairs[0]))
                partitions.setdefault(key,[]).append(i); continue
        unchanged.append(i)
    merged_groups = [groups[i] for i in unchanged]
    merged_models = [models[i] for i in unchanged]
    for (_,a,b), ids in partitions.items():
        if len(ids)==1:
            merged_groups.append(groups[ids[0]]);merged_models.append(models[ids[0]]);continue
        union = np.unique(np.concatenate([groups[i] for i in ids]))
        new_groups,new_models = verify(extra['points'],extra['pairs'],(union,),*geometry,
                                       cancel=cancel,reflection=True)
        for group,model in zip(new_groups,new_models):
            if np.linalg.det(np.asarray(model['matrix'])[:2,:2])<0:
                merged_groups.append(group)
                merged_models.append(dict(model,source_panels=[a+1,b+1],merged_fragments=len(ids)))
    return tuple(merged_groups),tuple(merged_models)


def analyze(engine, params, regions, compare, cancel, progress):
    from .cloning2 import Cloning2Engine, COMBINED, EXTENDED, SYMMETRIC, EXTENDED_SYMMETRIC, palette
    defaults = (SYMMETRIC,6000,600.,5.,.3,50.,'Similarity',3.,6,8,8,False,2.,False,False,False,(),())
    effective = (*params[:18], *defaults[len(params):18])
    extended = params[0] == EXTENDED_SYMMETRIC
    normal = (EXTENDED if extended else COMBINED, *effective[1:11], False, *effective[12:])
    method = engine.analyze_extended if extended else engine.analyze_combined
    base = method(normal, regions, compare, cancel, lambda n,s: progress(n//2,s))
    if not hasattr(engine, '_mirror_engines'):
        engine._mirror_engines = [Cloning2Engine(engine.image), Cloning2Engine(engine.image)]
    detail = None
    if effective[6] != 'None':
        check(cancel)
        if not hasattr(engine, '_detail_image'):
            engine._detail_image = detail_image(engine.image)
        detail = engine._detail_image
    # Keep the established mirror passes first, then append new scale hypotheses.
    passes = [(0, effective[9]), (1, effective[9])]
    if extended:
        passes += [(1, target) for target in base['extension_bins'] if target != effective[9]]
    if not hasattr(engine, '_mirror_scale_engines'):
        engine._mirror_scale_engines = {}
    children = list(engine._mirror_engines)
    for _, target in passes[2:]:
        if target not in engine._mirror_scale_engines:
            engine._mirror_scale_engines[target] = Cloning2Engine(engine.image)
        children.append(engine._mirror_scale_engines[target])
    stop = Event(); lock = Lock(); values = [0]*len(passes)
    cancelled = lambda: cancel() or stop.is_set()
    names = base['source_algorithms']

    def run(index):
        i, target = passes[index]
        child = children[index]
        # Canonical SIFT also covers a vertical reflection (horizontal + 180°).
        settings = (names[i], *normal[1:11], True, *normal[12:], 4,
                    'quarter_turn' if i else None, target)
        before = child.counts.copy()
        def update(n, text):
            with lock:
                values[index] = n; progress(50 + sum(values)//(2*len(passes)), text)
        extra = child.analyze_dense(settings, regions, compare, cancelled, update)
        groups, models = extra['groups'], extra['models']
        if models:
            ids = [j for j,m in enumerate(models) if np.linalg.det(np.asarray(m['matrix'])[:2,:2]) < 0]
            groups, models = tuple(groups[j] for j in ids), tuple(models[j] for j in ids)
            groups, models = coalesce_global(extra,groups,models,effective[17],effective[6:9],cancelled)
            if target != effective[9]:
                from .sift_frames import transformed_groups
                groups, models = transformed_groups(groups, models, effective[9], target, True)
            groups, models = corroborated_groups(detail, extra['points'], groups, models, cancelled)
        if models:
            kept=[]
            for j,model in enumerate(models):
                source=extra['points'][model['source_point_indices'],:2]
                displacement=np.median(np.linalg.norm(project(source,np.asarray(model['matrix']))-source,axis=1))
                if displacement+1e-6>=effective[3]:kept.append(j)
            groups,models=tuple(groups[j] for j in kept),tuple(models[j] for j in kept)
        return dict(extra,groups=groups,models=models), {k: child.counts[k]-before[k] for k in before}

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix='cm2-mirror') as pool:
        futures = [pool.submit(run,i) for i in range(len(passes))]
        try:
            answers = [f.result() for f in futures]
        except BaseException:
            stop.set()
            for f in futures: f.cancel()
            raise
    result = dict(base,params=params,group_variants=tuple(['normal']*len(base['groups'])))
    for (i,target),(extra,counts) in zip(passes, answers):
        check(cancel)
        for key,value in counts.items(): engine.counts[key] += value
        point_offset, pair_offset = len(result['points']),len(result['pairs'])
        pairs = extra['pairs'].copy(); pairs[:,:2] += point_offset
        pairs = np.concatenate((result['pairs'],pairs))
        groups = (*result['groups'],*(g+pair_offset for g in extra['groups']))
        colors,bases = palette(groups,pairs,params[4]); models = []
        for model in extra['models']:
            updated = dict(model,algorithm=names[i],variant='reflection')
            if extended:
                updated['descriptor_frame'] = f'quarter_turn/bin={target}' if i else 'normal'
            for key in ('source_point_indices','destination_point_indices'):
                updated[key] = [j+point_offset for j in updated[key]]
            models.append(updated)
        result = dict(result,points=np.concatenate((result['points'],extra['points'])),pairs=pairs,
                      pair_search_regions=np.concatenate((result['pair_search_regions'],extra['pair_search_regions'])),
                      groups=groups,models=(*result['models'],*models),colors=colors,bases=bases,
                      group_algorithms=(*result['group_algorithms'],*([names[i]]*len(extra['groups']))),
                      pair_algorithms=np.r_[result['pair_algorithms'],np.full(len(extra['pairs']),i,np.uint8)],
                      group_variants=(*result['group_variants'],*(['reflection']*len(extra['groups']))),
                      dense_maps=(*result['dense_maps'],*({**f,'algorithm':names[i],'variant':'reflection',
                                    **(dict(source_bin=effective[9],target_bin=target,descriptor_frame=f'quarter_turn/bin={target}' if i else 'normal') if extended else {})} for f in extra['dense_maps'])))
        if extended:
            frame = f'quarter_turn/bin={target}' if i else 'normal'
            result['group_frames'] = (*result['group_frames'], *([frame]*len(extra['groups'])))
        for key in ('total_features','candidate_comparisons','dense_count','dense_consistent_count'):
            result[key] += extra[key]
    result['mirror_policy'] = dict(version=1,normal_preserved=True,sift_frame='quarter_turn',
                                  detail=dict(enabled=detail is not None,**DETAIL_POLICY),maximum_overlap=.8,
                                  overlap_filter_stage='display',base_pair_count=len(base['pairs']))
    if extended:
        result['mirror_policy'].update(version=2,extended_preserved=True,
                                       sift_bins=tuple(target for i,target in passes if i),
                                       passes=tuple(dict(algorithm=names[i],source_bin=effective[9],target_bin=target,
                                                         frame='quarter_turn' if i else 'normal') for i,target in passes))
    progress(100,'Normal and reflected copies retained')
    return result
