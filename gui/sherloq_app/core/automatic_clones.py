"""Shared geometry/display for automatic clone search; detectors stay independent."""
import colorsys
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import cv2 as cv
import numpy as np

from .cloning2 import EXTENDED_SYMMETRIC, biome_sides, supported_selection
from .auto_zones import diagonal as zone_diagonal
from .copy_overlap import overlap

from .sift_panels import NAME as SIFT_SOURCE

D2PRL_SOURCE = 'D2PRL'
CLASSICAL_SOURCES = ('PatchMatch Zernike', 'PatchMatch SIFT', SIFT_SOURCE)
AI_SOURCES = ('Forgeryscope Auto', D2PRL_SOURCE)
SOURCES = ('Forgeryscope Auto', *CLASSICAL_SOURCES, D2PRL_SOURCE)

def source_label(source):
    return 'SIFT + G2NN + RANSAC' if source == SIFT_SOURCE else source


def parameters(shape, regions, envelope, disabled, cpu=False):
    """Extended normal and mirror passes, with per-zone distances/exclusions."""
    active = tuple(r for i, r in enumerate(regions) if i not in disabled)
    excluded = tuple(r for i, r in enumerate(regions) if i in disabled and r != envelope)
    guides = tuple(r for r in active if r != envelope) if envelope in regions else ()
    diagonal = math.hypot(*shape[:2])
    radius = round(max((zone_diagonal(r) for r in active), default=diagonal), 2)
    params = (EXTENDED_SYMMETRIC, 6000, radius, 5., .3, min(50., diagonal),
              'Similarity', 3., 6, 8, 8, True, 2., cpu, True, True, excluded, guides)
    forge = dict(variant='Forgeryscope Auto', regions=(envelope,) if envelope in active else (),
                 selection_present=True, compare=False, excluded=excluded)
    return params, active, forge


def _entry(source, polygons, count, provenance):
    polygons = [np.asarray(p, np.float32).reshape(-1, 2) for p in polygons]
    if any(len(p) < 3 or not np.isfinite(p).all() for p in polygons):
        return None
    polygons = [cv.convexHull(p).reshape(-1, 2) for p in polygons]
    if any(len(p) < 3 or cv.contourArea(p) <= 0 for p in polygons):
        return None
    # Stable identifiers preserve hidden states across incremental branch results.
    canonical = sorted(tuple(sorted(map(tuple, np.round(p, 3).tolist()))) for p in polygons)
    identity=[source,canonical]
    if 'branch' in provenance:identity.append(provenance['branch'])
    if 'search_context' in provenance:identity.append(provenance['search_context'])
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]
    hue = (int(key[:8], 16) / 2**32 + (SOURCES.index(source) if source in SOURCES else 3) / 3) % 1
    color = tuple(round(v * 255) for v in colorsys.hsv_to_rgb(hue, .8, .95)[::-1])
    return dict(id=key, source=source, polygons=[p.tolist() for p in polygons],
                count=int(count), color=color, provenance=provenance)


def point_entries(result, source, low, high, maximum_overlap, split=False):
    """Keep search contexts and panel relations before drawing endpoint hulls."""
    from .cloning2 import biomes
    from .clone_relations import pair_relations, polygon_key
    found=[]
    tolerance=result.get('params',(None,)*5+(50.,))[5]
    models=result.get('models',());regions=result.get('regions',())
    relations=pair_relations(result) if result['groups'] else np.empty((0,2),int)
    owners=result.get('pair_search_regions')
    if owners is None:owners=np.full(len(result.get('pairs',())), -2, np.int32)
    partitions=result.get('biome_partitions',())
    for index,group in enumerate(result['groups']):
        model=models[index] if index<len(models) else {}
        parts=(group,)
        if split and not model.get('source_panels'):
            parts=tuple(group[g] for g in biomes(result['points'],result['pairs'][group],tolerance))
        name=source or result['group_algorithms'][index]
        for part_number,part in enumerate(parts):
            keys=np.column_stack((owners[part],relations[part]))
            for key in np.unique(keys,axis=0):
                subset=part[(keys==key).all(1)]
                selected=supported_selection(result,subset,low,high,3,maximum_overlap=maximum_overlap)
                if not len(selected):continue
                owner=int(key[0]);origin=regions[owner] if 0<=owner<len(regions) else None
                context=('roi:'+polygon_key(np.asarray(origin)) if origin is not None else
                         'compare:'+':'.join(polygon_key(np.asarray(r)) for r in regions) if owner==-1 else
                         'whole-image' if owner>=0 else 'unspecified')
                provenance=dict(group=index,part=part_number,parts=len(parts),
                                search_context=context,search_region=origin,
                                search_region_index=owner,endpoint_region_indices=key[1:].tolist())
                if partitions:provenance['partition']=partitions[index]
                item=_entry(name,biome_sides(result['points'],result['pairs'],selected),len(selected),provenance)
                original=_entry(name,biome_sides(result['points'],result['pairs'],subset),len(subset),provenance)
                if item and original and overlap(*item['polygons'])<maximum_overlap:
                    item.update(id=original['id'],color=original['color'],search_context=context)
                    if source and result.get('bases'):item['color']=result['bases'][index]
                    item['label']=source_label(name)
                    found.append(item)
    return tuple(found)


def entries(patchmatch=None, forgeryscope=None, low=10., high=float('inf'), maximum_overlap=.8, sift=None, d2prl=None, d2prl_minimum=500):
    found = {}
    if patchmatch is not None:
        for item in point_entries(patchmatch, None, low, high, maximum_overlap, split=True):
            found[item['id']] = item
    if sift is not None:
        for item in point_entries(sift, SIFT_SOURCE, low, high, maximum_overlap):
            found[item['id']] = item
    if forgeryscope is not None:
        for zone_index, zone in enumerate(forgeryscope['metadata']['zones']):
            offset = np.asarray(zone.get('origin', [0, 0]))
            matches=[(i,m,m.get('branch','microscopy')) for i,m in enumerate(zone.get('comparisons',()))]
            matches += [(i,{**m,'accepted':True,'supported':False},'lanes') for i,m in enumerate(zone.get('lane_pairs',()))]
            for index, match, branch in matches:
                supported=bool(match.get('supported',False))
                if not match.get('accepted',supported):
                    continue
                polygons=[np.asarray(match.get(f'display_polygon{i}',match[f'polygon{i}']))+offset for i in (0,1)]
                item = _entry(SOURCES[0], polygons, match.get('inliers',0) if supported else 1,
                              dict(zone=zone_index,comparison=index,score=match['score'],branch=branch,
                                   evidence='geometry' if supported else 'similarity',accepted=True))
                if item:
                    title={'microscopy':'Microscopy','blots':'Western blots','lanes':'Lanes'}[branch]
                    item['label']=f"Forgeryscope Auto · {title} · {'Geometry' if supported else 'Similarity'}"
                    if not supported:item['count_kind']='pairs'
                    centers = []
                    for polygon in item['polygons']:
                        moments = cv.moments(np.asarray(polygon, np.float32))
                        centers.append(np.array([moments['m10'], moments['m01']]) / moments['m00'])
                    distance = float(np.linalg.norm(centers[0] - centers[1]))
                    if not low <= distance <= high or overlap(*item['polygons']) >= maximum_overlap:
                        continue
                    item['center_distance_px'] = distance
                    item['search_context']='forgeryscope-zone:'+str(zone_index)
                    found[item['id']] = item
    from .d2prl_regions import regions as d2prl_regions
    for item in d2prl_regions(d2prl,d2prl_minimum):found[item['id']]=item
    return tuple(found.values())


def visible(entries, source, hidden, focused):
    return tuple(e for e in entries if (not source or e['source'] == source)
                 and e['id'] not in hidden and (not focused or e['id'] == focused))


def render(request):
    image, biomes, excluded = request[:3]
    mode, opacity = request[3:5] if len(request)>3 else ('biomes', .45)
    if mode != 'biomes':
        from .clone_corroboration import render_heat
        return render_heat(image, biomes, excluded, mode, opacity, request[5] if len(request)>5 else None)
    from .clone_corroboration import unique_envelopes
    biomes = unique_envelopes(biomes)
    # Paint broad regions first. Small retained evidence and its contour remain
    # on top even when they belong to the same method and search context.
    biomes = sorted(biomes,key=lambda e:sum(cv.contourArea(np.asarray(p,np.float32)) for p in e['polygons']),reverse=True)
    out = image.copy()
    for item in biomes:
        if 'pixel_mask' in item:
            from .d2prl_regions import paint
            paint(out,item)
            continue
        for side,polygon in enumerate(item['polygons']):
            points = np.rint(polygon).astype(np.int32)
            x, y, w, h = cv.boundingRect(points)
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(out.shape[1], x+w), min(out.shape[0], y+h)
            if x1 <= x0 or y1 <= y0:
                continue
            roi = out[y0:y1, x0:x1]; overlay = roi.copy()
            cv.fillConvexPoly(overlay, points - (x0, y0), item['color'], cv.LINE_AA)
            alpha=item.get('biome_fill_alpha',[.25]*len(item['polygons']))[side]
            cv.addWeighted(overlay, alpha, roi, 1-alpha, 0, dst=roi)
            cv.polylines(out,[points],True,item['color'],2,cv.LINE_AA)
    # Hulls can bridge a disabled region; never paint its excluded pixels.
    for polygon in excluded:
        p = np.asarray(polygon, int); x0,y0 = p.min(0); x1,y1 = p.max(0) + 1
        x0,y0 = max(0,x0),max(0,y0); x1,y1 = min(out.shape[1],x1),min(out.shape[0],y1)
        out[y0:y1,x0:x1] = image[y0:y1,x0:x1]
    return out


def export(request):
    """Export complete detector arrays and the display snapshot without pickling."""
    filename, snapshot = request
    if 'corroboration' in snapshot:
        from .clone_corroboration import votes, unique_envelopes, context_counts
        config=snapshot['corroboration']
        snapshot=dict(snapshot,corroboration=dict(config,counts=votes(snapshot['image_shape'],config['entries'],config['excluded']),context_counts=context_counts(snapshot['image_shape'],config['entries'],config['excluded']),deduplicated_envelopes=unique_envelopes(config['entries'])))
    arrays = {}
    def encode(value, name='root'):
        if isinstance(value, np.ndarray):
            arrays[name] = value
            return {'array': name}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): encode(v, name+'_'+str(k)) for k,v in value.items()}
        if isinstance(value, (tuple, list, set)):
            return [encode(v, name+'_'+str(i)) for i,v in enumerate(value)]
        return value
    arrays['metadata_json'] = np.asarray(json.dumps(encode(snapshot), ensure_ascii=False))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('wb', dir=Path(filename).parent, delete=False) as stream:
            temporary = stream.name
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, filename)
        return filename
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
