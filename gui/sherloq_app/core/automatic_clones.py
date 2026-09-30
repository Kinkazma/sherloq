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

SOURCES = ('Forgeryscope Auto', 'PatchMatch Zernike', 'PatchMatch SIFT')


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
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]
    hue = (int(key[:8], 16) / 2**32 + (SOURCES.index(source) if source in SOURCES else 3) / 3) % 1
    color = tuple(round(v * 255) for v in colorsys.hsv_to_rgb(hue, .8, .95)[::-1])
    return dict(id=key, source=source, polygons=[p.tolist() for p in polygons],
                count=int(count), color=color, provenance=provenance)


def entries(patchmatch=None, forgeryscope=None, low=10., high=float('inf'), maximum_overlap=.8):
    found = {}
    if patchmatch is not None:
        for index, group in enumerate(patchmatch['groups']):
            selected = supported_selection(patchmatch, group, low, high, 3, maximum_overlap=maximum_overlap)
            if not len(selected):
                continue
            source = patchmatch['group_algorithms'][index]
            item = _entry(source, biome_sides(patchmatch['points'], patchmatch['pairs'], selected),
                          len(selected), dict(group=index))
            if item:
                if overlap(*item['polygons']) >= maximum_overlap:
                    continue
                # Slider changes alter the hull, not the identity/check state.
                original = _entry(source, biome_sides(patchmatch['points'], patchmatch['pairs'], group),
                                  len(group), dict(group=index))
                if original:
                    item.update(id=original['id'], color=original['color'])
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
                    found[item['id']] = item
    return tuple(found.values())


def visible(entries, source, hidden, focused):
    return tuple(e for e in entries if (not source or e['source'] == source)
                 and e['id'] not in hidden and (not focused or e['id'] == focused))


def render(request):
    image, biomes, excluded = request
    out = image.copy()
    for item in biomes:
        for polygon in item['polygons']:
            points = np.rint(polygon).astype(np.int32)
            x, y, w, h = cv.boundingRect(points)
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(out.shape[1], x+w), min(out.shape[0], y+h)
            if x1 <= x0 or y1 <= y0:
                continue
            roi = out[y0:y1, x0:x1]; overlay = roi.copy()
            cv.fillConvexPoly(overlay, points - (x0, y0), item['color'], cv.LINE_AA)
            cv.addWeighted(overlay, .25, roi, .75, 0, dst=roi)
            cv.polylines(out, [points], True, item['color'], 2, cv.LINE_AA)
    # Hulls can bridge a disabled region; never paint its excluded pixels.
    for polygon in excluded:
        p = np.asarray(polygon, int); x0,y0 = p.min(0); x1,y1 = p.max(0) + 1
        x0,y0 = max(0,x0),max(0,y0); x1,y1 = min(out.shape[1],x1),min(out.shape[0],y1)
        out[y0:y1,x0:x1] = image[y0:y1,x0:x1]
    return out


def export(request):
    """Export complete detector arrays and the display snapshot without pickling."""
    filename, snapshot = request
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
