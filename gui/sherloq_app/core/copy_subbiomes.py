"""Reconnect surviving geometric evidence, without imposing a biome size cap."""
import colorsys
import numpy as np
from .cloning import check


def refine(points, pairs, groups, models, colors, bases, tolerance, cancel=lambda: False):
    from .cloning2 import biomes
    from .copy_geometry import project
    refined, fitted, shades, provenance = [], [], [], []
    colors = colors.copy()
    for parent, group in enumerate(groups):
        check(cancel)
        parts = tuple(group[g] for g in biomes(points, pairs[group], tolerance, cancel))
        for number, part in enumerate(parts):
            # Leave every connected parent and its display colour exactly alone.
            model = models[parent] if models else None
            color = bases[parent]
            if len(parts) > 1:
                h, s, v = colorsys.rgb_to_hsv(*(np.asarray(color)[::-1] / 255.))
                h = (h + (number / max(1, len(parts)-1) - .5) * .045) % 1
                color = tuple(round(c * 255) for c in colorsys.hsv_to_rgb(h, s, v)[::-1])
                # Preserve descriptor-dependent darkness while changing the family shade.
                for row in part:
                    _, sat, val = colorsys.rgb_to_hsv(*(colors[row][::-1] / 255.))
                    colors[row] = [round(c*255) for c in colorsys.hsv_to_rgb(h, sat, val)[::-1]]
                if model is not None:
                    positions = {int(row): i for i, row in enumerate(group)}
                    ids = [positions[int(row)] for row in part]
                    source = np.asarray(model['source_point_indices'])[ids]
                    target = np.asarray(model['destination_point_indices'])[ids]
                    a, b = points[source,:2], points[target,:2]
                    matrix = np.asarray(model['matrix'])
                    error = np.maximum(np.linalg.norm(project(a,matrix)-b,axis=1),
                                       np.linalg.norm(project(b,np.linalg.inv(matrix))-a,axis=1))
                    model = dict(model, source_point_indices=source.tolist(), destination_point_indices=target.tolist(),
                                 inliers=len(part), parent_inliers=len(group), post_geometry_partition=True,
                                 distinct_centres=[len(np.unique(np.rint(a),axis=0)),len(np.unique(np.rint(b),axis=0))],
                                 median_error_px=float(np.median(error)), maximum_error_px=float(error.max()))
            refined.append(part);shades.append(color)
            if model is not None:fitted.append(model)
            provenance.append(dict(parent=parent, part=number, parts=len(parts), parent_matches=len(group)))
    return tuple(refined), tuple(fitted), colors, tuple(shades), tuple(provenance)
