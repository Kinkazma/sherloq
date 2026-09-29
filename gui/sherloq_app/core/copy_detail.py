"""Pixel-detail corroboration for supplementary geometric copy hypotheses.

The normal detector is unaffected. This is a heuristic rejection of additional
matches, not a probability of tampering. Small alignment residuals are allowed;
textureless patches cannot corroborate a transformation.
"""
import cv2 as cv
import numpy as np
from .cloning import check
from .copy_geometry import project

DETAIL_POLICY = dict(minimum_ncc=.8, minimum_samples=6, maximum_samples=32,
                     patch_width=9, alignment_radius=1., alignment_step=.5,
                     highpass_sigma=1.2)


def detail_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY).astype(np.float32)
    result = gray - cv.GaussianBlur(gray, (0, 0), DETAIL_POLICY['highpass_sigma'])
    result.flags.writeable = False
    return result


def corroborate(detail, points, model, cancel=lambda: False):
    """Median local NCC under an already fitted source-to-target mapping.

    At most 32 centres in stable model order, with up to 25 subpixel offsets
    each. Centres may overlap: sample count must not be read as independent
    statistical evidence. Neither image nor descriptors are resized here.
    """
    check(cancel)
    ids = np.asarray(model['source_point_indices'], int)
    if len(ids) > 32:
        ids = ids[np.linspace(0, len(ids) - 1, 32, dtype=int)]
    if not len(ids):
        return dict(accepted=False, ncc=0., samples=0)
    centers = points[ids, :2]
    yy, xx = np.mgrid[-4:5, -4:5]
    offsets = np.c_[xx.ravel(), yy.ravel()]
    src = centers[:, None, :] + offsets[None, :, :]
    dst = project(src.reshape(-1, 2), np.asarray(model['matrix'])).reshape(src.shape)
    h, w = detail.shape
    valid = ((src[:, :, 0].min(1) >= 0) & (src[:, :, 1].min(1) >= 0)
             & (src[:, :, 0].max(1) < w - 1) & (src[:, :, 1].max(1) < h - 1)
             & (dst[:, :, 0].min(1) >= 1) & (dst[:, :, 1].min(1) >= 1)
             & (dst[:, :, 0].max(1) < w - 2) & (dst[:, :, 1].max(1) < h - 2))
    src, dst = src[valid], dst[valid]
    if len(src) < 6:
        return dict(accepted=False, ncc=0., samples=len(src))

    def sample(xy):
        return cv.remap(detail, xy[:, :, 0].astype(np.float32),
                        xy[:, :, 1].astype(np.float32), cv.INTER_LINEAR,
                        borderMode=cv.BORDER_CONSTANT)

    a = sample(src)
    a -= a.mean(1, keepdims=True)
    norm = np.linalg.norm(a, axis=1)
    textured = norm >= 1
    if textured.sum() < 6:
        return dict(accepted=False, ncc=0., samples=int(textured.sum()))
    a, dst, norm = a[textured], dst[textured], norm[textured]
    best = np.full(len(a), -1.)
    for y in (-1., -.5, 0., .5, 1.):
        for x in (-1., -.5, 0., .5, 1.):
            check(cancel)
            b = sample(dst + [x, y])
            b -= b.mean(1, keepdims=True)
            n = np.linalg.norm(b, axis=1)
            score = (a * b).sum(1) / np.maximum(norm * n, 1e-9)
            best = np.maximum(best, score)
    value = float(np.median(best))
    return dict(accepted=value >= DETAIL_POLICY['minimum_ncc'], ncc=value, samples=len(best))


def corroborated_groups(detail, points, groups, models, cancel=lambda: False):
    # Geometry explicitly disabled: there is no mapping to verify. Preserve the
    # user's raw-inspection mode rather than inventing a geometric constraint.
    if not models:
        return groups, models
    accepted_groups, accepted_models = [], []
    for group, model in zip(groups, models):
        score = corroborate(detail, points, model, cancel)
        if score['accepted']:
            accepted_groups.append(group)
            accepted_models.append(dict(model, detail_corroboration=score))
    return tuple(accepted_groups), tuple(accepted_models)
