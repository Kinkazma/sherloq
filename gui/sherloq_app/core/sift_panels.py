"""SHERLOQ panel/text alternative to the unavailable full luc_pub ensemble."""
from .cloning import check

NAME = 'SIFT + G2NN + RANSAC + Panels + Text'


def analyze(engine, params, regions, compare, cancel, progress):
    from .auto_zones import detect_panels, enclosing
    from .cloning2 import palette
    from .copy_overlap import reject_self, MAX_OVERLAP
    from .sift_g2nn import NAME as CLASSIC
    from .text_regions import detect_text, polygons
    key = ('panels-text-v2', params, regions, compare)
    cached = engine.results.get(key)
    if cached is not None:
        return cached
    progress(0, 'Locating panels and text')
    auto = not regions and not compare
    if auto:
        panels = engine.features.get(('automatic-panels-v1',))
        if panels is None:
            panels = detect_panels(engine.image, cancel)
            check(cancel)
            engine.features.put(('automatic-panels-v1',), panels)
        envelope = enclosing(panels)
        regions = tuple(dict.fromkeys((*panels, envelope))) if panels else ()
    else:
        panels, envelope = (), None
    boxes = engine.features.get(('text-regions-v1',))
    if boxes is None:
        boxes = detect_text(engine.image, cancel, lambda n, s: progress(n // 10, s))
        check(cancel)
        engine.features.put(('text-regions-v1',), boxes)
    text = polygons(boxes)
    settings = list(params)
    settings[0] = CLASSIC
    settings[16] = (*params[16], *text)
    # The new profile always preserves independent within-zone feature budgets.
    settings = settings[:18] + [1, True]
    result = engine.analyze(tuple(settings), regions, compare, cancel,
                            lambda n, s: progress(10 + n * (45 if params[11] else 90) // 100, s))
    if params[11]:
        from .sift_reflection import analyze as reflected, append
        extra=reflected(engine,tuple(settings),regions,compare,cancel,
                        lambda n,s:progress(55+n*40//100,s))
        result=append(result,extra)
    groups, models, rejected = reject_self(result['points'], result['pairs'], result['groups'], result['models'], params[3], cancel)
    colors, bases = palette(groups, result['pairs'], params[4])
    from .copy_subbiomes import refine
    groups, models, colors, bases, partitions = refine(result['points'], result['pairs'], groups, models,
                                                       colors, bases, params[5], cancel)
    output = dict(result, params=params, groups=groups, models=models, colors=colors, bases=bases,
                  biome_partitions=partitions,
                  self_match_filter=dict(maximum_overlap=MAX_OVERLAP, metric='intersection_over_smaller_hull', minimum_model_displacement_px=params[3]),
                  rejected_biomes=rejected,
                  preprocessing=dict(version=2, reflections=bool(params[11]), implementation='SHERLOQ alternative, not the luc_pub competition ensemble',
                                     automatic_panels=auto, panels=panels, envelope=envelope,
                                     text_engine='Tesseract / eng / PSM 11', text_boxes=boxes,
                                     text_exclusions=text, graph_exclusion=False))
    check(cancel)
    engine.results.put(key, output)
    return output
