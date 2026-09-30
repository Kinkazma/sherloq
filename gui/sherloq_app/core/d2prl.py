"""D2PRL's released 448px/40-iteration inference, local weights only.

The three returned tensors are union probability, target residual and source
residual, NOT three class logits. Preserve the author's postprocessing.
"""
from pathlib import Path
import numpy as np
import cv2 as cv

SIDE = 448
ITERATIONS = 40
SEED = 22


def load(device):
    import torch
    from .clone_models import ROOT, WEIGHTS, runtime, verified
    runtime()
    from sherloq_clone_models.d2prl.models_D2PRL import DPM
    assets = ROOT / 'third_party/research/clone_detectors/01_d2prl'
    signatures = {}
    for name in ('ZM_polar_k13.mat', 'VV_mvf7.mat', 'VV_mvf9.mat', 'VV_mvf11.mat'):
        p, sha = verified(assets / name)
        signatures[str(p.relative_to(ROOT))] = sha
    path, sha = verified(WEIGHTS / '01_d2prl/d2prl.pth')
    signatures[str(path.relative_to(ROOT))] = sha
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(SEED)
        model = DPM(SIDE, 1, ITERATIONS, assets)
        rng_state = torch.get_rng_state()
    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    # These deterministic grids are ordinary attributes upstream. Register them
    # as non-persistent buffers so MPS does not upload/synchronise them every
    # evaluation. No checkpoint key, random draw or numerical operation changes.
    for module in model.patchmatch.modules():
        for name, value in list(vars(module).items()):
            if isinstance(value, torch.Tensor) and not isinstance(value, torch.nn.Parameter):
                delattr(module, name)
                module.register_buffer(name, value, persistent=False)
    if device == 'mps':
        from types import MethodType
        from .d2prl_ops import evaluate_batched
        for evaluator in (model.patchmatch.evaluate_ZM, model.patchmatch.evaluate_CNN):
            evaluator.forward = MethodType(evaluate_batched, evaluator)
    return dict(model=model.eval().to(device), device=device, side=SIDE,
                kind='d2prl', weights=signatures, rng_state=rng_state)


def postprocess(raw, min_component=500):
    """Original post_1c/post_3c, on the native model grid before resizing."""
    from skimage.morphology import remove_small_objects
    union, target, source = raw
    if not 0 <= min_component <= SIDE * SIDE:
        raise ValueError('Invalid minimum component size')
    mask = remove_small_objects(np.rint(union) > .5, min_size=min_component)
    t = (target > 0).astype(np.float32)
    s = (source > 0).astype(np.float32)
    both = remove_small_objects(t + s > 0, min_size=min_component)
    t = t * both
    s = both.astype(np.float32) - t
    signed = cv.filter2D(t - s, -1, np.ones((50, 50)), borderType=cv.BORDER_CONSTANT)
    t = (signed > 0) & both
    s = both & ~t
    return mask.astype(np.uint8), t.astype(np.float32), s.astype(np.float32)


def refilter(request):
    """Rebuild masks from cached native grids, never rerun neural inference.

    Independent zones keep their own connected components, then combine in
    original coordinates. Return fresh arrays so rendering/export can overlap.
    """
    result, minimum = request
    from copy import deepcopy
    updated = dict(result)
    metadata = deepcopy(result['metadata'])
    updated['metadata'] = metadata
    for name in ('mask', 'target', 'source'):
        updated[name] = np.zeros_like(result[name])
    for raw, box, zone in zip(result['raw_probabilities'], metadata['boxes'], metadata['zones']):
        x0, y0, x1, y1 = box
        masks = postprocess(raw, minimum)
        for name, values in zip(('mask', 'target', 'source'), masks):
            area = updated[name][y0:y1, x0:x1]
            np.maximum(area, cv.resize(values, (x1-x0, y1-y0), interpolation=cv.INTER_NEAREST), out=area)
        zone['min_component'] = minimum
        zone['status'] = 'ok' if masks[0].any() else 'empty'
    for name in ('mask', 'target', 'source'):
        updated[name] *= result['analyzed']
    metadata['min_component'] = minimum
    metadata['status'] = 'ok' if updated['mask'].any() else 'empty'
    return updated


def predict(image, loaded, progress=lambda *args: None):
    import torch
    from torchvision import transforms as T
    # Tensor resize (bilinear antialiased), not PIL/bicubic used by training fork.
    x = T.Compose([T.ToPILImage(), T.ToTensor(), T.Resize((SIDE, SIDE))])(
        cv.cvtColor(image, cv.COLOR_BGR2RGB))[None].to(loaded['device'])
    count = [0]
    def step(*_):
        count[0] += 1
        progress(count[0], 2 * ITERATIONS)
    hook = loaded['model'].patchmatch.evaluate_CNN.register_forward_hook(step)
    mps_state = torch.mps.get_rng_state() if loaded['device'] == 'mps' else None
    try:
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.set_rng_state(loaded['rng_state'])
            if loaded['device'] == 'mps': torch.mps.manual_seed(SEED)
            tensors = loaded['model'](x)
            raw = np.stack([t[0, 0].cpu().numpy() for t in tensors])
    finally:
        hook.remove()
        if mps_state is not None: torch.mps.set_rng_state(mps_state)
    if not np.isfinite(raw).all():
        raise ValueError('D2PRL returned non-finite values.')
    mask, target, source = postprocess(raw)
    return dict(raw=raw, map=raw[0], mask=mask, target=target, source=source)
