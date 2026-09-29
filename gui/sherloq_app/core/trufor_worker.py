"""Isolated official TruFor inference; atomic raw checkpoints or NPZ output."""
import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path


class ModelCache:
    def __init__(self):
        self.value = None
        self.key = None

    def clear(self):
        self.value = None
        self.key = None
        import gc
        gc.collect()
        if 'torch' in sys.modules:
            torch = sys.modules['torch']
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    def get(self, root, device):
        weights = root / 'weights/trufor-state.pt'
        stat = weights.stat()
        key = (device, stat.st_size, stat.st_mtime_ns)
        if self.key == key:
            return (*self.value, True)
        self.clear()
        import torch
        from config import _C
        from models.cmx.builder_np_conf import myEncoderDecoder
        torch.set_num_threads(8)
        if device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('Apple Metal is unavailable. Select CPU.')
        cfg = _C.clone()
        cfg.merge_from_file(str(root / 'src/trufor.yaml'))
        cfg.freeze()
        weights = root / 'weights/trufor-state.pt'
        with weights.open('rb') as stream:
            weights_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
        checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
        model = myEncoderDecoder(cfg=cfg)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        del checkpoint
        model.eval().to(device)
        self.value = torch, model, weights_sha256
        self.key = key
        return (*self.value, False)


def run(args, models):
    if args.output_dir and (Path(args.output_dir)/'result.json').exists():
        raise FileExistsError('A completed TruFor result already exists in this directory.')
    started = time.monotonic()
    root = Path(__file__).resolve().parents[2] / 'TruFor_main' / 'test_docker'
    sys.path.insert(0, str(root / 'src'))
    import numpy as np
    if Path(args.input).suffix.lower() == '.npy':
        bgr = np.load(args.input, mmap_mode='r', allow_pickle=False)
        if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError('Expected an 8-bit BGR image.')
        pixels = np.ascontiguousarray(bgr[:, :, ::-1])
    else:
        from PIL import Image
        with Image.open(args.input) as image:
            pixels = np.array(image.convert('RGB'))
    # The first stride-4 embedding needs eight cells for its reduction kernel.
    if min(pixels.shape[:2]) < 29:
        raise ValueError('TruFor needs at least 29 pixels in each image dimension.')
    prepared = time.monotonic()
    print('Loading TruFor…', flush=True)
    torch, model, weights_sha256, reused = models.get(root, args.device)
    from models.cmx.encoders.dual_segformer import Attention
    stages = [module for module in model.modules() if isinstance(module, Attention)]
    completed = 0
    sampled_driver = sampled_tensors = 0
    def stage_finished(*_):
        nonlocal completed, sampled_driver, sampled_tensors
        if args.device == 'mps':
            sampled_driver = max(sampled_driver, torch.mps.driver_allocated_memory())
            sampled_tensors = max(sampled_tensors, torch.mps.current_allocated_memory())
        completed += 1
        print(f'Progress {completed}/{len(stages)}', flush=True)
    handles = [stage.register_forward_hook(stage_finished) for stage in stages]
    loaded = time.monotonic()
    # Keep the original tensor conversion and /256 normalization exactly.
    rgb = torch.tensor(pixels.transpose(2, 0, 1).copy(), dtype=torch.float32).unsqueeze(0) / 256.0
    print(f'Analysing {pixels.shape[1]} × {pixels.shape[0]} pixels on {args.device}…', flush=True)
    try:
        with torch.inference_mode():
            pred, conf, det, npp = model(rgb.to(args.device))
            result = {
                'map': torch.softmax(pred[0], dim=0)[1].cpu().numpy(),
                'conf': torch.sigmoid(conf[0, 0]).cpu().numpy(),
                'score': float(torch.sigmoid(det).item()),
                'np++': npp[0, 0].cpu().numpy(),
                'imgsize': np.array(pixels.shape[:2]),
                'device': args.device,
                'seconds': time.monotonic() - started,
            }
    finally:
        for handle in handles:
            handle.remove()
    if not all(np.isfinite(result[key]).all() for key in ('map', 'conf', 'score', 'np++')):
        raise RuntimeError('TruFor produced non-finite values; retry with CPU.')
    result.update(model_reused=reused, model_sha256=weights_sha256, prepare_seconds=prepared-started,
                  model_seconds=loaded-prepared, inference_seconds=time.monotonic()-loaded,
                  sampled_mps_driver_bytes=sampled_driver, sampled_mps_tensor_bytes=sampled_tensors)
    if args.output_dir:
        folder = Path(args.output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        for key in ('map', 'conf', 'np++'):
            temporary = folder / (key+'.partial.npy')
            np.save(temporary, result[key], allow_pickle=False)
            temporary.replace(folder / (key+'.npy'))
        metadata = {key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in result.items() if key not in ('map', 'conf', 'np++')}
        temporary = folder/'result.partial.json'
        temporary.write_text(json.dumps(metadata))
        temporary.replace(folder/'result.json')
    else:
        destination = Path(args.output)
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.npz', delete=False) as stream:
            temporary = Path(stream.name)
        try:
            np.savez_compressed(temporary, **result)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Completed in {result['seconds']:.1f} s; score={result['score']:.6f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--input')
    output = parser.add_mutually_exclusive_group()
    output.add_argument('--output')
    output.add_argument('--output-dir')
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    args = parser.parse_args()
    models = ModelCache()
    if not args.serve:
        if not args.input or not (args.output or args.output_dir):
            parser.error('--input and one output are required')
        run(args, models)
        return
    from types import SimpleNamespace
    import gc
    for line in sys.stdin:
        request = None
        try:
            request = json.loads(line)
            if request.get('device') not in ('cpu', 'mps'):
                raise ValueError('Unknown TruFor backend.')
            result = run(SimpleNamespace(input=request['input'], output=None,
                         output_dir=request['output_dir'], device=request['device']), models)
            del result
            gc.collect()
            torch = sys.modules['torch']
            idle_driver = idle_tensors = 0
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
                torch.mps.empty_cache()
                idle_driver = torch.mps.driver_allocated_memory()
                idle_tensors = torch.mps.current_allocated_memory()
            print('SHERLOQ '+json.dumps(dict(event='done', id=request['id'],
                  idle_mps_driver_bytes=idle_driver, idle_mps_tensor_bytes=idle_tensors)), flush=True)
        except Exception as exc:
            models.clear()
            print('SHERLOQ '+json.dumps(dict(event='error', id=request.get('id') if isinstance(request,dict) else None,
                  message=str(exc))), flush=True)
    models.clear()


if __name__ == '__main__':
    main()
