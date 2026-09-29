"""Isolate SciPy's GIL-holding kernels; persist completed metrics for cancellation."""
import os
if os.name == 'posix' and os.getpgrp() != os.getpid():
    os.setsid()  # Helpers inherit this group so cancellation can stop them too.
import sys
import json
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
import numpy as np
import cv2 as cv
from gui.sherloq_app.core.comparison import ComparisonEngine


def main():
    folder = Path(sys.argv[1])
    engine = ComparisonEngine(np.load(folder / 'evidence.npy', mmap_mode='r'), np.load(folder / 'reference.npy', mmap_mode='r'))
    engine.helper_folder = folder / 'helpers'
    engine.helper_folder.mkdir(exist_ok=True)
    checkpoint = folder / 'checkpoint.json'
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        engine.values.update({name: float.fromhex(value) for name, value in saved['values'].items()})
        for name in saved['maps']:
            engine.maps[name] = np.load(folder / (name + '.npy'), mmap_mode='r')

    def save(values, maps, errors, progress):
        for name, array in maps.items():
            path = folder / (name + '.npy')
            if not path.exists():
                temporary = folder / (name + '.partial.npy')
                np.save(temporary, array, allow_pickle=False)
                temporary.replace(path)
        data = {'values': {name: float(value).hex() for name, value in values.items()},
                'errors': errors, 'maps': list(maps)}
        temporary = folder / 'checkpoint.partial.json'
        temporary.write_text(json.dumps(data))
        temporary.replace(checkpoint)
        print(json.dumps({'progress': progress}), flush=True)

    engine.checkpoint = save
    processes = {}
    try:
        # The two slowest independent reductions keep their original arithmetic
        # and run on separate CPU cores. Fresh interpreters avoid fork/OpenCV
        # thread hazards. They inherit our process group for reliable cancel.
        if engine.evidence.shape[0] * engine.evidence.shape[1] >= 1_048_576:
            paths = []
            for name, image in [('evidence-gray', engine.evidence), ('reference-gray', engine.reference)]:
                path = folder / (name + '.npy')
                if not path.exists():
                    temporary = folder / (name + '.partial.npy')
                    np.save(temporary, cv.cvtColor(image, cv.COLOR_BGR2GRAY), allow_pickle=False)
                    temporary.replace(path)
                paths.append(str(path))
            engine.gray_pair = tuple(np.load(path, mmap_mode='r') for path in paths)
            script = Path(__file__).with_name('comparison_metric_worker.py')
            for name in ('msssim', 'vifp'):
                if name not in engine.values:
                    processes[name] = subprocess.Popen([sys.executable, str(script), name, *paths], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            def independent(name):
                process = processes[name]
                try:
                    output, errors = process.communicate(timeout=3600)
                except subprocess.TimeoutExpired:
                    process.kill();process.communicate()
                    raise RuntimeError('Comparison metric timed out.')
                if process.returncode:
                    raise RuntimeError(errors.decode(errors='replace')[-4096:] or 'Comparison metric failed.')
                return float.fromhex(json.loads(output)['value'])
            engine.metric_runner = independent
        result = engine.compute()
        save(result['values'], engine.maps, result['errors'], (20, ''))
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.kill()
            process.communicate()



if __name__ == '__main__':
    main()
