"""One independent, unchanged Sewar reduction in the comparison process group."""
import os
import sys
import json
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
import numpy as np
import sewar
name, first, second = sys.argv[1:]
if name not in ('msssim', 'vifp'):
    raise ValueError('Unsupported independent comparison metric.')
x = np.load(first, mmap_mode='r')
y = np.load(second, mmap_mode='r')
result = getattr(sewar, name)(x, y).real
print(json.dumps({'value': float(result).hex()}), flush=True)
