from pathlib import Path
import sys,json,time,resource,importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core.contrast import ContrastEngine
from gui.sherloq_app.core import memory_resources as mr
spec=importlib.util.spec_from_file_location('gui.sherloq_app.core._huge_contrast_ref',Path(__file__).with_name('contrast_before.py'));old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
rng=np.random.default_rng(832);image=rng.integers(0,256,(8000,12000,3),dtype=np.uint8);params=(32,2)
limit=mr.MEMORY.limit;mr.MEMORY.limit=256*mr.MiB
try:
 start=time.perf_counter();actual=ContrastEngine(image).compute(params);seconds=time.perf_counter()-start;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
finally:mr.MEMORY.limit=limit
start=time.perf_counter();expected=old.ContrastEngine(image).compute(params);ram_seconds=time.perf_counter()-start
for first in range(0,len(actual),128):assert np.array_equal(actual[first:first+128],expected[first:first+128])
r=dict(passed=True,pixels=96000000,bit_exact=True,bounded_seconds=seconds,reference_seconds=ram_seconds,bounded_peak_rss_bytes=peak)
Path(__file__).with_name('huge-contrast-results.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
