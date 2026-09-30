from pathlib import Path
import sys,json,time,resource,importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core.interactive import WaveletEngine
from gui.sherloq_app.core import memory_resources as mr
spec=importlib.util.spec_from_file_location('gui.sherloq_app.core._huge_threshold_ref',Path(__file__).with_name('interactive_before_wavelet.py'));old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
rng=np.random.default_rng(413);image=rng.integers(0,256,(8000,12000,3),dtype=np.uint8);params=('db8',20,3,'soft')
limit=mr.MEMORY.limit;mr.MEMORY.limit=256*mr.MiB
try:
 start=time.perf_counter();actual=WaveletEngine(image).compute(params);seconds=time.perf_counter()-start;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
finally:mr.MEMORY.limit=limit
start=time.perf_counter();expected=old.WaveletEngine(image).compute(params);ram_seconds=time.perf_counter()-start
for first in range(0,len(actual),128):assert np.array_equal(actual[first:first+128],expected[first:first+128])
r=dict(passed=True,pixels=96000000,bit_exact=True,bounded_seconds=seconds,reference_seconds=ram_seconds,bounded_peak_rss_bytes=peak)
Path(__file__).with_name('huge-threshold-results.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
