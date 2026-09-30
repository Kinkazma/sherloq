from pathlib import Path
import sys,json,time,resource,importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core.illuminant import IlluminantEngine
from gui.sherloq_app.core import memory_resources as mr
spec=importlib.util.spec_from_file_location('gui.sherloq_app.core._huge_illuminant_ref',Path(__file__).with_name('illuminant_before.py'));old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
rng=np.random.default_rng(832);image=rng.integers(0,256,(8000,12000,3),dtype=np.uint8);params=((32,1,True,True),0)
limit=mr.MEMORY.limit;mr.MEMORY.limit=256*mr.MiB
try:
 start=time.perf_counter();actual=IlluminantEngine(image).compute(params);seconds=time.perf_counter()-start;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
finally:mr.MEMORY.limit=limit
start=time.perf_counter();expected=old.IlluminantEngine(image).compute(params);ram_seconds=time.perf_counter()-start
for a,b in zip((actual[0],*actual[1]),(expected[0],*expected[1])):
 for first in range(0,len(a),128):assert np.array_equal(a[first:first+128],b[first:first+128])
r=dict(passed=True,pixels=96000000,bit_exact=True,bounded_seconds=seconds,reference_seconds=ram_seconds,bounded_peak_rss_bytes=peak)
Path(__file__).with_name('huge-illuminant-results.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
