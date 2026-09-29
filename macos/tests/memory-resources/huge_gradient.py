from pathlib import Path
import sys,time,resource,json,gc
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core.memory_resources import TemporaryArrays,MiB,MEMORY,resident_memory
from gui.sherloq_app.core.gradient import GradientEngine
from gui.sherloq_app.core.histogram import analyze_histogram
store=TemporaryArrays(128*MiB);shape=(8000,12000);image=store.array((*shape,3),np.uint8);rng=np.random.default_rng(208)
for start in range(0,len(image),128):
 image[start:start+128]=rng.integers(0,256,image[start:start+128].shape,dtype=np.uint8);store.checkpoint()
store.checkpoint(force=True);limit=MEMORY.limit;MEMORY.limit=64*MiB;params=(95,3,True,True)
start=time.perf_counter()
try:actual=GradientEngine(image).compute(params)
finally:MEMORY.limit=limit
elapsed=time.perf_counter()-start;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(dict(stage='bounded_complete',seconds=elapsed,peak=peak),flush=True)
# Independent regular in-memory path, compared in chunks after peak capture.
start=time.perf_counter();reference=GradientEngine(image).compute(params);normal=time.perf_counter()-start
for y in range(0,len(image),128):assert np.array_equal(actual[y:y+128],reference[y:y+128])
del reference;gc.collect();store.checkpoint(force=True)
start=time.perf_counter();hist,unique,percent=analyze_histogram(image);hist_seconds=time.perf_counter()-start
assert (hist.sum(1)==shape[0]*shape[1]).all()
r=dict(passed=True,image_pixels=shape[0]*shape[1],gradient_bit_exact=True,bounded_peak_rss_bytes=peak,bounded_seconds=elapsed,ram_seconds=normal,histogram_count_exact=True,histogram_seconds=hist_seconds,colors=unique)
Path(__file__).with_name('huge-gradient-results.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
