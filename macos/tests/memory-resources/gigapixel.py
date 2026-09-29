"""One billion noisy pixels, mapped source and actual public color engine."""
from pathlib import Path
import sys,json,time,resource,importlib.util,gc
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core import memory_resources as mr
from gui.sherloq_app.core.color_spaces import SpaceEngine
spec=importlib.util.spec_from_file_location('gui.sherloq_app.core._gigapixel_ref',Path(__file__).with_name('color_spaces_before.py'));old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
store=mr.TemporaryArrays(64*mr.MiB);shape=(25000,40000,3);image=store.array(shape,np.uint8);rng=np.random.default_rng(321);start=time.perf_counter()
for dst,_,_ in mr.tiles(shape,(512,512)):
 image[dst]=rng.integers(0,256,image[dst].shape,dtype=np.uint8);store.checkpoint()
store.checkpoint(force=True);print(dict(stage='source_generated',seconds=time.perf_counter()-start),flush=True)
limit=mr.MEMORY.limit;mr.MEMORY.limit=64*mr.MiB;params=('cmyk',2)
try:
 start=time.perf_counter();out=SpaceEngine(image).compute(params);elapsed=time.perf_counter()-start;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
finally:mr.MEMORY.limit=limit
assert isinstance(out,np.memmap);store.watch(out)
for dst,_,_ in mr.tiles(shape,(512,512)):
 expected=old.SpaceEngine(np.ascontiguousarray(image[dst])).compute(params)
 assert np.array_equal(out[dst],expected);store.checkpoint()
r=dict(passed=True,pixels=shape[0]*shape[1],bit_exact=True,reference='pointwise previous engine in blocks; full RAM reference not allocated',seconds=elapsed,peak_rss_bytes=peak)
Path(__file__).with_name('gigapixel-results.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
