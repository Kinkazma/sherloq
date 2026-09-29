from pathlib import Path
import os,sys,json,tempfile,gc
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core.image_buffers import all_finite
from gui.sherloq_app.ui.trufor_job import _Files
from gui.sherloq_app.ui.research_job import read_result,prepare
from gui.sherloq_app.core import memory_resources as mr
for shape in [(0,3),(),(23,45),(17,18,3)]:
 a=np.ones(shape,np.float32);assert all_finite(a)
 if a.size:a.flat[-1]=np.nan;assert not all_finite(a)
a=np.ones((32,35,3),np.float32);a[3,4,1]=np.inf
assert not all_finite(a[::-1,::2,::-1]);assert all_finite(a[:,1::2])
files=_Files(np.zeros((12,13,3),np.uint8));folder=files.path/'case';folder.mkdir();expected=np.arange(12*13,dtype=np.float32).reshape(12,13)
np.save(folder/'map.npy',expected);(folder/'result.json').write_text(json.dumps(dict(arrays={'map':dict(shape=[12,13],dtype='float32')})))
key,result=read_result((files,'case'));assert isinstance(result['map'],np.memmap) and not result['map'].flags.writeable
files.retire();assert not files.path.exists();assert np.array_equal(result['map'],expected)
# Source staging has a capacity check, not the former arbitrary 512 MiB cap.
with tempfile.TemporaryDirectory() as root:
 source=Path(root)/'source.bin'
 with source.open('wb') as f:f.truncate(513*mr.MiB)
 files=_Files(np.zeros((3,4,3),np.uint8));files.source_filename=source
 try:
  prepare(files);assert (files.path/'source.image').stat().st_size==513*mr.MiB
 finally:files.retire()
report=dict(passed=True,finite_validation=True,readonly_results=True,mappings_survive_retirement=True,source_over_512MiB=True)
Path(__file__).with_name('worker-buffers-results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
