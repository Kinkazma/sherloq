"""Budget releases claims after errors/cancellation and serializes large jobs."""
import sys,threading,time,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core.dense_parallel import WorkspaceBudget
from gui.sherloq_app.core.dense_memory import workspace_bytes
from gui.sherloq_app.core.jpeg_curve import Cancelled
b=WorkspaceBudget();size=workspace_bytes(3510,4387,1,8,True,'metal',6,True)
entered=threading.Event();release=threading.Event();second=threading.Event()
def first():
 with b.claim(size,lambda:False):entered.set();assert release.wait(5)
def next_job():
 with b.claim(size,lambda:False):second.set()
with ThreadPoolExecutor(2) as pool:
 f=pool.submit(first);assert entered.wait(5);g=pool.submit(next_job)
 time.sleep(.15);assert not second.is_set();release.set();f.result();g.result()
assert b.used==0 and b.max_active==1 and b.peak<=b.limit
try:
 with b.claim(size,lambda:False):raise RuntimeError('fixture')
except RuntimeError:pass
assert b.used==0
try:
 with b.claim(size,lambda:True):raise AssertionError('cancel ignored')
except Cancelled:pass
assert b.used==0
r=dict(passed=True,large_fields_serialized=True,exception_releases=True,cancel_releases=True,peak=b.peak,limit=b.limit)
Path(__file__).with_name('budget-results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
