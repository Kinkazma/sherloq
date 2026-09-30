from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core import memory_resources as mr
budget=mr.MemoryCoordinator(128*mr.MiB,available=lambda:8*mr.GiB);seen=[]
def bounded():
 store=mr.TemporaryArrays(8*mr.MiB);seen.append(store.working_bytes);return 17
assert budget.execute(256*mr.MiB,32*mr.MiB,lambda:0,bounded)==17
assert seen[-1]==104*mr.MiB and budget.peak==128*mr.MiB
with budget.claim(budget.plan(90*mr.MiB)):
 assert budget.execute(256*mr.MiB,32*mr.MiB,lambda:0,bounded)==17
 assert seen[-1]==14*mr.MiB
assert mr.TemporaryArrays(8*mr.MiB).working_bytes==8*mr.MiB and budget.used==0
# A forced fallback may never be re-labelled RAM to evade its minimum buffers.
budget.limit=16*mr.MiB
try:
 with budget.claim(mr.MemoryPlan('bounded',64*mr.MiB,1,0,64*mr.MiB)):pass
except MemoryError:pass
else:raise AssertionError('bounded workspace bypassed')
r=dict(passed=True,optional_pages_use_free_capacity=True,minimum_job_not_blocked=True,context_restored=True,minimum_bound_enforced=True)
Path(__file__).with_name('page-allowance-results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
