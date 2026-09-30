from pathlib import Path
import sys,time,json
from threading import Event
from PySide6.QtWidgets import QApplication
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'source'))
from gui.sherloq_app.ui.model_job import ModelJob,DOWNLOAD_POOL
from gui.sherloq_app.core.model_store import Cancelled
app=QApplication([]);began=Event();stopped=Event();ready=[];errors=[];updates=[]
class Fake:
 def ensure(self,feature,cancel,progress):
  began.set()
  if feature=='slow':
   while not cancel():time.sleep(.005)
   stopped.set();progress(2**33,2**34,'obsolete')
   raise Cancelled('cancelled')
  progress(2**33,2**34,'current')
job=ModelJob(app,Fake);job.ready.connect(lambda:ready.append(True));job.failed.connect(errors.append);job.progress.connect(lambda n,s:updates.append((n,s)))
def spin(until):
 deadline=time.monotonic()+5
 while not until():app.processEvents();time.sleep(.005);assert time.monotonic()<deadline
job.request('slow');spin(began.is_set);job.cancel();job.request('fast');spin(lambda:bool(ready))
assert stopped.is_set() and ready==[True] and not errors
assert len(updates)==1 and updates[0][0]==50 and 'current' in updates[0][1],updates
job.shutdown();DOWNLOAD_POOL.waitForDone(1000)
result=dict(passed=True,cancel_reaches_transfer=True,stale_completion_ignored=True,stale_progress_ignored=True,large_byte_counts=True)
Path(__file__).with_name('ui-cancel.json').write_text(json.dumps(result,indent=2));print(result)
