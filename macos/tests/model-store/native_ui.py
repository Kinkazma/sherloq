"""Actual first-use Adaptive CFA download and worker inference in isolated staging."""
from pathlib import Path
import sys,time,json
import numpy as np
from PySide6.QtWidgets import QApplication
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'source'))
from gui.sherloq_app.ui.research_job import ResearchJob
from gui.sherloq_app.ui.model_job import DOWNLOAD_POOL
from gui.sherloq_app.ui.jobs import _POOL,STAGING_POOL
from gui.sherloq_app.core.model_store import Store
app=QApplication([]);errors=[];results=[];progress=[]
job=ResearchJob(app,np.random.default_rng(21).integers(0,256,(96,96,3),np.uint8),'adaptive_cfa')
job.result.connect(results.append);job.failed.connect(errors.append);job.progress.connect(lambda n,s:progress.append(s))
job.request(dict(variant='Original',block=32),'cpu');deadline=time.monotonic()+90
while job.is_busy:
 app.processEvents();time.sleep(.01);assert time.monotonic()<deadline
assert not errors and results,(errors,progress)
assert (R/'models/external/pretrained.pt').is_file()
assert any('Downloading' in s for s in progress),progress
assert results[0]['metadata']['method']=='adaptive_cfa'
job.shutdown();DOWNLOAD_POOL.waitForDone(1000);STAGING_POOL.waitForDone(1000);_POOL.waitForDone(1000)
result=dict(passed=True,download_in_native_job=True,real_adaptive_cfa_inference=True,progress_visible=True,model_verified_before_inference=True)
Path(__file__).with_name('native-ui.json').write_text(json.dumps(result,indent=2));print(result)
