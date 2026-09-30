"""Real complete analysis: PatchMatch mirror/scales + Forge Auto + ELA/Ghosts.

Small synthetic image keeps this a wiring check, not a quality benchmark.
"""
import sys,time,json,tempfile
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import QApplication
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'source'))
from gui.sherloq_app.tools.tampering.complete_analysis import CompleteAnalysisWidget
from gui.sherloq_app.ui.jobs import _POOL
from gui.sherloq_app.core.automatic_clones import export
app=QApplication([]);errors=[];sys.excepthook=lambda kind,value,tb:errors.append(str(value))
rng=np.random.default_rng(913)
image=rng.integers(30,220,(128,256,3),np.uint8)
image[30:78,140:188]=image[20:68,20:68][:,::-1]
w=CompleteAnalysisWidget(image,autostart=False)
start=time.perf_counter();w.run_whole_image()
last=None;deadline=time.monotonic()+240
while True:
 app.processEvents();time.sleep(.01)
 state=tuple(w.states.values())
 if state!=last:print(w.states,flush=True);last=state
 assert not errors and not w.errors,(errors,w.errors)
 assert time.monotonic()<deadline,w.states
 if all(v=='Complete' for v in state) and not any(j.is_busy for j in (w.job,w.forge,w.d2_job,w.sift_job,w.ela_job,w.prepare,w.draw)):break
assert set(w.results)=={'patchmatch','forgeryscope','ela','sift','d2prl'}
assert w.results['forgeryscope']['metadata']['variant']=='Forgeryscope Auto'
assert len(w.results['forgeryscope']['metadata']['zones'])==1
pm=w.results['patchmatch']
assert pm['mirror_policy']['extended_preserved']
assert set(pm['mirror_policy']['sift_bins'])=={6,8,10,12}
assert w.ela_ghost.isChecked()
assert w.results['d2prl']['metadata']['variant']=='D2PRL'
assert len(w.results['d2prl']['metadata']['zones'])==1
assert w.d2_minimum.value()==500
assert 'finished' in w.status.text().lower() or 'terminés' in w.status.text().lower(),w.status.text()
with tempfile.TemporaryDirectory() as folder:
 path=export((str(Path(folder)/'result.npz'),dict(results=w.results,configuration=w.submitted,biomes=w.biomes)))
 with np.load(path,allow_pickle=False) as saved:
  assert np.array_equal(saved['root_results_forgeryscope_mask'],w.results['forgeryscope']['mask'])
  assert np.array_equal(saved['root_results_patchmatch_pairs'],pm['pairs'])
report=dict(passed=True,real_patchmatch=True,real_forgeryscope_auto=True,real_ela_ghosts=True,real_sift_panels=True,real_d2prl=True,
 seconds=time.perf_counter()-start,shape=list(image.shape),mirror_scales=True,export_exact=True,
 patchmatch_pairs=len(pm['pairs']),forgeryscope_status=w.results['forgeryscope']['metadata']['status'])
w.close();_POOL.waitForDone(10000);app.processEvents();assert not errors,errors
Path(__file__).with_name('combined-native-results.json').write_text(json.dumps(report,indent=2));print(report)
