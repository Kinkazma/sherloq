"""Real worker/cache and Qt panel checks of the installed statistics policy."""
from pathlib import Path
import sys, json, os, tempfile, subprocess, time
import numpy as np
import cv2 as cv
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source'))
from gui.noiseprint.post_em import getSpamFromNoiseprint
from gui.noiseprint.utility.stable_covariance import STATISTICS_POLICY
fixture = np.load(OUT/'fixtures/singular-chain.npz')
expected = np.load(OUT/'fixtures/singular-chain-stable.npz')
image = cv.cvtColor(np.rint(fixture['gray']*255).astype(np.uint8), cv.COLOR_GRAY2BGR)
spam, valid, r0, r1, size = getSpamFromNoiseprint(fixture['noise'], fixture['gray'])


def populate(folder, backend, stale=False):
    for name, array in dict(image=image, gray=fixture['gray'], noise=fixture['noise'],
                            spam=spam, valid=valid, range0=r0, range1=r1).items():
        np.save(folder/f'{name}.npy', array)
    (folder/'model.json').write_text(json.dumps(dict(model=95, automatic=False, backend=backend)))
    (folder/'spam-complete.json').write_text(json.dumps(dict(imgsize=size)))
    if stale:
        np.save(folder/'map.npy', np.zeros(spam.shape[:2]))
        np.save(folder/'map-display.npy', np.zeros_like(image))
        (folder/'result.json').write_text(json.dumps(dict(model=95, automatic=False, backend=backend, stage='map')))


records = []
for backend in ['cpu', 'mps']:
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        populate(folder, backend, stale=True)
        times = {name:(folder/f'{name}.npy').stat().st_mtime_ns for name in ['noise','spam']}
        command = [sys.executable, str(ROOT/'source/gui/sherloq_app/core/splicing_worker.py'), tmp, 'map', '95', backend]
        env = dict(os.environ, OPENBLAS_NUM_THREADS='8')
        for repeat in [False, True]:
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, result.stderr
            m = np.load(folder/'map.npy')
            delta = float(np.max(abs(m/np.max(m)-expected['map']/np.max(expected['map']))))
            assert delta < 1e-5, delta
            marker = json.loads((folder/'map-statistics.json').read_text())
            assert marker['statistics_policy'] == STATISTICS_POLICY
            assert json.loads((folder/'result.json').read_text())['statistics_policy'] == STATISTICS_POLICY
            assert all((folder/f'{n}.npy').stat().st_mtime_ns == t for n,t in times.items())
            if repeat: assert (folder/'map.npy').stat().st_mtime_ns == mtime
            mtime = (folder/'map.npy').stat().st_mtime_ns
        records.append(dict(backend=backend, normalized_error=delta, old_cache_recomputed=True,
                            residual_and_spam_reused=True, stable_map_reused=True))

# Exercise the actual panel: CPU/GPU selector, process delivery, export, close.
from PySide6.QtWidgets import QApplication, QFileDialog
from gui.sherloq_app.tools.tampering.splicing import SplicingWidget
app = QApplication([])
w = SplicingWidget(image)
w.model_combo.setCurrentIndex(w.model_combo.findData(95))
assert not w.cpu_button.isChecked()
errors = []
w.job.failed.connect(errors.append)
w.estimate_noise()
folder = Path(w.job.context.folder.name)
# Supply a retained stage before dispatching the queued worker start.
while w.job.context.writing.is_set(): time.sleep(.002)
populate(folder, 'mps')
def wait():
    deadline = time.monotonic()+60
    while w.job.is_busy:
        app.processEvents(); time.sleep(.002)
        assert time.monotonic() < deadline, 'Qt worker timeout'
    app.processEvents(); assert not errors, errors
wait()
assert w.noise is not None
np.save(folder/'map.npy', np.zeros(spam.shape[:2]))
np.save(folder/'map-display.npy', np.zeros_like(image))
w.compute_map(); wait()
assert np.max(abs(w.raw_map/np.max(w.raw_map)-expected['map']/np.max(expected['map']))) < 1e-5
assert w.map_viewer.export_button.isEnabled()
export = OUT/'panel-export.png'
original = QFileDialog.getSaveFileName
try:
    QFileDialog.getSaveFileName = lambda *a, **k:(str(export), 'PNG')
    w.map_viewer.export_image()
finally:
    QFileDialog.getSaveFileName = original
assert np.array_equal(cv.imread(str(export)), w.map)
mtime = (folder/'map.npy').stat().st_mtime_ns
w.job.request('map', 95, 'mps'); wait()
assert (folder/'map.npy').stat().st_mtime_ns == mtime
w.cpu_button.setChecked(True)
assert w.map is None and not folder.exists()
w.shutdown()
report = dict(passed=True, workers=records, qt_panel=True, stale_cache_rejected=True,
              cached_policy_reused=True, exported_png_exact=True, backend_invalidation=True,
              default_gpu=True)
(OUT/'delivery-results.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
