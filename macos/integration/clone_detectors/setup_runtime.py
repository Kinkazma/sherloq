"""Recreate the isolated inference environment from pinned offline wheels.

The main application's locked environment supplies Torch/NumPy/OpenCV. Additions
are private to this worker. No conflicting OpenCV wheel is installed.
"""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).parent
base_python=ROOT/'venv/bin/python';dest=BASE/'.venv-validation'
def run(*args):return subprocess.check_output(list(map(str,args)),text=True)
if not (dest/'bin/python').exists():subprocess.run([str(base_python),'-m','venv',str(dest)],check=True)
python=dest/'bin/python'
base_site=json.loads(run(base_python,'-c','import sysconfig,json;print(json.dumps(sysconfig.get_path("purelib")))'))
site=Path(json.loads(run(python,'-c','import sysconfig,json;print(json.dumps(sysconfig.get_path("purelib")))')))
(site/'sherloq-base.pth').write_text(base_site+'\n')
subprocess.run([str(python),'-m','pip','install','--no-index','--no-deps','--require-hashes','--find-links',str(BASE/'wheelhouse'),'-r',str(BASE/'runtime-requirements.lock')],check=True)
check=subprocess.run([str(python),'-m','pip','check'],capture_output=True,text=True)
expected={'albumentations 2.0.8 requires opencv-python-headless, which is not installed.','albucore 0.0.24 requires opencv-python-headless, which is not installed.','ultralytics 8.4.164 requires opencv-python, which is not installed.'}
errors=set(check.stdout.strip().splitlines())-expected-{'No broken requirements found.'}
if errors:raise RuntimeError('Dependency errors: '+repr(errors))
subprocess.run([str(python),'-c','import torch,cv2,albumentations,einops,ultralytics,shapely; assert cv2.__version__ == "4.11.0"; print("Worker imports OK; cv2 provided by existing opencv-contrib-python-headless")'],check=True,env=__import__('os').environ|{'NO_ALBUMENTATIONS_UPDATE':'1','YOLO_CONFIG_DIR':str(BASE/'yolo-config'),'YOLO_AUTOINSTALL':'false','YOLO_OFFLINE':'true'})
print('Ready:',python)
