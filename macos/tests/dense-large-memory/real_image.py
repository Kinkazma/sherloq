"""Full-resolution, default extended+mirror profile; private image input via argv."""
import sys,time,json,resource,hashlib
from pathlib import Path
import cv2 as cv
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core.cloning2 import Cloning2Engine
from gui.sherloq_app.core.automatic_clones import parameters
from gui.sherloq_app.core.dense_parallel import BUDGET
im=cv.imread(sys.argv[1]);assert im is not None
h,w=im.shape[:2];zone=((0,0),(w-1,0),(w-1,h-1),(0,h-1));params,regions,_=parameters(im.shape,(zone,),zone,set())
cv.setNumThreads(8);start=time.monotonic();last=[None]
def progress(n,text):
 if last[0]!=(n,text):
  last[0]=(n,text);print(json.dumps(dict(seconds=round(time.monotonic()-start,2),progress=n,stage=text,peak_gib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3,2),reserved_gib=round(BUDGET.used/1024**3,2))),flush=True)
cancel_file=Path(__file__).with_name('cancel-requested')
engine=Cloning2Engine(im)
r=engine.analyze(params,regions,False,cancel_file.exists,progress)
result=dict(passed=True,shape=im.shape,params=params,seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,workspace_peak_bytes=BUDGET.peak,workspace_limit=BUDGET.limit,point_count=len(r['points']),pairs=len(r['pairs']),groups=len(r['groups']),dense_maps=len(r['dense_maps']),backend=r.get('backend'),mirror_policy=r.get('mirror_policy'),sha_points=hashlib.sha256(r['points'].tobytes()).hexdigest(),sha_pairs=hashlib.sha256(r['pairs'].tobytes()).hexdigest())
Path(__file__).with_name('real-image-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
