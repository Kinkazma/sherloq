"""Resident clone worker with offline execution and atomic NumPy/JSON results."""
import sys,os,json,time,gc,socket,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'source'))
config=ROOT/'integration/clone_detectors/yolo-config';config.mkdir(exist_ok=True)
os.environ.update(YOLO_CONFIG_DIR=str(config),YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',NO_ALBUMENTATIONS_UPDATE='1',HF_HUB_OFFLINE='1',PYTORCH_MPS_FAST_MATH='0')
def no_network(*args,**kwargs):raise RuntimeError('Aucun accès réseau pendant une analyse de clones.')
socket.socket.connect=no_network;socket.create_connection=no_network
import numpy as np
from gui.sherloq_app.core.clone_detectors import Models,analyze
from gui.sherloq_app.core.macos_activity import user_activity

def run(request,models):
    import torch
    torch.set_num_threads(8);start=time.perf_counter();folder=Path(request['output_dir']);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():raise ValueError('Résultat déjà terminé.')
    image=np.load(request['input'],allow_pickle=False,mmap_mode='r');print('Loading model',flush=True)
    result=analyze(image,request['params'],request['device'],models,lambda d,n:print(f'Progress {d}/{n}',flush=True))
    metadata=result.pop('metadata');metadata.update(seconds=time.perf_counter()-start,params=request['params'],image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),arrays={})
    for name,array in result.items():
        if not np.isfinite(array).all():raise ValueError('Résultat non fini : '+name)
        partial=folder/(name+'.partial.npy');np.save(partial,array,allow_pickle=False);partial.replace(folder/(name+'.npy'));metadata['arrays'][name]=dict(shape=list(array.shape),dtype=str(array.dtype))
    path=folder/'result.partial.json';path.write_text(json.dumps(metadata));path.replace(folder/'result.json')
    return metadata

def main():
    models=Models()
    for line in sys.stdin:
        request=None
        try:
            request=json.loads(line)
            with user_activity('SHERLOQ clone detector'):
                run(request,models)
            event=dict(event='done',id=request['id'])
        except Exception as exc:
            models.clear();event=dict(event='error',id=request.get('id') if isinstance(request,dict) else None,message=str(exc))
        print('SHERLOQ '+json.dumps(event),flush=True);gc.collect()
    models.clear()
if __name__=='__main__':main()
