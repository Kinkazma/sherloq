"""Isolated resident pretrained analyses; raw arrays are committed before metadata."""
import sys,json,gc,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'source'))

class Models:
    def __init__(self):self.key=None;self.model=None;self.hashes={};self.intermediates={}
    def clear(self):
        self.key=None;self.model=None;self.intermediates.clear();gc.collect()
        if 'torch' in sys.modules and sys.modules['torch'].backends.mps.is_available():sys.modules['torch'].mps.empty_cache()
    def get(self,method,variant,device):
        key=(method,None if method=='safire' else variant,device)
        if key==self.key:return self.model,True
        self.clear()
        if method=='adaptive_cfa':
            from gui.sherloq_app.core.adaptive_cfa import load,WEIGHTS
            files=[WEIGHTS[variant]];model=load(variant,device)
        elif method=='catnet':
            from gui.sherloq_app.core.catnet import load
            files=['CAT_full_v2.pth.tar'];model=load(device)
        elif method=='safire':
            from gui.sherloq_app.core.safire import load
            files=['safire.pth'];model=load(device)
        elif method=='focal':
            from gui.sherloq_app.core.focal import load
            files=['FOCAL_ViT_weights.pth','FOCAL_HRNet_weights.pth'];model=load(device)
        elif method=='adaifl':
            from gui.sherloq_app.core.adaifl import load
            files=['AdaIFL_v0.pth'];model=load(device)
        else:raise ValueError('Unknown analysis method.')
        self.hashes={}
        for filename in files:
            with (ROOT/'models/external'/filename).open('rb') as stream:self.hashes[filename]=hashlib.file_digest(stream,'sha256').hexdigest()
        self.model=model;self.key=key;return model,False

def run(request,models):
    import torch
    torch.set_num_threads(8)
    device=request['device']
    if device not in ('cpu','mps'):raise ValueError('Invalid backend.')
    if device=='mps' and not torch.backends.mps.is_available():raise ValueError('Metal is unavailable. Select CPU.')
    folder=Path(request['output_dir']);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():raise ValueError('A completed analysis already exists.')
    from gui.sherloq_app.core.image_buffers import image_sha256
    image=np.load(request['input'],mmap_mode='r',allow_pickle=False)
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:raise ValueError('Expected an 8-bit BGR image.')
    started=time.perf_counter();method=request['method'];params=request['params'];print('Loading model',flush=True)
    model,reused=models.get(method,params['variant'],device);loaded=time.perf_counter()
    if method=='adaptive_cfa':
        from gui.sherloq_app.core.adaptive_cfa import predict
        result=predict(image,model,block=params['block'],tile=0 if device=='cpu' and image.shape[0]*image.shape[1]<=4_000_000 else 512,progress=lambda d,n:print(f'Progress {d}/{n}',flush=True))
    elif method=='catnet':
        from gui.sherloq_app.core.catnet import predict,source_for
        path,source_meta=source_for(image,request.get('source'),folder)
        result=predict(path,model);result['metadata'].update(source_meta)
        if result['map'].shape!=image.shape[:2]:raise ValueError('CAT-Net output does not align with the opened image.')
    elif method=='safire':
        from gui.sherloq_app.core.safire import predict
        result=predict(image,model,side=params['side'],groups=params['groups'],kind='dbscan' if 'DBSCAN' in params['variant'] else 'kmeans',eps=params['eps'],minimum=params['minimum'],binary=params['variant']=='Binaire',cache=models.intermediates,progress=lambda d,n:print(f'Progress {d}/{n}',flush=True))
    elif method in ('focal','adaifl'):
        from importlib import import_module
        result=import_module('gui.sherloq_app.core.'+method).predict(image,model)
    else:raise ValueError('Unknown analysis method.')
    metadata=result.pop('metadata');metadata.update(method=method,params=params,device=device,model_reused=reused,weights=models.hashes,model_seconds=loaded-started,inference_seconds=time.perf_counter()-loaded,seconds=time.perf_counter()-started,image_sha256=image_sha256(image),arrays={})
    for name,array in result.items():
        array=np.asarray(array)
        if not np.isfinite(array).all():raise ValueError(f'Invalid output {name}.')
        temporary=folder/f'{name}.partial.npy';np.save(temporary,array,allow_pickle=False);temporary.replace(folder/f'{name}.npy');metadata['arrays'][name]=dict(shape=list(array.shape),dtype=str(array.dtype))
    temporary=folder/'result.partial.json';temporary.write_text(json.dumps(metadata));temporary.replace(folder/'result.json')
    return metadata

def main():
    models=Models()
    for line in sys.stdin:
        request=None
        try:
            request=json.loads(line);run(request,models);gc.collect()
            import torch
            if torch.backends.mps.is_available():torch.mps.synchronize();torch.mps.empty_cache()
            event=dict(event='done',id=request['id'])
        except Exception as exc:
            models.clear();event=dict(event='error',id=request.get('id') if isinstance(request,dict) else None,message=str(exc))
        print('SHERLOQ '+json.dumps(event),flush=True)
    models.clear()
if __name__=='__main__':main()
