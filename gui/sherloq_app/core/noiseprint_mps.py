"""Metal inference from the original Noiseprint checkpoints (float32, approximate).

No training or weight approximation. TensorFlow is only needed on the first
lossless conversion of a checkpoint; subsequent runs load cached NumPy weights.
"""
import hashlib
import os
import sys
import tempfile
from pathlib import Path
import numpy as np


def load_weights(quality,cache_root=None):
    prefix=Path(__file__).resolve().parents[2]/'noiseprint/nets'/f'net_jpg{quality}'/'model'
    source=[prefix.with_suffix('.index'),Path(str(prefix)+'.data-00000-of-00001')]
    digest=hashlib.sha256()
    for path in source:
        with path.open('rb') as stream:
            while chunk:=stream.read(1024*1024):digest.update(chunk)
    fingerprint=digest.hexdigest()
    if cache_root is None:
        cache_root=(Path.home()/'Library/Caches/SHERLOQ/noiseprint' if sys.platform=='darwin'
                    else Path(os.environ.get('XDG_CACHE_HOME',Path.home()/'.cache'))/'sherloq/noiseprint')
    cache_root=Path(cache_root);cache_root.mkdir(parents=True,exist_ok=True)
    cached=cache_root/f'model-{quality}.npz'
    if cached.exists():
        try:
            with np.load(cached,allow_pickle=False) as stored:
                if str(stored['checkpoint_sha256'])==fingerprint and int(stored['format'])==1:
                    return {name:stored[name] for name in stored.files if name not in ('checkpoint_sha256','format')}
        except (OSError,ValueError,KeyError):
            pass
    import tensorflow as tf
    reader=tf.train.load_checkpoint(str(prefix));weights={}
    for level in range(17):
        prefix_name=f'level_{level}/'
        for name,suffix in [('weight','conv/weights'),('bias','bias/beta')]:
            weights[f'{level}_{name}']=reader.get_tensor(prefix_name+suffix)
        if 0<level<16:
            for name in ('moving_mean','moving_variance','gamma'):
                weights[f'{level}_{name}']=reader.get_tensor(prefix_name+'bn/'+name)
    handle=tempfile.NamedTemporaryFile(dir=cache_root,prefix=f'model-{quality}-',suffix='.npz',delete=False)
    temporary=Path(handle.name);handle.close()
    try:
        np.savez(temporary,**weights,checkpoint_sha256=np.asarray(fingerprint),format=np.asarray(1))
        temporary.replace(cached)
    finally:
        temporary.unlink(missing_ok=True)
    return weights


def gen_noise_mps(image,quality,progress=None,cache_dir=None):
    import torch
    if not torch.backends.mps.is_available():
        raise ValueError('Metal is unavailable. Select the CPU reference backend.')
    weights=load_weights(quality)
    model=[]
    for level in range(17):
        prefix=f'{level}_'
        values={key[len(prefix):]:torch.from_numpy(value).to('mps')
                for key,value in weights.items() if key.startswith(prefix)}
        values['weight']=values['weight'].permute(3,2,0,1).contiguous()
        model.append(values)
    def infer(clip):
        x=torch.from_numpy(np.array(clip,copy=True))[None,None].to('mps')
        for level,values in enumerate(model):
            x=torch.nn.functional.conv2d(x,values['weight'],padding=1)
            if 0<level<16:
                inv=torch.rsqrt(values['moving_variance']+1e-5)*values['gamma']
                x=x*inv[None,:,None,None]-values['moving_mean'][None,:,None,None]*inv[None,:,None,None]
            x=x+values['bias'][None,:,None,None]
            if level<16:x=torch.relu(x)
        return x[0,0].cpu().numpy()
    # Preserve the reference tile shape and overlap; do not silently downsample.
    step=1024 if image.size>1050000 else max(image.shape)
    overlap=34;cache=Path(cache_dir) if cache_dir is not None else None
    if cache is not None:cache.mkdir(parents=True,exist_ok=True)
    positions=[(y,x) for y in range(0,image.shape[0],step) for x in range(0,image.shape[1],step)]
    result=np.zeros(image.shape,np.float32)
    with torch.no_grad():
        for index,(y,x) in enumerate(positions):
            height=min(step,image.shape[0]-y);width=min(step,image.shape[1]-x)
            path=cache/f'{y}-{x}.npy' if cache is not None else None
            if path is not None and path.exists():
                block=np.load(path,allow_pickle=False)
                if block.shape!=(height,width) or block.dtype!=np.float32:
                    raise ValueError('Invalid cached Metal noise tile.')
            else:
                clip=image[max(0,y-overlap):min(image.shape[0],y+step+overlap),
                           max(0,x-overlap):min(image.shape[1],x+step+overlap)]
                block=infer(clip)
                if y>0:block=block[overlap:,:]
                if x>0:block=block[:,overlap:]
                block=block[:height,:width]
                if path is not None:
                    temporary=path.with_suffix('.partial.npy');np.save(temporary,block,allow_pickle=False);temporary.replace(path)
            result[y:y+height,x:x+width]=block
            if progress is not None:progress(index+1,len(positions))
    return result
