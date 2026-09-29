"""Process boundary for TensorFlow/EM; atomic files retain completed stages."""
import os
import sys
import json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
# Accelerated mode avoids oversubscribing tiny BLAS solves. This may change
# float64 values by roundoff; the CPU reference mode keeps original threading.
if len(sys.argv)>4 and sys.argv[4]=='mps':
    os.environ['OPENBLAS_NUM_THREADS']='1'
    import threading
    threading.stack_size(8 * 1024 * 1024)
import numpy as np
import cv2 as cv
from gui.sherloq_app.core.splicing import estimate_model,noise_display


def save(path,array):
    temporary=path.with_suffix('.partial.npy')
    np.save(temporary,array,allow_pickle=False)
    temporary.replace(path)


def write_json(path,value):
    temporary=path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def main():
    folder=Path(sys.argv[1]);stage=sys.argv[2];requested=int(sys.argv[3])
    backend=sys.argv[4] if len(sys.argv)>4 else 'cpu'
    if backend not in ('cpu','mps'):raise ValueError('Unknown Noiseprint backend.')
    def progress(value,text):
        print(json.dumps(dict(progress=[value,text])),flush=True)
    image=np.load(folder/'image.npy',mmap_mode='r',allow_pickle=False)
    metadata=folder/'model.json'
    if metadata.exists():
        info=json.loads(metadata.read_text());quality=info['model']
    else:
        quality=estimate_model(image,progress) if requested==0 else requested
        if not 51<=quality<=101:
            raise ValueError(f'Estimated quality {quality} has no installed Noiseprint model. Choose a model explicitly (51–100, or 101 for uncompressed).')
        info=dict(model=int(quality),automatic=requested==0,backend=backend)
        write_json(metadata,info)
    info.setdefault('backend',backend)
    gray_path=folder/'gray.npy'
    if not gray_path.exists():
        save(gray_path,cv.cvtColor(image,cv.COLOR_BGR2GRAY).astype(np.float32)/255)
    gray=np.load(gray_path,mmap_mode='r',allow_pickle=False)
    noise_path=folder/'noise.npy'
    if not noise_path.exists():
        progress(20,f'Loading Noiseprint model {quality}')
        report=lambda done,total:progress(20+done*60//total,f'Noiseprint tile {done}/{total}')
        if backend=='mps':
            from gui.sherloq_app.core.noiseprint_mps import gen_noise_mps
            noise=gen_noise_mps(gray,quality,cache_dir=folder/'tiles-mps',progress=report)
        else:
            from gui.noiseprint.noiseprint import genNoiseprint
            noise=genNoiseprint(gray,quality,model_name='net',cache_dir=folder/'tiles-cpu',progress=report)
        save(noise_path,noise)
    else:
        noise=np.load(noise_path,mmap_mode='r',allow_pickle=False)
    if not (folder/'noise-display.npy').exists():
        save(folder/'noise-display.npy',noise_display(noise))
    if stage=='map' and not (folder/'map-display.npy').exists():
        if min(gray.shape)<100:
            raise ValueError('Too few valid blocks: the heatmap needs at least 100 × 100 pixels.')
        progress(82,'Computing local noise statistics')
        from gui.noiseprint.post_em import getSpamFromNoiseprint,EMgu_img
        from gui.noiseprint.noiseprint_blind import genMappUint8
        # No TensorFlow import when a completed noise estimate already exists.
        spam_path=folder/'spam.npy'
        if not (folder/'spam-complete.json').exists():
            spam,valid,r0,r1,imgsize=getSpamFromNoiseprint(noise,gray)
            for name,array in [('spam',spam),('valid',valid),('range0',r0),('range1',r1)]:
                save(folder/(name+'.npy'),array)
            write_json(folder/'spam-complete.json',dict(imgsize=list(imgsize)))
        else:
            spam,valid,r0,r1=[np.load(folder/(name+'.npy'),mmap_mode='r',allow_pickle=False)
                             for name in ['spam','valid','range0','range1']]
            imgsize=tuple(json.loads((folder/'spam-complete.json').read_text())['imgsize'])
        if np.sum(valid)<50:
            raise ValueError('Too few valid blocks for a splicing map. The noise estimate remains available.')
        progress(90,'Fitting the original statistical model')
        mapp,other=EMgu_img(spam,valid,extFeat=range(32),seed=0,maxIter=100,replicates=10,outliersNlogl=42,
            workers=4 if backend=='mps' else 1,
            progress=lambda done,total:progress(90+done*9//total,f'Statistical fit {done}/{total}'))
        if not np.isfinite(mapp).all():
            raise ValueError('The statistical model returned non-finite values; no heatmap exported.')
        save(folder/'map.npy',mapp)
        render=cv.applyColorMap(genMappUint8(mapp,valid,r0,r1,imgsize),cv.COLORMAP_JET)
        save(folder/'map-display.npy',render)
    write_json(folder/'result.json',dict(info,stage=stage))
    progress(100,'Noiseprint analysis ready')


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        import traceback
        (Path(sys.argv[1])/'worker-error.txt').write_text(traceback.format_exc())
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        sys.exit(1)
