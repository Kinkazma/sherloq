"""IPOL Noisesniffer: bounded DCT batches, corrected flat-block test, pure output.

CPU reference arithmetic and sorting are preserved. See vendor/noisesniffer
for the original Apache-2.0 sources and reports/X03-noisesniffer.md for fixes.
"""
from functools import lru_cache
import json,math
from pathlib import Path
if __name__ == "__main__":
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import cv2 as cv
import numpy as np
from scipy.stats import binom
from scipy.special import gammaln
from numpy.lib.stride_tricks import sliding_window_view
from gui.sherloq_app.vendor.noisesniffer import functions as ref


def save_array(path,array):
    temporary=path.with_suffix('.partial.npy');np.save(temporary,array,allow_pickle=False);temporary.replace(path)


def statistics(image,w,folder=None,progress=lambda *x:None):
    """Per-window statistics, with identical per-block DCT/reduction order."""
    if w not in (3,5,7,8) or min(image.shape[:2])<w:raise ValueError('Block size must be 3, 5, 7 or 8 and fit the image.')
    folder=Path(folder) if folder else None
    names=('valid','means','variance')
    if folder and all((folder/f'{n}-{w}.npy').exists() for n in names):
        return tuple(np.load(folder/f'{n}-{w}.npy',mmap_mode='r',allow_pickle=False) for n in names)
    rgb=np.ascontiguousarray(image[:,:,::-1],dtype=np.float64)
    valid=ref.compute_valid_blocks_indices(rgb,w);means=ref.all_image_means(rgb,w)
    rows,cols=image.shape[0]-w+1,image.shape[1]-w+1
    variance=np.empty((3,rows*cols),np.float32)
    # Batch rows bound both the input block copy and DCT temporaries.
    batch=max(1,32768//cols)
    for ch in range(3):
        windows=sliding_window_view(rgb[:,:,ch],(w,w))
        for y in range(0,rows,batch):
            end=min(rows,y+batch);blocks=windows[y:end].reshape(-1,w,w)
            variance[ch,y*cols:end*cols]=ref.compute_low_freq_var(blocks,w)
            progress(int(60*(ch+(end/rows))/3),'DCT statistics')
    result=(valid,means,variance)
    if folder:
        for n,array in zip(names,result):save_array(folder/f'{n}-{w}.npy',array)
    return result


def select(image,w,b,n,m,stats):
    valid,means,variance=stats
    if not len(valid):return np.empty(0,np.int64),np.empty(0,np.int64)
    b=ref.update_samples_per_bin(b,len(valid));bins=int(round(len(valid)/b));V=[];S=[]
    rgb=np.ascontiguousarray(image[:,:,::-1],dtype=np.float64);cols=image.shape[1]-w+1
    for ch in range(3):
        windows=sliding_window_view(rgb[:,:,ch],(w,w));ordered=ref.sort_blocks_means(ch,means,valid)
        for i in range(bins):
            ids=ref.bin_block_list(bins,i,ordered,b);chosen=ref.select_blocks_VL(b,n,variance[ch],ids)
            blocks=windows[chosen//cols,chosen%cols]
            std=np.std(blocks,axis=(1,2));indices=chosen[np.argsort(std)]
            # Original bin_is_valid received block indices instead of std values.
            if np.count_nonzero(std==0)<int(b*n*m):
                V.append(indices);S.append(indices[:int(b*n*m)])
    return (np.concatenate(V) if V else np.empty(0,np.int64),np.concatenate(S) if S else np.empty(0,np.int64))


def counts(shape,w,W,V,S):
    h,width=shape[:2];gh,gw=h//W+1,width//W+1;cols=width-w+1
    def histogram(ids):
        cells=(ids//cols//W)*gw+(ids%cols//W)
        return np.bincount(cells,minlength=gh*gw).reshape(gh,gw).astype(np.float64)
    return histogram(V),histogram(S)


def log_tail(K,N,w,m):
    """Binomial survival in log space; avoids 1-CDF cancellation/underflow."""
    k=int(np.floor(K/w**2));n=int(np.ceil(N/w**2))
    if k<=0:return 0.
    if k>n:return -math.inf
    result=float(binom.logsf(k-1,n,m))
    if math.isfinite(result):return result
    # SciPy may underflow in an extreme upper tail. Sum the PMF ratios
    # relative to the first term, where all terms decrease monotonically.
    first=gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1)+k*math.log(m)+(n-k)*math.log1p(-m)
    term=total=1.
    for j in range(k,n):
        term*=((n-j)/(j+1))*m/(1-m);total+=term
        if term<total*1e-16:break
    return float(first+math.log(total))


def regions(shape,w,W,m,all_blocks,red_blocks):
    h,width=shape[:2];gh,gw=all_blocks.shape;mask=np.zeros((gh,gw),np.uint8);found=[]
    tail=lru_cache(maxsize=32768)(lambda K,N:log_tail(K,N,w,m))
    def neighbours(y,x):
        # Preserve original visitation order, but reject invalid border cells.
        for a,b in ((y+1,x),(y-1,x),(y,x+1),(y,x-1)):
            if 0<=a<gh and 0<=b<gw:yield a,b
    for y in range(gh):
        for x in range(gw):
            if not ref.seed_crit_satisfied(y,x,all_blocks,red_blocks,m,mask):continue
            cells=[(y,x)];seen=set(cells);N=all_blocks[y,x];K=red_blocks[y,x];before=0
            while before!=len(cells):
                before=len(cells)
                for a,b in cells:
                    for q in neighbours(a,b):
                        if q in seen:continue
                        nb=all_blocks[q];kb=red_blocks[q];R=len(cells)
                        if tail(K,N)-math.log(R)>math.log(4)+tail(K+kb,N+nb)-math.log(R+1):
                            cells.append(q);seen.add(q);N+=nb;K+=kb
            R=len(cells);log_nfa=math.log(.5*w*w)+2*math.log(h*width/W**2)+math.log(.316915/R)+R*math.log(4.062570)+tail(K,N)
            if log_nfa<0:
                for q in cells:mask[q]=255
                found.append(dict(cells=cells,log10_nfa=log_nfa/math.log(10),selected_blocks=int(K),all_blocks=int(N)))
    full=np.repeat(np.repeat(mask,W,axis=0),W,axis=1)[:h,:width]
    return full,found


def distribution(image,w,V,S):
    output=image.copy();h,width=image.shape[:2];cols=width-w+1
    # Difference-array rectangle union is equivalent to repeated solid fills.
    for ids,color in ((V,(255,255,255)),(S,(0,0,255))):
        delta=np.zeros((h+1,width+1),np.int32);y=ids//cols;x=ids%cols
        np.add.at(delta,(y,x),1);np.add.at(delta,(y+w,x),-1);np.add.at(delta,(y,x+w),-1);np.add.at(delta,(y+w,x+w),1)
        covered=delta.cumsum(axis=0,dtype=np.int32).cumsum(axis=1,dtype=np.int32)[:h,:width]>0
        output[covered]=color
    return output


def analyze(image,params=(3,100,20000,.1,.5),folder=None,progress=lambda *x:None):
    w,W,b,n,m=params
    if W<1 or b<1 or not 0<n<=1 or not 0<m<1:raise ValueError('Invalid Noisesniffer settings.')
    stats=statistics(image,w,folder,progress);V,S=select(image,w,b,n,m,stats);progress(65,'Regional noise analysis')
    total,low=counts(image.shape,w,W,V,S);mask,found=regions(image.shape,w,W,m,total,low)
    progress(95,'Rendering distributions');paint=distribution(image,w,V,S)
    return dict(mask=mask,distribution=paint,all_blocks=total,low_noise_blocks=low,selected=V,low_noise=S,
                metadata=dict(method='IPOL Noisesniffer',parameters=params,valid_blocks=len(stats[0]),selected_blocks=len(V),low_noise_blocks=len(S),regions=found,inconclusive=not len(V)))


def main():
    import sys,time
    folder=Path(sys.argv[1]);params=tuple(json.loads(sys.argv[2]));start=time.perf_counter()
    image=np.load(folder/'image.npy',mmap_mode='r',allow_pickle=False)
    result=analyze(image,params,folder)
    for key,value in result.items():
        if isinstance(value,np.ndarray):save_array(folder/f'{key}.npy',value)
    import hashlib
    meta=result['metadata'];meta['decoded_bgr8_sha256']=hashlib.sha256(memoryview(np.ascontiguousarray(image))).hexdigest();meta['seconds']=time.perf_counter()-start
    p=folder/'result.partial.json';p.write_text(json.dumps(meta));p.replace(folder/'result.json')
    print(json.dumps(meta))

if __name__=='__main__':main()
