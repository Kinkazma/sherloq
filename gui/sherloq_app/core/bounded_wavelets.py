"""Global separable wavelets; batches never cut a transformed axis."""
import cv2 as cv
import numpy as np
import pywt
from .memory_resources import TemporaryArrays,MiB,tiles
from .bounded_ops import map_local,extrema
from .wavelet_threshold import threshold_band


def oriented(array,axis,store):
    if (array.flags.f_contiguous if axis==0 else array.flags.c_contiguous):return array
    out=store.array(array.shape[::-1],array.dtype).T if axis==0 else store.array(array.shape,array.dtype)
    map_local(array,lambda a:a,(out,),store=store)
    return out


def axis_transform(low,wavelet,axis,store,high=None,high_only=False):
    inverse=high is not None;shape=list(low.shape)
    low=oriented(low,axis,store)
    if inverse:high=oriented(high,axis,store)
    shape[axis]=2*shape[axis]-wavelet.rec_len+2 if inverse else pywt.dwt_coeff_len(shape[axis],wavelet.dec_len,'symmetric')
    def allocate():return store.array(shape[::-1],np.float64).T if axis==0 else store.array(shape,np.float64)
    outputs=tuple(allocate() for _ in range(1 if inverse or high_only else 2))
    other=1-axis;batch=max(1,min(256,8*MiB//max(1,low.shape[axis]*32)))
    store.watch(low,high)
    for first in range(0,low.shape[other],batch):
        index=[slice(None),slice(None)];index[other]=slice(first,first+batch);index=tuple(index)
        if inverse:values=(pywt.idwt(low[index],high[index],wavelet,axis=axis),)
        else:
            values=pywt.dwt(low[index],wavelet,axis=axis)
            if high_only:values=(values[1],)
        for out,value in zip(outputs,values):out[index]=value
        del values;store.checkpoint()
    return outputs[0] if inverse or high_only else outputs


def decompose(image,wavelet,store):
    current=image;details=[]
    level=pywt.dwt_max_level(min(image.shape),wavelet.dec_len)
    for _ in range(level):
        low,high=axis_transform(current,wavelet,0,store)
        current,vertical=axis_transform(low,wavelet,1,store)
        horizontal,diagonal=axis_transform(high,wavelet,1,store)
        details.append((horizontal,vertical,diagonal));del low,high
    return [current,*reversed(details)]


def reconstruct(coefficients,wavelet,store,threshold=0,level=0,mode='soft',maxima=None):
    current=coefficients[0];count=len(coefficients)-1
    for index,details in enumerate(coefficients[1:]):
        # Same odd-size truncation as waverec2 before every idwt2 call.
        shape=details[0].shape
        crop=tuple(slice(None,-1 if a==d+1 else None) for a,d in zip(current.shape,shape))
        current=current[crop]
        if threshold and count-index<=level:
            changed=[]
            for channel,band in enumerate(details):
                if maxima is None:
                    lo,hi=extrema(band,store=store);maximum=max(abs(lo),abs(hi))
                else:maximum=maxima[index][channel]
                if maximum:
                    out=store.array(band.shape,np.float64)
                    map_local(band,lambda a:threshold_band(a,threshold/100*maximum,mode),(out,),store=store)
                    changed.append(out)
                else:changed.append(band)
            details=changed
        horizontal,vertical,diagonal=details
        low=axis_transform(current,wavelet,1,store,vertical)
        high=axis_transform(horizontal,wavelet,1,store,diagonal)
        current=axis_transform(low,wavelet,0,store,high)
        del low,high
    return current


def compute(engine,params):
    name,threshold,level,mode=params
    if threshold==0 or level==0:threshold,level,mode=0,0,'soft'
    key=name,threshold,level,mode;cached=engine.results.get(key)
    if cached is not None:return cached
    wavelet=pywt.Wavelet(name);store=TemporaryArrays(32*MiB)
    transform=engine.transforms.get(name)
    if transform is None:
        coeffs=decompose(engine.image[:,:,0],wavelet,store)
        maxima=[tuple(max(abs(x) for x in extrema(band,store=store)) for band in octave) for octave in coeffs[1:]]
        engine.transforms.put(name,(coeffs,maxima))
    else:coeffs,maxima=transform;store.watch(coeffs)
    level=min(level,len(coeffs)-1)
    if threshold and level:
        prefix=engine.prefixes.get((name,level))
        if prefix is None:
            prefix=reconstruct(coeffs[:-level],wavelet,store)
            engine.prefixes.put((name,level),prefix)
        else:store.watch(prefix)
        coeffs=[prefix,*coeffs[-level:]];maxima=maxima[-level:]
    values=reconstruct(coeffs,wavelet,store,threshold,level,mode,maxima)
    h,w=engine.image.shape[:2];result=store.array(engine.image.shape,np.uint8)
    map_local(values[:h,:w],lambda a:cv.cvtColor(a.astype(np.uint8),cv.COLOR_GRAY2BGR),(result,),store=store)
    store.checkpoint(force=True)
    return engine.results.put(key,result)
