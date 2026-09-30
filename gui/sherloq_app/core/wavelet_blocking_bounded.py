"""Separable db8 transforms along complete axes, preserving boundary phase."""
import cv2 as cv
import numpy as np
import pywt
from .memory_resources import TemporaryArrays,MiB,tiles
from .bounded_ops import map_local,normalize_u8
from .bounded_wavelets import axis_transform


def detail(engine,store):
    cached=engine.details.get('db8')
    if cached is not None:store.watch(cached);return cached
    gray=engine.gray.get('gray')
    if gray is None:
        gray=cv.imread(engine.filename,cv.IMREAD_GRAYSCALE)
        if gray is None:
            gray=store.array(engine.image.shape[:2],np.uint8)
            map_local(engine.image,lambda a:cv.cvtColor(a,cv.COLOR_BGR2GRAY),(gray,),store=store)
            engine.source_mode='loaded image grayscale'
        engine.gray.put('gray',gray)
    store.watch(gray)
    wavelet=pywt.Wavelet('db8')
    high=axis_transform(gray,wavelet,0,store,high_only=True)
    result=axis_transform(high,wavelet,1,store,high_only=True)
    return engine.details.put('db8',result)


def compute(engine,blocksize):
    store=TemporaryArrays(32*MiB);coeff=detail(engine,store)
    if blocksize<1 or blocksize>min(coeff.shape):
        raise ValueError(f'Block size must be between 1 and {min(coeff.shape)}')
    noise=engine.maps.get(blocksize)
    if noise is None:
        nr,nc=(size//blocksize for size in coeff.shape)
        noise=store.array((nr,nc),np.float64)
        # Batch whole median blocks, not wavelet inputs; retain original grid.
        cells=max(1,min(32,int((MiB/max(8,blocksize*blocksize*8))**.5)))
        for dst,_,_ in tiles(noise.shape,(cells,cells)):
            y,x=dst;roi=coeff[y.start*blocksize:y.stop*blocksize,x.start*blocksize:x.stop*blocksize]
            blocks=roi.reshape(y.stop-y.start,blocksize,x.stop-x.start,blocksize).transpose(0,2,1,3)
            absolute=np.array(blocks,order='C',copy=True).reshape(y.stop-y.start,x.stop-x.start,blocksize*blocksize)
            np.abs(absolute,out=absolute)
            noise[dst]=np.median(absolute,axis=2,overwrite_input=True)/.6745;store.checkpoint()
        engine.maps.put(blocksize,noise)
    else:store.watch(noise)
    display=engine.displays.get(blocksize)
    if display is None:
        normalized=normalize_u8(noise,store,direct=True)
        h,w=engine.image.shape[:2];resized=store.array((h,w),np.uint8)
        cv.resize(normalized,(w,h),dst=resized,interpolation=cv.INTER_NEAREST)
        display=store.array(engine.image.shape,np.uint8)
        map_local(resized,lambda a:cv.cvtColor(a,cv.COLOR_GRAY2BGR),(display,),store=store)
        engine.displays.put(blocksize,display)
    store.checkpoint(force=True)
    return display,noise
