"""Ordered Metal dense descriptors, CPU preparation/normalization reference.

No half precision, changed coefficients, reduction tree or pixel resampling.
The CPU remains available via the existing CPU checkbox.
"""
import ctypes as ct
import sys
import os
import mmap
from functools import lru_cache
import numpy as np
from .dense_copy import library,ptr

@lru_cache(1)
def available():
    if sys.platform!='darwin':return False
    try:return bool(library().sherloq_metal_available())
    except (OSError,AttributeError):return False

def sift_enabled():
    # Shared output / ordered parallel normalization validated on M1 Max.
    # Full-board ABCCBA and ABBA runs preserve every output array.
    # CPU checkbox and explicit diagnostic override retain the reference path.
    return os.environ.get('SHERLOQ_SIFT_METAL','1')=='1' and available()

@lru_cache(16)
def coefficients(patch):
    filters=np.empty((24,2*patch,2*patch),np.float32);weights=np.empty(12,np.float32)
    fn=library().sherloq_zernike_filters;fn.argtypes=[ct.c_int,ct.POINTER(ct.c_float),ct.POINTER(ct.c_float)]
    if fn(patch,ptr(filters),ptr(weights)):raise ValueError('Invalid Zernike patch radius.')
    return filters,weights

def features(image,patch=8,flip=False):
    h,w=image.shape[:2]
    if min(h,w)<=3*patch or not 2<=patch<=32:raise ValueError('Invalid dense descriptor dimensions/settings.')
    if h*w*12*4*(4 if flip else 3)+h*w*32>16*1024**3:raise ValueError('Dense descriptor working set exceeds 16 GiB.')
    filters,weights=coefficients(patch)
    gray=image.astype(np.float32).sum(axis=2)*np.float32(1/np.sqrt(np.float32(3)))
    fn=library().sherloq_metal_zernike;floatp=ct.POINTER(ct.c_float)
    fn.argtypes=[floatp,floatp,floatp,ct.c_int,ct.c_int,ct.c_int,ct.c_int,floatp,ct.c_char_p];fn.restype=ct.c_int
    result=[]
    for mirror in range(2 if flip else 1):
        source=np.ascontiguousarray(gray[:,::-1]) if mirror else gray
        out=np.empty((h,w,12),np.float32);error=ct.create_string_buffer(1024)
        if fn(ptr(source),ptr(filters),ptr(weights),w,h,patch,0,ptr(out),error):raise RuntimeError(error.value.decode(errors='replace'))
        if mirror:out=np.ascontiguousarray(out[:,::-1])
        out/=np.maximum(np.linalg.norm(out,axis=2,keepdims=True),1e-12);result.append(out)
    return result[0],result[1] if flip else result[0],0.


def features_sift(image,patch=8,flip=False):
    h,w=image.shape[:2]
    if min(h,w)<=3*patch or not 3<=patch<=32:raise ValueError('Invalid dense SIFT dimensions/settings (patch >= 3).')
    shape=(h-3*patch,w-3*patch,128)
    if np.prod(shape)*4*(4 if flip else 3)+h*w*128>16*1024**3:raise ValueError('Dense descriptor working set exceeds 16 GiB. Select smaller regions or use Zernike.')
    gray=image.astype(np.float32).sum(axis=2)*np.float32(1/np.sqrt(np.float32(3)))
    lib=library();fp=ct.POINTER(ct.c_float)
    prepare=lib.sherloq_sift_inputs;prepare.argtypes=[fp,ct.c_int,ct.c_int,ct.c_int,ct.c_int,fp,fp];prepare.restype=ct.c_int
    compute=lib.sherloq_metal_sift_shared;compute.argtypes=[fp,fp,ct.c_int,ct.c_int,ct.c_int,fp,ct.c_char_p,ct.c_size_t];compute.restype=ct.c_int
    normalize=lib.sherloq_sift_normalize;normalize.argtypes=[fp,ct.c_int];normalize.restype=None
    result=[]
    for mirror in range(2 if flip else 1):
        gradients=np.empty((8,h,w),np.float32);weights=np.empty(4,np.float32)
        if prepare(ptr(gray),w,h,patch,mirror,ptr(gradients),ptr(weights)):raise RuntimeError('SIFT preparation failed.')
        size=int(np.prod(shape))*4;size=((size+mmap.PAGESIZE-1)//mmap.PAGESIZE)*mmap.PAGESIZE
        shared=mmap.mmap(-1,size);out=np.ndarray(shape,np.float32,buffer=shared);error=ct.create_string_buffer(1024)
        if compute(ptr(gradients),ptr(weights),w,h,patch,ptr(out),error,size):raise RuntimeError(error.value.decode(errors='replace'))
        del gradients
        normalize(ptr(out),shape[0]*shape[1])
        if mirror:out=np.ascontiguousarray(out[:,::-1])
        out/=np.maximum(np.linalg.norm(out,axis=2,keepdims=True),1e-12);result.append(out)
    return result[0],result[1] if flip else result[0],1.5*patch
