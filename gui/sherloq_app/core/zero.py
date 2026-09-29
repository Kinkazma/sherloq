"""ZERO AGPL-3.0+ CPU adapter. Run in an isolated process: upstream can exit()."""
import ctypes as ct
import json,sys,time
from pathlib import Path
if __name__=='__main__':sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import cv2 as cv
import numpy as np
from gui.sherloq_app.core.noisesniffer import save_array

class Region(ct.Structure):
    _fields_=[('x0',ct.c_int),('y0',ct.c_int),('x1',ct.c_int),('y1',ct.c_int),('grid',ct.c_int),('lnfa',ct.c_double)]

def ptr(a,t):return a.ctypes.data_as(ct.POINTER(t))

def analyze(image,jpeg=None,reference=False):
    h,w=image.shape[:2]
    if min(h,w)<16:raise ValueError('ZERO needs an image at least 16 × 16 pixels.')
    if h*w>100_000_000:raise ValueError('ZERO working set exceeds the 100 MP limit.')
    if jpeg is not None and jpeg.shape!=image.shape:raise ValueError('JPEG99 dimensions differ from the image.')
    # C expects RGB planes, not interleaved BGR. No resize or DCT replacement.
    source=np.ascontiguousarray(image[:,:,::-1].transpose(2,0,1),dtype=np.float64)
    companion=np.ascontiguousarray(jpeg[:,:,::-1].transpose(2,0,1),dtype=np.float64) if jpeg is not None else None
    arrays={n:np.zeros((h,w),np.float64 if n.startswith('luminance') else np.int32) for n in ('luminance','luminance_jpeg','votes','votes_jpeg','mask_f','mask_f_reg','mask_m','mask_m_reg')}
    arrays['votes_jpeg'].fill(-1);lnfa=np.zeros(64,np.float64)
    # Each disjoint detected region uses min_size > 64 pixels for X,Y >=16.
    # This conservative bound avoids two 32-byte structures for every pixel.
    capacity=h*w//64+1
    foreign=(Region*capacity)();missing=(Region*capacity)();nf=ct.c_int();nm=ct.c_int()
    lib=ct.CDLL(str(Path(__file__).resolve().parents[4]/'native/runtime/libsherloq_zero.dylib'));fn=lib.sherloq_reference_zero if reference else lib.zero
    dp=ct.POINTER(ct.c_double);ip=ct.POINTER(ct.c_int);rp=ct.POINTER(Region)
    fn.argtypes=[dp,dp,dp,dp,ip,ip,dp,rp,ip,rp,ip,ip,ip,ip,ip,ct.c_int,ct.c_int,ct.c_int,ct.c_int];fn.restype=ct.c_int
    main=fn(ptr(source,ct.c_double),ptr(companion,ct.c_double) if companion is not None else None,ptr(arrays['luminance'],ct.c_double),ptr(arrays['luminance_jpeg'],ct.c_double),ptr(arrays['votes'],ct.c_int),ptr(arrays['votes_jpeg'],ct.c_int),ptr(lnfa,ct.c_double),foreign,ct.byref(nf),missing,ct.byref(nm),*[ptr(arrays[n],ct.c_int) for n in ('mask_f','mask_f_reg','mask_m','mask_m_reg')],w,h,3,3 if jpeg is not None else 0)
    def describe(regs,n):return [dict(x0=r.x0,y0=r.y0,x1=r.x1,y1=r.y1,grid=r.grid,log10_nfa=r.lnfa) for r in regs[:n]]
    arrays['grid_log10_nfa']=lnfa;arrays['metadata']=dict(method='ZERO IPOL 2021/390',main_grid=main,foreign_regions=describe(foreign,nf.value),missing_regions=describe(missing,nm.value),jpeg99_used=jpeg is not None,missing_grid_analyzed=main>=0 and jpeg is not None,reference=reference)
    return arrays


def jpeg99(image):
    # Explicit JPEG99 companion; fixed 4:4:4 sampling avoids hidden defaults.
    ok,encoded=cv.imencode('.jpg',image,[cv.IMWRITE_JPEG_QUALITY,99,cv.IMWRITE_JPEG_SAMPLING_FACTOR,cv.IMWRITE_JPEG_SAMPLING_FACTOR_444])
    if not ok:raise ValueError('Cannot encode the JPEG99 reference.')
    decoded=cv.imdecode(encoded,cv.IMREAD_COLOR)
    if decoded is None:raise ValueError('Cannot decode the JPEG99 reference.')
    return decoded,encoded


def main():
    import hashlib
    folder=Path(sys.argv[1]);settings=json.loads(sys.argv[2]);image=np.load(folder/'image.npy',mmap_mode='r',allow_pickle=False);start=time.perf_counter();companion=None
    if settings[0]:
        companion,encoded=jpeg99(image);(folder/'reference99.jpg').write_bytes(encoded.tobytes())
    result=analyze(image,companion,settings[1]);meta=result['metadata'];meta['seconds']=time.perf_counter()-start;meta['decoded_bgr8_sha256']=hashlib.sha256(memoryview(np.ascontiguousarray(image))).hexdigest();meta['jpeg99_encoder']=f'OpenCV {cv.__version__} 4:4:4 quality99' if companion is not None else None
    if companion is not None:meta['jpeg99_sha256']=hashlib.sha256(encoded).hexdigest()
    for key,value in result.items():
        if isinstance(value,np.ndarray):save_array(folder/f'{key}.npy',value)
    p=folder/'result.partial.json';p.write_text(json.dumps(meta));p.replace(folder/'result.json');print(json.dumps(meta))
if __name__=='__main__':main()
