"""Dense adapters: compact descriptors, complete-axis stripes and global fields.

The RAM array path remains in dense_copy / metal_dense. Native address spaces
and mmap are backend details, not APIs that can be copied into a browser.
"""
import ctypes as ct
import math,gc
from functools import lru_cache
from pathlib import Path
import numpy as np
import cv2 as cv
from .dense_copy import library,ptr as p,CALLBACK,integer_sum_filter
from .memory_resources import TemporaryArrays,tiles
from .cloning import check
from .jpeg_curve import Cancelled
from . import sift_frames as sf
fp=ct.POINTER(ct.c_float)

class Compact(ct.Structure):
    _fields_=[('hist',fp),('norms',fp),('turns',ct.POINTER(ct.c_ubyte)),('weights',ct.c_float*4)]+[(n,ct.c_int) for n in ('width','height','patch','offset','view_width','view_height','mirror','quarter')]
    @property
    def shape(self):return (self.view_height,self.view_width,128)
    def diversity(self):
        full=self.memory_arrays[3].reshape(self.height-3*self.patch,self.width-3*self.patch)
        if self.mirror:full=full[:,::-1]
        return full[self.offset:self.offset+self.view_height,self.offset:self.offset+self.view_width].view(bool)

@lru_cache(1)
def native():
    lib=library()
    if not hasattr(lib,'sherloq_patchmatch_compact'):
        # An already-open Qt application may still own the previous dylib.
        path=Path(__file__).resolve().parents[4]/'native/runtime/libsherloq_dense_stream.dylib'
        lib=ct.CDLL(str(path))
    lib.sherloq_sift_inputs.argtypes=[fp,ct.c_int,ct.c_int,ct.c_int,ct.c_int,fp,fp]
    lib.vl_imconvcoltri_f.argtypes=[fp,ct.c_size_t,fp,*([ct.c_size_t]*5),ct.c_uint]
    lib.sherloq_sift_pack_range.argtypes=[fp,fp,ct.c_int,ct.c_int,ct.c_int,ct.c_size_t,ct.c_int,fp]
    lib.sherloq_sift_norm_factors.argtypes=[fp,ct.c_int,fp]
    lib.sherloq_metal_sift_columns.argtypes=[fp,ct.c_int,ct.c_int,ct.c_int,fp,ct.c_char_p]
    args=[ct.POINTER(Compact),ct.POINTER(Compact),ct.POINTER(ct.c_ubyte),ct.c_int,ct.c_int,ct.c_int,ct.c_float,ct.c_float,ct.c_int,ct.c_uint32,ct.POINTER(ct.c_int),fp,ct.POINTER(ct.c_uint64),CALLBACK,ct.c_char_p,ct.c_float,ct.c_float,fp,fp,ct.c_size_t]
    lib.sherloq_patchmatch_compact.argtypes=args
    lib.sherloq_patchmatch_bounded.argtypes=library().sherloq_patchmatch_metric.argtypes
    lib.sherloq_dense_area_filter.argtypes=[ct.POINTER(ct.c_ubyte),ct.c_int,ct.c_int,ct.c_int,CALLBACK,ct.c_char_p]
    return lib

def stripe(data,patch,gpu=True):
 lib=native();data=np.ascontiguousarray(data);rows,columns=data.shape;out=np.empty_like(data)
 if gpu:
  err=ct.create_string_buffer(1024)
  if lib.sherloq_metal_sift_columns(p(data),columns,rows,patch,p(out),err):raise RuntimeError(err.value.decode())
 else:lib.vl_imconvcoltri_f(p(out),columns,p(data),columns,rows,columns,patch,1,1)
 return out

def prepare_sift(im,patch,mirror=False,quarter=False,support=None,*,tile=512,stripe_bytes=16*1024**2,store=None,gpu=True,progress=lambda x:None,cancel=lambda:False):
 from gui.sherloq_app.core.cloning import check
 lib=native();store=store or TemporaryArrays();h,w=im.shape[:2];dh,dw=h-3*patch,w-3*patch
 if min(dh,dw)<=0:raise ValueError('Image too small')
 weights=np.empty(4,np.float32)
 halo=max(3*patch+1,math.ceil(3*float(np.float32(math.sqrt((patch*patch)//9-.25))))+1)
 # Prepare full-height stripes, then filter every complete column. Store each
 # stripe contiguously; a separate blocked transpose avoids disk-page thrash.
 gray_columns=store.array((w,h))
 for x in range(0,w,512):
  for y in range(0,h,512):
   check(cancel);pixels=im[y:y+512,x:x+512]
   gray_columns[x:x+512,y:y+512]=(pixels.astype(np.float32).sum(2)*np.float32(1/np.sqrt(np.float32(3)))).T
   store.checkpoint()
 progress('gray')
 vertical=store.array((w,h,8));cols=max(32,min(tile,stripe_bytes//(h*8*4)))
 for x in range(0,w,cols):
  check(cancel);x1=min(w,x+cols);lo=max(0,x-halo);hi=min(w,x1+halo)
  gray=np.ascontiguousarray(gray_columns[lo:hi].T)
  g=np.empty((8,h,hi-lo),np.float32)
  if lib.sherloq_sift_inputs(p(gray),hi-lo,h,patch,mirror,p(g),p(weights)):raise RuntimeError('SIFT stripe failed')
  crop=g[:,:,hi-x1:hi-x] if mirror else g[:,:,x-lo:x1-lo]
  data=np.ascontiguousarray(crop.transpose(1,2,0)).reshape(h,-1)
  out=stripe(data,patch,gpu).reshape(h,x1-x,8)
  if mirror:vertical[w-x1:w-x]=out.transpose(1,0,2)
  else:vertical[x:x1]=out.transpose(1,0,2)
  del gray,g,crop,data,out;store.checkpoint()
 del gray_columns;gc.collect();progress('vertical')
 hist=store.array((h,w,8))
 for y in range(0,h,256):
  for x in range(0,w,256):
   check(cancel);hist[y:y+256,x:x+256]=vertical[x:x+256,y:y+256].transpose(1,0,2)
  store.checkpoint()
 del vertical;gc.collect();progress('transposed')
 rows=max(1,stripe_bytes//(w*8*4))
 for y in range(0,h,rows):
  check(cancel);b=hist[y:y+rows];data=np.ascontiguousarray(b.transpose(1,0,2)).reshape(w,-1)
  out=stripe(data,patch,gpu).reshape(w,len(b),8)
  hist[y:y+rows]=out.transpose(1,0,2)
  del b,data,out;store.checkpoint()
 progress('horizontal')
 norms=store.array((dh*dw,3));turns=store.array((dh*dw,),np.uint8);diverse=store.array((dh*dw,),np.uint8)
 for start in range(0,dh*dw,8192):
  check(cancel);n=min(8192,dh*dw-start);raw=np.empty((n,128),np.float32)
  lib.sherloq_sift_pack_range(p(hist),p(weights),w,h,patch,start,n,p(raw))
  factors=norms[start:start+n];lib.sherloq_sift_norm_factors(p(raw),n,p(factors))
  factors[:,2]=np.maximum(np.linalg.norm(raw,axis=1),1e-12);raw/=factors[:,2,None]
  turns[start:start+n]=raw.reshape(-1,4,4,8).sum((1,2)).reshape(-1,4,2).sum(2).argmax(1)
  if quarter:raw=sf.quarter_turn_frame(raw.reshape(1,n,128)).reshape(n,128)
  diverse[start:start+n]=sf.orientation_diversity(raw.reshape(1,n,128)).ravel()
  if start%(8192*16)==0:store.checkpoint()
 store.checkpoint(force=True);support=support or patch;off=3*(support-patch)//2
 f=Compact(p(hist),p(norms),p(turns,ct.c_ubyte),(ct.c_float*4)(*weights),w,h,patch,off,w-3*support,h-3*support,mirror,quarter)
 f.memory_arrays=(hist,norms,turns,diverse);f.store=store
 return f


def prepare_zernike(image,patch,flip,store,cancel=lambda:False,gpu=True):
    from . import metal_dense,dense_copy
    shape=(*image.shape[:2],12);a=store.array(shape);b=store.array(shape) if flip else a
    for dst,src,crop in tiles(image.shape,(512,512),3*patch+1):
        check(cancel);pixels=np.ascontiguousarray(image[src])
        first,second,_=metal_dense.features(pixels,patch,flip) if gpu else dense_copy.features(pixels,0,patch,flip)
        a[dst]=first[crop]
        if flip:b[dst]=second[crop]
        store.checkpoint()
    return a,b,0.


def descriptors(image,method,patch,target_patch,flip,quarter,store,cancel,backend):
    if not method:return prepare_zernike(image,patch,flip,store,cancel,backend=='metal')
    support=max(patch,target_patch)
    a=prepare_sift(image,patch,False,quarter,support,store=store,gpu=backend=='metal',cancel=cancel)
    b=prepare_sift(image,target_patch,flip,quarter,support,store=store,gpu=backend=='metal',cancel=cancel) if flip or patch!=target_patch else a
    return a,b,1.5*support


def field(first,second,mask,radius,minimum,iterations,compare,seed,cancel,gap,axes,store):
    check(cancel);store.watch(first,second,mask);h,w,d=first.shape
    if second.shape!=first.shape or mask.shape!=(h,w):raise ValueError('Dense field shape mismatch')
    matches=store.array((h,w),np.int32);distances=store.array((h,w),np.float32)
    count=ct.c_uint64();error=ct.create_string_buffer(1024);failures=[]
    def checkpoint():
        try:
            if cancel():return 1
            store.checkpoint();return 0
        except Exception as exc:failures.append(exc);return 1
    callback=CALLBACK(checkpoint)
    if axes is not None:axes=tuple(np.ascontiguousarray(a,np.float32) for a in axes)
    tail=(int(compare),minimum,radius,iterations,seed,p(matches,ct.c_int),p(distances),ct.byref(count),callback,error,*gap,p(axes[0]) if axes is not None else None,p(axes[1]) if axes is not None else None)
    if isinstance(first,Compact):
        status=native().sherloq_patchmatch_compact(ct.byref(first),ct.byref(second),p(mask,ct.c_ubyte),w,h,*tail,16384)
    else:
        status=native().sherloq_patchmatch_bounded(p(first),p(second),p(mask,ct.c_ubyte),w,h,d,*tail)
    if failures:raise failures[0]
    if status==1:raise Cancelled()
    if status:raise ValueError(error.value.decode(errors='replace'))
    check(cancel);return matches,distances,count.value


def coherence(targets,squared,threshold,error_threshold,radius,minimum,store,cancel=lambda:False):
    h,w=targets.shape;selected=store.array((h,w),bool,zero=True);errors=store.array((h,w),np.float32,zero=True)
    yy,xx=np.mgrid[-radius:radius+1,-radius:radius+1]
    kernel=((xx*xx+yy*yy)<=radius*radius).astype(np.float64);count=kernel.sum();moment=(kernel*xx*xx).sum()
    rows=max(1,min(256,16*1024**2//max(1,w*8*8)))
    for start in range(0,h,rows):
        check(cancel);lo=max(0,start-radius);hi=min(h,start+rows+radius);t=targets[lo:hi]
        valid=(t>=0)&(squared[lo:hi]<=threshold*threshold)
        if not np.any(valid):continue
        complete=cv.filter2D(valid.astype(np.float64),-1,kernel,borderType=cv.BORDER_CONSTANT)>count-.5
        residual=np.zeros(t.shape,np.float64)
        for axis in (0,1):
            delta=((t%w-np.arange(w)[None,:]) if axis==0 else (t//w-np.arange(lo,hi)[:,None])).astype(np.float64)
            delta[~valid]=0;total=integer_sum_filter(delta,kernel)
            residual+=integer_sum_filter(delta*delta,kernel)-total*total/count
            for coords in (xx,yy):
                linear=integer_sum_filter(delta,kernel*coords);residual-=linear*linear/moment
        error=np.sqrt(np.maximum(residual,0)/count);crop=slice(start-lo,min(h,start+rows)-lo)
        selected[start:start+rows]=(complete&(error<=error_threshold))[crop];errors[start:start+rows]=error[crop];store.checkpoint()
    failures=[]
    def checkpoint():
        try:
            if cancel():return 1
            store.checkpoint();return 0
        except Exception as e:failures.append(e);return 1
    cb=CALLBACK(checkpoint);error=ct.create_string_buffer(1024)
    status=native().sherloq_dense_area_filter(p(selected,ct.c_ubyte),w,h,minimum,cb,error)
    if failures:raise failures[0]
    if status==1:raise Cancelled()
    if status:raise ValueError(error.value.decode(errors='replace'))
    return selected,errors


def allowed_mask(crop,dh,dw,shift,regions,excluded,texture,patch,store,cancel):
    allowed=store.array((dh,dw),np.uint8,zero=True)
    index=np.rint(shift+np.arange(dh)).astype(int);columns=np.rint(shift+np.arange(dw)).astype(int)
    if not regions:
        for y in range(0,dh,128):allowed[y:y+128]=1;store.checkpoint()
    for polygons,subtract in ((regions,False),(excluded,True)):
        for j,polygon in enumerate(polygons):
            check(cancel);mask=store.array(crop.shape[:2],np.uint8,zero=True)
            if len(polygon)>=3:cv.fillPoly(mask,[np.rint(polygon).astype(np.int32)],255)
            for y in range(0,dh,128):
                selected=mask[np.ix_(index[y:y+128],columns)]>0
                if subtract:allowed[y:y+128][selected]=0
                else:allowed[y:y+128]|=selected.astype(np.uint8)*(1<<j)
                store.checkpoint()
            del mask
    if texture>0:
        gray=store.array(crop.shape[:2],np.float32)
        for y in range(0,len(gray),128):
            check(cancel);gray[y:y+128]=crop[y:y+128].astype(np.float32).mean(axis=2);store.checkpoint()
        size=2*patch+1
        for y in range(0,dh,128):
            check(cancel);selected=index[y:y+128];lo=max(0,int(selected[0])-patch);hi=min(len(gray),int(selected[-1])+patch+1)
            block=np.ascontiguousarray(gray[lo:hi]);deviation=np.sqrt(np.maximum(cv.boxFilter(block*block,-1,(size,size))-cv.boxFilter(block,-1,(size,size))**2,0))
            allowed[y:y+128][deviation[np.ix_(selected-lo,columns)]<texture]=0;store.checkpoint()
    return allowed


def memory_plan(height,width,method,patch,flip,backend,target_patch=None,quarter=False,cap=None):
    from .dense_memory import workspace_bytes
    from .dense_parallel import BUDGET
    from .memory_resources import MEMORY,MiB
    cap=min(BUDGET.limit,MEMORY.capacity()) if cap is None else cap
    original=workspace_bytes(height,width,method,patch,flip,backend,target_patch,quarter)
    # Respect the old reference bridge's internal 16 GiB guard on its fast path.
    if original<=cap and (backend=='metal' and method==1 or original<=16*1024**3):return 'ram',original
    pixels=int(height)*int(width)
    # Compact descriptors + global outputs + bounded native scratch. No crop or
    # quality reduction. Keep these intermediates in RAM whenever they fit.
    paired=bool(flip or (target_patch is not None and target_patch!=patch))
    compact=pixels*((110 if paired else 64) if method else (111 if paired else 63))+256*MiB
    if compact<=cap:return 'compact',compact
    # Global polygons and rank/select pools still have a small per-pixel cost;
    # reserve it rather than advertising a fixed process-RSS guarantee.
    mapped=256*MiB+pixels*2+max(height,width)*2048
    if mapped>cap:raise MemoryError('The bounded dense workspace exceeds currently available memory.')
    return 'mapped',mapped
