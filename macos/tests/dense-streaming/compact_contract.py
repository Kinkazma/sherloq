"""Candidate only: numerical reconstruction and unchanged global NNF."""
from pathlib import Path
import ctypes as ct,sys,time,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source'))
from gui.sherloq_app.core import dense_copy as dc,metal_dense as md,sift_frames as sf
fp=ct.POINTER(ct.c_float);bp=ct.POINTER(ct.c_ubyte);ip=ct.POINTER(ct.c_int)
class Compact(ct.Structure):
 _fields_=[('hist',fp),('norms',fp),('turns',bp),('weights',ct.c_float*4)]+[(n,ct.c_int) for n in ('width','height','patch','offset','view_width','view_height','mirror','quarter')]
lib=ct.CDLL(str(Path(__file__).with_name('candidate.dylib')))
lib.sherloq_sift_inputs.argtypes=[fp,ct.c_int,ct.c_int,ct.c_int,ct.c_int,fp,fp]
lib.vl_imconvcoltri_f.argtypes=[fp,ct.c_size_t,fp,*([ct.c_size_t]*5),ct.c_uint]
lib.sherloq_sift_pack_range.argtypes=[fp,fp,ct.c_int,ct.c_int,ct.c_int,ct.c_size_t,ct.c_int,fp]
lib.sherloq_sift_norm_factors.argtypes=[fp,ct.c_int,fp]
lib.sherloq_sift_unpack.argtypes=[ct.POINTER(Compact),ct.c_int,ct.c_int,fp]
lib.sherloq_patchmatch_compact.argtypes=[ct.POINTER(Compact),ct.POINTER(Compact),bp,ct.c_int,ct.c_int,ct.c_int,ct.c_float,ct.c_float,ct.c_int,ct.c_uint32,ip,fp,ct.POINTER(ct.c_uint64),dc.CALLBACK,ct.c_char_p,ct.c_float,ct.c_float,fp,fp,ct.c_size_t]
p=dc.ptr

def prepare(im,patch,mirror=False,quarter=False,support=None):
 h,w=im.shape[:2];dh,dw=h-3*patch,w-3*patch
 gray=im.astype(np.float32).sum(2)*np.float32(1/np.sqrt(np.float32(3)))
 g=np.empty((8,h,w),np.float32);weights=np.empty(4,np.float32)
 assert lib.sherloq_sift_inputs(p(gray),w,h,patch,mirror,p(g),p(weights))==0
 tmp=np.empty((w,h),np.float32)
 for plane in g:
  lib.vl_imconvcoltri_f(p(tmp),h,p(plane),w,h,w,patch,1,5)
  lib.vl_imconvcoltri_f(p(plane),w,p(tmp),h,w,h,patch,1,5)
 hist=np.ascontiguousarray(g.transpose(1,2,0));norms=np.empty((dh*dw,3),np.float32)
 raw=np.empty((dh*dw,128),np.float32)
 lib.sherloq_sift_pack_range(p(hist),p(weights),w,h,patch,0,len(raw),p(raw))
 lib.sherloq_sift_norm_factors(p(raw),len(raw),p(norms))
 norms[:,2]=np.maximum(np.linalg.norm(raw,axis=1),1e-12);raw/=norms[:,2,None]
 turns=raw.reshape(-1,4,4,8).sum((1,2)).reshape(-1,4,2).sum(2).argmax(1).astype(np.uint8)
 support=support or patch;off=3*(support-patch)//2
 f=Compact(p(hist),p(norms),p(turns,ct.c_ubyte),(ct.c_float*4)(*weights),w,h,patch,off,w-3*support,h-3*support,mirror,quarter)
 f.owners=(hist,norms,turns)
 return f

def unpack(f):
 a=np.empty((f.view_height,f.view_width,128),np.float32);lib.sherloq_sift_unpack(ct.byref(f),0,a.shape[0]*a.shape[1],p(a));return a

def field(a,b,mask,cache=4096,compare=False):
 h,w=mask.shape;matches=np.empty((h,w),np.int32);d=np.empty((h,w),np.float32);n=ct.c_uint64();err=ct.create_string_buffer(1024);cb=dc.CALLBACK(lambda:0)
 t=time.perf_counter();code=lib.sherloq_patchmatch_compact(ct.byref(a),ct.byref(b),p(mask,ct.c_ubyte),w,h,compare,5,1e5,3,729,p(matches,ct.c_int),p(d),ct.byref(n),cb,err,0,0,None,None,cache)
 assert code==0,err.value
 return (matches,d,n.value),time.perf_counter()-t

def main():
 rng=np.random.default_rng(45);rows=[]
 for shape in ((83,97),(207,251)):
  im=rng.integers(0,256,(*shape,3),dtype=np.uint8)
  for patch in (6,8,12):
   for flip in (False,True):
    for quarter in (False,True):
     support=max(8,patch);off=3*(support-patch)//2;dh,dw=shape[0]-3*support,shape[1]-3*support
     f=prepare(im,patch,flip,quarter,support);actual=unpack(f)
     ref=md.features_sift(im,patch,False,mirror_only=flip)[0]
     if quarter:ref=sf.quarter_turn_frame(ref)
     ref=np.ascontiguousarray(ref[off:off+dh,off:off+dw]);delta=float(np.max(abs(actual-ref)))
     assert np.array_equal(actual,ref),(shape,patch,flip,quarter,delta)
     rows.append(dict(shape=shape,patch=patch,flip=flip,quarter=quarter,max_error=delta))
  a=prepare(im,8);b=prepare(im,8,True);aa=unpack(a);bb=unpack(b);mask=np.ones(aa.shape[:2],np.uint8)
  for compare in (False,True):
   if compare:mask[:,mask.shape[1]//2:]=2
   t=time.perf_counter();ref=dc.field(aa,bb,mask,1e5,5,iterations=3,compare=compare);direct=time.perf_counter()-t
   for cache in (1,4096):
    result,elapsed=field(a,b,mask,cache,compare)
    assert all(np.array_equal(x,y) for x,y in zip(result,ref))
    rows.append(dict(shape=shape,compare=compare,cache=cache,field_exact=True,direct_s=direct,compact_s=elapsed))
 print(json.dumps(rows,indent=2),flush=True);Path(__file__).with_name('compact-contract-results.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
