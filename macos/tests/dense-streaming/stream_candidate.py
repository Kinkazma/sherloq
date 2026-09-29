"""Experimental bounded preparation; not installed in the application."""
import ctypes as ct, math, mmap, tempfile, os, shutil, weakref, gc
import numpy as np
from compact_contract import Compact,lib,p,fp
from gui.sherloq_app.core import sift_frames as sf
lib.sherloq_metal_sift_columns.argtypes=[fp,ct.c_int,ct.c_int,ct.c_int,fp,ct.c_char_p]

from gui.sherloq_app.core.memory_resources import TemporaryArrays as Store

def stripe(data,patch,gpu=True):
 data=np.ascontiguousarray(data);rows,columns=data.shape;out=np.empty_like(data)
 if gpu:
  err=ct.create_string_buffer(1024)
  if lib.sherloq_metal_sift_columns(p(data),columns,rows,patch,p(out),err):raise RuntimeError(err.value.decode())
 else:lib.vl_imconvcoltri_f(p(out),columns,p(data),columns,rows,columns,patch,1,1)
 return out

def prepare(im,patch,mirror=False,quarter=False,support=None,*,tile=512,stripe_bytes=16*1024**2,store=None,gpu=True,progress=lambda x:None,cancel=lambda:False):
 from gui.sherloq_app.core.cloning import check
 store=store or Store();h,w=im.shape[:2];dh,dw=h-3*patch,w-3*patch
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
 f.owners=(hist,norms,turns,diverse);f.store=store
 return f
