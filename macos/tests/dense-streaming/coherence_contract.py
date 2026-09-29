from pathlib import Path
import ctypes as ct,json,time
import numpy as np,cv2 as cv
from compact_contract import lib,p,dc
from gui.sherloq_app.core.memory_resources import TemporaryArrays
lib.sherloq_dense_area_filter.argtypes=[ct.POINTER(ct.c_ubyte),ct.c_int,ct.c_int,ct.c_int,dc.CALLBACK,ct.c_char_p]
def area(mask,minimum):
 h,w=mask.shape;out=mask.astype(np.uint8);err=ct.create_string_buffer(1024);cb=dc.CALLBACK(lambda:0)
 assert lib.sherloq_dense_area_filter(p(out,ct.c_ubyte),w,h,minimum,cb,err)==0,err.value
 return out.astype(bool)
def coherence(targets,squared,threshold,error_threshold,radius,minimum,rows=53):
 store=TemporaryArrays();h,w=targets.shape;selected=store.array((h,w),bool);errors=store.array((h,w),np.float32)
 yy,xx=np.mgrid[-radius:radius+1,-radius:radius+1];kernel=((xx*xx+yy*yy)<=radius*radius).astype(np.float64);count=kernel.sum();moment=(kernel*xx*xx).sum()
 for start in range(0,h,rows):
  lo=max(0,start-radius);hi=min(h,start+rows+radius);t=targets[lo:hi];valid=(t>=0)&(squared[lo:hi]<=threshold*threshold)
  complete=cv.filter2D(valid.astype(np.float64),-1,kernel,borderType=cv.BORDER_CONSTANT)>count-.5
  residual=np.zeros(t.shape,np.float64)
  for axis in (0,1):
   delta=((t%w-np.arange(w)[None,:]) if axis==0 else (t//w-np.arange(lo,hi)[:,None])).astype(np.float64);delta[~valid]=0
   total=dc.integer_sum_filter(delta,kernel)
   residual+=dc.integer_sum_filter(delta*delta,kernel)-total*total/count
   for coords in (xx,yy):
    linear=dc.integer_sum_filter(delta,kernel*coords);residual-=linear*linear/moment
  error=np.sqrt(np.maximum(residual,0)/count);crop=slice(start-lo,min(h,start+rows)-lo)
  selected[start:start+rows]=(complete&(error<=error_threshold))[crop];errors[start:start+rows]=error[crop];store.checkpoint()
 err=ct.create_string_buffer(1024);cb=dc.CALLBACK(lambda:0)
 assert lib.sherloq_dense_area_filter(p(selected,ct.c_ubyte),w,h,minimum,cb,err)==0
 return selected,errors

def main():
 rng=np.random.default_rng(728);n=0
 for shape in ((1,1),(1,27),(40,1),(33,43),(201,302)):
  for density in (.03,.3,.7,1.):
   mask=rng.random(shape)<density
   for minimum in (1,2,7,100,10000):
    _,labels,stats,_=cv.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=4);keep=stats[:,cv.CC_STAT_AREA]>=minimum;keep[0]=False
    assert np.array_equal(keep[labels],area(mask,minimum)),(shape,density,minimum);n+=1
 reports=[]
 for shape in ((103,151),(513,727)):
  h,w=shape;targets=rng.integers(-1,h*w,shape,dtype=np.int32);scores=rng.random(shape,dtype=np.float32)
  y,x=np.mgrid[:h,:w];targets[15:h-15,20:w-50]=((y+10)*w+x+23)[15:h-15,20:w-50];scores[15:h-15,20:w-50]=.01
  for radius in (2,6):
   expected=dc.coherent_mask(targets,scores,.9,3,radius,10);actual=coherence(targets,scores,.9,3,radius,10)
   assert np.array_equal(expected[0],actual[0]),(shape,radius,'selection')
   delta=float(np.max(abs(expected[1]-actual[1])));assert delta==0,(shape,radius,delta)
   reports.append(dict(shape=shape,radius=radius,selection_exact=True,error_max=delta))
 r=dict(passed=True,component_cases=n,coherence=reports);Path(__file__).with_name('coherence-results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
if __name__=='__main__':main()
