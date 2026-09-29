/* Derived from VLFeat dense SIFT / triangular filtering.
Copyright (C) 2007-11, Andrea Vedaldi and Brian Fulkerson
Copyright (C) 2012-13, The VLFeat Team
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the
   distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/
#include "src/vlfeat/vl/dsift.c"
int sherloq_sift_inputs(const float* source,int width,int height,int patch,int flip,float* grads,float* weights){
 if(patch<3 || patch>32 || width<=3*patch || height<=3*patch)return -1;
 float* temp=malloc((size_t)width*height*sizeof(float));if(!temp)return -1;
 float sigma=sqrt((patch*patch)/9-0.25);
 vl_imsmooth_f(temp,width,source,width,height,width,sigma,sigma);
 if(flip)for(int y=0;y<height;++y)for(int x=0;x<width/2;++x){float v=temp[y*width+x];temp[y*width+x]=temp[y*width+width-1-x];temp[y*width+width-1-x]=v;}
 VlDsiftFilter* self=vl_dsift_new_basic(width,height,1,patch);const float* im=temp;int t,x,y;
  /* update buffers */
  // Only gradients are needed here; avoid allocating the unused 128-D field.
  self->grads=vl_malloc(8*sizeof(float*));
  if(!self->grads){vl_dsift_delete(self);free(temp);return -1;}
  for(t=0;t<8;++t)self->grads[t]=grads+(size_t)t*width*height;
  // These planes belong to the caller. Delete only the pointer array.
  self->numGradAlloc=0;

  /* clear integral images */
  for (t = 0 ; t < self->geom.numBinT ; ++t)
    memset (self->grads[t], 0,
            sizeof(float) * self->imWidth * self->imHeight) ;

#undef at
#define at(x,y) (im[(y)*self->imWidth+(x)])

  /* Compute gradients, their norm, and their angle */

  for (y = 0 ; y < self->imHeight ; ++ y) {
    for (x = 0 ; x < self->imWidth ; ++ x) {
      float gx, gy ;
      float angle, mod, nt, rbint ;
      int bint ;

      /* y derivative */
      if (y == 0) {
        gy = at(x,y+1) - at(x,y) ;
      } else if (y == self->imHeight - 1) {
        gy = at(x,y) - at(x,y-1) ;
      } else {
        gy = 0.5F * (at(x,y+1) - at(x,y-1)) ;
      }

      /* x derivative */
      if (x == 0) {
        gx = at(x+1,y) - at(x,y) ;
      } else if (x == self->imWidth - 1) {
        gx = at(x,y) - at(x-1,y) ;
      } else {
        gx = 0.5F * (at(x+1,y) - at(x-1,y)) ;
      }

      /* angle and modulus */
      angle = vl_fast_atan2_f (gy,gx) ;
      mod = vl_fast_sqrt_f (gx*gx + gy*gy) ;

      /* quantize angle */
      nt = vl_mod_2pi_f (angle) * (self->geom.numBinT / (2*VL_PI)) ;
      bint = (int) vl_floor_f (nt) ;
      rbint = nt - bint ;

      /* write it back */
      self->grads [(bint    ) % self->geom.numBinT][x + y * self->imWidth] = (1 - rbint) * mod ;
      self->grads [(bint + 1) % self->geom.numBinT][x + y * self->imWidth] = (    rbint) * mod ;
    }
  }


 for(t=0;t<4;++t)weights[t]=_vl_dsift_get_bin_window_mean(patch,4,t,2.0)*patch;
 vl_dsift_delete(self);free(temp);return 0;
}
static void normalize_range(float* descriptors,int count){
 // Match the scalar ARM64 FMA reduction in the original vl_dsift_process.
 #pragma clang fp contract(off)
 for(int i=0;i<count;++i){float* d=descriptors+i*128;
  for(int repeat=0;repeat<2;++repeat){
   float norm=0;for(int k=0;k<128;++k)norm=fmaf(d[k],d[k],norm);
   norm=vl_fast_sqrt_f(norm)+VL_EPSILON_F;
   for(int k=0;k<128;++k)d[k]/=norm;
   if(repeat==0)for(int k=0;k<128;++k)if(d[k]>.2f)d[k]=.2f;
  }
 }
}

#ifdef __APPLE__
#include <dispatch/dispatch.h>
typedef struct {float* data;int count;} NormalizeContext;
static void normalize_task(void* raw,size_t block){
 NormalizeContext* ctx=(NormalizeContext*)raw;int start=(int)block*16384;
 int length=ctx->count-start;if(length>16384)length=16384;
 normalize_range(ctx->data+(size_t)start*128,length);
}
#endif
void sherloq_sift_normalize(float* descriptors,int count){
#ifdef __APPLE__
 if(count>=65536){
  NormalizeContext ctx={descriptors,count};
  dispatch_apply_f(((size_t)count+16383)/16384,dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT,0),&ctx,normalize_task);
  return;
 }
#endif
 normalize_range(descriptors,count);
}

/* Streaming path: keep normalization factors instead of 128 floats/pixel. */
static void norm_factors_range(float* descriptors,int count,float* factors){
 #pragma clang fp contract(off)
 for(int i=0;i<count;++i){float* d=descriptors+(size_t)i*128;
  for(int repeat=0;repeat<2;++repeat){
   float norm=0;for(int k=0;k<128;++k)norm=fmaf(d[k],d[k],norm);
   norm=vl_fast_sqrt_f(norm)+VL_EPSILON_F;factors[(size_t)i*3+repeat]=norm;
   for(int k=0;k<128;++k)d[k]/=norm;
   if(repeat==0)for(int k=0;k<128;++k)if(d[k]>.2f)d[k]=.2f;
  }
 }
}
#ifdef __APPLE__
typedef struct {float* data;float* factors;int count;} FactorContext;
static void factor_task(void* raw,size_t block){
 FactorContext* ctx=(FactorContext*)raw;int start=(int)block*1024;
 int n=ctx->count-start;if(n>1024)n=1024;
 norm_factors_range(ctx->data+(size_t)start*128,n,ctx->factors+(size_t)start*3);
}
#endif
void sherloq_sift_norm_factors(float* descriptors,int count,float* factors){
#ifdef __APPLE__
 if(count>=4096){FactorContext ctx={descriptors,factors,count};
  dispatch_apply_f(((size_t)count+1023)/1024,dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT,0),&ctx,factor_task);return;
 }
#endif
 norm_factors_range(descriptors,count,factors);
}
void sherloq_sift_pack_range(const float* hist,const float* weights,int width,int height,int patch,size_t first,int count,float* output){
 #pragma clang fp contract(off)
 int dw=width-3*patch;
 for(int i=0;i<count;++i){size_t pixel=first+i;int x=pixel%dw,y=pixel/dw;
  for(int by=0;by<4;++by)for(int bx=0;bx<4;++bx)for(int bin=0;bin<8;++bin)
   output[(size_t)i*128+(by*4+bx)*8+bin]=(weights[bx]*weights[by])*hist[((size_t)(y+by*patch)*width+x+bx*patch)*8+bin];
 }
}
