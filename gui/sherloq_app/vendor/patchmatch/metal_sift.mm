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
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstring>
#include <algorithm>
#include <mutex>
#include <unistd.h>
#include <cstdint>
static int sift_compute(const float* grads,const float* weights,int width,int height,int patch,float* output,char* error,size_t sharedBytes){
 @autoreleasepool{
  if(patch<3 || patch>32 || width<=3*patch || height<=3*patch){strcpy(error,"Invalid SIFT dimensions/settings");return -1;}
  const int fused=1; // Match VLFeat's ARM64 fused tail expression.
  static id<MTLDevice> device=MTLCreateSystemDefaultDevice();static id<MTLCommandQueue> queue=[device newCommandQueue];
  static id<MTLComputePipelineState> tri=nil,pack=nil;
  static std::mutex pipeline_lock;
  {std::lock_guard<std::mutex> guard(pipeline_lock);
  if(!tri || !pack){
   const char* source=R"MSL(
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
kernel void triangle(device const float* input [[buffer(0)]],device float* scratch [[buffer(1)]],device float* output [[buffer(2)]],constant int4& cfg [[buffer(3)]],constant int& fused [[buffer(4)]],uint id [[thread_position_in_grid]]){
 int w=cfg.x,h=cfg.y,p=cfg.z,axis=cfg.w,columns=axis?h:w,rows=axis?w:h;
 if(id>=uint(columns*8))return;
 int bin=id/columns,col=id%columns;
 device float* b=scratch+bin*columns*(rows+p)+col+columns*p;
 #define B(y) b[(y)*columns]
 #define I(y) input[bin*w*h+(axis?col*w+(y):(y)*w+col)]
 B(rows-1)=I(rows-1);
 for(int y=rows-2;y>=0;--y)B(y)=B(y+1)+I(y);
 for(int y=-1;y>=-p;--y)B(y)=B(y+1)+I(0);
 for(int y=-p;y<rows-p;++y)B(y)=B(y)-B(y+p);
 for(int y=rows-p;y<rows;++y)B(y)=fused?fma(-B(rows-1),float(rows-p-y),B(y)):B(y)-B(rows-1)*float(rows-p-y);
 for(int y=-p+1;y<rows;++y)B(y)=B(y)+B(y-1);
 float scale=1.f/float(p*p);
 for(int y=rows-1;y>=0;--y)output[bin*w*h+(axis?col*w+y:y*w+col)]=scale*(B(y)-B(y-p));
}
kernel void packing(device const float* hist [[buffer(0)]],device const float* weights [[buffer(1)]],device float* out [[buffer(2)]],constant int4& cfg [[buffer(3)]],uint id [[thread_position_in_grid]]){
 int w=cfg.x,h=cfg.y,p=cfg.z,dw=w-3*p,dh=h-3*p;if(id>=uint(dw*dh*128))return;
 int bin=id%8,bx=(id/8)%4,by=(id/32)%4,pixel=id/128,x=pixel%dw,y=pixel/dw;
 out[id]=(weights[bx]*weights[by])*hist[bin*w*h+(y+by*p)*w+x+bx*p];
}
)MSL";
   NSError* e=nil;MTLCompileOptions* options=[MTLCompileOptions new];if(@available(macOS 15.0,*)){options.mathMode=MTLMathModeSafe;}else{options.fastMathEnabled=NO;}
   id<MTLLibrary> library=[device newLibraryWithSource:[NSString stringWithUTF8String:source] options:options error:&e];
   if(!library){strncpy(error,[[e description] UTF8String],1023);return -1;}
   tri=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"triangle"] error:&e];
   pack=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"packing"] error:&e];
   if(!tri||!pack){strncpy(error,[[e description] UTF8String],1023);return -1;}
  }
  }
  if(!device||!queue||!tri||!pack){strcpy(error,"Metal unavailable");return -1;}
  size_t histBytes=size_t(width)*height*8*sizeof(float),outBytes=size_t(width-3*patch)*(height-3*patch)*128*sizeof(float);
  id<MTLBuffer> a=[device newBufferWithBytes:grads length:histBytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> b=[device newBufferWithLength:histBytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> scratch=[device newBufferWithLength:size_t(width*height+patch*std::max(width,height))*8*sizeof(float) options:MTLResourceStorageModeShared];
  // Shared output belongs to a dedicated page-rounded mmap retained by Python.
  // Legacy callers keep the copy path; pointer alignment alone is insufficient.
  bool shared=sharedBytes>=outBytes && sharedBytes%getpagesize()==0 && uintptr_t(output)%getpagesize()==0;
  id<MTLBuffer> result=shared?[device newBufferWithBytesNoCopy:output length:sharedBytes options:MTLResourceStorageModeShared deallocator:nil]:[device newBufferWithLength:outBytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> weightsBuffer=[device newBufferWithBytes:weights length:4*sizeof(float) options:MTLResourceStorageModeShared];
  if(!a||!b||!scratch||!result||!weightsBuffer){strcpy(error,"Allocation failed");return -1;}
  id<MTLCommandBuffer> command=[queue commandBuffer];
  for(int axis=0;axis<2;++axis){
   id<MTLComputeCommandEncoder> enc=[command computeCommandEncoder];[enc setComputePipelineState:tri];
   [enc setBuffer:axis?b:a offset:0 atIndex:0];[enc setBuffer:scratch offset:0 atIndex:1];[enc setBuffer:axis?a:b offset:0 atIndex:2];
   int cfg[4]={width,height,patch,axis};[enc setBytes:cfg length:sizeof(cfg) atIndex:3];[enc setBytes:&fused length:sizeof(fused) atIndex:4];
   [enc dispatchThreads:MTLSizeMake(8*(axis?height:width),1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[enc endEncoding];
  }
  id<MTLComputeCommandEncoder> enc=[command computeCommandEncoder];[enc setComputePipelineState:pack];[enc setBuffer:a offset:0 atIndex:0];[enc setBuffer:weightsBuffer offset:0 atIndex:1];[enc setBuffer:result offset:0 atIndex:2];int cfg[4]={width,height,patch,0};[enc setBytes:cfg length:sizeof(cfg) atIndex:3];[enc dispatchThreads:MTLSizeMake(outBytes/sizeof(float),1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[enc endEncoding];
  [command commit];[command waitUntilCompleted];
  if(command.status==MTLCommandBufferStatusError){strncpy(error,[[command.error description] UTF8String],1023);return -1;}
  if(!shared)memcpy(output,result.contents,outBytes);return 0;
 }
}

extern "C" int sherloq_metal_sift(const float* grads,const float* weights,int width,int height,int patch,float* output,char* error){
 return sift_compute(grads,weights,width,height,patch,output,error,0);
}
extern "C" int sherloq_metal_sift_shared(const float* grads,const float* weights,int width,int height,int patch,float* output,char* error,size_t sharedBytes){
 return sift_compute(grads,weights,width,height,patch,output,error,sharedBytes);
}
