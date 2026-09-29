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
// One batch of complete columns. Never split the prefix recurrence.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mutex>
#include <cstring>
extern "C" int sherloq_metal_sift_columns(const float* input,int columns,int rows,int patch,float* output,char* error){
 @autoreleasepool {
  if(columns<1||rows<patch||patch<3||patch>32){strcpy(error,"Invalid SIFT stripe");return -1;}
  static id<MTLDevice> device=MTLCreateSystemDefaultDevice();
  static id<MTLCommandQueue> queue=[device newCommandQueue];
  static id<MTLComputePipelineState> pipeline=nil;static std::mutex lock;
  {std::lock_guard<std::mutex> guard(lock);if(!pipeline){
   const char* source=R"MSL(
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
kernel void stripe(device const float* input [[buffer(0)]],device float* scratch [[buffer(1)]],device float* output [[buffer(2)]],constant int4& cfg [[buffer(3)]],uint id [[thread_position_in_grid]]){
 int columns=cfg.x,rows=cfg.y,p=cfg.z;if(id>=uint(columns))return;
 device float* b=scratch+id+columns*p;
 #define B(y) b[(y)*columns]
 #define I(y) input[(y)*columns+id]
 B(rows-1)=I(rows-1);
 for(int y=rows-2;y>=0;--y)B(y)=B(y+1)+I(y);
 for(int y=-1;y>=-p;--y)B(y)=B(y+1)+I(0);
 for(int y=-p;y<rows-p;++y)B(y)=B(y)-B(y+p);
 for(int y=rows-p;y<rows;++y)B(y)=fma(-B(rows-1),float(rows-p-y),B(y));
 for(int y=-p+1;y<rows;++y)B(y)=B(y)+B(y-1);
 float scale=1.f/float(p*p);
 for(int y=rows-1;y>=0;--y)output[y*columns+id]=scale*(B(y)-B(y-p));
}
)MSL";
   NSError* e=nil;MTLCompileOptions* options=[MTLCompileOptions new];
   if(@available(macOS 15.0,*)){options.mathMode=MTLMathModeSafe;}else{options.fastMathEnabled=NO;}
   id<MTLLibrary> lib=[device newLibraryWithSource:[NSString stringWithUTF8String:source] options:options error:&e];
   if(lib)pipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"stripe"] error:&e];
   if(!pipeline){strncpy(error,e.description.UTF8String ?: "Metal unavailable",1023);return -1;}
  }}
  size_t bytes=size_t(columns)*rows*4;
  id<MTLBuffer> a=[device newBufferWithBytes:input length:bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> b=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> scratch=[device newBufferWithLength:size_t(columns)*(rows+patch)*4 options:MTLResourceStorageModeShared];
  if(!a||!b||!scratch){strcpy(error,"Metal stripe allocation failed");return -1;}
  id<MTLCommandBuffer> cmd=[queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];
  [enc setComputePipelineState:pipeline];[enc setBuffer:a offset:0 atIndex:0];[enc setBuffer:scratch offset:0 atIndex:1];[enc setBuffer:b offset:0 atIndex:2];
  int cfg[4]={columns,rows,patch,0};[enc setBytes:cfg length:sizeof(cfg) atIndex:3];
  [enc dispatchThreads:MTLSizeMake(columns,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[enc endEncoding];[cmd commit];[cmd waitUntilCompleted];
  if(cmd.status==MTLCommandBufferStatusError){strncpy(error,cmd.error.description.UTF8String,1023);return -1;}
  memcpy(output,b.contents,bytes);return 0;
 }
}
