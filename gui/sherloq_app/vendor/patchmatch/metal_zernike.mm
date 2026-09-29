// SHERLOQ ordered Zernike convolution. GPL-3.0-or-later.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <complex>
#include <cstring>
#include <mutex>
extern "C" int sherloq_metal_available(){return MTLCreateSystemDefaultDevice()!=nil;}
extern "C" int sherloq_metal_zernike(const float* input,const float* filters,const float* weights,int width,int height,int patch,int fused,float* output,char* error){
 @autoreleasepool {
  static id<MTLDevice> device=MTLCreateSystemDefaultDevice();
  static id<MTLCommandQueue> queue=[device newCommandQueue];
  static id<MTLComputePipelineState> pipeline=nil;
  static std::mutex pipeline_lock;
  {std::lock_guard<std::mutex> guard(pipeline_lock);
  if(!pipeline){
   NSString* source=@"#include <metal_stdlib>\nusing namespace metal;\n#pragma clang fp contract(off)\nkernel void convolution(device const float* im [[buffer(0)]],device const float* coeff [[buffer(1)]],device float2* out [[buffer(2)]],constant int4& size [[buffer(3)]],constant int& fused [[buffer(4)]],uint id [[thread_position_in_grid]]){int w=size.x,h=size.y,p=size.z; if(id>=uint(w*h*12))return;int n=id%12,i=id/12,px=i%w,py=i/w;float2 sum=0;for(int x=max(px-p,0);x<min(px+p,w);++x)for(int y=max(py-p,0);y<min(py+p,h);++y){int k=x-px+p+(y-py+p)*2*p;float v=im[x+y*w];float2 c=float2(coeff[(2*n)*4*p*p+k],coeff[(2*n+1)*4*p*p+k]);if(fused)sum=fma(float2(v),c,sum);else sum=sum+v*c;}out[id]=sum;}";
   NSError* e=nil;MTLCompileOptions* options=[MTLCompileOptions new];if(@available(macOS 15.0,*)){options.mathMode=MTLMathModeSafe;}else{options.fastMathEnabled=NO;}
   id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&e];
   if(!library){strncpy(error,[[e description] UTF8String],1023);return -1;}
   pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"convolution"] error:&e];
   if(!pipeline){strncpy(error,[[e description] UTF8String],1023);return -1;}
  }
  }
  if(!device || !queue || !pipeline){strcpy(error,"Metal unavailable");return -1;}
  size_t count=size_t(width)*height*12;
  id<MTLBuffer> im=[device newBufferWithBytes:input length:width*height*sizeof(float) options:MTLResourceStorageModeShared];
  id<MTLBuffer> cf=[device newBufferWithBytes:filters length:24*4*patch*patch*sizeof(float) options:MTLResourceStorageModeShared];
  id<MTLBuffer> result=[device newBufferWithLength:count*2*sizeof(float) options:MTLResourceStorageModeShared];
  if(!im||!cf||!result){strcpy(error,"Metal allocation failed");return -1;}
  id<MTLCommandBuffer> command=[queue commandBuffer];id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];[encoder setBuffer:im offset:0 atIndex:0];[encoder setBuffer:cf offset:0 atIndex:1];[encoder setBuffer:result offset:0 atIndex:2];
  int size[4]={width,height,patch,0};[encoder setBytes:size length:sizeof(size) atIndex:3];[encoder setBytes:&fused length:sizeof(fused) atIndex:4];
  [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[encoder endEncoding];[command commit];[command waitUntilCompleted];
  if(command.status==MTLCommandBufferStatusError){strncpy(error,[[command.error description] UTF8String],1023);return -1;}
  float* v=(float*)result.contents;
  for(size_t i=0;i<count;++i)output[i]=std::abs(weights[i%12]*std::complex<float>(v[2*i],v[2*i+1]));
  return 0;
 }
}
