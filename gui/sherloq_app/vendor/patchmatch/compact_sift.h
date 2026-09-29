// Lossless reconstruction of normalized dense SIFT. GPL-3.0-or-later.
#pragma once
#include <cstdint>
#include <vector>
#include <algorithm>
struct CompactSift {
 const float* hist; // Pixel-major H x W x 8 filtered orientation histograms.
 const float* norms; // Full descriptor grid: preclip, postclip, final NumPy norm.
 const unsigned char* turns;
 float weights[4];
 int width,height,patch,offset,view_width,view_height,mirror,quarter;
};
static inline void compact_sift_read(const CompactSift& f,int index,float* out){
 #pragma clang fp contract(off)
 int dw=f.width-3*f.patch;
 int x=index%f.view_width+f.offset,y=index/f.view_width+f.offset;
 if(f.mirror)x=dw-1-x;
 size_t pixel=size_t(y)*dw+x;
 const float* norms=f.norms+pixel*3;
 int turn=f.quarter?f.turns[pixel]:0;
 for(int by=0;by<4;++by)for(int bx=0;bx<4;++bx)for(int bin=0;bin<8;++bin){
  int xx=bx,yy=by;
  if(turn==1){xx=3-by;yy=bx;}else if(turn==2){xx=3-bx;yy=3-by;}else if(turn==3){xx=by;yy=3-bx;}
  float v=(f.weights[xx]*f.weights[yy])*f.hist[(size_t(y+yy*f.patch)*f.width+x+xx*f.patch)*8+(bin+2*turn)%8];
  v/=norms[0];if(v>.2f)v=.2f;v/=norms[1];v/=norms[2];
  out[(by*4+bx)*8+bin]=v;
 }
}
class CompactSiftCache {
 const CompactSift* field;size_t slots;std::vector<int> ids;std::vector<float> data;
 public:
 CompactSiftCache(const CompactSift* f,size_t count):field(f),slots(f?std::max(size_t(1),count):0),ids(slots,-1),data(slots*128){}
 const float* get(int i){size_t slot=uint32_t(i)*uint64_t(2654435761u)%slots;
  float* out=data.data()+slot*128;if(ids[slot]!=i){compact_sift_read(*field,i,out);ids[slot]=i;}return out;}
};
