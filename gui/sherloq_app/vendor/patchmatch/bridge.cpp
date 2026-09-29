// SHERLOQ ARM64 descriptor bridge. GPL-3.0-or-later, as the IPOL engine.
#include "src/Utilities/FeatManager/zMManager.h"
#include "src/Utilities/FeatManager/siftManager.h"
#include <memory>
#include <exception>
#include <cstring>
#include "compact_sift.h"
#include "candidate_pool.h"
extern "C" int sherloq_dense_features(const float* image,int width,int height,int method,int patch,int flip,float* first,float* second,char* error) {
 try {
  if(width<=3*patch || height<=3*patch || patch<2 || patch>32 || (method==1 && patch<3) || method<0 || method>1)throw std::runtime_error("Invalid dense descriptor dimensions/settings");
  ImageSize size{(unsigned)width,(unsigned)height,1,(unsigned)(width*height),(unsigned)(width*height)};
  std::vector<float> pixels(image,image+width*height);
  std::unique_ptr<FeatManager> manager;
  if(method==0)manager.reset(new ZMManager(pixels,size,patch,5,26,32,1,flip));
  else manager.reset(new SiftManager(pixels,size,patch,flip));
  manager->copyFeatures(first,second);return 0;
 }catch(const std::exception& e){std::strncpy(error,e.what(),1023);error[1023]=0;return -1;}
}

#include <cstdint>
#include <limits>
#include <cmath>
#include <algorithm>
// Spatially constrained zero/first-order PatchMatch. Candidate validity is
// checked before descriptor distance; absent matches stay -1, never identity.
// Specialize the SIFT loop bound without changing accumulation order. Keep
// Zernike on the original runtime loop: specializing it can alter FP contraction.
template<bool Early, int FixedDimensions=0, bool Compact=false, bool Bounded=Compact> static int patchmatch_impl(const float* first,const float* second,const unsigned char* mask,
 int width,int height,int requested_dimensions,int compare,float minimum,float maximum,int iterations,
 uint32_t seed,int* matches,float* distances,uint64_t* comparisons,int (*cancel)(),char* error,float gapx,float gapy,const float* xmap,const float* ymap,const CompactSift* packed_first=nullptr,const CompactSift* packed_second=nullptr,size_t cache_slots=0) {
 const int dimensions=FixedDimensions?FixedDimensions:requested_dimensions;
 try {
  if(width<=0||height<=0||int64_t(width)*height>std::numeric_limits<int>::max())throw std::runtime_error("Dense index range exceeds 32-bit addressing");
  const int count=width*height;
  CompactSiftCache cache_a(packed_first,cache_slots),cache_b(packed_second,cache_slots);
  if(count<=0 || dimensions<=0 || iterations<1 || maximum<minimum)throw std::runtime_error("Invalid PatchMatch settings");
  const double lo=double(minimum)*minimum,hi=double(maximum)*maximum;
  CandidatePool<Bounded> pool[2]={CandidatePool<Bounded>(count),CandidatePool<Bounded>(count)};
  for(int i=0;i<count;++i){
   if(Bounded && (i&65535)==0 && cancel && cancel())return 1;
   matches[i]=-1;distances[i]=std::numeric_limits<float>::infinity();if(mask[i]&1)pool[0].add(i);if(mask[i]&2)pool[1].add(i);
  }
  pool[0].finish();pool[1].finish();
  *comparisons=0;
  auto random=[](uint32_t& state){state^=state<<13;state^=state>>17;state^=state<<5;return state;};
  auto allowed=[&](int i,int j){
   if(j<0 || j>=count || i==j || !mask[i] || !mask[j])return false;
   if(compare && !(((mask[i]&1)&&(mask[j]&2))||((mask[i]&2)&&(mask[j]&1))))return false;
   double dx=xmap?double(xmap[i%width])-xmap[j%width]:i%width-j%width;
   double dy=ymap?double(ymap[i/width])-ymap[j/width]:i/width-j/width;
   if(compare){double sign=((mask[i]&1)&&(mask[j]&2))?1.:-1.;dx+=sign*gapx;dy+=sign*gapy;}
   double d=dx*dx+dy*dy;return d>=lo && d<=hi;
  };
  auto offer=[&](int i,int j){
   if(!allowed(i,j))return;
   // An already winning candidate cannot improve itself; zero is the lower
   // bound of this sum of squares. Preserve candidate accounting and ties.
   if(Early && (matches[i]==j || distances[i]==0.f)){++*comparisons;return;}
   const float* a=Compact?cache_a.get(i):first+size_t(i)*dimensions;
   const float* b=Compact?cache_b.get(j):second+size_t(j)*dimensions;
   float distance=0;
   // Nonnegative partial sums cannot beat the current best once they reach it.
   // Keep the original operation order and tie rule; only skip rejected tails.
   const float best=distances[i];
   if(Early){
    #pragma clang fp contract(off)
    for(int start=0;start<dimensions;start+=16){
     int end=std::min(start+16,dimensions);
     for(int k=start;k<end;++k){float d=a[k]-b[k];distance+=d*d;}
     if(distance>=best)break;
    }
   }else{
    for(int k=0;k<dimensions;++k){float d=a[k]-b[k];distance+=d*d;}
   }
   ++*comparisons;
   if(distance<distances[i]){distances[i]=distance;matches[i]=j;}
  };
  for(int i=0;i<count;++i){
   if((i&(Bounded?127:4095))==0 && cancel && cancel())return 1;
   if(!mask[i])continue;
   const auto& candidates=pool[compare && (mask[i]&1)?1:0];if(candidates.empty())continue;
   uint32_t state=seed ^ (uint32_t(i)+1)*2654435761u;if(!state)state=1;
   for(int attempt=0;attempt<64 && matches[i]<0;++attempt)offer(i,candidates[random(state)%candidates.size()]);
  }
  const float removed_x=xmap?width-1-(xmap[width-1]-xmap[0]):0.f;
  const float removed_y=ymap?height-1-(ymap[height-1]-ymap[0]):0.f;
  const float search_max=maximum+std::max(std::abs(gapx)+removed_x,std::abs(gapy)+removed_y);
  for(int iteration=0;iteration<iterations;++iteration){
   int sign=iteration%2?-1:1;
   for(int step=0;step<count;++step){
    if((step&(Bounded?127:4095))==0 && cancel && cancel())return 1;
    int i=sign>0?step:count-1-step;if(!mask[i])continue;
    int x=i%width,y=i/width;
    // Neighbor displacements and their first-order extrapolation.
    const int dxs[4]={-sign,0,-sign,sign},dys[4]={0,-sign,-sign,-sign};
    for(int n=0;n<4;++n){
     int nx=x+dxs[n],ny=y+dys[n];if(nx<0||nx>=width||ny<0||ny>=height)continue;
     int p=ny*width+nx,m=matches[p];if(m<0)continue;
     int tx=x+(m%width-nx),ty=y+(m/width-ny);
     if(tx>=0&&tx<width&&ty>=0&&ty<height)offer(i,ty*width+tx);
     int n2x=nx+dxs[n],n2y=ny+dys[n];if(n2x<0||n2x>=width||n2y<0||n2y>=height)continue;
     int q=n2y*width+n2x,m2=matches[q];if(m2<0)continue;
     tx=x+2*(m%width-nx)-(m2%width-n2x);ty=y+2*(m/width-ny)-(m2/width-n2y);
     if(tx>=0&&tx<width&&ty>=0&&ty<height)offer(i,ty*width+tx);
    }
    uint32_t state=seed ^ (uint32_t(i)+1)*2654435761u ^ (uint32_t(iteration)+1)*2246822519u;if(!state)state=1;
    int m=matches[i],cx=m<0?x:m%width,cy=m<0?y:m/width;
    for(int window=std::min(int(std::ceil(search_max)),std::max(width,height));window>=1;window/=2){
     int xmin=std::max({0,cx-window,int(std::ceil(x-search_max))}),xmax=std::min({width-1,cx+window,int(std::floor(x+search_max))});
     int ymin=std::max({0,cy-window,int(std::ceil(y-search_max))}),ymax=std::min({height-1,cy+window,int(std::floor(y+search_max))});
     if(xmin>xmax||ymin>ymax)continue;
     int tx=xmin+random(state)%(xmax-xmin+1),ty=ymin+random(state)%(ymax-ymin+1);offer(i,ty*width+tx);
    }
   }
   // Re-evaluate reverse descriptors too: reflection matching need not be symmetric.
   for(int i=0;i<count;++i){
    if(Bounded && (i&127)==0 && cancel && cancel())return 1;
    if(matches[i]>=0)offer(matches[i],i);
   }
  }
  return 0;
 }catch(const std::exception& e){std::strncpy(error,e.what(),1023);error[1023]=0;return -1;}
}

extern "C" int sherloq_patchmatch_metric(const float* first,const float* second,const unsigned char* mask,
 int width,int height,int dimensions,int compare,float minimum,float maximum,int iterations,
 uint32_t seed,int* matches,float* distances,uint64_t* comparisons,int (*cancel)(),char* error,float gapx,float gapy,const float* xmap,const float* ymap) {
 if(dimensions==128)return patchmatch_impl<true,128>(first,second,mask,width,height,dimensions,compare,minimum,maximum,iterations,seed,matches,distances,comparisons,cancel,error,gapx,gapy,xmap,ymap);
 return patchmatch_impl<false>(first,second,mask,width,height,dimensions,compare,minimum,maximum,iterations,seed,matches,distances,comparisons,cancel,error,gapx,gapy,xmap,ymap);
}

extern "C" int sherloq_patchmatch_gap(const float* first,const float* second,const unsigned char* mask,
 int width,int height,int dimensions,int compare,float minimum,float maximum,int iterations,
 uint32_t seed,int* matches,float* distances,uint64_t* comparisons,int (*cancel)(),char* error,float gapx,float gapy) {
 return sherloq_patchmatch_metric(first,second,mask,width,height,dimensions,compare,minimum,maximum,iterations,
   seed,matches,distances,comparisons,cancel,error,gapx,gapy,nullptr,nullptr);
}

// Preserve the original ABI and exact no-gap candidate traversal.
extern "C" int sherloq_patchmatch(const float* first,const float* second,const unsigned char* mask,
 int width,int height,int dimensions,int compare,float minimum,float maximum,int iterations,
 uint32_t seed,int* matches,float* distances,uint64_t* comparisons,int (*cancel)(),char* error) {
 return sherloq_patchmatch_gap(first,second,mask,width,height,dimensions,compare,minimum,maximum,iterations,
   seed,matches,distances,comparisons,cancel,error,0.f,0.f);
}

// The direct-array specialization above is retained for the RAM fast path.
extern "C" int sherloq_patchmatch_compact(const CompactSift* first,const CompactSift* second,const unsigned char* mask,
 int width,int height,int compare,float minimum,float maximum,int iterations,
 uint32_t seed,int* matches,float* distances,uint64_t* comparisons,int (*cancel)(),char* error,float gapx,float gapy,const float* xmap,const float* ymap,size_t cache_slots) {
 return patchmatch_impl<true,128,true>(nullptr,nullptr,mask,width,height,128,compare,minimum,maximum,iterations,seed,matches,distances,comparisons,cancel,error,gapx,gapy,xmap,ymap,first,second,cache_slots);
}
extern "C" void sherloq_sift_unpack(const CompactSift* field,int first,int count,float* out){
 for(int i=0;i<count;++i)compact_sift_read(*field,first+i,out+size_t(i)*128);
}

// Global 4-connected area filtering with a row frontier. Components crossing
// arbitrary stripes remain connected. Only small undecided components retain
// coordinates; accepted components no longer need their pixel lists.
extern "C" int sherloq_dense_area_filter(unsigned char* mask,int width,int height,int minimum,int (*cancel)(),char* error){
 struct Node{int parent;bool accepted;std::vector<size_t> pending;};
 struct Run{int first,last,node;};
 try{
  if(width<=0||height<=0||minimum<1)throw std::runtime_error("Invalid component settings");
  if(minimum==1)return 0;
  std::vector<Node> nodes;std::vector<Run> previous;
  auto root=[&](int i){while(nodes[i].parent!=i){nodes[i].parent=nodes[nodes[i].parent].parent;i=nodes[i].parent;}return i;};
  auto join=[&](int a,int b){a=root(a);b=root(b);if(a==b)return a;
   if(nodes[a].pending.size()<nodes[b].pending.size())std::swap(a,b);
   nodes[b].parent=a;nodes[a].accepted=nodes[a].accepted||nodes[b].accepted;
   if(!nodes[a].accepted){nodes[a].pending.insert(nodes[a].pending.end(),nodes[b].pending.begin(),nodes[b].pending.end());nodes[a].accepted=nodes[a].pending.size()>=size_t(minimum);}
   nodes[b].pending.clear();if(nodes[a].accepted)nodes[a].pending.clear();return a;};
  for(int y=0;y<height;++y){
   if(cancel&&cancel())return 1;
   std::vector<Run> current;size_t p=0;
   for(int x=0;x<width;){if(!mask[size_t(y)*width+x]){++x;continue;}
    int first=x;while(x<width&&mask[size_t(y)*width+x])++x;int last=x-1;
    while(p<previous.size()&&previous[p].last<first)++p;
    int id=-1;
    for(size_t j=p;j<previous.size()&&previous[j].first<=last;++j)id=id<0?root(previous[j].node):join(id,previous[j].node);
    if(id<0){id=int(nodes.size());nodes.push_back({id,false,{}});}
    Node& n=nodes[id];
    if(!n.accepted){
     if(n.pending.size()+size_t(last-first+1)>=size_t(minimum)){n.accepted=true;n.pending.clear();}
     else for(int col=first;col<=last;++col)n.pending.push_back(size_t(y)*width+col);
    }
    current.push_back({first,last,id});
   }
   std::vector<int> remap(nodes.size(),-1);std::vector<Node> next;
   for(Run& r:current){int id=root(r.node);if(remap[id]<0){remap[id]=int(next.size());next.push_back({remap[id],nodes[id].accepted,std::move(nodes[id].pending)});}r.node=remap[id];}
   for(size_t id=0;id<nodes.size();++id)if(nodes[id].parent==int(id)&&remap[id]<0&&!nodes[id].accepted)for(size_t pixel:nodes[id].pending)mask[pixel]=0;
   nodes.swap(next);previous.swap(current);
  }
  for(const Node& n:nodes)if(!n.accepted)for(size_t pixel:n.pending)mask[pixel]=0;
  return 0;
 }catch(const std::exception& e){strncpy(error,e.what(),1023);error[1023]=0;return -1;}
}

extern "C" int sherloq_patchmatch_bounded(const float* first,const float* second,const unsigned char* mask,
 int width,int height,int dimensions,int compare,float minimum,float maximum,int iterations,
 uint32_t seed,int* matches,float* distances,uint64_t* comparisons,int (*cancel)(),char* error,float gapx,float gapy,const float* xmap,const float* ymap) {
 if(dimensions==128)return patchmatch_impl<true,128,false,true>(first,second,mask,width,height,dimensions,compare,minimum,maximum,iterations,seed,matches,distances,comparisons,cancel,error,gapx,gapy,xmap,ymap);
 return patchmatch_impl<false,0,false,true>(first,second,mask,width,height,dimensions,compare,minimum,maximum,iterations,seed,matches,distances,comparisons,cancel,error,gapx,gapy,xmap,ymap);
}
