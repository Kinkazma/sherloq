// Sorted candidate rank/select without a 4-byte index per valid pixel.
#pragma once
#include <vector>
#include <cstdint>
#include <algorithm>
template<bool Bounded> class CandidatePool;
template<> class CandidatePool<false>:public std::vector<int>{
 public:CandidatePool(int){}void finish(){}void add(int i){push_back(i);}
};
template<> class CandidatePool<true>{
 int length;size_t total=0;std::vector<uint64_t> words;std::vector<uint32_t> prefix;
 public:
 CandidatePool(int n):length(n){}
 void add(int i){if(words.empty())words.resize((size_t(length)+63)/64);words[size_t(i)/64]|=uint64_t(1)<<(i%64);++total;}
 void finish(){
  if(total==size_t(length)){std::vector<uint64_t>().swap(words);return;}
  uint32_t sum=0;
  for(size_t i=0;i<words.size();i+=16){prefix.push_back(sum);for(size_t j=i;j<std::min(i+16,words.size());++j)sum+=__builtin_popcountll(words[j]);}
  prefix.push_back(sum);
 }
 bool empty()const{return total==0;}size_t size()const{return total;}
 int operator[](size_t rank)const{
  if(total==size_t(length))return int(rank);
  size_t group=std::upper_bound(prefix.begin(),prefix.end(),uint32_t(rank))-prefix.begin()-1;
  rank-=prefix[group];size_t index=group*16;
  for(;;++index){unsigned count=__builtin_popcountll(words[index]);if(rank<count)break;rank-=count;}
  uint64_t bits=words[index];while(rank--){bits&=bits-1;}
  return int(index*64+__builtin_ctzll(bits));
 }
};
