// SHERLOQ: independent pixel work on macOS's shared CPU pool. GPL-3.0-or-later.
#pragma once
#include <dispatch/dispatch.h>
#include <algorithm>
template<class Function> void parallel_pixels(int size,Function function) {
 struct Context {int size;Function* function;};Context context{size,&function};
 dispatch_apply_f((size+1023)/1024,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),&context,
  [](void* pointer,size_t chunk){auto* c=static_cast<Context*>(pointer);int end=std::min(c->size,int((chunk+1)*1024));for(int id=chunk*1024;id<end;++id)(*c->function)(id);});
}
