"""Shared resources for large-image adapters (CPU, GPU staging, and caches).

This is not an automatic out-of-core replacement for arbitrary NumPy/native
operations. Each algorithm supplies a bounded execution plan that preserves its
spatial and numerical semantics. Unadapted engines must not claim spill support.
The browser implementation uses the same planning contract with a different
storage backend; native mmap is not a browser API.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass,replace
from functools import lru_cache
from pathlib import Path
from threading import Condition, RLock
import ctypes as ct
import math
import mmap
import os
import shutil
import sys
import tempfile
import time
import weakref
import numpy as np

MiB=1024**2
GiB=1024**3
_PAGE_ALLOWANCE=ContextVar('sherloq_page_allowance',default=0)


@lru_cache(1)
def physical_memory():
    try:return os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
    except (ValueError,OSError,AttributeError):return 8*GiB


@lru_cache(1)
def _system():
    lib=ct.CDLL('/usr/lib/libSystem.B.dylib')
    lib.mach_host_self.restype=ct.c_uint
    lib.host_statistics64.argtypes=[ct.c_uint,ct.c_int,ct.c_void_p,ct.POINTER(ct.c_uint)]
    lib.msync.argtypes=[ct.c_void_p,ct.c_size_t,ct.c_int]
    lib.proc_pidinfo.argtypes=[ct.c_int,ct.c_int,ct.c_uint64,ct.c_void_p,ct.c_int]
    return lib


def available_memory():
    """Conservative host estimate; excludes compressed/purgeable speculation."""
    if sys.platform=='darwin':
        data=(ct.c_uint*64)();count=ct.c_uint(64);lib=_system()
        if lib.host_statistics64(lib.mach_host_self(),4,data,ct.byref(count))==0:
            return (int(data[0])+int(data[2]))*os.sysconf('SC_PAGE_SIZE')
    if sys.platform.startswith('linux'):
        try:
            for line in Path('/proc/meminfo').read_text().splitlines():
                if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
        except OSError:pass
    return physical_memory()//4


def resident_memory():
    if sys.platform=='darwin':
        # proc_taskinfo: six uint64 + twelve int32; resident_size is field 2.
        data=(ct.c_uint64*12)()
        if _system().proc_pidinfo(os.getpid(),4,0,data,ct.sizeof(data))==ct.sizeof(data):return int(data[1])
    if sys.platform.startswith('linux'):
        try:return int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
        except (OSError,ValueError,IndexError):pass
    return 0


@dataclass(frozen=True)
class MemoryPlan:
    mode: str
    reservation: int
    full_bytes: int
    temporary_bytes: int
    bounded_bytes: int | None = None


class MemoryCoordinator:
    """Shared admission, with an algorithm-provided bounded alternative.

    A lease covers working buffers, not retained output or all process memory.
    Native/model adapters must include their GPU buffers in the estimate on
    unified-memory hosts. No resizing or reduced iteration count is permitted.
    """
    def __init__(self,limit=None,available=available_memory):
        self.limit=int(limit if limit is not None else physical_memory()*0.8)
        self.available=available;self.used=0;self.peak=0;self.condition=Condition()
    def capacity(self):
        headroom=min(2*GiB,physical_memory()//16)
        return max(16*MiB,min(self.limit,int(self.available())-headroom))
    def plan(self,full_bytes,bounded_bytes=None,temporary_bytes=0):
        cap=self.capacity();full_bytes=int(full_bytes)
        if full_bytes<=cap:return MemoryPlan('ram',full_bytes,full_bytes,int(temporary_bytes),bounded_bytes)
        if bounded_bytes is None:
            raise MemoryError('This algorithm has no validated bounded-memory plan yet.')
        bounded_bytes=int(bounded_bytes)
        if bounded_bytes>cap:
            raise MemoryError('The minimum working buffers exceed currently available memory.')
        return MemoryPlan('bounded',bounded_bytes,full_bytes,int(temporary_bytes),bounded_bytes)
    @contextmanager
    def claim(self,plan,cancel=lambda:False):
        from .cloning import check
        with self.condition:
            while self.used+plan.reservation>self.capacity():
                check(cancel)
                if plan.mode=='bounded' and plan.bounded_bytes is not None:
                    # Optional hot pages must never make a minimum-sized job
                    # wait behind another job when its real scratch fits.
                    room=max(plan.bounded_bytes,self.capacity()-self.used)
                    plan=replace(plan,reservation=min(plan.reservation,room))
                if plan.reservation>self.capacity():
                    if plan.mode=='bounded':
                        if plan.bounded_bytes is None or plan.bounded_bytes>self.capacity():
                            raise MemoryError('The minimum working buffers exceed currently available memory.')
                        plan=replace(plan,reservation=self.capacity())
                    else:plan=self.plan(plan.full_bytes,plan.bounded_bytes,plan.temporary_bytes)
                if self.used+plan.reservation<=self.capacity():break
                self.condition.wait(.1)
            check(cancel);self.used+=plan.reservation;self.peak=max(self.peak,self.used)
        try:yield plan
        finally:
            with self.condition:
                self.used-=plan.reservation;self.condition.notify_all()

    def execute(self,full_bytes,bounded_bytes,ram,bounded,cancel=lambda:False):
        """Choose a validated implementation; recover one reported RAM OOM.

        Admission is the first protection. If another process consumes RAM
        after that check, recover allocation errors without changing analysis
        parameters. An OS process kill cannot be caught by this mechanism.
        """
        plan=self.plan(full_bytes,bounded_bytes)
        def with_page_allowance(plan):
            with self.condition:
                room=max(int(bounded_bytes),self.capacity()-self.used)
                return replace(plan,reservation=min(room,int(bounded_bytes)+256*MiB))
        def run_bounded(admitted):
            token=_PAGE_ALLOWANCE.set(max(0,admitted.reservation-int(bounded_bytes)))
            try:return bounded()
            finally:_PAGE_ALLOWANCE.reset(token)
        if plan.mode=='bounded':plan=with_page_allowance(plan)
        with self.claim(plan,cancel) as admitted:
            if admitted.mode=='bounded':return run_bounded(admitted)
            try:return ram()
            except Exception as error:
                if not isinstance(error,MemoryError):
                    import cv2 as cv
                    if not isinstance(error,cv.error) or error.code!=cv.Error.StsNoMem:raise
        # Leave the exception scope before retrying so failed native/Python
        # frames do not keep their temporary allocations alive.
        import gc
        from .cache_budget import GLOBAL_CACHE_BUDGET
        GLOBAL_CACHE_BUDGET.trim(0);gc.collect()
        plan=MemoryPlan('bounded',int(bounded_bytes),int(full_bytes),0,int(bounded_bytes))
        with self.claim(with_page_allowance(plan),cancel) as admitted:return run_bounded(admitted)


MEMORY=MemoryCoordinator()


_storage_lock=RLock()
_storage_pending={}


def require_disk_space(size,directory=None):
    """Check real free storage plus unwritten mapped reservations, not file MB."""
    folder=directory or tempfile.gettempdir()
    with _storage_lock:
        pending=sum(max(0,n-os.fstat(fd).st_blocks*512) for fd,n in _storage_pending.values())
        if int(size)+pending+GiB>shutil.disk_usage(folder).free:
            raise OSError(f'Not enough temporary disk space for {int(size)} bytes of lossless working data.')


class TemporaryArrays:
    """Unlinked, lossless, disk-backed arrays; views retain their storage.

    Bounded adapters call checkpoint between batches. Dirty pages are flushed
    before advising eviction; never apply this to the normal in-RAM path.
    No persistent user file is modified or deleted. Storage is reclaimed on
    last view release, including exceptions and cancellation.
    """
    def __init__(self,working_bytes=256*MiB,directory=None,*,mapped=True):
        self.working_bytes=int(working_bytes)+_PAGE_ALLOWANCE.get();self.directory=directory;self.mapped=bool(mapped)
        self.arrays=[];self.base_rss=resident_memory();self.peak_bytes=0;self.bytes=0
        self.last_check=0.;self.lock=RLock()
    def array(self,shape,dtype=np.float32,*,zero=False):
        shape=tuple(map(int,shape));dtype=np.dtype(dtype)
        if dtype.hasobject or any(n<0 for n in shape):raise ValueError('Invalid temporary array type/shape')
        size=math.prod(shape)*dtype.itemsize
        if not size:return np.zeros(shape,dtype) if zero else np.empty(shape,dtype)
        if not self.mapped:return np.zeros(shape,dtype) if zero else np.empty(shape,dtype)
        with _storage_lock:
            require_disk_space(size,self.directory)
            with tempfile.TemporaryFile(prefix='sherloq-working-',dir=self.directory) as file:
                file.truncate(size)
                array=np.memmap(file,shape=shape,dtype=dtype,mode='r+')
                fd=os.dup(file.fileno())
            token=id(array._mmap);_storage_pending[token]=(fd,size)
        with self.lock:
            self.bytes+=size;self.peak_bytes=max(self.peak_bytes,self.bytes)
            self.arrays.append(weakref.ref(array._mmap))
        owner=weakref.ref(self)
        def released():
            with _storage_lock:
                held=_storage_pending.pop(token,None)
                if held is not None:os.close(held[0])
            current=owner()
            if current is not None:
                with current.lock:current.bytes-=size
        weakref.finalize(array._mmap,released)
        # New anonymous file pages are zero; no whole-image initialization.
        return array
    def watch(self, *values):
        """Also release pages of cached/borrowed mappings used by this job."""
        if not self.mapped:return
        seen=set()
        with self.lock:
            known={id(r()) for r in self.arrays if r() is not None}
            def visit(value):
                if id(value) in seen:return
                seen.add(id(value))
                if isinstance(value,mmap.mmap):
                    if id(value) not in known:
                        self.arrays.append(weakref.ref(value));known.add(id(value))
                elif isinstance(value,np.ndarray):
                    if isinstance(value,np.memmap):visit(value._mmap)
                    elif value.base is not None:visit(value.base)
                elif hasattr(value,'memory_arrays'):visit(value.memory_arrays)
                elif isinstance(value,(tuple,list)):
                    for item in value:visit(item)
                elif isinstance(value,dict):
                    for item in value.values():visit(item)
            for value in values:visit(value)

    def checkpoint(self,*,force=False):
        if not self.mapped:return
        now=time.monotonic()
        if not force and now-self.last_check<.1:return
        self.last_check=now
        if not force and resident_memory()-self.base_rss<self.working_bytes:return
        with self.lock:
            living=[]
            for reference in self.arrays:
                mapping=reference()
                if mapping is None:continue
                living.append(reference)
                if sys.platform=='darwin':
                    # MADV_DONTNEED alone leaves clean shared pages resident on
                    # Darwin. MS_SYNC|MS_INVALIDATE writes and invalidates them.
                    # Worker results may be borrowed read-only np.load maps.
                    # ctypes.from_buffer rejects those; a one-byte NumPy view
                    # exposes the address without copying or requiring writes.
                    address=np.frombuffer(mapping,dtype=np.uint8,count=1).ctypes.data
                    if _system().msync(address,len(mapping),0x12):
                        raise OSError('Unable to flush temporary working pages')
                else:
                    mapping.flush()
                    if hasattr(mapping,'madvise'):mapping.madvise(mmap.MADV_DONTNEED)
            self.arrays=living


def tiles(shape,tile_shape,halo=0):
    """Finite-support operators only: output slices, input slices, crop slices."""
    h,w=map(int,shape[:2]);th,tw=map(int,tile_shape)
    if min(th,tw)<=0 or halo<0:raise ValueError('Invalid tile geometry')
    for y in range(0,h,th):
        for x in range(0,w,tw):
            y1=min(h,y+th);x1=min(w,x+tw)
            lo_y=max(0,y-halo);hi_y=min(h,y1+halo);lo_x=max(0,x-halo);hi_x=min(w,x1+halo)
            yield ((slice(y,y1),slice(x,x1)),(slice(lo_y,hi_y),slice(lo_x,hi_x)),
                   (slice(y-lo_y,y1-lo_y),slice(x-lo_x,x1-lo_x)))


_pressure_lock=RLock()
_pressure_last=0.
def release_idle_caches_under_pressure():
    """Cheap/throttled common safeguard; does not adapt an algorithm itself."""
    global _pressure_last
    now=time.monotonic()
    with _pressure_lock:
        if now-_pressure_last<1.:return
        _pressure_last=now
        reserve=min(2*GiB,physical_memory()//16)
        free=available_memory()
        if free>=2*reserve:return
        from .cache_budget import GLOBAL_CACHE_BUDGET
        GLOBAL_CACHE_BUDGET.trim(max(0,GLOBAL_CACHE_BUDGET.bytes-(2*reserve-free)))
