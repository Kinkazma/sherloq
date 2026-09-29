"""Declare finite user-requested work to macOS, without preventing idle sleep."""
from contextlib import contextmanager
from functools import lru_cache
import ctypes as ct
import logging
import sys


@lru_cache(1)
def _runtime():
    # Typed Objective-C calls are necessary on arm64; no PyObjC dependency.
    foundation = ct.CDLL('/System/Library/Frameworks/Foundation.framework/Foundation')
    objc = ct.CDLL('/usr/lib/libobjc.A.dylib')
    objc.objc_getClass.argtypes = [ct.c_char_p]; objc.objc_getClass.restype = ct.c_void_p
    objc.sel_registerName.argtypes = [ct.c_char_p]; objc.sel_registerName.restype = ct.c_void_p
    objc.objc_autoreleasePoolPush.restype = ct.c_void_p
    objc.objc_autoreleasePoolPop.argtypes = [ct.c_void_p]
    objc.objc_autoreleasePoolPop.restype = None
    def send(result, *args):
        return ct.CFUNCTYPE(result, ct.c_void_p, ct.c_void_p, *args)(('objc_msgSend', objc))
    return foundation, objc, send(ct.c_void_p), send(ct.c_void_p, ct.c_char_p), send(ct.c_void_p, ct.c_uint64, ct.c_void_p), send(None, ct.c_void_p)


@contextmanager
def user_activity(reason='SHERLOQ image analysis'):
    if sys.platform != 'darwin':
        yield
        return
    pool = token = None
    try:
        _, objc, message, string, begin, end = _runtime()
        pool = objc.objc_autoreleasePoolPush()
        process = message(objc.objc_getClass(b'NSProcessInfo'), objc.sel_registerName(b'processInfo'))
        text = string(objc.objc_getClass(b'NSString'), objc.sel_registerName(b'stringWithUTF8String:'), reason.encode('utf-8'))
        # NSActivityUserInitiatedAllowingIdleSystemSleep from NSProcessInfo.h.
        # No latency-critical flag, display assertion, or permanent preference.
        options = 0x00ffffff & ~(1 << 20)
        token = begin(process, objc.sel_registerName(b'beginActivityWithOptions:reason:'), options, text)
    except (OSError, AttributeError):
        logging.exception('Unable to declare macOS analysis activity')
    try:
        yield
    finally:
        if token:
            end(process, objc.sel_registerName(b'endActivity:'), token)
        if pool:
            objc.objc_autoreleasePoolPop(pool)
