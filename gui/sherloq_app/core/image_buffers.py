"""Small model inputs and image digests without full-image RGB/byte copies."""
import hashlib
import cv2 as cv
import numpy as np
from .memory_resources import TemporaryArrays,MiB


def resize_rgb(image,size):
    # Interpolation is channel-independent. Reverse channels on the small
    # result instead of making OpenCV copy a huge negative-stride RGB view.
    return cv.cvtColor(cv.resize(image,size),cv.COLOR_BGR2RGB)


def image_sha256(image):
    """Same digest as image.tobytes(order='C'), also for strided views."""
    state=hashlib.sha256();store=TemporaryArrays(32*MiB);store.watch(image)
    if image.flags.c_contiguous:
        data=memoryview(image).cast('B')
        for first in range(0,len(data),4*MiB):
            state.update(data[first:first+4*MiB]);store.checkpoint()
    else:
        # Preserve C-order row/channel layout, including reversed views.
        for row in image:
            flat=row.reshape(-1) if row.flags.c_contiguous else None
            if flat is not None:
                data=memoryview(flat).cast('B')
                for first in range(0,len(data),4*MiB):state.update(data[first:first+4*MiB])
            else:
                for first in range(0,len(row),4096):
                    state.update(memoryview(np.ascontiguousarray(row[first:first+4096])).cast('B'))
            store.checkpoint()
    return state.hexdigest()


def all_finite(array):
    """Validation without a full-size temporary boolean map."""
    store=TemporaryArrays(32*MiB);store.watch(array)
    iterator=np.nditer(array,flags=['external_loop','buffered','zerosize_ok'],
                       op_flags=['readonly'],buffersize=262144)
    for chunk in iterator:
        if not np.isfinite(chunk).all():return False
        store.checkpoint()
    return True
