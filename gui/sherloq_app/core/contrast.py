"""Exact blockwise contrast indicators; retain analysis separately from display."""
import cv2 as cv
import numpy as np
from .interactive import ArrayCache
from .utility import pad_image, compute_hist, gray_to_bgr
from .jpeg_curve import Cancelled

_WINDOW = np.arange(256).astype(np.float32)
_WINDOW[:8] = (1-np.cos(np.pi*_WINDOW[:8]/8))/2
_WINDOW[-8:] = (1+np.cos(np.pi*(_WINDOW[-8:]+8-255)/8))/2
_WINDOW[8:-8] = 1
_WEIGHT = ((np.arange(256)-128)/128)**2


def histogram_error(gray):
    hist = compute_hist(gray)*_WINDOW
    hist = cv.normalize(hist, None, 0, 1, cv.NORM_MINMAX)
    dft = np.fft.fftshift(cv.dft(hist, flags=cv.DFT_COMPLEX_OUTPUT))
    mag = cv.magnitude(dft[:, :, 0], dft[:, :, 1])
    mag = cv.normalize(mag, None, 0, 1, cv.NORM_MINMAX).flatten()
    # Each stencil is unchanged; only the 252 Python iterations disappear.
    left = 2*hist[1:253]-hist[:252]
    right = 2*hist[3:255]-hist[4:256]
    diff = np.max(np.abs(hist[2:254]-(left+right)/2))
    ed = np.sum(mag)
    if ed == 0:
        return 0
    error = np.sum(mag*_WEIGHT)/ed
    error = 1 if error > .185 else error/.185
    return error*np.sqrt(diff)


class ContrastEngine:
    def __init__(self, image):
        self.image = image
        self.prepared = ArrayCache(256)
        self.maps = ArrayCache(32)
        self.renders = ArrayCache(128)
        # Only one unfinished block size is retained, for cancellation/resume.
        self.partial = None

    def prepare(self, block):
        height, width = self.image.shape[:2]
        # Preserve historical zero padding, including a whole extra block at
        # divisible dimensions: it is part of the original border contract.
        key = (height+block-height%block, width+block-width%block)
        value = self.prepared.get(key)
        if value is None:
            self.prepared.reserve(key[0]*key[1]*9)
            color = pad_image(self.image, block)
            gray = cv.cvtColor(color, cv.COLOR_BGR2GRAY)
            kx, ky = cv.getDerivKernels(1, 1, 1)
            bd, gd, rd = [cv.sepFilter2D(c, cv.CV_32F, kx, ky) for c in cv.split(color)]
            del color
            tri = np.subtract(gd, rd)
            np.abs(tri, out=tri)
            scratch = np.subtract(gd, bd)
            np.abs(scratch, out=scratch)
            tri += scratch
            np.subtract(rd, bd, out=scratch)
            np.abs(scratch, out=scratch)
            tri += scratch
            tri /= 3
            del scratch
            # The derivatives are private temporaries. Reuse blue for avg,
            # keeping exactly the historical ((abs(B)+abs(G))+abs(R))/3 order.
            np.abs(bd, out=bd)
            np.abs(gd, out=gd)
            np.abs(rd, out=rd)
            bd += gd
            bd += rd
            bd /= 3
            value = self.prepared.put(key, (gray, tri, bd))
        return value

    def analyze(self, block, cancel=lambda:False, progress=lambda *a:None):
        if block not in (32, 64, 128, 256):
            raise ValueError('Block size must be 32, 64, 128 or 256')
        if cancel():
            raise Cancelled()
        cached = self.maps.get(block)
        if cached is not None:
            return cached
        gray, tri, avg = self.prepare(block)
        rows, cols = gray.shape
        nr, nc = rows//block, cols//block
        if self.partial is None or self.partial[0] != block:
            # Preserve the historical extra zero row/column before medianBlur.
            arrays = tuple(np.zeros((nr+1,nc+1), np.float32) for _ in range(3))
            self.partial = [block, 0, arrays]
        _, first, arrays = self.partial
        errors, similarities, joint = arrays
        for index in range(first,nr*nc):
            if cancel():
                raise Cancelled()
            r,c = divmod(index,nc)
            ys,xs = slice(r*block,(r+1)*block),slice(c*block,(c+1)*block)
            error = histogram_error(gray[ys,xs])
            avg_m = np.mean(avg[ys,xs])
            if avg_m == 0:
                chsim = 0
            else:
                chsim = np.mean(tri[ys,xs])/avg_m
                chsim = 1 if chsim > .75 else chsim/.75
            errors[r,c], similarities[r,c], joint[r,c] = error,chsim,error*chsim
            self.partial[1] = index+1
            # One update per block row; no GUI objects or event pumping here.
            if c == nc-1:
                progress((r+1)*100//nr, 'Analyzing contrast blocks')
        if cancel():
            raise Cancelled()
        self.partial = None
        return self.maps.put(block, arrays)

    def render(self, block, mode, maps):
        if mode not in (0,1,2):
            raise ValueError('Unknown contrast indicator')
        key = block,mode
        value = self.renders.get(key)
        if value is None:
            self.renders.reserve(self.image.nbytes)
            plane = cv.medianBlur(cv.convertScaleAbs(maps[mode], None, 255),3)
            h,w = self.image.shape[:2]
            value = gray_to_bgr(cv.resize(plane,None,None,block,block,cv.INTER_NEAREST)[:h,:w])
            self.renders.put(key,value)
        return value

    def compute(self, params, cancel=lambda:False, progress=lambda *a:None):
        from .memory_resources import MEMORY,MiB
        def bounded():
            from .contrast_bounded import compute
            return compute(self,params,cancel,progress)
        return MEMORY.execute(self.image.shape[0]*self.image.shape[1]*64,64*MiB,
                              lambda:self._compute(params,cancel,progress),bounded,cancel)

    def _compute(self, params, cancel=lambda:False, progress=lambda *a:None):
        block, mode = params
        maps = self.analyze(block,cancel,progress)
        if cancel():
            raise Cancelled()
        return self.render(block,mode,maps)
