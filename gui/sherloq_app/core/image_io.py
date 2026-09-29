"""Pure image decoding and atomic export; preserve the BGR8 analysis contract."""
import os,struct,tempfile
from pathlib import Path
import cv2 as cv
import numpy as np
from .jpeg_curve import Cancelled

RAW_EXTENSIONS={'arw','cr2','cr3','crw','dcr','dng','erf','fff','kdc','mos','mrw','nef','nrw','orf','pef','raf','raw','rw2','rwl','sr2','srf','srw','x3f'}


def image_metadata(filename):
    result=dict(source_bits=None,alpha=False,icc=False,orientation=1,frames=1)
    try:
        from PIL import Image
        with Image.open(filename) as image:
            result.update(format=image.format,source_size=list(image.size),alpha='A' in image.getbands() or 'transparency' in image.info,icc=bool(image.info.get('icc_profile')),orientation=int(image.getexif().get(274,1)),frames=int(getattr(image,'n_frames',1)))
            if image.format=='TIFF':
                bits=image.tag_v2.get(258,8);result['source_bits']=max(bits) if isinstance(bits,tuple) else int(bits)
            elif image.mode.startswith('I;16'):result['source_bits']=16
            elif image.mode in ('I','F'):result['source_bits']=32
            else:result['source_bits']=8
        if result.get('format')=='PNG':
            with open(filename,'rb') as stream:header=stream.read(26)
            if len(header)==26 and header[:8]==b'\x89PNG\r\n\x1a\n' and header[12:16]==b'IHDR':result['source_bits']=header[24]
    except Exception:
        # Metadata support may be narrower than the image decoder's support.
        pass
    return result


def decode_image(filename,cancel=lambda:False):
    if cancel():raise Cancelled()
    filename=os.fspath(filename);ext=Path(filename).suffix.lower().lstrip('.')
    meta=image_metadata(filename)
    if ext in RAW_EXTENSIONS:
        import rawpy
        with rawpy.imread(filename) as raw:
            meta.update(format='RAW',source_bits=None,raw_white_level=int(raw.white_level),raw_value_bits=int(raw.white_level).bit_length(),source_size=[int(raw.sizes.raw_width),int(raw.sizes.raw_height)])
            image=cv.cvtColor(raw.postprocess(no_auto_bright=True,use_camera_wb=True),cv.COLOR_RGB2BGR)
    elif ext=='gif':
        capture=cv.VideoCapture(filename)
        try:
            meta['frames']=max(meta['frames'],int(capture.get(cv.CAP_PROP_FRAME_COUNT)))
            success,image=capture.read()
            if not success:raise ValueError('Unable to decode GIF.')
            if image.ndim==2:image=cv.cvtColor(image,cv.COLOR_GRAY2BGR)
        finally:capture.release()
    else:image=cv.imread(filename,cv.IMREAD_COLOR)
    if image is None:raise ValueError('Unable to load image.')
    if image.ndim!=3 or image.shape[2] not in (3,4) or image.dtype!=np.uint8:raise ValueError('The analysis image must decode to 8-bit colour.')
    if image.shape[2]==4:meta['alpha']=True;image=cv.cvtColor(image,cv.COLOR_BGRA2BGR)
    if cancel():raise Cancelled()
    image=np.ascontiguousarray(image)
    meta.update(analysis_shape=list(image.shape),analysis_dtype='uint8',icc_applied=False,alpha_retained=False)
    return filename,os.path.basename(filename),image,meta


def save_image(filename,image,cancel=lambda:False):
    destination=Path(filename);temporary=None
    try:
        if cancel():raise Cancelled()
        with tempfile.NamedTemporaryFile(dir=destination.parent,prefix='.sherloq-export-',suffix=destination.suffix,delete=False) as stream:temporary=Path(stream.name)
        if not cv.imwrite(str(temporary),image):raise OSError('The image encoder could not write this file.')
        if cancel():raise Cancelled()
        os.replace(temporary,destination)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
    return str(destination)


def decode_image_isolated(filename, cancel=lambda: False, timeout=180):
    """Keep codecs holding Python's GIL outside the GUI process.

    Called from an I/O thread. Cancellation reaps the child before deleting its
    outputs; no file borrowed by a result survives the temporary directory.
    """
    import json
    import subprocess
    import sys
    import time

    if cancel():
        raise Cancelled()
    with tempfile.TemporaryDirectory(prefix='sherloq-decode-') as directory:
        with open(Path(directory) / 'stderr.log', 'wb') as errors:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name('image_decode_worker.py')),
                 os.fspath(filename), directory],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=errors)
            deadline = time.monotonic() + timeout
            try:
                while process.poll() is None:
                    if cancel():
                        raise Cancelled()
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Image decoding timed out.')
                    try:
                        process.wait(timeout=0.02)
                    except subprocess.TimeoutExpired:
                        pass
                if cancel():
                    raise Cancelled()
                if process.returncode:
                    raise ValueError('Unable to decode RAW image.')
                image = np.load(Path(directory) / 'image.npy', allow_pickle=False)
                metadata = json.loads((Path(directory) / 'metadata.json').read_text())
                if cancel():
                    raise Cancelled()
                return os.fspath(filename), os.path.basename(filename), image, metadata
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
