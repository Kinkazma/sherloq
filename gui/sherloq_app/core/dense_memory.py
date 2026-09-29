"""Bound temporary descriptor storage without changing pixels or reductions."""
import numpy as np


def normalize_inplace(field):
    # Each descriptor uses exactly the original float32 norm/reduction order.
    # A whole-image norm first materializes a descriptor-sized square array.
    rows=field.reshape(-1,field.shape[-1])
    for start in range(0,len(rows),8192):
        block=rows[start:start+8192]
        block/=np.maximum(np.linalg.norm(block,axis=1,keepdims=True),1e-12)


def mirror_inplace(field):
    # The output is newly owned; no cached/user array is mutated.
    for row in field:row[:]=row[::-1].copy()
    return field


def compact_inplace(field,offset,height,width):
    """Pack an aligned crop into its own buffer, with only one row of scratch."""
    if offset==0 and field.shape[:2]==(height,width):return field
    flat=field.reshape(-1);stride=width*field.shape[2]
    for y in range(height):
        flat[y*stride:(y+1)*stride]=field[y+offset,offset:offset+width].copy().ravel()
    return flat[:height*stride].reshape(height,width,field.shape[2])


def workspace_bytes(height,width,method,patch,flip=False,backend='cpu',target_patch=None,quarter_turn=False):
    """Conservative reservation for active fields, including descriptor work.

    Metal builds only the two fields actually matched. Quarter-turn and mirror
    transforms reuse fresh output buffers; CPU retains its existing bridge.
    This is a workspace estimate, not a process RSS or cache limit.
    """
    pixels=int(height)*int(width);target=patch if target_patch is None else target_patch
    paired=target!=patch;dims=128 if method else 12
    def field_bytes(size):
        border=3*size if method else 0
        return max(0,height-border)*max(0,width-border)*dims*4
    first=field_bytes(patch);second=field_bytes(target)
    if backend=='metal' and method==1:
        descriptors=first+(second if flip or paired else 0)
        # Python gradient plane, native gradient/preparation/Metal scratch,
        # grayscale, masks, correspondence/coherence work, and row/block scratch.
        return descriptors+pixels*192+128*1024**2
    copies=6 if paired else 4 if flip else 3
    return pixels*(dims*4*copies+64)
