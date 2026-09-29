"""CAT-Net v2: original stored JPEG coefficients, pinned full checkpoint."""
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4]

def source_for(image,source,folder):
    import cv2 as cv,hashlib
    from PIL import Image
    original=False
    if source:
        with Image.open(source) as header:original=header.format=='JPEG'
    if original:
        decoded=cv.imread(str(source),cv.IMREAD_COLOR)
        if decoded is None or not np.array_equal(decoded,image):raise ValueError('The JPEG no longer matches the opened image. Reload it before analysis.')
        path=Path(source)
    else:
        path=Path(folder)/'companion-q100.jpg';Image.fromarray(image[:,:,::-1]).save(path,quality=100,subsampling=0)
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return path,dict(jpeg_source='original' if original else 'companion_q100_444',jpeg_sha256=digest)

def load(device):
    import torch
    from ..vendor.catnet.config import _C
    from ..vendor.catnet.network_CAT import CAT_Net
    cfg=_C.clone();cfg.merge_from_file(str(Path(__file__).resolve().parents[1]/'vendor/catnet/CAT_full.yaml'));cfg.freeze()
    model=CAT_Net(cfg)
    # The full checkpoint must cover every tensor; no random/pretrained fallback.
    # Published checkpoint also stores a NumPy scalar metric. Permit only these
    # numeric reconstruction types, retaining the restricted weights loader.
    with torch.serialization.safe_globals([np.core.multiarray.scalar,np.dtype,np.dtypes.Float64DType]):
        state=torch.load(ROOT/'models/external/CAT_full_v2.pth.tar',weights_only=True,map_location='cpu')['state_dict']
    model.load_state_dict(state,strict=True)
    model.memory_dct=bounded_dct;model.memory_head=bounded_head
    return model.eval().to(device)

def bounded_dct(coeff,first,tail):
    """Exact receptive field (eight-pixel halo) without full 64-channel output."""
    out=coeff.new_empty((coeff.shape[0],4,*coeff.shape[-2:]));height=coeff.shape[-2]
    for y in range(0,height,128):
        end=min(y+128,height);lo=max(0,y-8);hi=min(height,end+8)
        part=tail(first(coeff[:,:,lo:hi]));out[:,:,y:end]=part[:,:,y-lo:end-lo]
    return out

def bounded_head(features,head):
    """Global bilinear coordinates, then pointwise classifier in row strips."""
    import torch
    import torch.nn.functional as F
    h,w=features[0].shape[-2:];out=features[0].new_empty((features[0].shape[0],2,h,w))
    xs=(torch.arange(w,device=out.device,dtype=torch.float32)+.5)*(2/w)-1
    for lo in range(0,h,32):
        hi=min(lo+32,h);ys=(torch.arange(lo,hi,device=out.device,dtype=torch.float32)+.5)*(2/h)-1
        gy,gx=torch.meshgrid(ys,xs,indexing='ij');grid=torch.stack((gx,gy),-1)[None].expand(out.shape[0],-1,-1,-1)
        parts=[features[0][:,:,lo:hi]]
        for f in features[1:]:
            fh,fw=f.shape[-2:]
            # Metal lacks grid_sample's border mode. Clamp to border pixel
            # centres first, then use its supported zero-padding kernel.
            bounded_grid=torch.stack((grid[...,0].clamp(-1+1/fw,1-1/fw),grid[...,1].clamp(-1+1/fh,1-1/fh)),-1)
            parts.append(F.grid_sample(f,bounded_grid,mode='bilinear',padding_mode='zeros',align_corners=False))
        out[:,:,lo:hi]=head(torch.cat(parts,1))
    return out

def prepare(path):
    import torch,jpeglib
    from PIL import Image
    with Image.open(path) as im:
        if im.format!='JPEG' or im.mode not in ('L','RGB'):raise ValueError('CAT-Net requires a grayscale or RGB JPEG.')
        orientation=int(im.getexif().get(274,1));rgb=np.array(im.convert('RGB'))
    h,w=rgb.shape[:2];ph,pw=(h+7)//8*8,(w+7)//8*8
    jpeg=jpeglib.read_dct(str(path));blocks=jpeg.Y
    if blocks is None:raise ValueError('JPEG luminance coefficients are unavailable.')
    coeff=blocks.transpose(0,2,1,3).reshape(blocks.shape[0]*8,blocks.shape[1]*8)[:ph,:pw]
    if coeff.shape!=(ph,pw):raise ValueError('Unexpected JPEG coefficient dimensions.')
    table=jpeg.qt[jpeg.quant_tbl_no[0]].astype(np.float32)[None,None]
    padded=np.full((ph,pw,3),127.5,np.float32);padded[:h,:w]=rgb
    normalized=(torch.tensor(padded.transpose(2,0,1),dtype=torch.float32)-127.5)/127.5
    # Exact one-hot equivalent of the published comparisons (|DCT| clipped at20).
    magnitude=np.minimum(np.abs(coeff.astype(np.int32)),20)
    volume=torch.from_numpy((np.arange(21)[:,None,None]==magnitude).astype(np.float32))
    return torch.cat((normalized,volume))[None],torch.from_numpy(table),dict(source_shape=[h,w],padded_shape=[ph,pw],orientation=orientation)

def orient(array,orientation):
    if orientation==2:return array[:,::-1].copy()
    if orientation==3:return array[::-1,::-1].copy()
    if orientation==4:return array[::-1].copy()
    if orientation==5:return array.T.copy()
    if orientation==6:return np.rot90(array,-1).copy()
    if orientation==7:return array.T[::-1,::-1].copy()
    if orientation==8:return np.rot90(array,1).copy()
    return array.copy()

def predict(path,model,bounded=None):
    import torch
    import torch.nn.functional as F
    image,table,meta=prepare(path);device=next(model.parameters()).device
    model.memory_bounded=image.shape[-2]*image.shape[-1]>4_000_000 if bounded is None else bounded
    meta['memory_bounded']=model.memory_bounded
    with torch.inference_mode():
        logits=model(image.to(device),table.to(device));prob=torch.softmax(logits[0],dim=0)[1]
        # Native output is quarter resolution, before any visualization resizing.
        raw=prob.cpu().numpy();full=F.interpolate(prob[None,None],size=tuple(meta['padded_shape']),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
    h,w=meta['source_shape'];return dict(map=orient(full[:h,:w],meta['orientation']),native_map=raw,metadata=meta)
