"""Independent clone and ELA evidence on original or cached ELA pixels."""
import hashlib
import colorsys
import cv2 as cv
import numpy as np
from .automatic_clones import SOURCES as CLONE_SOURCES, entries as clone_entries, render as render_clones
from .ela_biomes import segment
from .ela_energy import energy_color

ELA_SOURCE = 'ELA biomes'
SOURCES = (*CLONE_SOURCES, ELA_SOURCE)


def ela_entries(base, threshold, minimum, regions, excluded, image_shape, energy_thresholds=None):
    if base is None:return (),None
    b=base['metadata']['block'];rows,cols=base['score'].shape
    selected=np.zeros(image_shape[:2],np.uint8)
    for p in regions:cv.fillPoly(selected,[np.rint(p).astype(np.int32)],1)
    for p in excluded:cv.fillPoly(selected,[np.rint(p).astype(np.int32)],0)
    cells=selected[:rows*b,:cols*b].reshape(rows,b,cols,b).all(axis=(1,3))
    result=segment({**base,'supported':base['supported']&cells,'energy_allowed':selected.astype(bool)},threshold,minimum,energy_thresholds=energy_thresholds)
    result['metadata']={**result['metadata'],'analysis_scope':'full_image_reference_profiles_selected_complete_cells',
                        'energy_analysis_scope':'detected_panel_reference_selected_pixels',
                        'selected_regions':regions,'excluded_regions':excluded}
    entries=[]
    for region in result['metadata']['regions']:
        if 'kind' in region:
            x,y,w,h=region['bbox']
            mask=(result['energy_labels'][y:y+h,x:x+w]==region['id']).astype(np.uint8)
            key=f"ela-energy-{region['energy_region']}-{region['kind']}"
            color=energy_color(region)
            contours,_=cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
            entries.append(dict(id=key,source=ELA_SOURCE,label=region['sources'][0],kind=region['kind'],
                count=region['pixels'],color=color,pixel_mask=mask,origin=(x,y),
                polygons=[(p.reshape(-1,2)+(x,y)).tolist() for p in contours],
                provenance=dict(region=region['id'],score=region['score'],sources=region['sources'],
                    dominant_descriptor=region['dominant_descriptor'],classification='descriptive_energy_contrast',energy_region=region['energy_region'],components=region['components'])))
            continue
        coords=np.argwhere(result['labels']==region['id']).astype(np.int32)
        key='ela-'+hashlib.sha256(coords.tobytes()+str(b).encode()).hexdigest()[:24]
        # Stable hue, independent of the blue UI extension marker.
        hue=int(key[-8:],16)/2**32
        color=tuple(round(v*255) for v in colorsys.hsv_to_rgb(hue,.7,.95)[::-1])
        mask=(result['labels']==region['id']).astype(np.uint8)
        contours,_=cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        # Polygons are only descriptive. Rendering/picking use exact cell sets.
        polygons=[(p.reshape(-1,2)*b+b/2).tolist() for p in contours]
        entries.append(dict(id=key,source=ELA_SOURCE,count=len(coords),color=color,
            cells=coords,block=b,polygons=polygons,provenance=dict(region=region['id'],
                score=region['score'],sources=region.get('sources',['ELA']),dominant_descriptor=region['dominant_descriptor'])))
    return tuple(entries),result


def prepare(request):
    patch,forge,base,low,high,overlap,threshold,minimum,regions,excluded,shape=request[:11]
    energy_thresholds=request[11] if len(request)>11 else None
    clones=clone_entries(patch,forge,low,high,overlap)
    ela,result=ela_entries(base,threshold,minimum,regions,excluded,shape,energy_thresholds)
    return clones+ela,result


class CompletePreparation:
    """One retained result per branch; LatestJob serializes calls per panel.

    Detector dictionaries are immutable after publication. Keep strong references
    so identity keys cannot be reused after a detector result is released.
    """
    def __init__(self):
        self.clone_key=None;self.clone_inputs=None;self.clones=()
        self.ela_key=None;self.ela_base=None;self.ela=((),None)
        self.counts=dict(clones=0,ela=0)
    def __call__(self,request):
        patch,forge,base,low,high,overlap,threshold,minimum,regions,excluded,shape=request[:11]
        energy=request[11] if len(request)>11 else None
        key=(id(patch),id(forge),low,high,overlap)
        if key!=self.clone_key:
            clones=clone_entries(patch,forge,low,high,overlap)
            self.clone_inputs=(patch,forge);self.clones=clones;self.clone_key=key
            self.counts['clones']+=1
        key=(id(base),threshold,minimum,regions,excluded,shape,energy)
        if key!=self.ela_key:
            value=ela_entries(base,threshold,minimum,regions,excluded,shape,energy)
            self.ela_base=base;self.ela=value;self.ela_key=key;self.counts['ela']+=1
        entries,result=self.ela
        return self.clones+entries,result


def render(request):
    image,biomes,excluded=request[:3]
    ela,mode=request[3:] if len(request)>3 else (None,0)
    if ela is not None and mode in (1,2,3,4):image=ela
    if mode==2:return image.copy()
    out=render_clones((image,tuple(e for e in biomes if e['source']!=ELA_SOURCE),excluded))
    cell_boundaries=[]
    for entry in biomes:
        if entry['source']!=ELA_SOURCE:continue
        if 'pixel_mask' in entry:
            x,y=entry['origin'];mask=entry['pixel_mask'].copy();h,w=mask.shape
            for p in excluded:cv.fillPoly(mask,[np.rint(np.asarray(p)-(x,y)).astype(np.int32)],0)
            roi=out[y:y+h,x:x+w];keep=mask.astype(bool)
            mixed=cv.addWeighted(roi,.72,np.full_like(roi,entry['color']),.28,0)
            roi[keep]=mixed[keep]
            contours,_=cv.findContours(mask,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
            border=np.zeros_like(mask);cv.drawContours(border,contours,-1,1,1)
            roi[(border&mask).astype(bool)]=entry['color']
            continue
        b=entry['block'];mask=np.zeros(image.shape[:2],np.uint8)
        for y,x in entry['cells']:mask[y*b:(y+1)*b,x*b:(x+1)*b]=1
        for p in excluded:cv.fillPoly(mask,[np.rint(p).astype(np.int32)],0)
        for y,x in entry['cells']:
            roi=out[y*b:(y+1)*b,x*b:(x+1)*b]
            alpha=cv.addWeighted(roi,.72,np.full_like(roi,entry['color']),.28,0)
            keep=mask[y*b:(y+1)*b,x*b:(x+1)*b].astype(bool);roi[keep]=alpha[keep]
        contours,_=cv.findContours(mask,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
        # Draw only inside the selected evidence, preserving holes/exclusions.
        outline=np.zeros_like(mask);cv.drawContours(outline,contours,-1,1,1)
        cell_boundaries.append(((outline&mask).astype(bool),entry['color']))
    # Energy layers can overlap cell biomes. Keep the complete cell perimeter
    # above all fills, including inner holes and boundaries created by exclusions.
    for boundary,color in cell_boundaries:out[boundary]=color
    return out
