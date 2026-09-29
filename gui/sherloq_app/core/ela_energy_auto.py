"""Exploratory, scale-independent automatic settings; not a forgery classifier."""
import numpy as np

# Provisional conservative calibration: same estimators, three global constants.
CONSERVATIVE_TAIL_FACTOR = .2
CONSERVATIVE_SHADOW_GAIN = 2.05
CONSERVATIVE_HIGHLIGHT_GAIN = 2.47


def estimate(base, cancel=lambda: False):
    from .ela_biomes import check
    from .ela_energy import ENERGY_FLOOR
    check(cancel)
    plane=np.median(base['energy_planes'],axis=0)
    scope=base['energy_scope'];samples=[]
    # Deterministic area-proportional sampling after per-panel normalization.
    step=max(1,int(np.ceil(plane.size/131072)))
    flat=plane.ravel()[::step];ids=scope.ravel()[::step]
    for region in base['energy_summary']:
        check(cancel)
        values=flat[ids==region['id']]
        if values.size<32:continue
        center=float(np.median(values))
        samples.append(np.log((values+ENERGY_FLOOR)/(center+ENERGY_FLOOR)))
    fallback=dict(quantiles=[.1,.9],thresholds=[2.,2.],method='quantile_tail_knees_v1',status='insufficient_spread',sample_count=0)
    if not samples:return fallback
    values=np.concatenate(samples);q=np.quantile(values,np.linspace(0,1,1001))
    fallback['sample_count']=int(values.size)
    if q[950]-q[50]<.08:return fallback
    # Chord distance on normalized quantile tails. Unlike a drawn angle this
    # criterion is invariant to graph aspect ratio and log-energy translation.
    def knee(tail,reverse=False):
        y=tail[::-1] if reverse else tail
        span=abs(float(y[-1]-y[0]))
        if span<.08:return 100,0.
        y=(y-y[0])/(y[-1]-y[0]);x=np.linspace(0,1,len(y))
        delta=y-x;index=int(np.argmax(delta))
        prominence=float(delta[index])
        return (index if prominence>=.1 and 1<=index<=99 else 100),prominence
    lo,lp=knee(q[:101]);hi,hp=knee(q[900:],True)
    bounds=(lo/1000,1-hi/1000)
    check(cancel)
    return dict(quantiles=list(bounds),thresholds=[2.,2.],
                method=fallback['method'],status='estimated',sample_count=int(values.size),
                tail_prominence=[lp,hp])


def deviations(maps,cancel=lambda:False):
    """Two-class Otsu on each positive score distribution, independently."""
    from skimage.filters import threshold_otsu
    from .ela_biomes import check
    thresholds=[]
    for key in ('energy_low_score','energy_high_score'):
        check(cancel)
        raw=maps[key].ravel();step=max(1,int(np.ceil(raw.size/131072)))
        positive=raw[::step];positive=positive[positive>0]
        if positive.size<256 or np.quantile(positive,.95)<1:
            thresholds.append(2.);continue
        # Bound the influence of isolated extreme scores before histogramming.
        values=np.minimum(positive,np.quantile(positive,.999))
        cutoff=float(threshold_otsu(values,nbins=256)) if np.ptp(values)>1e-6 else float(values[0])
        sensitive=round(float(np.clip(cutoff,1.,8.)),1)
        thresholds.append(sensitive)
    return thresholds
