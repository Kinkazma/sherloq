"""Bounded exact deduplication and display sampling of a dense match field."""
import numpy as np
from .cloning import check


def sample_unique_links(targets,squared,selected,limit,*,cancel=lambda:False,checkpoint=lambda:None,block_size=65536):
    """Each source has one target: a duplicate can only be its reverse link.

    Preserve the former stable distance sort tie rule, then ascending source
    order and linspace display sampling. No global pair table/sort is needed.
    The selection itself and the dense field remain unchanged.
    """
    target=targets.reshape(-1);distance=squared.reshape(-1);selection=selected.reshape(-1)
    def accepted():
        for start in range(0,len(target),block_size):
            check(cancel);rows=np.flatnonzero(selection[start:start+block_size])+start
            other=target[rows]
            # Defensive validity keeps this helper safe for external callers.
            valid=(other>=0)&(other<len(target));rows=rows[valid];other=other[valid]
            reciprocal=(selection[other]!=0)&(target[other]==rows)
            better=(distance[other]<distance[rows])|((distance[other]==distance[rows])&(other<rows))
            yield rows[~(reciprocal&better)]
            checkpoint()
    total=sum(len(rows) for rows in accepted())
    cap=max(1,int(limit));ranks=np.linspace(0,total-1,min(cap,total),dtype=np.int64) if total else np.empty(0,np.int64)
    sampled=np.empty(len(ranks),np.int64);offset=0;written=0
    for rows in accepted():
        stop=np.searchsorted(ranks,offset+len(rows),side='left')
        sampled[written:stop]=rows[ranks[written:stop]-offset]
        written=stop;offset+=len(rows)
    return sampled,total
