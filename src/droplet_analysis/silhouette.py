"""Complete bright cavities through a bounded, mostly observed boundary."""
import cv2
import numpy as np
from scipy import ndimage as ndi


def complete_bright_cavities(mask, image, distance, allowed, radius, min_boundary_fraction=.75,
                             solid=None, attachment_tolerance=2.5):
    """Bridge a narrow bright opening only if it seals an interior cavity.

    Airborne components are processed separately. Wall-supported fragments can
    belong to the same bright hollow rim and share a wall closure. Closing is not
    accepted as liquid: most of the added region's perimeter must already be
    observed liquid, and the bridge must be bright and outside the contact band.
    Open notches, isolated bright patches and separate drops are not joined.
    """
    repaired = np.zeros_like(mask)
    if radius <= 0:
        return mask.copy(), repaired
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
    h,w = mask.shape
    attached = np.unique(labels[mask & (distance<=attachment_tolerance)]) if solid is not None else np.array([],int)
    attached = attached[attached>0]
    groups = [(np.array([label]),False) for label in range(1,count) if label not in attached and stats[label,4]>=50]
    if len(attached):
        groups.append((attached,True))
    for members,wall_supported in groups:
        selected=stats[members]
        x,y=selected[:,0].min(),selected[:,1].min()
        bw,bh=(selected[:,0]+selected[:,2]).max()-x,(selected[:,1]+selected[:,3]).max()-y
        x0,y0,x1,y1 = max(0,x-radius-1),max(0,y-radius-1),min(w,x+bw+radius+1),min(h,y+bh+radius+1)
        component = np.isin(labels[y0:y1,x0:x1],members)
        closed = cv2.morphologyEx(component.astype(np.uint8),cv2.MORPH_CLOSE,kernel).astype(bool)
        bridge = (closed & ~component & (image[y0:y1,x0:x1] >= 235) &
                  (distance[y0:y1,x0:x1] > 7) & allowed[y0:y1,x0:x1])
        reference = component | (solid[y0:y1,x0:x1] if wall_supported else False)
        joined = reference | bridge
        enclosed = ndi.binary_fill_holes(joined) & ~joined
        if not enclosed.any():
            continue
        additions = bridge | enclosed
        n, regions = cv2.connectedComponents(additions.astype(np.uint8),connectivity=8)
        for index in np.unique(regions[enclosed]):
            region = regions == index
            if not index or np.count_nonzero(region & enclosed) < 12:
                continue
            boundary = ndi.binary_dilation(region) & ~region
            observed = np.count_nonzero(boundary & reference)/max(1,np.count_nonzero(boundary))
            if observed >= min_boundary_fraction:
                repaired[y0:y1,x0:x1] |= region
    repaired &= allowed & ~mask
    return mask | repaired, repaired
