"""Current-image checks for wall shadow, open specular rims and thin tails."""
import cv2
import numpy as np
from scipy import ndimage as ndi


def wall_air_gap(analyzer,image,mask,occupied=None):
    """A bright rise between the wall and a thick body is resolved air/blur.

    Transparent thin tails are not judged by this thick-front test. A contact
    seed needs a flat or darkening near-wall intensity, not just a dark delta
    relative to the bright dry-wall reference.
    """
    wall=analyzer.wall
    selected=(wall.offsets>=1)&(wall.offsets<=3)
    near=cv2.remap(image.astype(np.float32),wall.map_x[:,selected],wall.map_y[:,selected],cv2.INTER_LINEAR)
    rise=near.max(axis=1)-near[:,0]
    if occupied is None:
        indices=np.flatnonzero(wall.offsets>=8)[::4]
        sample=cv2.remap(mask.astype(np.uint8),wall.map_x[:,indices],wall.map_y[:,indices],cv2.INTER_NEAREST)>.5
        height=np.where(sample&analyzer.ray_valid[:,indices],wall.offsets[indices][None,:],0).max(axis=1)
    else:height=np.where(occupied,wall.offsets[None,:],0).max(axis=1)
    noise=analyzer.air_gap_noise
    resolved=np.flatnonzero(height>=8)
    front=wall.s>wall.s[resolved[-1]]+2 if len(resolved) else np.zeros(len(wall.s),bool)
    gap=((height>=8)|front)&(rise>np.maximum(10,6*noise))
    return gap,rise,height


def remove_wall_shadow(analyzer,image,mask):
    # Once a long film is attached, a translucent near-wall layer can also
    # brighten outward. Keep its silhouette evidence; the separate contact
    # measurement still rejects an image-resolved air transition. Applying the
    # compact-drop shadow rule to the whole attached film erased valid gray rim.
    attached=False
    if mask.any():
        yy,xx=analyzer.optical_band_pixels
        ids=analyzer.optical_band_indices[mask[yy,xx]&(analyzer.wall.distance[yy,xx]<=2.5)]
        width=float(np.max(analyzer.wall.distance[mask]))
        if len(ids)>1 and np.ptp(analyzer.wall.s[ids])>=1.25*width:
            attached=True
    gap,rise,height=wall_air_gap(analyzer,image,mask)
    yy,xx=analyzer.optical_band_pixels;indices=analyzer.optical_band_indices
    removed=np.zeros_like(mask)
    if attached:
        resolved=np.flatnonzero(height>=8)
        front=analyzer.wall.s>analyzer.wall.s[resolved[-1]]+2 if len(resolved) else np.zeros(len(analyzer.wall.s),bool)
        # Only a thin, brightening extension ahead of the resolved front can
        # be removed; gray attached body/tail above it remains untouched.
        y,x=analyzer.front_band_pixels;idx=analyzer.front_band_indices
        removed[y,x]=mask[y,x]&(gap&front)[idx]
    else:removed[yy,xx]=mask[yy,xx]&gap[indices]
    return mask&~removed,removed


def complete_front_reflection(analyzer,image,mask):
    """Close a small bright opening with the wall included in the closure.

    The original cavity routine closed the liquid alone, so an opening next to
    the wall could never seal. Accept only mostly observed bright pockets near
    a resolved wall-attached body. Do not create contact pixels in the protected
    7 px wall band. A local observed-point hull only bounds candidate additions;
    it is never filled wholesale or used to interpolate output measurements.
    """
    wall=analyzer.wall;added=np.zeros_like(mask)
    if not np.any(mask&(wall.distance<=2.5)):return mask.copy(),added
    width=float(np.max(wall.distance[mask]))
    if width<12:return mask.copy(),added
    band_y,band_x=analyzer.optical_band_pixels
    ids=analyzer.optical_band_indices[mask[band_y,band_x]&(wall.distance[band_y,band_x]<=2.5)]
    # A compact drop at first contact has no long attached film to close. The
    # wall must not become an artificial edge of its otherwise round front.
    if len(ids)<2 or np.ptp(wall.s[ids])<1.25*width:return mask.copy(),added
    yy,xx=np.nonzero(mask&(wall.distance>=max(8,width*.3)))
    if not len(xx):return mask.copy(),added
    margin=23;x0=max(0,int(xx.min())-margin);x1=min(mask.shape[1],int(xx.max())+margin+1)
    y0=max(0,int(yy.min())-margin);y1=min(mask.shape[0],int(yy.max())+margin+1)
    local=mask[y0:y1,x0:x1];solid=wall.solid[y0:y1,x0:x1];reference=local|solid
    envelope=np.zeros(local.shape,np.uint8)
    ly,lx=np.nonzero(local)
    hull=cv2.convexHull(np.c_[lx,ly].astype(np.int32))
    cv2.fillConvexPoly(envelope,hull,1)
    evidence=solid|(local&(image[y0:y1,x0:x1]<235))
    direct=ndi.binary_fill_holes(reference)&~reference
    accepted=direct.copy()
    for radius in [8,12,20]:
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
        closed=cv2.morphologyEx(reference.astype(np.uint8),cv2.MORPH_CLOSE,kernel)>0
        candidate=ndi.binary_fill_holes(closed)&~reference&(envelope>0)
        count,labels,stats,_=cv2.connectedComponentsWithStats(candidate.astype(np.uint8),8)
        for label in range(1,count):
            area=stats[label,cv2.CC_STAT_AREA]
            if area<12 or area>2*width*width:continue
            x,y,w,h=stats[label,:4]
            if x==0 or y==0 or x+w>=local.shape[1] or y+h>=local.shape[0]:continue
            if max(stats[label,cv2.CC_STAT_WIDTH],stats[label,cv2.CC_STAT_HEIGHT])>3*width:continue
            region=labels==label;boundary=ndi.binary_dilation(region)&~region
            observed=np.count_nonzero(boundary&evidence)/max(1,boundary.sum())
            bright=np.mean(image[y0:y1,x0:x1][region]>=235)
            if observed>=.70 and bright>=.65:accepted|=region
    added[y0:y1,x0:x1]=accepted&(envelope>0)
    added&=analyzer.allowed&~mask&(wall.distance>7)
    return mask|added,added


def recover_thin_tail(analyzer,image,mask,neighbors):
    """Keep current optical evidence also supported on both temporal sides."""
    if len(neighbors)!=2:return mask.copy(),np.zeros_like(mask)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,9))
    past=cv2.dilate(neighbors[0].astype(np.uint8),kernel)>0
    future=cv2.dilate(neighbors[1].astype(np.uint8),kernel)>0
    delta=analyzer.background-image.astype(float)
    signal=delta>=np.maximum(analyzer.settings.dark_support,4*analyzer.noise)
    add=past&future&signal&analyzer.allowed&(analyzer.wall.distance<=6)&~mask
    # Separate unsupported dry-wall speckles cannot establish a new tail.
    count,labels,stats,_=cv2.connectedComponentsWithStats(add.astype(np.uint8),8)
    keep=stats[:,cv2.CC_STAT_AREA]>=3;keep[0]=False;add&=keep[labels]
    revised,shadow=remove_wall_shadow(analyzer,image,mask|add)
    add&=revised
    return revised,add


def trim_front_extension(analyzer,image,mask):
    """Reject a forward wall wedge beyond the current visible dark front rim.

    A filled bright wedge may be wider than the old protected 7 px wall band.
    Use the observed rim outside wall blur to bound only downstream additions;
    do not impose a round head or a curvature bound on its side lobes.
    """
    removed=np.zeros_like(mask);wall=analyzer.wall
    if not mask.any():return mask.copy(),removed
    width=float(wall.distance[mask].max())
    y,x=analyzer.optical_band_pixels;idx=analyzer.optical_band_indices[mask[y,x]&(wall.distance[y,x]<=2.5)]
    if len(idx)<2 or np.ptp(wall.s[idx])<1.25*width:return mask.copy(),removed
    yy,xx=np.nonzero(mask);dist=wall.distance[yy,xx]
    delta=analyzer.background[yy,xx]-image[yy,xx].astype(float)
    body=dist>=max(8,.25*width)
    if not body.any():return mask.copy(),removed
    contrast=float(np.percentile(delta[body],90))
    visible=body&(image[yy,xx]<235)&(delta>=np.maximum(max(analyzer.settings.dark_seed,.4*contrast),4*analyzer.noise[yy,xx]))
    if np.count_nonzero(visible)<12:return mask.copy(),removed
    # Fill the observed liquid alone first. Its enclosed white interior stays
    # liquid; a bright pocket closed only by the wall may instead be exterior
    # air below the observed rim. Remove that entire unsupported pocket rather
    # than cutting the head with an artificial straight line.
    front_y=float(yy[visible].max())
    x0=max(0,int(xx.min())-1);x1=min(mask.shape[1],int(xx.max())+2)
    y0=max(0,int(yy.min())-1);y1=min(mask.shape[0],int(yy.max())+2)
    local=mask[y0:y1,x0:x1];values=image[y0:y1,x0:x1]
    observed=local&(values<235)
    enclosed=ndi.binary_fill_holes(observed)
    candidate=local&~enclosed
    count,labels,stats,_=cv2.connectedComponentsWithStats(candidate.astype(np.uint8),8)
    for label in range(1,count):
        if stats[label,cv2.CC_STAT_AREA]<3:continue
        region=labels==label;ry,rx=np.nonzero(region)
        distances=wall.distance[y0+ry,x0+rx]
        if y0+ry.max()<=front_y+2 or distances.min()>7:continue
        if distances.max()>max(12,.65*width):continue
        if np.mean(values[region]>=235)<.8:continue
        removed[y0+ry,x0+rx]=True
    return mask&~removed,removed


def reject_transient_tail(analyzer,image,mask,neighbors):
    """Remove a weak upstream tail excursion absent on both temporal sides."""
    removed=np.zeros_like(mask)
    if len(neighbors)!=2 or not mask.any():return mask.copy(),removed
    wall=analyzer.wall;yy,xx=np.nonzero(mask);distance=wall.distance[yy,xx]
    body=distance>=8
    if not body.any():return mask.copy(),removed
    _,idx=analyzer.wall_tree.query(np.c_[xx,yy]);body_start=float(wall.s[idx[body]].min())
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,13))
    supported=(cv2.dilate(neighbors[0].astype(np.uint8),kernel)>0)|(cv2.dilate(neighbors[1].astype(np.uint8),kernel)>0)
    delta=analyzer.background[yy,xx]-image[yy,xx].astype(float)
    body_delta=delta[body];threshold=max(analyzer.settings.dark_seed,.48*float(np.percentile(body_delta,90)))
    weak=delta<threshold
    drop=(wall.s[idx]<body_start)&(distance<=6)&weak&~supported[yy,xx]
    removed[yy[drop],xx[drop]]=True
    return mask&~removed,removed


def limit_valid_view(results,wall,roi):
    """End formal measurements before the main silhouette reaches wall end."""
    bottom=min(float(wall.points[-1,1]),float(roi[3]-1))
    first=next((i for i,r in enumerate(results) if r['phase']=='postimpact' and r.get('bbox') and r['bbox'][3]-1>=bottom-2),None)
    for i,r in enumerate(results):
        r['within_valid_view']=first is None or i<first
        if not r['within_valid_view']:
            r['phase']='after_valid_view'
            r['pre_truncation_measurement_status']=r['measurement_status']
            r['measurement_status']='outside_valid_wall_view'
            r['flags']=sorted(set(r['flags']+['outside_valid_wall_view']))
    return dict(status='ended_at_wall_boundary' if first is not None else 'complete_source_sequence',
        first_excluded_frame=results[first]['frame'] if first is not None else None,
        last_valid_frame=results[first-1]['frame'] if first is not None and first else results[-1]['frame'] if first is None else None,
        wall_end_y_px=bottom,boundary_tolerance_px=2,
        definition='Formal series ends before the first postimpact main silhouette reaches the lower valid wall/ROI boundary. Later source frames and provisional geometry remain inspectable, excluded from postimpact CSV and plots.')
