"""Bidirectional evidence for saturated-boundary completion and event checks."""
import cv2
import numpy as np
from scipy import ndimage as ndi
from .core import contours_of, intervals
from .silhouette import complete_bright_cavities


class ContactTracker:
    """Forward-only contact hysteresis with independent image-based seeds.

    Weak optical bridges cannot initiate attachment. A seeded interval may
    continue while a resolved, overlapping interval remains in the next frame.
    A disappearance or missing frame resets it; future contact never promotes
    earlier shadow frames. No sequence name, frame offset, or annotation enters
    this decision.
    """
    def __init__(self):
        self.previous_frame = None
        self.previous_segments = []

    def update(self, analyzer, result, mask, aligned, height, possible, confident):
        wall = analyzer.wall
        seeds = analyzer.contact_seed_profile(mask, analyzer.background-aligned.astype(np.float32))
        previous = self.previous_segments if self.previous_frame == result['frame']-1 else []
        candidates = result['contact_segments']
        accepted = []
        possible_runs = intervals(possible)
        for segment in candidates:
            start, end = segment['s_start_px'], segment['s_end_px']
            # Components of the possible-contact profile are separated by real
            # resolved gaps. Thin sections within one component share evidence.
            group = next(((a,b) for a,b in possible_runs if wall.s[b] >= start and wall.s[a] <= end), None)
            gs,ge = (wall.s[group[0]],wall.s[group[1]]) if group else (start,end)
            select = (wall.s >= gs) & (wall.s <= ge) & confident
            seeded = any(b-a+1 >= analyzer.settings.min_contact_length_px for a,b in intervals(seeds & select))
            overlaps = any(min(ge, old['s_end_px'])-max(gs, old['s_start_px']) >= analyzer.settings.min_contact_length_px
                           for old in previous)
            candidate = dict(segment, strong_support_length_px=int(np.count_nonzero(seeds & select)),
                             contact_evidence='seeded' if seeded else 'continued' if overlaps else 'weak_only')
            if seeded or overlaps:
                accepted.append(candidate)
        result['contact_candidates'] = [dict(s) for s in candidates]
        result['unconfirmed_contact_candidate_count'] = len(candidates)-len(accepted)
        # Preserve possible endpoint ranges only for accepted components.
        accepted_mask = np.zeros_like(possible)
        for start,end in intervals(possible):
            if any(s['s_start_px'] <= wall.s[end] and s['s_end_px'] >= wall.s[start] for s in accepted):
                accepted_mask[start:end+1] = True
        possible = possible & accepted_mask
        confident = confident & accepted_mask
        height,maximum,trunc = analyzer.outer_thickness(mask,confident)
        result.update(contact_segments=accepted, contact_length_px=sum(s['length_px'] for s in accepted),
                      confident_contact_length_px=float(confident.sum()),
                      possible_contact_length_px=float(possible.sum()),
                      uncertain_contact_length_px=float(np.count_nonzero(possible & ~confident)),
                      maximum_visible_thickness=maximum)
        result['flags'] = [f for f in result['flags'] if f not in ['contact_boundary_uncertain','thickness_contact_uncertain','normal_search_truncated','thickness_outer_envelope']]
        if result['unconfirmed_contact_candidate_count']:
            result['flags'].append('contact_evidence_weak')
        if result['uncertain_contact_length_px'] > 8:
            result['flags'].append('contact_boundary_uncertain')
        if maximum and not maximum['supported_contact']:
            result['flags'].append('thickness_contact_uncertain')
        if maximum and maximum['spans_visible_gap']:
            result['flags'].append('thickness_outer_envelope')
        if trunc:
            result['flags'].append('normal_search_truncated')
        self.previous_frame, self.previous_segments = result['frame'], accepted
        return height, possible, confident


def warp_neighbor(current, neighbor, neighbor_mask, box):
    x0,y0,x1,y1 = box
    a = current[y0:y1,x0:x1]
    b = neighbor[y0:y1,x0:x1]
    flow = cv2.calcOpticalFlowFarneback(a,b,None,.5,3,21,4,7,1.5,0)
    yy,xx = np.indices(a.shape,dtype=np.float32)
    return cv2.remap(neighbor_mask[y0:y1,x0:x1].astype(np.uint8),xx+flow[...,0],yy+flow[...,1],cv2.INTER_NEAREST).astype(bool)


def repair_highlights(analyzer, current, current_mask, neighbors, weights=(.5,.5)):
    """Infer bright missing outer boundaries supported by past AND future.

    Direct boundary additions stay outside the contact band. Enclosed interiors
    may extend to the wall; measurement separately preserves spatial contact
    evidence. Caller supplies one contiguous neighbor on each temporal side.
    """
    if len(neighbors) != 2 or not np.any(current_mask):
        return current_mask.copy(), np.zeros_like(current_mask)
    all_masks = current_mask | neighbors[0][1] | neighbors[1][1]
    yy,xx = np.nonzero(all_masks)
    h,w = current.shape
    box = [max(0,int(xx.min())-24),max(0,int(yy.min())-24),min(w,int(xx.max())+25),min(h,int(yy.max())+25)]
    x0,y0,x1,y1 = box
    past = warp_neighbor(current,*neighbors[0],box)
    future = warp_neighbor(current,*neighbors[1],box)
    local = current_mask[y0:y1,x0:x1]
    eligible = ((current[y0:y1,x0:x1] >= 235) &
                (analyzer.wall.distance[y0:y1,x0:x1] > 7) &
                analyzer.allowed[y0:y1,x0:x1])
    # Small closing of the two predictions tolerates subpixel flow differences.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    past = cv2.morphologyEx(past.astype(np.uint8),cv2.MORPH_CLOSE,kernel).astype(bool)
    future = cv2.morphologyEx(future.astype(np.uint8),cv2.MORPH_CLOSE,kernel).astype(bool)
    # Work with boundary distances, not a pixelwise intersection that follows
    # flow noise into the bright cavity. Both predictions must support the
    # boundary within a small geometric tolerance. Actual temporal distances
    # determine the interpolation weights when anchors are asymmetric.
    def signed(mask):
        distance=ndi.distance_transform_edt(mask)-ndi.distance_transform_edt(~mask)
        return ndi.gaussian_filter(distance,1.)
    dp,df=signed(past),signed(future)
    tolerance=analyzer.settings.temporal_boundary_tolerance_px
    consensus=(weights[0]*dp+weights[1]*df>=0) & (dp>=-tolerance) & (df>=-tolerance)
    add = consensus & eligible & ~local
    # All additions must be near current-frame liquid evidence.
    missing_depth=ndi.distance_transform_edt(~local)
    add &= missing_depth <= analyzer.settings.temporal_evidence_distance_px
    # A subpixel mismatch around an otherwise complete boundary is not a
    # missing cavity. Require a connected missing patch with depth beyond the
    # prediction tolerance; this prevents repeated outward contour inflation.
    count,labels,stats,_=cv2.connectedComponentsWithStats(add.astype(np.uint8),connectivity=8)
    substantial=np.unique(labels[add & (missing_depth>tolerance+1)])
    keep=np.isin(np.arange(count),substantial) & (stats[:,cv2.CC_STAT_AREA]>=analyzer.settings.min_component_area)
    keep[0]=False
    add=keep[labels]
    if not add.any():
        return current_mask.copy(),np.zeros_like(current_mask)
    repaired = current_mask.copy()
    repaired[y0:y1,x0:x1] |= add
    repaired = ndi.binary_fill_holes(repaired | analyzer.wall.solid) & analyzer.allowed
    # A partial temporal bridge can close a bright cavity that the spatial pass
    # could not close. Reuse exactly the same boundary-evidence requirement.
    repaired,_=complete_bright_cavities(repaired,current,analyzer.wall.distance,analyzer.allowed,
                                      analyzer.settings.highlight_gap_radius,analyzer.settings.cavity_boundary_fraction,
                                      analyzer.wall.solid,analyzer.settings.wall_uncertainty_px+.5)
    return repaired, repaired & ~current_mask


def update_geometry(analyzer, result, mask, aligned, repair, contact_mask=None):
    """Re-measure after a contour correction; all derived geometry is refreshed."""
    delta = analyzer.background - aligned.astype(np.float32)
    segments, maximum, height, possible, confident, trunc = analyzer.measure(mask, delta,contact_mask)
    yy,xx = np.nonzero(mask)
    result.update(area_px2=int(len(xx)), contours=contours_of(mask),
        centroid=[float(xx.mean()),float(yy.mean())] if len(xx) else None,
        bbox=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)] if len(xx) else None,
        contact_segments=segments, contact_length_px=sum(s['length_px'] for s in segments),
        confident_contact_length_px=float(np.count_nonzero(confident)),
        uncertain_contact_length_px=float(np.count_nonzero(possible & ~confident)),
        possible_contact_length_px=float(np.count_nonzero(possible)),
        maximum_visible_thickness=maximum,
        minimum_wall_gap_px=float(np.min(analyzer.wall.distance[mask])) if len(xx) else None,
        temporal_repair_area_px2=int(np.count_nonzero(repair)))
    result['flags'] = [f for f in result['flags'] if f not in ['contact_boundary_uncertain','thickness_contact_uncertain','normal_search_truncated']]
    if result['uncertain_contact_length_px'] > 8:
        result['flags'].append('contact_boundary_uncertain')
    if maximum and not maximum['supported_contact']:
        result['flags'].append('thickness_contact_uncertain')
    if trunc:
        result['flags'].append('normal_search_truncated')
    if np.count_nonzero(repair) > 20:
        result['flags'].append('temporal_highlight_repair')
    return height,possible,confident


def select_anchors(results, index, radius=4):
    """Use nearest low-reconstruction frames on both sides of a bright gap."""
    if index == 0 or index == len(results)-1:
        return None
    chosen=[]
    for direction in [-1,1]:
        fallback=index+direction
        selected=fallback
        for distance in range(1,radius+1):
            j=index+direction*distance
            if j<0 or j>=len(results):
                break
            if results[j]['frame'] != results[index]['frame']+direction*distance:
                break
            neighborhood=[r['area_px2'] for r in results[max(0,j-radius):min(len(results),j+radius+1)]]
            typical=float(np.percentile(neighborhood,75))
            if results[j]['repaired_boundary_area_px2'] <= 20 and results[j]['area_px2'] >= .9*typical:
                selected=j
                break
        if results[fallback]['frame'] != results[index]['frame']+direction:
            return None
        chosen.append(selected)
    return chosen


def audit_sequence(results, settings, fps=None):
    """Flag temporal inconsistencies without replacing measurements by averages."""
    for previous,current in zip(results,results[1:]):
        if current['frame'] != previous['frame'] + 1:
            current['flags'].append('missing_previous_frame')
            continue
        a,b = previous['area_px2'],current['area_px2']
        if min(a,b) > 100 and abs(a-b)/max(a,b) > .2:
            current['flags'].append('area_jump')
        p,c = previous['contact_segments'],current['contact_segments']
        if p and c:
            movement = np.linalg.norm(np.array(current['centroid'])-np.array(previous['centroid']))
            jump = max(abs(c[0]['s_start_px']-p[0]['s_start_px']),abs(c[-1]['s_end_px']-p[-1]['s_end_px']))
            if jump > max(8,2*movement+4):
                current['flags'].append('contact_endpoint_jump')
        hp,hc = previous['maximum_visible_thickness'],current['maximum_visible_thickness']
        if hp and hc and abs(hp['value_px']-hc['value_px']) > max(5,hp['value_px']*.2):
            current['flags'].append('thickness_jump')
    positive = [r['confident_contact_length_px'] >= settings.min_contact_length_px for r in results]
    first = None
    run = settings.persistence_frames
    for i in range(len(results)-run+1):
        consecutive = all(results[j]['frame'] == results[i]['frame']+j-i for j in range(i,i+run))
        if consecutive and all(positive[i:i+run]):
            first = i
            break
    event = {'status': 'not_detected', 'first_contact_frame': None, 'previous_frame': None,
             'bracket_frames': None, 'manual_reviewed': False,
             'definition': 'First sustained image-resolved contact; optical evidence, not microscopic wetting.'}
    if first is not None:
        frame = results[first]['frame']
        before = results[first-1]['frame'] if first else None
        event.update(status='candidate_requires_review',first_contact_frame=frame,previous_frame=before,
                     bracket_frames=[before,frame],confirmation_frames=[r['frame'] for r in results[first:first+run]])
        if before is None:
            event['status'] = 'contact_present_at_sequence_start'
        event['earlier_transient_candidates'] = [results[i]['frame'] for i in range(first) if positive[i]]
        for r in results:
            r['relative_time_s'] = (r['frame']-frame)/fps if fps and fps>0 else None
            r['impact_candidate'] = r['frame'] == frame
    else:
        for r in results:
            r['relative_time_s'] = None
            r['impact_candidate'] = False
    for r in results:
        r['flags'] = sorted(set(r['flags']))
    return event
