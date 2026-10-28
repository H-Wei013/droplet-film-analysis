"""Derived impact origin; does not change segmentation, timing or geometry."""
import numpy as np

EXTENSION='impact-contact-arc-midpoint-v1'
DEFINITION='Wall arc midpoint of the contact endpoint span in the selected first-impact frame. Not the straight-line midpoint of endpoint coordinates.'
COLUMNS=['impact_frame','impact_point_status','impact_point_x_px','impact_point_y_px','impact_point_s_px','impact_point_s_mm','impact_point_raw_x_px','impact_point_raw_y_px']


def impact_point(manifest):
    number=manifest['impact']['first_contact_frame']
    result=dict(definition=DEFINITION,frame=number,status='impact_not_detected',
        coordinate_system='aligned_reference_px',point_px=None,s_px=None,s_mm=None,raw_point_px=None,
        source='contact_endpoint_span_arc_midpoint')
    if number is None:return result
    frame=next((r for r in manifest['frames'] if r['frame']==number),None)
    if frame is None:result['status']='impact_frame_missing';return result
    segments=frame['contact_segments']
    if not segments:result['status']='no_contact_in_impact_frame';return result
    start=min(s['s_start_px'] for s in segments);end=max(s['s_end_px'] for s in segments)
    middle=(start+end)/2
    wall=manifest['wall'];ss=np.asarray(wall['s_px']);points=np.asarray(wall['points'])
    if not ss[0]<=start<=end<=ss[-1]:raise ValueError('Impact contact span exceeds reference wall')
    point=[float(np.interp(middle,ss,points[:,j])) for j in [0,1]]
    shift=frame.get('registration_translation_px');scale=(manifest.get('physical_calibration') or {}).get('mm_per_px')
    result.update(status='defined',point_px=point,s_px=float(middle),s_mm=float(middle*scale) if scale is not None else None,
        raw_point_px=[float(point[j]+shift[j]) for j in [0,1]] if shift is not None else None,
        contact_start_s_px=float(start),contact_end_s_px=float(end))
    return result


def attach_impact_point(manifest):
    manifest['impact_point']=impact_point(manifest)
    extensions=manifest.setdefault('data_extensions',[])
    if EXTENSION not in extensions:extensions.append(EXTENSION)
    manifest.setdefault('definitions',{})['impact_point']=DEFINITION
    return manifest


def csv_fields(point):
    xy=point.get('point_px') or [None,None];raw=point.get('raw_point_px') or [None,None]
    return dict(impact_frame=point.get('frame'),impact_point_status=point['status'],
        impact_point_x_px=xy[0],impact_point_y_px=xy[1],impact_point_s_px=point.get('s_px'),
        impact_point_s_mm=point.get('s_mm'),impact_point_raw_x_px=raw[0],impact_point_raw_y_px=raw[1])
