from pathlib import Path
import cv2
import numpy as np
from .core import save_image

COLORS = {'wall': (190, 90, 190), 'contour': (130, 160, 0), 'contact': (240, 150, 10),
          'start': (240, 50, 20), 'end': (0, 190, 240), 'thickness': (40, 40, 230),
          'uncertain': (0, 155, 245)}


def point(value):
    return tuple(np.rint(value).astype(int))


def dashed_line(image, a, b, color, thickness=1):
    a,b = np.asarray(a),np.asarray(b)
    length = np.linalg.norm(b-a)
    if length < 1:
        return
    for start in np.arange(0, length, 5):
        cv2.line(image, point(a+(b-a)*start/length), point(a+(b-a)*min(start+2,length)/length), color, thickness, cv2.LINE_AA)


def overlay(raw, wall, result, measurements=True):
    image = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    cv2.polylines(image, [np.rint(wall.points).astype(np.int32)], False, COLORS['wall'], 1, cv2.LINE_AA)
    for contour in result['contours']:
        cv2.polylines(image, [np.array(contour, np.int32)], True, COLORS['contour'], 1, cv2.LINE_AA)
    if not measurements:
        return image
    for segment in result['contact_segments']:
        select = (wall.s >= segment['s_start_px']) & (wall.s <= segment['s_end_px'])
        cv2.polylines(image, [np.rint(wall.points[select]).astype(np.int32)], False, COLORS['contact'], 2, cv2.LINE_AA)
        for key in ['start_range','end_range']:
            dashed_line(image, *segment[key], COLORS['uncertain'], 2)
        cv2.circle(image, point(segment['start']), 3, COLORS['start'], -1, cv2.LINE_AA)
        cv2.circle(image, point(segment['end']), 3, COLORS['end'], -1, cv2.LINE_AA)
    maximum = result['maximum_visible_thickness']
    if maximum:
        cv2.line(image, point(maximum['start']), point(maximum['end']), COLORS['thickness'], 2, cv2.LINE_AA)
    return image


def sheet(rows, path, title, width=1000):
    panels = []
    for frame, raw, wall, result in rows:
        h,w = raw.shape
        box = result.get('bbox')
        if box:
            left,top,right,bottom = box
            # Include wall endpoints even when a faint film extends above the body.
            for seg in result['contact_segments']:
                top = min(top, int(seg['start'][1]))
                bottom = max(bottom, int(seg['end'][1])+1)
                right = max(right, int(seg['start'][0])+1, int(seg['end'][0])+1)
            left,top = max(0,left-25),max(0,top-25)
            right,bottom = min(w,right+55),min(h,bottom+25)
        else:
            left,top,right,bottom = int(w*.35),int(h*.3),int(w*.65),int(h*.8)
        crops = [cv2.cvtColor(raw,cv2.COLOR_GRAY2BGR)[top:bottom,left:right],
                 overlay(raw,wall,result,False)[top:bottom,left:right],
                 overlay(raw,wall,result,True)[top:bottom,left:right]]
        height = min(420, max(200, bottom-top))
        panel = np.full((height+52,width,3),250,np.uint8)
        col_width = width//3
        for i,crop in enumerate(crops):
            scale = min((col_width-16)/crop.shape[1],height/crop.shape[0])
            resized = cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST if scale>1 else cv2.INTER_AREA)
            rh,rw = resized.shape[:2]
            x=i*col_width+(col_width-rw)//2
            panel[52:52+rh,x:x+rw] = resized
            cv2.putText(panel,['Raw','External silhouette','Contact + visible thickness'][i],(i*col_width+10,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(30,30,30),1,cv2.LINE_AA)
        status = ', '.join(result['flags']) or 'no automatic flags (not human verified)'
        cv2.putText(panel,f'frame {frame}: {status}',(10,43),cv2.FONT_HERSHEY_SIMPLEX,.43,(70,70,130),1,cv2.LINE_AA)
        panels.append(panel)
    header = np.full((55,width,3),250,np.uint8)
    cv2.putText(header,title,(12,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(30,30,30),1,cv2.LINE_AA)
    cv2.putText(header,'Green: liquid | pink: wall | blue/yellow: endpoints | orange: uncertain | red: thickness',(12,45),cv2.FONT_HERSHEY_SIMPLEX,.42,(60,60,60),1,cv2.LINE_AA)
    save_image(path,np.vstack([header]+panels))
