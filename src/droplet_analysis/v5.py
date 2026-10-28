"""One complete main film, endpoint-span contact, and solid-safe normal rays."""
from copy import deepcopy
import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from .core import Analyzer as SpatialAnalyzer, contours_of, intervals, keep_runs
from .silhouette import complete_bright_cavities
from .optical import remove_wall_shadow,complete_front_reflection,wall_air_gap,trim_front_extension


def segment(wall, a, b):
    start, end = wall.point_at(a), wall.point_at(b)
    return dict(s_start_px=float(a), s_end_px=float(b), start=start, end=end,
                length_px=float(b-a), start_range=[start,start], end_range=[end,end])


class Analyzer(SpatialAnalyzer):
    def __init__(self, references, settings):
        super().__init__(references, settings)
        # A ray cannot go through solid or leave the image and re-enter it.
        wall = self.wall
        solid = wall.sample(wall.solid, nearest=True) > .5
        blocked = solid & (wall.offsets[None,:] > 1)
        self.ray_valid = np.logical_and.accumulate(wall.sample_valid & ~blocked, axis=1)
        self.previous_main = None
        self.previous_centroid = None
        self.previous_number = None
        self.last_fragment_count=0
        self.wall_tree=cKDTree(wall.points)
        yy,xx=np.nonzero(self.allowed&(wall.distance<=3)&(np.indices(wall.distance.shape)[0]>=wall.y_min)&(np.indices(wall.distance.shape)[0]<=wall.y_max))
        self.optical_band_pixels=(yy,xx)
        _,self.optical_band_indices=self.wall_tree.query(np.c_[xx,yy])
        fy,fx=np.nonzero(self.allowed&(wall.distance<=7)&(np.indices(wall.distance.shape)[0]>=wall.y_min)&(np.indices(wall.distance.shape)[0]<=wall.y_max))
        self.front_band_pixels=(fy,fx)
        _,self.front_band_indices=self.wall_tree.query(np.c_[fx,fy])
        selected=(wall.offsets>=1)&(wall.offsets<=3)
        self.air_gap_noise=cv2.remap(self.noise.astype(np.float32),wall.map_x[:,selected],wall.map_y[:,selected],cv2.INTER_LINEAR).max(axis=1)
        self.last_shadow_area=0;self.last_front_repair_area=0
        self.last_front_trim_area=0
        self.contact_history_number=None;self.contact_seed_streak=0;self.contact_confirmed=False;self.contact_history_span=None

    def segment(self,aligned):
        liquid,filled,repaired,delta=super().segment(aligned)
        liquid,shadow=remove_wall_shadow(self,aligned,liquid)
        self.last_shadow_area=int(shadow.sum());self.last_front_repair_area=0
        return liquid,filled&liquid,repaired&liquid,delta

    def main_component(self, liquid, number, image=None):
        self.last_front_trim_area=0
        count, labels, stats, centers = cv2.connectedComponentsWithStats(liquid.astype(np.uint8), 8)
        if count <= 1:
            self.previous_main = None
            self.last_fragment_count=0
            return liquid.copy(), [], 'missing'
        body_count = np.bincount(labels[liquid & (self.wall.distance > 4)], minlength=count)
        candidates = [i for i in range(1,count) if body_count[i] >= self.settings.min_component_area]
        if not candidates:
            candidates = list(range(1,count))
        if self.previous_main is not None and self.previous_number == number-1:
            near = cv2.dilate(self.previous_main.astype(np.uint8), np.ones((31,31), np.uint8)) > 0
            overlap = np.bincount(labels[near & liquid], minlength=count)
            eligible = [i for i in candidates if overlap[i] >= self.settings.min_component_area]
            chosen = max(eligible, key=lambda i:(overlap[i],body_count[i])) if eligible else max(candidates,key=lambda i:body_count[i])
            status = 'tracked' if eligible else 'reacquired'
        else:
            chosen = max(candidates,key=lambda i:body_count[i])
            status = 'initialized'
        selected={chosen}
        near_wall=np.unique(labels[liquid & (self.wall.distance<=self.settings.wall_uncertainty_px+.5)])
        # Saturation can cut one wall-attached liquid into multiple dark pieces.
        # Associate only nearby pieces that themselves reach the wall; an
        # airborne satellite remains excluded even when it overlaps the past.
        if chosen in near_wall:
            changed=True
            while changed:
                changed=False;main=np.isin(labels,list(selected));distance=ndi.distance_transform_edt(~main)
                for label in candidates:
                    if label in selected or label not in near_wall:continue
                    if distance[labels==label].min()<=15:
                        selected.add(label);changed=True
        main = np.isin(labels,list(selected))
        # A resolved gray lobe can be separated by a sub-threshold optical neck.
        # Require current-image gray support across a <=3 px connection; nearby
        # airborne drops separated by a white gap stay independent.
        if image is not None and chosen in near_wall:
            distance,indices=ndi.distance_transform_edt(~main,return_indices=True)
            for label in candidates:
                if label in selected or label in near_wall:continue
                cy,cx=np.nonzero(labels==label);k=int(np.argmin(distance[cy,cx]))
                if distance[cy[k],cx[k]]>3:continue
                ey,ex=indices[:,cy[k],cx[k]];count=max(abs(int(ex)-int(cx[k])),abs(int(ey)-int(cy[k])))+1
                bx=np.rint(np.linspace(cx[k],ex,count)).astype(int);by=np.rint(np.linspace(cy[k],ey,count)).astype(int)
                bridge=(image[by,bx]<245)&((self.background[by,bx]-image[by,bx].astype(float))>=np.maximum(self.settings.dark_support,4*self.noise[by,bx]))
                if bridge.all():selected.add(label);main[cy,cx]=True;main[by,bx]=True
        if image is not None and len(selected)>1:
            main,_=complete_bright_cavities(main,image,self.wall.distance,self.allowed,20,
                self.settings.cavity_boundary_fraction,self.wall.solid,self.settings.wall_uncertainty_px+.5)
        if image is not None:
            main,front=complete_front_reflection(self,image,main)
            self.last_front_repair_area=int(front.sum())
            main,front_removed=trim_front_extension(self,image,main)
            self.last_front_trim_area=int(front_removed.sum())
        self.last_fragment_count=len(selected)
        satellites = contours_of(liquid & ~main)
        self.previous_main, self.previous_centroid, self.previous_number = main.copy(), centers[chosen], number
        return main, satellites, status

    def outer_thickness(self, liquid, contact_selection):
        wall, cfg = self.wall, self.settings
        occupied = (wall.sample(liquid, nearest=True) > .5) & self.ray_valid
        has = occupied.any(axis=1)
        height = np.zeros(len(wall.s),np.float64)
        if not np.any(contact_selection):
            return height.astype(np.float32), None, False
        # The mask already selects the main liquid. Restrict the normal origins
        # to its actual wall projection, rather than the whole curved target.
        domain = has
        points, normals = wall.points.astype(float), wall.normals.astype(float)
        ray_limit = np.maximum(0,self.ray_valid.sum(axis=1)-1)*cfg.normal_step_px
        for contour in contours_of(liquid):
            vertices=np.asarray(contour,float);vectors=np.roll(vertices,-1,axis=0)-vertices
            q=vertices[None,:,:]-points[:,None,:]
            den=normals[:,0,None]*vectors[None,:,1]-normals[:,1,None]*vectors[None,:,0]
            safe=abs(den)>1e-10
            t=np.divide(q[:,:,0]*vectors[None,:,1]-q[:,:,1]*vectors[None,:,0],den,out=np.zeros_like(den),where=safe)
            u=np.divide(q[:,:,0]*normals[:,1,None]-q[:,:,1]*normals[:,0,None],den,out=np.zeros_like(den),where=safe)
            valid=safe&(t>0)&(t<=ray_limit[:,None]+1e-6)&(u>=-1e-8)&(u<=1+1e-8)&domain[:,None]
            height=np.maximum(height,np.where(valid,t,0).max(axis=1))
        # An overhanging front is eligible outside the local contact endpoints.
        # Reject remote, nonlocal normal intersections using the nearest wall
        # distance, rather than restricting thickness to the contact span.
        endpoints=points+normals*height[:,None]
        nearest,_=self.wall_tree.query(endpoints)
        height[height-nearest>1.0]=0
        truncated=bool(np.any(domain & occupied[:,-1]))
        best=None
        if height.max()>0:
            k=int(np.argmax(height));start=points[k];end=start+normals[k]*height[k]
            indices=np.flatnonzero(occupied[k]);lo,hi=indices[[0,-1]]
            best=dict(value_px=float(height[k]),s_px=float(wall.s[k]),start=start.tolist(),end=end.tolist(),
                      supported_contact=True,local_contact=bool(contact_selection[k]),
                      spans_visible_gap=bool(wall.offsets[lo]>cfg.wall_uncertainty_px+.5 or not occupied[k,lo:hi+1].all()),
                      definition='main_film_solid_safe_wall_normal_outer_boundary')
        return height.astype(np.float32),best,truncated

    def measure_main(self, main, aligned, number, repair=None):
        wall,cfg=self.wall,self.settings
        delta=self.background-aligned.astype(float)
        samples=(wall.sample(main,nearest=True)>.5)&self.ray_valid
        first=np.argmax(samples,axis=1)
        possible=samples.any(axis=1)&(wall.offsets[first]<=cfg.wall_uncertainty_px+.5)
        # Resolve thin tails using their own near-wall evidence, independent of
        # the front body's much stronger contrast. No imposed monotonic growth.
        d=wall.sample(delta);noise=wall.sample(self.noise)
        band=(wall.offsets>=.5)&(wall.offsets<=5)
        local_signal=np.max(d[:,band],axis=1)
        threshold=np.maximum(cfg.dark_support,3*np.max(noise[:,band],axis=1))
        observed=possible&(local_signal>=threshold)
        gap,rise,optical_height=wall_air_gap(self,aligned,main,samples)
        observed&=~gap
        observed=keep_runs(observed,2)
        # A long film with an independently seeded wall connection can have a
        # refractive bright transition at its front. Retain optical gaps as
        # evidence gaps, while its current silhouette supplies span endpoints.
        ids=np.flatnonzero(possible)
        width=float(wall.distance[main].max()) if main.any() else 0
        attached=len(ids)>1 and np.ptp(wall.s[ids])>=1.25*width
        contrast=wall.sample(self.contrast_field(main,delta),nearest=True).max(axis=1)
        ref=wall.sample(self.background);inside=wall.points-wall.normals*7
        solid=cv2.remap(self.background,inside[:,0:1],inside[:,1:2],cv2.INTER_LINEAR).ravel()
        outside_index=min(int(round(7/cfg.normal_step_px)),len(wall.offsets)-1)
        index=np.argmax((ref>=(solid+ref[:,outside_index])[:,None]/2)&(wall.offsets[None,:]<=1),axis=1)
        at_wall=d[np.arange(len(index)),index]
        uniform=(optical_height>=8)&~gap
        strong=keep_runs(observed&((uniform&(at_wall>=np.maximum(cfg.dark_seed,.42*contrast)))|
            (~uniform&(at_wall>=np.maximum(cfg.dark_support,cfg.contact_seed_fraction*contrast)))),cfg.min_contact_length_px)
        consecutive=self.contact_history_number==number-1
        history=self.contact_history_span if consecutive and self.contact_confirmed else None
        overlap=bool(history and len(ids) and min(history[1],float(wall.s[ids[-1]]))-max(history[0],float(wall.s[ids[0]]))>=cfg.min_contact_length_px)
        continued=attached and overlap
        use_outline=attached and (strong.sum()>=cfg.min_contact_length_px or continued)
        endpoint_geometry=keep_runs(possible&((local_signal>=threshold)|(optical_height>=8)),2) if use_outline else observed
        self.contact_seed_streak=(self.contact_seed_streak+1 if consecutive else 1) if strong.sum()>=2 else 0
        self.contact_confirmed=(self.contact_seed_streak>=cfg.persistence_frames) or continued
        self.contact_history_number=number
        self.contact_history_span=(float(wall.s[ids[0]]),float(wall.s[ids[-1]])) if len(ids) else None
        evidence=[]
        for i,j in intervals(observed):
            evidence.append(segment(wall,max(0,wall.s[i]-.5),min(wall.s[-1],wall.s[j]+.5)))
        official=[]
        endpoint_segments=[segment(wall,max(0,wall.s[i]-.5),min(wall.s[-1],wall.s[j]+.5)) for i,j in intervals(endpoint_geometry)]
        if endpoint_segments:
            official=[segment(wall,endpoint_segments[0]['s_start_px'],endpoint_segments[-1]['s_end_px'])]
        profile,maximum,trunc=self.outer_thickness(main,observed)
        yy,xx=np.nonzero(main);area=len(xx)
        flags=[]
        evidence_length=sum(s['length_px'] for s in evidence)
        length=official[0]['length_px'] if official else 0.
        gaps=max(0,length-evidence_length)
        after=np.arange(samples.shape[1])[None,:]>=first[:,None]
        stops=(~samples)&after;stop=np.argmax(stops,axis=1);stop[~stops.any(axis=1)]=samples.shape[1]
        contiguous_height=np.maximum(0,stop*cfg.normal_step_px-cfg.normal_step_px/2)
        resolved=observed&(contiguous_height>=cfg.min_film_thickness_px)
        endpoint_uncertainty=0.
        if official and resolved.any():
            i,j=np.flatnonzero(resolved)[[0,-1]]
            a,b=official[0]['s_start_px'],official[0]['s_end_px']
            sa,sb=max(a,wall.s[i]-.5),min(b,wall.s[j]+.5)
            official[0]['start_range']=[wall.point_at(a),wall.point_at(sa)]
            official[0]['end_range']=[wall.point_at(sb),wall.point_at(b)]
            endpoint_uncertainty=max(0,sa-a)+max(0,b-sb)
            if endpoint_uncertainty>8:flags.append('contact_boundary_uncertain')
        if gaps>2:flags.append('contact_span_contains_uncertain_gaps')
        if trunc:flags.append('normal_search_truncated')
        if maximum and maximum['spans_visible_gap']:flags.append('thickness_outer_envelope')
        if repair is not None and repair.sum()>20:flags.append('temporal_highlight_repair')
        if area:
            x0,y0,x1,y1=self.roi
            if xx.min()<=x0 or yy.min()<=y0 or xx.max()>=x1-1 or yy.max()>=y1-1:flags.append('liquid_at_roi_boundary')
        result=dict(frame=number,area_px2=int(area),centroid=[float(xx.mean()),float(yy.mean())] if area else None,
            bbox=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)] if area else None,
            contours=contours_of(main),contact_segments=official,contact_length_px=float(length),
            contact_evidence_segments=evidence,automatic_contact_segments=deepcopy(official),
            thickness_support_segments=deepcopy(evidence),uncertain_contact_segments=[],
            contact_evidence_length_px=float(evidence_length),contact_span_gap_px=float(gaps),
            endpoint_uncertainty_px=float(endpoint_uncertainty),
            confident_contact_length_px=float(evidence_length),possible_contact_length_px=float(possible.sum()),
            uncertain_contact_length_px=float(gaps),maximum_visible_thickness=maximum,
            minimum_wall_gap_px=float(wall.distance[main].min()) if area else None,
            contact_selection='main_film_contact_endpoint_span',manual_reviewed=False,contact_seed_length_px=int(strong.sum()),
            measurement_status='valid' if maximum else 'no_contact' if not official else 'no_valid_thickness',flags=flags)
        result['contact_air_gap_rejected_length_px']=int(np.count_nonzero(possible&gap))
        result['contact_seed_method']='near_wall_intensity_continuity_with_darkening_evidence'
        result['contact_endpoint_geometry_segments']=endpoint_segments
        result['contact_endpoint_method']='continued_attached_silhouette_span' if use_outline and continued and strong.sum()<cfg.min_contact_length_px else 'seeded_attached_silhouette_span' if use_outline else 'optical_evidence_span'
        return result,profile,observed,possible


def detect_impact(results,settings,fps):
    positive=[bool(r['contact_segments']) and r.get('contact_seed_length_px',2)>=2 for r in results]
    first=None
    for i in range(len(results)-settings.persistence_frames+1):
        block=results[i:i+settings.persistence_frames]
        if all(positive[i:i+settings.persistence_frames]) and all(r['frame']==results[i]['frame']+j for j,r in enumerate(block)):
            first=i;break
    event=dict(status='not_detected',first_contact_frame=None,previous_frame=None,manual_reviewed=False,
               definition='Earliest image-supported main-film contact confirmed by subsequent consecutive contact frames; not microscopic wetting.')
    if first is not None:
        before=results[first-1]['frame'] if first else None
        event.update(status='automatic_candidate_requires_review',first_contact_frame=results[first]['frame'],
                     previous_frame=before,bracket_frames=[before,results[first]['frame']],
                     confirmation_frames=[r['frame'] for r in results[first:first+settings.persistence_frames]],
                     confirmation_frame=results[first+settings.persistence_frames-1]['frame'])
        event['first_possible_contact_frame']=next((r['frame'] for r in results[:first+1] if r['contact_segments']),results[first]['frame'])
        event['optical_onset_range_frames']=[event['first_possible_contact_frame'],results[first]['frame']]
    for i,r in enumerate(results):
        r['impact_candidate']=first is not None and i==first
        r['relative_time_s']=(r['frame']-event['first_contact_frame'])/fps if first is not None and fps else None
        r['phase']='postimpact' if first is not None and i>=first else 'preimpact'
        if i and results[i-1]['frame']==r['frame']-1:
            prev=results[i-1];p,c=prev['contact_segments'],r['contact_segments']
            if p and c and max(abs(p[0][k]-c[0][k]) for k in ['s_start_px','s_end_px'])>10:r['flags'].append('contact_endpoint_jump')
            hp,hc=prev['maximum_visible_thickness'],r['maximum_visible_thickness']
            if hp and hc and abs(hp['value_px']-hc['value_px'])>max(5,.2*hp['value_px']):r['flags'].append('thickness_jump')
        r['flags']=sorted(set(r['flags']))
    return event
