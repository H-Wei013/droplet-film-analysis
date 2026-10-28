/* Shared by the self-contained review page and the offline result exporter. */
(function(root){
'use strict';
const clone=v=>JSON.parse(JSON.stringify(v));
function wallAt(wall,s){
  const ss=wall.s_px, pp=wall.points;let lo=0,hi=ss.length-1;
  while(hi-lo>1){const m=(lo+hi)>>1;if(ss[m]<=s)lo=m;else hi=m}
  const t=Math.max(0,Math.min(1,(s-ss[lo])/(ss[hi]-ss[lo])));
  return pp[lo].map((v,j)=>v+(pp[hi][j]-v)*t);
}
function segment(wall,a,b){const start=wallAt(wall,a),end=wallAt(wall,b);return {s_start_px:a,s_end_px:b,length_px:b-a,start,end,start_range:[start,start],end_range:[end,end]}}
function impactPoint(manifest,frames=manifest.frames,number=manifest.impact.first_contact_frame){
  const value={definition:'Wall arc midpoint of the contact endpoint span in the selected first-impact frame. Not the straight-line midpoint of endpoint coordinates.',
    frame:number,status:'impact_not_detected',coordinate_system:'aligned_reference_px',point_px:null,s_px:null,s_mm:null,raw_point_px:null,source:'contact_endpoint_span_arc_midpoint'};
  if(number===null||number===undefined)return value;
  const frame=frames.find(r=>r.frame===number);
  if(!frame){value.status='impact_frame_missing';return value}
  if(!frame.contact_segments.length){value.status='no_contact_in_impact_frame';return value}
  const start=Math.min(...frame.contact_segments.map(s=>s.s_start_px)),end=Math.max(...frame.contact_segments.map(s=>s.s_end_px));
  if(start<manifest.wall.s_px[0]||end>manifest.wall.s_px.at(-1)||end<start)throw Error('撞击首帧接触跨度超出壁面。');
  const middle=(start+end)/2,point=wallAt(manifest.wall,middle),shift=frame.registration_translation_px,scale=manifest.physical_calibration?.mm_per_px;
  return {...value,status:'defined',point_px:point,s_px:middle,s_mm:Number.isFinite(scale)&&scale>0?middle*scale:null,
    raw_point_px:shift?point.map((v,j)=>v+shift[j]):null,contact_start_s_px:start,contact_end_s_px:end};
}
function rayClear(wall,start,end){
  const pp=wall.points;
  for(let j=1,n=Math.max(2,Math.ceil(Math.hypot(end[0]-start[0],end[1]-start[1])*2));j<n;j++){
    const t=j/n,x=start[0]+(end[0]-start[0])*t,y=start[1]+(end[1]-start[1])*t;
    if(y<pp[0][1]||y>pp.at(-1)[1])return false;
    let lo=0,hi=pp.length-1;while(hi-lo>1){const k=(lo+hi)>>1;if(pp[k][1]<=y)lo=k;else hi=k}
    const f=(y-pp[lo][1])/(pp[hi][1]-pp[lo][1]),boundary=pp[lo][0]+f*(pp[hi][0]-pp[lo][0]);
    if(x>boundary+1)return false;
  }return true;
}
function uncertainty(automatic,official,wall){
  const out=[];
  for(const s of automatic){
    const a=s.s_start_px,b=s.s_end_px;
    let pieces=[[a,b]];
    if(official){const x=official.s_start_px,y=official.s_end_px;pieces=[];if(a<x)pieces.push([a,Math.min(b,x)]);if(b>y)pieces.push([Math.max(a,y),b])}
    for(const [x,y] of pieces)if(y>x)out.push({...segment(wall,x,y),reason:'outside_official_contact'});
  }
  return out;
}
function finalize(frame,wall){
  const r=clone(frame),auto=r.automatic_contact_segments||clone(r.contact_segments);
  const manual=!!r.contact_manually_corrected;
  const candidates=manual?r.contact_segments:auto;
  const primary=[...candidates].sort((a,b)=>b.length_px-a.length_px||a.s_start_px-b.s_start_px)[0];
  r.automatic_contact_segments=auto;
  r.thickness_support_segments=r.thickness_support_segments||clone(auto);
  r.contact_segments=primary?[segment(wall,primary.s_start_px,primary.s_end_px)]:[];
  r.contact_length_px=primary?primary.length_px:0;
  r.confident_contact_length_px=r.contact_evidence_length_px??r.contact_length_px;
  r.uncertain_contact_segments=uncertainty(auto,r.contact_segments[0],wall);
  r.uncertain_contact_length_px=r.uncertain_contact_segments.reduce((n,s)=>n+s.length_px,0);
  r.possible_contact_length_px=r.contact_length_px+r.uncertain_contact_length_px;
  r.contact_selection=manual?'manual':'main_film_contact_endpoint_span';
  r.measurement_status=r.within_valid_view===false?'outside_valid_wall_view':r.maximum_visible_thickness?'valid':'no_valid_thickness';
  if(r.uncertain_contact_segments.length&&!r.flags.includes('contact_boundary_uncertain'))r.flags.push('contact_boundary_uncertain');
  return r;
}
function validateEdit(edit,manifest){
  if(!edit||typeof edit!=='object'||Array.isArray(edit))throw Error('复核记录格式错误。');
  if(edit.contact_intervals!==undefined){
    if(!Array.isArray(edit.contact_intervals)||edit.contact_intervals.length>1)throw Error('每帧只能指定一段接触段。');
    for(const v of edit.contact_intervals)if(!Array.isArray(v)||v.length!==2||!v.every(Number.isFinite)||v[0]<0||v[1]<=v[0]||v[1]>manifest.wall.s_px.at(-1))throw Error('接触端点超出参考壁面。');
  }
  if(edit.contours!==undefined){
    if(!Array.isArray(edit.contours))throw Error('液体轮廓格式错误。');
    for(const c of edit.contours)if(!Array.isArray(c)||c.length<3||c.some(p=>!Array.isArray(p)||p.length!==2||!p.every(Number.isFinite)||p.some((v,k)=>v<0||v>=manifest.image_size[k])))throw Error('轮廓需要至少三个位于图内的有效点。');
  }
}
function inside(p,poly){
  let yes=false;
  for(let i=0,j=poly.length-1;i<poly.length;j=i++){
    const a=poly[i],b=poly[j];
    if(((a[1]>p[1])!==(b[1]>p[1]))&&p[0]<(b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0])yes=!yes;
  }return yes;
}
function hits(origin,normal,poly){
  const out=[];
  for(let i=0;i<poly.length;i++){
    const a=poly[i],b=poly[(i+1)%poly.length],vx=b[0]-a[0],vy=b[1]-a[1],qx=a[0]-origin[0],qy=a[1]-origin[1];
    const den=normal[0]*vy-normal[1]*vx;if(Math.abs(den)<1e-10)continue;
    const t=(qx*vy-qy*vx)/den,u=(qx*normal[1]-qy*normal[0])/den;
    if(t>=-1e-8&&u>=-1e-8&&u<=1+1e-8)out.push(Math.max(0,t));
  }
  return out.sort((a,b)=>a-b);
}
function touches(a,b){
  const cross=(p,q,r)=>(q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0]);
  if(inside(a[0],b)||inside(b[0],a))return true;
  for(let i=0;i<a.length;i++)for(let j=0;j<b.length;j++){
    const p=a[i],q=a[(i+1)%a.length],r=b[j],s=b[(j+1)%b.length];
    if(Math.max(p[0],q[0])+1e-8<Math.min(r[0],s[0])||Math.max(r[0],s[0])+1e-8<Math.min(p[0],q[0])||Math.max(p[1],q[1])+1e-8<Math.min(r[1],s[1])||Math.max(r[1],s[1])+1e-8<Math.min(p[1],q[1]))continue;
    if(cross(p,q,r)*cross(p,q,s)<=1e-8&&cross(r,s,p)*cross(r,s,q)<=1e-8)return true;
  }return false;
}
function attachedContours(contours,manifest,support){
  const wall=manifest.wall,points=wall.points,ss=wall.s_px,normals=wall.normals;
  const tol=manifest.settings.wall_uncertainty_px+.5;
  const attached=new Set();
  for(let index=0;index<contours.length;index++){
    const poly=contours[index];
    let found=false;
    for(let k=0;k<points.length&&!found;k++){
      if(!support.some(s=>ss[k]>=s.s_start_px&&ss[k]<=s.s_end_px))continue;
      const h=hits(points[k],normals[k],poly);
      found=h.length>0&&(h[0]<=tol||inside(points[k],poly));
    }
    if(found)attached.add(index);
  }
  // Appended polygons that overlap an attached polygon form the same liquid.
  let changed=true;while(changed){changed=false;for(let i=0;i<contours.length;i++)if(!attached.has(i)&&[...attached].some(j=>touches(contours[i],contours[j]))){attached.add(i);changed=true}}
  return attached;
}
function measureThickness(contours,manifest,support){
  const wall=manifest.wall,points=wall.points,ss=wall.s_px,normals=wall.normals;
  const tol=manifest.settings.wall_uncertainty_px+.5,maxDistance=manifest.settings.max_normal_distance_px;
  const attached=[...attachedContours(contours,manifest,support)].map(i=>contours[i]);
  let best=null,truncated=false;
  for(let k=0;k<points.length;k++)for(const poly of attached){
    if(!support.length)continue;
    const h=hits(points[k],normals[k],poly);if(!h.length)continue;
    // A distant ray intersection beyond the search domain is not a measured
    // liquid run. Only clip a far edge after liquid has entered the domain.
    if(h[0]>maxDistance&&!inside(points[k],poly))continue;
    const far=h.at(-1);if(far>maxDistance)truncated=true;
    const value=Math.min(far,maxDistance);
    const endpoint=[points[k][0]+normals[k][0]*value,points[k][1]+normals[k][1]*value];
    if(!rayClear(wall,points[k],endpoint))continue;
    let nearest=Infinity;for(const p of points)nearest=Math.min(nearest,Math.hypot(p[0]-endpoint[0],p[1]-endpoint[1]));
    if(value-nearest>1)continue;
    if(!best||value>best.value_px){
      if(value<=0)continue;
      const start=points[k],n=normals[k];
      best={value_px:value,s_px:ss[k],start:clone(start),end:[start[0]+n[0]*value,start[1]+n[1]*value],supported_contact:true,
        local_contact:support.some(s=>ss[k]>=s.s_start_px&&ss[k]<=s.s_end_px),
        spans_visible_gap:h[0]>tol||h.length>2,definition:'wall_normal_outer_envelope_of_attached_component'};
    }
  }
  return {maximum:best,truncated};
}
function effective(frame,edit,manifest){
  if(!edit)return clone(frame);
  validateEdit(edit,manifest);
  let r=clone(frame);
  const hasContact=edit.contact_intervals!==undefined,hasContour=edit.contours!==undefined;
  if(hasContact){r.contact_segments=edit.contact_intervals.map(([a,b])=>segment(manifest.wall,a,b));r.contact_manually_corrected=true}
  if(hasContour)r.contours=clone(edit.contours);
  r=finalize(r,manifest.wall);
  // A contact edit alone cannot restrict the full-film thickness search.
  // Contour edits remeasure using all attachment evidence plus manual support.
  const support=[...r.thickness_support_segments,...(hasContact?r.contact_segments:[])];
  const originalAttached=hasContact?attachedContours(r.contours,manifest,r.thickness_support_segments):new Set();
  const newlyAttached=hasContact&&[...attachedContours(r.contours,manifest,support)].some(i=>!originalAttached.has(i));
  if(hasContour||newlyAttached){
    const measured=measureThickness(r.contours,manifest,support);
    r.maximum_visible_thickness=measured.maximum;
    r.flags=r.flags.filter(f=>!['normal_search_truncated','thickness_outer_envelope'].includes(f));
    if(measured.truncated)r.flags.push('normal_search_truncated');
    if(measured.maximum?.spans_visible_gap)r.flags.push('thickness_outer_envelope');
    if(hasContour)r.contour_manually_corrected=true;
  }
  r.manual_reviewed=!!edit.reviewed;r.review_note=String(edit.note||'');r.manual_correction_applied=hasContact||hasContour;
  r.measurement_status=r.within_valid_view===false?'outside_valid_wall_view':r.maximum_visible_thickness?'valid':'no_valid_thickness';
  return r;
}
function finalManifest(manifest,edits,impactOverride){
  const result=clone(manifest);
  const known=new Set(manifest.frames.map(f=>String(f.frame)));
  for(const [key,e] of Object.entries(edits)){if(!known.has(key))throw Error('复核文件含未知帧号。');validateEdit(e,manifest)}
  result.frames=manifest.frames.map(f=>effective(f,edits[String(f.frame)],manifest));
  if(impactOverride!==null&&impactOverride!==undefined){
    const i=result.frames.findIndex(f=>f.frame===impactOverride);if(i<0)throw Error('撞击帧不存在。');
    result.impact={...result.impact,first_contact_frame:impactOverride,previous_frame:i?result.frames[i-1].frame:null,manual_reviewed:true,status:'manually_reviewed'};
    const fps=manifest.camera_metadata.frame_rate;
    for(const r of result.frames){r.impact_candidate=r.frame===impactOverride;r.relative_time_s=fps?(r.frame-impactOverride)/fps:null}
  }
  const impactFrame=result.impact.first_contact_frame;
  for(const r of result.frames)r.phase=r.within_valid_view===false?'after_valid_view':impactFrame!==null&&r.frame>=impactFrame?'postimpact':'preimpact';
  result.impact_point=impactPoint(result);
  result.data_extensions=[...new Set([...(result.data_extensions||[]),'impact-contact-arc-midpoint-v1'])];
  result.definitions={...result.definitions,impact_point:result.impact_point.definition};
  result.export_kind='final_measurements';return result;
}
function csv(manifest){
  const columns=['frame','relative_time_s','contact_length_px','maximum_visible_thickness_px','contact_start_s_px','contact_end_s_px','contact_selection','manual_reviewed','measurement_status'];
  const rows=manifest.frames.map(r=>{const s=r.contact_segments[0];return [r.frame,r.relative_time_s??'',r.contact_length_px,r.maximum_visible_thickness?.value_px??'',s?.s_start_px??'',s?.s_end_px??'',r.contact_selection,r.manual_reviewed,r.measurement_status].join(',')});
  return '\ufeff'+[columns.join(','),...rows].join('\r\n');
}
const api={wallAt,segment,finalize,validateEdit,measureThickness,effective,finalManifest,csv,rayClear,impactPoint};
root.DropletGeometry=api;if(typeof module!=='undefined')module.exports=api;
})(globalThis);
