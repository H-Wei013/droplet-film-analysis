/* 人工厚度覆盖放在自动几何计算之后，避免接触或轮廓编辑悄悄改掉已确认的厚度。 */
(function(root){
'use strict';
const G=root.DropletGeometry,copy=v=>JSON.parse(JSON.stringify(v));
const baseEffective=G.effective,baseValidate=G.validateEdit,baseFinal=G.finalManifest;
function normalAt(manifest,s){
  const wall=manifest.wall,ss=wall.s_px;let k=0;
  while(k<ss.length-2&&ss[k+1]<s)k++;
  const t=Math.max(0,Math.min(1,(s-ss[k])/(ss[k+1]-ss[k])));
  const n=wall.normals[k].map((v,j)=>v+(wall.normals[k+1][j]-v)*t);
  const length=Math.hypot(...n);if(!(length>0))throw Error('此处壁面法线无效。');
  return n.map(v=>v/length);
}
function manualThickness(value,manifest){
  if(value===null)return null;
  if(!value||!Number.isFinite(value.wall_s_px)||value.wall_s_px<manifest.wall.s_px[0]||value.wall_s_px>manifest.wall.s_px.at(-1))
    throw Error('厚度起点超出参考壁面。');
  const p=value.outer_point;
  if(!Array.isArray(p)||p.length!==2||p.some((v,k)=>!Number.isFinite(v)||v<0||v>=manifest.image_size[k]))
    throw Error('厚度外缘点必须位于图像内。');
  const s=value.wall_s_px,start=G.wallAt(manifest.wall,s),n=normalAt(manifest,s);
  const distance=p.reduce((sum,v,j)=>sum+(v-start[j])*n[j],0);
  if(!(distance>0))throw Error('第二点应位于壁面外侧的液膜外缘，请重新选点。');
  const end=start.map((v,j)=>v+n[j]*distance);
  if(!G.rayClear(manifest.wall,start,end))throw Error('测量线穿过靶体，请在液膜附近重新选择壁面起点。');
  if(end.some((v,k)=>v<0||v>=manifest.image_size[k]))throw Error('法向投影端点超出图像，请重新选择壁面起点。');
  return {value_px:distance,s_px:s,start,end,selected_point:copy(p),supported_contact:true,
    local_contact:null,spans_visible_gap:null,manually_corrected:true,
    definition:'manual_wall_normal_thickness'};
}
function validate(edit,manifest){
  baseValidate(edit,manifest);
  if(Object.hasOwn(edit,'manual_thickness'))manualThickness(edit.manual_thickness,manifest);
}
function apply(r,edit,manifest){
  if(!edit||!Object.hasOwn(edit,'manual_thickness'))return r;
  r.automatic_maximum_visible_thickness=copy(r.automatic_maximum_visible_thickness??r.maximum_visible_thickness);
  r.maximum_visible_thickness=manualThickness(edit.manual_thickness,manifest);
  r.thickness_manually_corrected=true;r.thickness_selection='manual';
  r.manual_correction_applied=true;r.measurement_status=r.within_valid_view===false?'outside_valid_wall_view':r.maximum_visible_thickness?'valid':'no_valid_thickness';
  if(!r.flags.includes('manual_thickness_override'))r.flags.push('manual_thickness_override');
  return r;
}
G.validateEdit=validate;
G.manualThickness=manualThickness;
G.normalAt=normalAt;
G.effective=function(frame,edit,manifest){
  if(edit)validate(edit,manifest);
  return apply(baseEffective(frame,edit,manifest),edit,manifest);
};
G.finalManifest=function(manifest,edits,impact){
  for(const edit of Object.values(edits))validate(edit,manifest);
  const result=baseFinal(manifest,edits,impact);
  result.frames=result.frames.map(r=>apply(r,edits[String(r.frame)],manifest));
  result.manual_review_extension='wall-normal-two-click-v1';
  return result;
};
G.csv=function(manifest,options={}){
 const columns=['frame','relative_time_s','contact_length_px','maximum_visible_thickness_px','contact_start_s_px','contact_end_s_px','manual_reviewed','measurement_status','within_valid_view','phase','impact_frame','impact_point_x_px','impact_point_y_px','impact_point_s_px'];
 const origin=G.impactPoint(manifest),xy=origin.point_px||[];
 const frames=options.trace?manifest.frames:manifest.frames.filter(r=>r.phase==='postimpact'&&r.within_valid_view!==false);
 const rows=frames.map(r=>{const s=r.contact_segments[0],h=r.maximum_visible_thickness;return [r.frame,r.relative_time_s??'',r.contact_length_px,h?.value_px??'',s?.s_start_px??'',s?.s_end_px??'',r.manual_reviewed,r.measurement_status,r.within_valid_view!==false,r.phase,origin.frame??'',xy[0]??'',xy[1]??'',origin.s_px??''].join(',')});
 return '\ufeff'+[columns.join(','),...rows].join('\r\n');
};
if(typeof module!=='undefined')module.exports=G;
})(globalThis);
