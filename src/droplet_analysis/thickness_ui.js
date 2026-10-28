/* 沿用已有坐标映射与存储键，保留之前在浏览器中保存的接触修正。 */
'use strict';
let manualThicknessAnchor=null;
$('addThickness').onclick=()=>{
  stop();mode='thickness';polygon=[];contactPoints=[];manualThicknessAnchor=null;
  $('showThickness').checked=true;$('showWall').checked=true;saveLayers();
  $('editHint').textContent='先点击最大厚度处的壁面起点（吸附到壁面），再点击液膜外缘。第二点自动投影到该处法线；确认后红线和数值立即更新。';render();
};
$('resetThickness').onclick=()=>{
  const e=edits[String(frames[index].frame)];if(e){delete e.manual_thickness;e.reviewed=false}
  mode=null;manualThicknessAnchor=null;save();render();
};
$('clearThickness').onclick=()=>{
  edit().manual_thickness=null;edit().reviewed=false;mode=null;manualThicknessAnchor=null;save();render();
};
for(const id of ['contour','measure'])$(id).addEventListener('click',event=>{
  if(mode!=='thickness')return;
  event.stopImmediatePropagation();stop();
  const p=canvasPoint(event,event.currentTarget);if(p.some((v,k)=>v<0||v>=manifest.image_size[k]))return;
  if(manualThicknessAnchor===null){
    manualThicknessAnchor=nearestWall(p);
    $('editHint').textContent='壁面起点已选定。现在点击液膜外缘；会沿该起点的局部法线计算厚度。Esc 取消。';render();
  }else{
    const value={wall_s_px:manualThicknessAnchor,outer_point:p.map(v=>Math.round(v*10)/10)};
    try{
      const h=DropletGeometry.manualThickness(value,manifest);
      edit().manual_thickness=value;edit().reviewed=false;mode=null;manualThicknessAnchor=null;save();
      $('editHint').textContent='人工最大厚度已更新：'+h.value_px.toFixed(2)+' px。接触段保持当前设置；可单独撤销厚度修正。';render();
    }catch(error){$('editHint').textContent=error.message}
  }
},true);
const originalReviewDraw=draw;
draw=function(canvas,img,r,roi,kind,small=false){
  originalReviewDraw(canvas,img,r,roi,kind,small);
  if(kind&&mode==='thickness'&&manualThicknessAnchor!==null&&!small){
    const ctx=canvas.getContext('2d'),m=canvas._mapping,p=wallAt(manualThicknessAnchor);
    ctx.save();ctx.translate(m.ox,m.oy);ctx.scale(m.scale,m.scale);ctx.translate(-m.x0,-m.y0);
    circle(ctx,p,'#b42ac5');ctx.restore();
  }
};
const originalReviewRoi=roiFor;
roiFor=function(i){
  const box=originalReviewRoi(i);if(!$('crop').checked)return box;
  const h=effective(i).maximum_visible_thickness;
  const points=h?.manually_corrected?[h.start,h.end]:[];
  if(mode==='thickness'&&manualThicknessAnchor!==null)points.push(wallAt(manualThicknessAnchor));
  for(const p of points){box[0]=Math.max(0,Math.min(box[0],p[0]-25));box[1]=Math.max(0,Math.min(box[1],p[1]-25));box[2]=Math.min(manifest.image_size[0],Math.max(box[2],p[0]+25));box[3]=Math.min(manifest.image_size[1],Math.max(box[3],p[1]+25))}
  return box;
};

labels.manual_thickness_override='最大厚度已人工指定；原自动质量标记保留供追溯';
render();
