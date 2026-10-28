/* Curves use the same effective frames as overlays and exports, including edits. */
const seriesHost=document.createElement('section');seriesHost.className='bar';seriesHost.id='seriesPanel';
seriesHost.innerHTML='<div class="series-heading"><h2>厚度与接触长度序列 · 点击曲线跳帧</h2><div class="controls"><span class="muted">红点：建议复核；竖线：当前帧</span><label><input type="checkbox" id="seriesAll">含撞击前/终止后</label><button id="downloadThickness">保存厚度图</button><button id="downloadContact">保存长度图</button></div></div><p id="impactPointInfo" class="muted"></p><div class="series-charts"><canvas id="thicknessSeries" aria-label="最大厚度序列"></canvas><canvas id="contactSeries" aria-label="接触长度序列"></canvas></div>';
document.getElementById('navigationPanel').before(seriesHost);
const impactInfo=document.getElementById('impactPointInfo');
labels.unsupported_front_extension_removed='仅由壁面闭合的前方亮空气区域已排除，需复核';labels.transient_weak_tail_removed='前后帧均无支持的弱尾部突出已排除，需复核';
$('seriesAll').parentElement.lastChild.textContent='显示撞击前候选值';
const validWindowInfo=document.createElement('span');validWindowInfo.id='validWindowInfo';document.getElementById('locateImpact').after(validWindowInfo);
function fitWorkspace(){if(innerWidth<=1100)return;const top=document.getElementById('reviewWorkspace').getBoundingClientRect().top+scrollY,available=Math.max(270,innerHeight-top-12);document.documentElement.style.setProperty('--manual-stage',available+'px');document.documentElement.style.setProperty('--image-stage',Math.max(240,available-24)+'px')}
const fitObserver=new ResizeObserver(fitWorkspace);fitObserver.observe(seriesHost);fitObserver.observe(document.getElementById('navigationPanel'));window.addEventListener('resize',fitWorkspace);fitWorkspace();
const scaleMM=1;
labels.contact_span_contains_uncertain_gaps='两端跨度内含证据不足的部分';labels.main_film_reacquired='主液膜重新定位，需复核';labels.main_film_missing='主液膜缺测';labels.saturated_main_film_fragments_associated='强反光碎片已按贴壁证据关联，需复核';labels.wall_shadow_fringe_removed='近壁亮空气过渡已排除';labels.front_specular_completion='前端反光轮廓补全，需复核';labels.temporal_thin_tail_recovered='当前灰度及双侧邻帧支持的薄尾恢复';labels.outside_valid_wall_view='已到壁面有效边界，正式测量序列已终止';
function drawSeries(){
  const impact=impactOverride??manifest.impact.first_contact_frame;
  const all=frames.map((f,i)=>({r:effective(i),i}));
  const origin=currentImpactPoint();
  impactInfo.textContent=origin.status==='defined'?'● 撞击点：首帧 '+origin.frame+' 接触弧长中点；(x, y) = ('+origin.point_px.map(v=>v.toFixed(2)).join(', ')+') px；s = '+origin.s_px.toFixed(2)+' px。紫色点标注；横轴覆盖全部源帧，空白区间无正式测量。':'撞击点暂无有效位置：所选撞击帧缺少接触段，或未检测到撞击。';
  impactInfo.title=impactInfo.textContent;
  const viewWindow=manifest.analysis_window;validWindowInfo.hidden=!viewWindow||viewWindow.first_excluded_frame===null;
  if(!validWindowInfo.hidden){const after=frames[index].frame>=viewWindow.first_excluded_frame;validWindowInfo.textContent=(after?'当前帧仅供追溯；':'')+'正式测量截至 '+viewWindow.last_valid_frame+' 帧';validWindowInfo.style.color=after?'#a42828':'#58707d';validWindowInfo.style.fontWeight=after?'650':'normal';}
  const selected=all;
  if(!selected.length)return;
  for(const [id,title,get] of [
    ['thicknessSeries','Maximum main-film thickness (px)',r=>r.maximum_visible_thickness?.value_px],
    ['contactSeries','Contact endpoint-span arc length (px)',r=>r.measurement_status==='missing_main_film'?null:r.contact_length_px]]){
    const canvas=$(id),ctx=canvas.getContext('2d');canvas.width=1200;canvas.height=255;
    const left=80,right=1170,top=45,bottom=215,min=Math.min(...frames.map(f=>f.frame)),max=Math.max(min+1,...frames.map(f=>f.frame));
    const values=selected.map(({r})=>r.within_valid_view===false||(!$('seriesAll').checked&&(impact===null||r.frame<impact))?null:get(r)),peak=Math.max(.1,...values.filter(v=>v!==null&&v!==undefined).map(v=>v*scaleMM))*1.1;
    const xp=n=>left+(n-min)/(max-min)*(right-left),yp=v=>bottom-v/peak*(bottom-top);
    ctx.fillStyle='white';ctx.fillRect(0,0,1200,255);ctx.font='15px system-ui';ctx.fillStyle='#243b49';ctx.fillText(manifest.sequence+' | '+title,12,23);
    ctx.strokeStyle='#e3e9ed';ctx.lineWidth=1;
    for(let n=0;n<=4;n++){const y=bottom-n*(bottom-top)/4;ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();ctx.fillText((peak*n/4).toFixed(2),12,y+5)}
    ctx.save();ctx.textAlign='center';
    for(let k=0;k<=4;k++){const n=Math.round(min+(max-min)*k/4);ctx.fillText(String(n),xp(n),bottom+21)}
    ctx.fillText('Source frame number',625,252);ctx.restore();
    ctx.strokeStyle='#146b87';ctx.lineWidth=1.7;ctx.beginPath();let pen=false;
    selected.forEach(({r},j)=>{const v=values[j];if(v===null||v===undefined){pen=false;return}const x=xp(r.frame),y=yp(v*scaleMM);if(pen)ctx.lineTo(x,y);else ctx.moveTo(x,y);pen=true});ctx.stroke();
    selected.forEach(({r},j)=>{const v=values[j];if(v===null||v===undefined)return;if(r.flags.some(f=>['thickness_jump','contact_endpoint_jump','liquid_at_roi_boundary','normal_search_truncated','main_film_reacquired'].includes(f))){ctx.fillStyle='#c85a32';ctx.beginPath();ctx.arc(xp(r.frame),yp(v*scaleMM),2.5,0,2*Math.PI);ctx.fill()}});
    const current=frames[index].frame;ctx.strokeStyle='#303d46';ctx.beginPath();ctx.moveTo(xp(current),top);ctx.lineTo(xp(current),bottom);ctx.stroke();ctx.fillText('Frame '+current,970,23);
    canvas._series={left,right,min,max,selected,values,current,currentX:xp(current)};
  }
}
for(const id of ['thicknessSeries','contactSeries'])$(id).onclick=e=>{const c=$(id),m=c._series;if(!m)return;const x=(e.clientX-c.getBoundingClientRect().left)*c.width/c.getBoundingClientRect().width;const n=m.min+(x-m.left)/(m.right-m.left)*(m.max-m.min);const item=m.selected.reduce((a,b)=>Math.abs(b.r.frame-n)<Math.abs(a.r.frame-n)?b:a);stop();go(item.i)};
$('seriesAll').onchange=drawSeries;
for(const [button,id,name] of [['downloadThickness','thicknessSeries','thickness_vs_frame'],['downloadContact','contactSeries','contact_length_vs_frame']])$(button).onclick=()=>{const a=document.createElement('a');a.download=manifest.sequence+'.'+name+'.png';a.href=$(id).toDataURL('image/png');a.click()};
const renderWithoutSeries=render;render=function(){drawSeries();return renderWithoutSeries()};
$('sequence').textContent=manifest.sequence+' · '+frames.length+' 帧 · 像素单位 px';
drawSeries();
