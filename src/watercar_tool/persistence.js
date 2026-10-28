/* Disk is authoritative. Browser storage remains a recovery cache only. */
(async function(){
 'use strict';
 const button=document.getElementById('exportFolder'),panel=document.getElementById('manualPanel');
 if(!toolContext){button.disabled=true;button.title='请通过启动工具打开结果，以自动保存到磁盘';document.getElementById('saved').textContent='离线页面：浏览器暂存，须导出复核备份';return}
 panel.inert=true;
 let state=null,pending={},timer=null,chain=Promise.resolve(),failed=null,inFlight=0;
 const allBySignature=new Map(datasets.map(m=>[m.analysis_signature,m]));
 async function call(path,value){
  const response=await fetch(toolContext.api+'/api/'+path,value===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)});
  const result=await response.json();if(!response.ok)throw Error(result.error||'保存失败');return result;
 }
 function entry(m,framesValue,impactValue){return {schema_version:5,sequence:m.sequence,analysis_signature:m.analysis_signature,coordinate_system:'aligned_reference_px',frames:structuredClone(framesValue),impact_frame:impactValue}}
 function cache(value){localStorage.setItem('droplet-review-v5:'+value.sequence+':'+value.analysis_signature,JSON.stringify({frames:value.frames,impact_frame:value.impact_frame}))}
 const originalSave=save;
 function put(changes){
  for(const [key,value] of Object.entries(changes)){pending[key]=structuredClone(value);cache(value)}
  document.getElementById('saved').textContent='正在保存到结果目录…';
  clearTimeout(timer);timer=setTimeout(()=>flush().catch(showError),400);
 }
 function showError(error){failed=error;document.getElementById('saved').textContent='未保存到磁盘：'+error.message;document.getElementById('error').textContent=error.message}
 async function flush(){
  clearTimeout(timer);
  const changes=pending;pending={};
  if(!Object.keys(changes).length){await chain;if(failed)throw failed;return}
  inFlight++;
  chain=chain.then(async()=>{
   if(failed)throw failed;
   try{state=await call('reviews',{revision:state.revision,changes});document.getElementById('saved').textContent='已保存到结果目录 · '+state.saved_at;}
   catch(error){for(const [key,value]of Object.entries(changes))if(!pending[key])pending[key]=value;showError(error);throw error}
  }).finally(()=>{inFlight--});
  return chain;
 }
 window.addEventListener('beforeunload',event=>{if(Object.keys(pending).length||inFlight||failed){event.preventDefault();event.returnValue='存在未保存的复核，请先导出备份。'}});
 try{
  state=await call('reviews');
  for(const m of datasets){const value=state.sequences[m.analysis_signature]||entry(m,{},null);cache(value)}
  const loaded=state.sequences[manifest.analysis_signature];edits=loaded?.frames||{};impactOverride=loaded?.impact_frame??null;effectiveCache.clear();
  originalSave();document.getElementById('saved').textContent='已从结果目录恢复复核 · 版本 '+state.revision;
  save=function(){originalSave();put({[manifest.analysis_signature]:entry(manifest,edits,impactOverride)})};
  panel.inert=false;await render();
 }catch(error){showError(error);return}
 document.getElementById('import').onchange=async()=>{try{
  await flush();const file=document.getElementById('import').files[0];if(!file)return;const value=JSON.parse(await file.text());
  const items=value.sequences||[value],changes={},seen=new Set();
  for(const item of items){
   const m=allBySignature.get(item.analysis_signature);
   if(!m||m.sequence!==item.sequence||item.coordinate_system!=='aligned_reference_px'||seen.has(item.analysis_signature))throw Error('复核文件与当前结果不一致或包含重复组。');
   seen.add(item.analysis_signature);
   const prior=m===manifest?{frames:edits,impact_frame:impactOverride}:(state.sequences[item.analysis_signature]||{frames:{},impact_frame:null});
   const merged={...prior.frames,...item.frames},impact=item.impact_frame??prior.impact_frame;
   DropletGeometry.finalManifest(m,merged,impact);changes[item.analysis_signature]=entry(m,merged,impact);
  }
  put(changes);await flush();const current=state.sequences[manifest.analysis_signature];edits=current?.frames||{};impactOverride=current?.impact_frame??null;effectiveCache.clear();await render();document.getElementById('editHint').textContent='已导入并保存 '+items.length+' 个数据集。';
 }catch(error){showError(error)}};
 document.getElementById('exportAllReviews').onclick=async()=>{try{
  await flush();const values=Object.values(state.sequences).filter(x=>Object.keys(x.frames||{}).length||x.impact_frame!==null);
  download('all.review.json',JSON.stringify({schema_version:5,coordinate_system:'aligned_reference_px',sequences:values},null,2),'application/json');
 }catch(error){showError(error)}};
 button.onclick=async()=>{try{
  button.disabled=true;await flush();const value=finalData();
  const result=await call('export',{signature:manifest.analysis_signature,revision:state.revision,manifest:value});
  document.getElementById('editHint').textContent='修正结果已保存到：'+result.directory;
 }catch(error){showError(error)}finally{button.disabled=false}};
 // Downloaded final measurements must wait for a pending disk save as well.
 for(const id of ['exportData','exportCsv','exportTraceCsv']){const original=document.getElementById(id).onclick;document.getElementById(id).onclick=async()=>{try{await flush();original()}catch(error){showError(error)}}}
 window.toolReview={flush,get state(){return state},get failed(){return failed}};
})();
