/* Multi-dataset review archive; scientific data and storage keys stay unchanged. */
function reviewDocument(item,framesValue,impactValue){
  return {schema_version:5,manual_review_extension:'wall-normal-two-click-v1',sequence:item.sequence,
    analysis_signature:item.analysis_signature,coordinate_system:'aligned_reference_px',frames:framesValue,impact_frame:impactValue};
}
$('exportAllReviews').onclick=()=>{
  try{
    const sequences=[];
    for(const item of datasets){
      const stored=item===manifest?{frames:edits,impact_frame:impactOverride}:JSON.parse(localStorage.getItem('droplet-review-v5:'+item.sequence+':'+item.analysis_signature)||'{}');
      if(Object.keys(stored.frames||{}).length||stored.impact_frame!=null){
        DropletGeometry.finalManifest(item,stored.frames||{},stored.impact_frame??null);
        sequences.push(reviewDocument(item,stored.frames||{},stored.impact_frame??null));
      }
    }
    if(!sequences.length)throw Error('尚无修正或复核记录。请先编辑或勾选复核。');
    download('all.review.json',JSON.stringify({schema_version:5,coordinate_system:'aligned_reference_px',sequences},null,2),'application/json');
  }catch(error){$('error').textContent=error.message}
};
$('import').onchange=async()=>{
  try{
    const file=$('import').files[0];if(!file)return;
    const payload=JSON.parse(await file.text()),items=payload.sequences||[payload],pending=[],seen=new Set();
    for(const value of items){
      const item=datasets.find(v=>v.sequence===value.sequence);
      if(!item||seen.has(value.sequence)||value.coordinate_system!=='aligned_reference_px'||value.analysis_signature!==item.analysis_signature||!value.frames)
        throw Error('复核文件与数据集、坐标系或分析签名不匹配，或数据集重复。');
      seen.add(value.sequence);
      const stored=item===manifest?{frames:edits,impact_frame:impactOverride}:JSON.parse(localStorage.getItem('droplet-review-v5:'+item.sequence+':'+item.analysis_signature)||'{}');
      const merged={...(stored.frames||{})};
      for(const [key,editValue] of Object.entries(value.frames))merged[key]={...(merged[key]||{}),...editValue};
      const impact=Object.hasOwn(value,'impact_frame')?value.impact_frame:(stored.impact_frame??null);
      DropletGeometry.finalManifest(item,merged,impact);
      pending.push({item,frames:merged,impact_frame:impact});
    }
    for(const entry of pending){
      if(entry.item===manifest){edits=entry.frames;impactOverride=entry.impact_frame}
      localStorage.setItem('droplet-review-v5:'+entry.item.sequence+':'+entry.item.analysis_signature,JSON.stringify({frames:entry.frames,impact_frame:entry.impact_frame}));
    }
    save();render();$('editHint').textContent='已导入并保存 '+pending.length+' 个数据集的复核记录。';
  }catch(error){$('error').textContent=error.message}
};
