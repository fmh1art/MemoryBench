'use strict';
const $ = id => document.getElementById(id);
const rangeMode = window.ACWM_RANGE_MODE === true;
const markDefs = rangeMode ? [
  ['prefix_start','1 · 前缀开始',''],
  ['prefix_end','2 · 前缀结束（不含）',''],
  ['t','3 · 生成开始','action'],
  ['eval_end','4 · 生成结束（不含）','action']
] : [
  ['prefix_start','1 · 前缀开始',''], ['t','2 · t / 动作开始','action'],
  ['action_end','3 · 动作结束（不含）','action'], ['eval_start','4 · 评估开始','eval'],
  ['eval_end','5 · 评估结束（不含）','eval']
];
let cases=[], meta=null, episode=null, saved=[], current=0, dirty=false;
let frameRequest=0, displayedFrame=-1, preview=null, switching=false;
let fastPlayback=null;
const video=$('video');
const payloadKeys=[...(rangeMode?['annotation_scope','boundaries_linked']:[]),'id','case_id','view_id','revision','status','title','marks','evidence','target_description','expected_state','gate_criterion','references','notes'];
function fmt(s){if(!Number.isFinite(s))return '—';const m=Math.floor(s/60);return `${String(m).padStart(2,'0')}:${(s-m*60).toFixed(3).padStart(6,'0')}`;}
function toast(text,error=false){$('toast').textContent=text;$('toast').classList.toggle('error',error);$('toast').style.display='block';clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('toast').style.display='none',5000);}
async function api(url, options){const r=await fetch(url,options);if(!r.ok){let body;try{body=await r.json();}catch{body={detail:r.statusText};}throw new Error(typeof body.detail==='string'?body.detail:JSON.stringify(body.detail));}return r.json();}
function base(){return `/api/cases/${encodeURIComponent(meta.case_id)}/views/${meta.view_id}`;}
function imageURL(frame){return `${base()}/frames/${frame}.jpg`;}
function emptyEpisode(){return {id:crypto.randomUUID(),case_id:meta.case_id,view_id:meta.view_id,revision:0,status:'draft',annotation_scope:rangeMode?'prefix_generation':'keyframes',boundaries_linked:rangeMode?true:undefined,marks:rangeMode?{prefix_start:0,prefix_end:null,t:null,eval_end:null}:{prefix_start:0,t:null,action_end:null,eval_start:null,eval_end:null}};}
function backupKey(){return `acwm-memory:${rangeMode?'ranges:':''}${meta.case_id}:${meta.view_id}`;}
function setDirty(){if(!episode)return;dirty=true;$('saveState').textContent='有未保存修改';$('saveState').className='dirty';try{localStorage.setItem(backupKey(),JSON.stringify(episode));}catch{}renderRanges();}
function markClean(label){dirty=false;$('saveState').textContent=label;$('saveState').className='';}
async function preserve(){if(!dirty)return true;try{await save('draft',false);return true;}catch(e){toast(`草稿保存失败，已保留当前页面：${e.message}`,true);return false;}}
function frameAtTime(time){const pts=meta.media_seconds;let lo=0,hi=pts.length;while(lo<hi){const mid=(lo+hi)>>1;if(pts[mid]<=time+1e-7)lo=mid+1;else hi=mid;}return Math.max(0,lo-1);}
function mediaAt(frame){return frame===meta.frame_count?meta.media_end_seconds:meta.media_seconds[frame];}
function sensorAt(frame){return (frame===meta.frame_count?meta.sensor_end_timestamp_ns_estimate:meta.sensor_timestamp_ns[frame])/1e9;}
function isPlaying(){return fastPlayback!==null||!video.paused;}
function pausePlayback(){
  if(fastPlayback){cancelAnimationFrame(fastPlayback.raf);fastPlayback=null;}
  video.pause();
  $('play').textContent='播放';
}
function setNativeSpeed(rate){
  try{video.playbackRate=rate;return true;}
  catch{video.playbackRate=1;return false;}
}
function startFastPlayback(rate){
  // Browsers may reject playbackRate > 16. Seek by elapsed wall time instead;
  // allow only one seek at a time and update annotations from decoded frames.
  video.pause();
  const state={mediaStart:mediaAt(current),wallStart:performance.now(),lastSeek:-Infinity,raf:null};
  fastPlayback=state;
  $('play').textContent='暂停';
  function tick(now){
    if(fastPlayback!==state)return;
    let target=state.mediaStart+(now-state.wallStart)*rate/1000;
    const endFrame=preview?preview[1]:meta.frame_count;
    const endTime=mediaAt(endFrame);
    if(target>=endTime){
      if(preview&&$('loop').checked){
        const startTime=mediaAt(preview[0]);
        target=startTime+(target-startTime)%(endTime-startTime);
        state.mediaStart=target;state.wallStart=now;
      }else{seek(endFrame-1,true);return;}
    }
    if(!video.seeking&&now-state.lastSeek>=125){
      video.currentTime=target;
      state.lastSeek=now;
    }
    state.raf=requestAnimationFrame(tick);
  }
  state.raf=requestAnimationFrame(tick);
}
video.addEventListener('seeked',()=>{
  if(fastPlayback){current=frameAtTime(video.currentTime);updatePosition();}
});
function updatePosition(){if(!meta)return;$('frameInput').value=current;$('scrubber').value=current;$('mediaClock').textContent=`播放 ${fmt(mediaAt(current)-mediaAt(0))} / ${fmt(meta.media_duration_seconds)} · 帧 ${current} / ${meta.frame_count-1}`;$('sensorClock').textContent=`采集 ${fmt(sensorAt(current)-sensorAt(0))}`;$('playhead').style.left=`${100*current/meta.frame_count}%`;}
async function showFrame(frame){const token=++frameRequest, url=imageURL(frame);displayedFrame=-1;$('frameLoading').style.display='block';$('still').style.visibility='hidden';const img=new Image();img.src=url;try{await img.decode();if(token!==frameRequest)return;$('still').src=url;$('still').style.display='block';$('still').style.visibility='visible';displayedFrame=frame;$('frameLoading').style.display='none';}catch(e){if(token===frameRequest){$('frameLoading').textContent='精确帧加载失败，请重新定位';toast('精确帧加载失败，请重新定位',true);}}}
function seek(frame, keepPreview=false){if(!meta)return;if(!keepPreview)preview=null;current=Math.max(0,Math.min(meta.frame_count-1,Math.round(frame)));pausePlayback();const next=mediaAt(current+1);video.currentTime=mediaAt(current)+Math.min(.001,(next-mediaAt(current))/4);updatePosition();$('frameLoading').textContent='加载精确帧…';showFrame(current);}
async function play(){if(!meta)return;if(isPlaying()){seek(fastPlayback?current:frameAtTime(video.currentTime),true);return;}if(preview && (current<preview[0]||current>=preview[1]))seek(preview[0],true);try{++frameRequest;$('frameLoading').style.display='none';$('still').style.display='none';displayedFrame=-1;const rate=Number($('speed').value);if(!setNativeSpeed(rate)){startFastPlayback(rate);return;}await video.play();}catch(e){toast(`浏览器无法播放：${e.message}`,true);showFrame(current);}}
function onVideoFrame(_now, info){if(meta&&!video.paused){current=frameAtTime(info.mediaTime);if(preview&&current>=preview[1]-1){const span=preview;if($('loop').checked){seek(span[0],true);play();}else{seek(span[1]-1,true);}}updatePosition();}video.requestVideoFrameCallback(onVideoFrame);}
if(video.requestVideoFrameCallback)video.requestVideoFrameCallback(onVideoFrame);else video.addEventListener('timeupdate',()=>{if(meta&&!video.paused){current=frameAtTime(video.currentTime);updatePosition();}});
video.addEventListener('play',()=>$('play').textContent='暂停');video.addEventListener('pause',()=>$('play').textContent=isPlaying()?'暂停':'播放');video.addEventListener('ended',()=>{if(preview&&$('loop').checked){seek(preview[0],true);play();}else seek(meta.frame_count-1);});
function renderCases(){const q=$('search').value.toLowerCase();$('caseList').replaceChildren();for(const item of cases.filter(c=>c.id.toLowerCase().includes(q))){const b=document.createElement('button');b.className=`case${meta?.case_id===item.id?' active':''}`;const title=document.createElement('b');title.textContent=item.id.split('-memory')[0].replace('QUEST-','');const badge=document.createElement('span');badge.className='badge';badge.textContent=item.completed?`${item.completed} 完成`:item.drafts?`${item.drafts} 草稿`:'';const small=document.createElement('small');small.textContent=item.id.split('__')[1]||item.id;b.append(badge,title,small);b.onclick=()=>openCase(item.id);$('caseList').append(b);}$('caseCount').textContent=`${cases.length} 个`;}
function renderEpisodeOptions(){
  const select=$('episodeSelect');
  select.replaceChildren(new Option('新样本（未保存）',''));
  const rows=saved.filter(r=>r.view_id===meta.view_id);
  for(const [index,row] of rows.entries()){
    const label=row.title||`样本 ${index+1} · ${row.marks.t??'?'} → ${row.marks.eval_end??'?'}`;
    select.add(new Option(`${row.status==='complete'?'✓':'○'} ${label}`,row.id));
  }
  select.value=rows.some(r=>r.id===episode.id)?episode.id:'';
}
function renderForm(){if(rangeMode){$("caseName").value=episode.title??'';if(!Object.hasOwn(episode.marks,"prefix_end"))episode.marks.prefix_end=episode.marks.t;episode.boundaries_linked=episode.boundaries_linked??(episode.marks.prefix_end===episode.marks.t);$("linkBoundaries").checked=episode.boundaries_linked;}for(const [key] of markDefs)$(`mark-${key}`).value=episode.marks[key]??'';renderEpisodeOptions();renderRanges();}
async function openCase(caseId,viewId){if(switching)return;switching=true;try{if(!(await preserve()))return;pausePlayback();frameRequest++;const item=cases.find(c=>c.id===caseId);viewId=viewId||item.views[0].id;const newMeta=await api(`/api/cases/${encodeURIComponent(caseId)}/views/${viewId}`);const newSaved=await api(`/api/annotations?case_id=${encodeURIComponent(caseId)}`);const deleted=rangeMode?await api("/api/deleted-annotations"):[];meta=newMeta;saved=newSaved;const select=$('viewSelect');select.replaceChildren();for(const view of item.views)select.add(new Option(`${view.name} · ${view.width}×${view.height}`,view.id));select.value=viewId;episode=saved.filter(r=>r.view_id===viewId).at(-1)||emptyEpisode();let recovered=false;try{const backup=JSON.parse(localStorage.getItem(backupKey()));if(backup&&deleted.includes(backup.id))localStorage.removeItem(backupKey());if(backup&&!deleted.includes(backup.id)&&backup.case_id===caseId&&backup.view_id===viewId){const serverRow=saved.find(r=>r.id===backup.id);if(!serverRow||backup.revision>=serverRow.revision){episode=backup;recovered=true;}}}catch{}$('caseTitle').textContent=caseId;$('videoInfo').textContent=`${meta.frame_count.toLocaleString()} 帧 · 播放 ${fmt(meta.media_duration_seconds)} · 采集 ${fmt(meta.sensor_duration_seconds)} · ${meta.width}×${meta.height}`;$('stage').style.aspectRatio=`${meta.width}/${meta.height}`;$('stage').style.maxWidth=`${540*meta.width/meta.height}px`;$('scrubber').max=meta.frame_count-1;$('frameInput').max=meta.frame_count-1;for(const [key] of markDefs)$(`mark-${key}`).max=meta.frame_count;video.src=`${base()}/video`;video.load();renderForm();renderCases();markClean(episode.revision?`已保存 · 版本 ${episode.revision}`:'新样本');if(recovered){setDirty();toast('已恢复浏览器中的未保存草稿');}seek(episode.marks.t??0);history.replaceState(null,'',`?case=${encodeURIComponent(caseId)}&view=${viewId}`);}catch(e){toast(e.message,true);}finally{switching=false;}}
function setMark(key,frame=current){
  if(!episode)return;
  if(rangeMode&&key==='t'&&frame!==episode.marks.prefix_end){
    episode.boundaries_linked=false;$('linkBoundaries').checked=false;
  }
  episode.marks[key]=frame;$(`mark-${key}`).value=frame??'';
  if(rangeMode&&key==='prefix_end'&&episode.boundaries_linked){
    episode.marks.t=frame;$('mark-t').value=frame??'';
  }
  setDirty();
}
function spans(){const m=episode.marks;if(rangeMode)return {prefix:[m.prefix_start,m.prefix_end],action:[m.t,m.eval_end],evaluation:[null,null],generation:[m.t,m.eval_end]};return {prefix:[m.prefix_start,m.t],action:[m.t,m.action_end],evaluation:[m.eval_start,m.eval_end],generation:[m.t,m.eval_end]};}
function renderRanges(){if(!meta||!episode)return;for(const [key] of markDefs){const f=episode.marks[key];$(`time-${key}`).textContent=Number.isInteger(f)&&f>=0&&f<=meta.frame_count?`播放 ${fmt(mediaAt(f)-mediaAt(0))} · 采集 ${fmt(sensorAt(f)-sensorAt(0))}`:'尚未标记';}const all=spans();for(const [id,key] of [['prefixLane','prefix'],['actionLane','action'],['evalLane','evaluation']]){const [s,e]=all[key];const valid=s!==null&&e!==null&&e>s;$(id).style.display=valid?'block':'none';if(valid){$(id).style.left=`${s/meta.frame_count*100}%`;$(id).style.width=`${(e-s)/meta.frame_count*100}%`;}}const summary=[];for(const [key,label] of [['prefix','前缀'],['generation','生成'],['evaluation','评分']]){const [s,e]=all[key];if(s!==null&&e!==null&&e>s&&s>=0&&e<=meta.frame_count)summary.push(`${label} ${e-s} 帧 / ${(sensorAt(e)-sensorAt(s)).toFixed(2)} 秒（采集）`);}$('rangeSummary').textContent=summary.join('　·　')||'先观看视频，在右侧记录事件边界。';}
async function save(status, notify=true){if(!episode)throw new Error('尚未加载视频');episode.status=status;const payload=Object.fromEntries(payloadKeys.map(k=>[k,episode[k]]));const row=await api('/api/annotations',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});episode=row;saved=saved.filter(r=>r.id!==row.id).concat(row);try{localStorage.removeItem(backupKey());}catch{}markClean(`${status==='complete'?'已完成':'草稿已保存'} · 版本 ${row.revision}`);renderEpisodeOptions();$('validation').className=row.validation_errors.length?'':'ok';$('validation').textContent=row.validation_errors.length?`待完成：${row.validation_errors.join('；')}`:`${markDefs.length} 个边界已标注完成，可以导出。`;cases=await api('/api/cases');renderCases();if(notify)toast(status==='complete'?'标注已完成，可以导出':'草稿已保存');return row;}
for(const [key,label,type] of markDefs){const row=document.createElement('div');row.className=`markRow ${type}`;const title=document.createElement('b');title.textContent=label;const controls=document.createElement('div');controls.className='row';const input=document.createElement('input');input.id=`mark-${key}`;input.type='number';input.min='0';input.placeholder='帧号';input.onchange=()=>{setMark(key,input.value===''?null:Number(input.value));};const set=document.createElement('button');set.textContent='用当前帧';set.onclick=()=>setMark(key);const jump=document.createElement('button');jump.textContent='定位';jump.onclick=()=>{const f=episode.marks[key];if(f!==null)seek(f);};controls.append(input,set,jump);const small=document.createElement('small');small.id=`time-${key}`;row.append(title,controls,small);$('markRows').append(row);}
$('search').oninput=renderCases;$('viewSelect').onchange=()=>openCase(meta.case_id,$('viewSelect').value);
$('episodeSelect').onchange=async()=>{const id=$('episodeSelect').value;if(!(await preserve())){renderEpisodeOptions();return;}episode=saved.find(r=>r.id===id)||emptyEpisode();renderForm();markClean(episode.revision?`已保存 · 版本 ${episode.revision}`:'新样本');seek(episode.marks.t??0);};
$('newEpisode').onclick=async()=>{if(!(await preserve()))return;episode=emptyEpisode();renderForm();markClean('新样本');toast('已创建新样本，可继续标注同一视频');};
$('saveDraft').onclick=()=>save('draft').catch(e=>toast(e.message,true));$('saveComplete').onclick=()=>save('complete').catch(e=>{toast(e.message,true);$('validation').textContent=e.message;$('validation').className='';});
for(const [id,kind] of [['exportInfer','inference'],['exportEval','evaluation']])$(id).onclick=()=>{if(dirty){toast('请先保存或完成当前标注；导出只包含已完成事件',true);return;}window.location.href=`/api/export/${kind}`;toast('已导出所有已完成样本；草稿不参与导出');};
$('play').onclick=play;$('prevFrame').onclick=()=>seek(current-1);$('nextFrame').onclick=()=>seek(current+1);$('backSecond').onclick=()=>seek(frameAtTime(mediaAt(current)-1));$('forwardSecond').onclick=()=>seek(frameAtTime(mediaAt(current)+1));$('speed').onchange=()=>{if(isPlaying()){seek(fastPlayback?current:frameAtTime(video.currentTime),true);play();}else setNativeSpeed(Number($('speed').value));};$('frameInput').onchange=()=>seek(Number($('frameInput').value));
let scrubTimer;$('scrubber').oninput=()=>{clearTimeout(scrubTimer);const f=Number($('scrubber').value);pausePlayback();preview=null;current=f;updatePosition();scrubTimer=setTimeout(()=>seek(f),90);};$('scrubber').onchange=()=>{clearTimeout(scrubTimer);seek(Number($('scrubber').value));};$('timeline').onclick=e=>{const r=$('timeline').getBoundingClientRect();seek((e.clientX-r.left)/r.width*meta.frame_count);};
$('endAtVideo').onclick=()=>setMark('eval_end',meta.frame_count);
document.querySelectorAll('[data-preview]').forEach(b=>b.onclick=()=>{const span=spans()[b.dataset.preview];if(span.some(x=>x===null)||span[0]>=span[1]||span[0]<0||span[1]>meta.frame_count){toast('请先设置这一段的有效边界',true);return;}preview=span;seek(span[0],true);play();});
document.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='s'){e.preventDefault();save('draft').catch(err=>toast(err.message,true));return;}if(['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName)||!meta)return;if(e.key===' '){e.preventDefault();play();}else if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();const d=e.key==='ArrowRight'?1:-1;seek(e.shiftKey?frameAtTime(mediaAt(current)+d):current+d);}else if(/^[1-5]$/.test(e.key)&&Number(e.key)<=markDefs.length){e.preventDefault();setMark(markDefs[Number(e.key)-1][0]);}else if(e.key==='Escape'){preview=null;}});
window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue='';}});
async function init(){try{const [list,health]=await Promise.all([api('/api/cases'),api('/api/health')]);cases=list;$('outputPath').textContent=health.output_root;$('footerPath').textContent=health.output_root;renderCases();const params=new URLSearchParams(location.search);const item=cases.find(c=>c.id===params.get('case'))||cases[0];if(!item)throw new Error('没有找到 WMData RGB 视频');const view=item.views.some(v=>v.id===params.get('view'))?params.get('view'):undefined;await openCase(item.id,view);}catch(e){toast(e.message,true);$('caseTitle').textContent='加载失败';}}
init();

if(rangeMode)$('linkBoundaries').onchange=()=>{episode.boundaries_linked=$('linkBoundaries').checked;if(episode.boundaries_linked)setMark('prefix_end',episode.marks.prefix_end);else setDirty();};

if(rangeMode)$('caseName').oninput=()=>{episode.title=$('caseName').value;setDirty();renderEpisodeOptions();};

if(rangeMode)$('deleteEpisode').onclick=async()=>{
  if(!episode)return;
  const label=episode.title||'当前未命名样本';
  if(!window.confirm(`删除“${label}”？当前未保存修改也会丢弃。已保存的样本会移入回收目录，可恢复。`))return;
  const button=$('deleteEpisode');button.disabled=true;
  try{
    const id=episode.id;
    if(episode.revision>0)await api(`/api/annotations/${encodeURIComponent(id)}?revision=${episode.revision}`,{method:'DELETE'});
    try{localStorage.removeItem(backupKey());}catch{}
    saved=saved.filter(r=>r.id!==id);
    episode=saved.filter(r=>r.view_id===meta.view_id).at(-1)||emptyEpisode();
    preview=null;pausePlayback();renderForm();markClean(episode.revision?`已保存 · 版本 ${episode.revision}`:'新样本');
    $('validation').textContent='';cases=await api('/api/cases');renderCases();
    toast('样本已删除；已保存内容保留在回收目录');
  }catch(e){toast(e.message,true);}finally{button.disabled=false;}
};
