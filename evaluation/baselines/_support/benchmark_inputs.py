"""Adapt existing annotated evaluation cases; never open ground_truth RGB.
All indices refer to the source case. Future poses/hands are allowed controls.
"""
import argparse,json,math
from pathlib import Path
import av,cv2,numpy as np
try:
    from .prepare_case import video
except ImportError:
    from prepare_case import video

ACTION_KEYS=['inventory','ESC']+[f'hotbar.{i}' for i in range(1,10)]+['forward','back','left','right','cameraX','cameraY','jump','sneak','sprint','swapHands','attack','use','pickItem','drop']
TEXT={
 'open_drawer':'Open the target drawer by grasping its handle and pulling it outward, then keep the view steady.',
 'turn_view':'Turn the viewpoint toward the target area, then keep the view steady.',
 'close_drawer':'Push the target drawer closed, then keep the view steady.',
}

def read_selected(path,indices):
    wanted=set(map(int,indices));found={}
    with av.open(str(path)) as c:
        stream=c.streams.video[0];rate=stream.average_rate
        if rate is None:raise ValueError('Exported video needs a frame rate')
        # Exported case videos are CFR. Seek to each cluster, avoiding decoding minutes of unused prefix.
        remaining=sorted(wanted)
        while remaining:
            first=remaining[0];last=first
            for i in remaining[1:]:
                if i-last>120:break
                last=i
            c.seek(int(first/rate/stream.time_base),stream=stream,backward=True)
            for f in c.decode(video=0):
                i=round(f.pts*f.time_base*rate)
                if i in wanted:found[i]=f.to_ndarray(format='rgb24')
                if i>=last:break
            missing=[i for i in remaining if i<=last and i not in found]
            if missing:raise ValueError(f'Cannot resolve exported CFR frame {missing[0]} in {path}')
            remaining=[i for i in remaining if i not in found]
    if len(found)!=len(wanted):raise ValueError(f'Missing selected frames in {path}')
    return np.stack([found[int(i)] for i in indices])

def nearest(ts,target,lo,hi):
    # Bounds prevent a history timestamp from selecting the first future RGB.
    t=ts[lo:hi];j=np.searchsorted(t,target).clip(0,len(t)-1)
    prev=np.maximum(j-1,0);j=np.where(abs(t[prev]-target)<abs(t[j]-target),prev,j)
    return j+lo

def game_controls(poses,times,read_action,action_end,begin):
    """Optical c2w: +z forward, +x right; normalized Minecraft mouse deltas.
    Translation threshold .05 m/s, camera +/-20 degrees per output interval.
    Generic object manipulation maps to held use (approximate, not robot IK).
    """
    n=len(poses);a=np.zeros((n,25),np.float32)
    for i in range(1,n):
        rel=np.linalg.inv(poses[i-1])@poses[i];dt=max((times[i]-times[i-1])/1e9,1e-6)
        vel=rel[:3,3]/dt
        a[i,11:15]=[vel[2]>.05,vel[2]<-.05,vel[0]<-.05,vel[0]>.05]
        f=rel[:3,2];pitch=np.degrees(np.arcsin(np.clip(f[1],-1,1)))
        yaw=np.degrees(np.arctan2(f[0],f[2]))
        a[i,15:17]=np.clip([pitch/20,yaw/20],-1,1)
        # Minecraft use is only a coarse interaction proxy; no object-answer labels.
        if begin<=i<action_end and read_action in ['open_drawer','close_drawer']:a[i,22]=1
    return a

def prepare(case,out,model,context_frames=None,prediction_frames=None,prompt=None):
    case=Path(case).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((case/'case.json').read_text());m=meta['marks']
    ann=json.loads((case/'memory_annotation.json').read_text())
    read_action=ann['read_action']
    # Only the task's action instruction is admitted; never memory_target/success_criterion/GT.
    text=prompt or TEXT.get(read_action,ann['action_description'])
    raw=np.load(case/'input/aligned_conditions.npz');src=raw['source_frame_indices'];ts=raw['sensor_timestamp_ns']
    split=int(np.searchsorted(src,m['t']));end=int(np.searchsorted(src,m['eval_end']))
    assert 0<split<end and src[split-1]==m['t']-1
    specs={'Oasis':(20,16),'Hand2World':(16,77),'HyDRA':(16,77),'WorldMem':(10,32),
           'Ctrl-World':(5,6),'Genie-Envisioner':(16,4),'DreamDojo':(16,5),'iVideoGPT':(4,2),'Matrix-Game-3.0':(16,1)}
    fps,nctx=specs[model];nctx=context_frames or nctx
    if nctx<1:raise ValueError('context_frames must be positive')
    if model=='Hand2World' and (nctx-1)%4:raise ValueError('Hand2World context length must be 4k+1')
    hist_target=ts[split-1]-np.rint(np.arange(nctx-1,-1,-1)/fps*1e9).astype(np.int64)
    hist=nearest(ts,hist_target,0,split)
    if model=='Genie-Envisioner':hist=np.rint(np.linspace(0,split-1,nctx)).astype(int)
    # Timeline starts at t; duration is defined by registered sensor timestamps.
    duration=meta['sensor_duration_seconds']['generation']
    nf=prediction_frames or math.ceil(duration*fps)
    if nf<1:raise ValueError('prediction_frames must be positive')
    if prediction_frames and nf>math.ceil(duration*fps):raise ValueError('Requested horizon exceeds annotated interval')
    future_target=ts[split]+np.rint(np.arange(nf)/fps*1e9).astype(np.int64)
    future=nearest(ts,future_target,split,end)
    ix=np.r_[hist,future]
    rgb=read_selected(case/'input/rgb_prefix.mp4',hist) # prefix file contains no future RGB
    video(out/'context.mp4',rgb,fps);cv2.imwrite(str(out/'anchor.png'),cv2.cvtColor(rgb[-1],cv2.COLOR_RGB2BGR))
    poses=raw['c2w_opencv'][ix];times=ts[ix]
    ae=nctx+int(np.searchsorted(src[future],m['action_end']))
    controls=game_controls(poses,times,read_action,ae,nctx)
    np.save(out/'game_actions.npy',controls)
    arrays={'c2w':poses,'timestamp_ns':times,'source_frame_indices':src[ix],'context_rgb':rgb,'future_timestamp_ns':future_target}
    for key in raw.files:
        if key.endswith('_positions_ref') or key.endswith('_hand_valid'):arrays[key]=raw[key][ix]
    if model in ['Hand2World','Genie-Envisioner']:
        import pyarrow.parquet as pq
        source=Path('/mnt/vdc_5t/data/acwm/memory-office')/meta['source_case_id']
        streams=pq.read_table(source/'meta/streams.parquet').to_pylist()
        files={f['file_id']:f for f in pq.read_table(source/'meta/files.parquet').to_pylist()}
        rgbstream=next(s for s in streams if s['stream_id']==meta['rgb_stream_id'])
        cal=next(s for s in streams if s['schema_id']=='camera_calibration/pinhole@v1' and s['camera_frame_id']==rgbstream['camera_frame_id'])
        rows=[]
        for e in pq.read_table(source/'meta/extents.parquet').to_pylist():
            if e['stream_id']==cal['stream_id']:rows+=pq.ParquetFile(source/files[e['file_id']]['relative_path']).read_row_group(e['row_group_id']).to_pylist()
        rows.sort(key=lambda x:x['timestamp_ns']);kt=np.array([r['timestamp_ns'] for r in rows]);intr=[]
        for t in times:
            j=np.searchsorted(kt,t,side='right')-1
            if j<0 or rows[j]['status']!='valid':raise ValueError('Missing registered camera calibration')
            k=rows[j];intr.append([[k['fx'],0,k['cx']],[0,k['fy'],k['cy']],[0,0,1]])
        arrays['K']=np.array(intr,dtype=np.float32)
    np.savez_compressed(out/'conditions.npz',**arrays)
    # Rendered hands are actions, so historical and future hand files are allowed.
    if model=='Hand2World':
        hp=read_selected(case/'input/past_hands.mp4',hist)
        hf=read_selected(case/'input/future_hands.mp4',future-split)
        hands=np.concatenate([hp,hf]);pad=(-(len(hands)-1))%4
        if pad:hands=np.concatenate([hands,np.repeat(hands[-1:],pad,axis=0)])
        video(out/'hands.mp4',hands,fps)
        # case exporter uses source calibration; resolve camera intrinsics from its DA3 file separately.
    (out/'prompt.txt').write_text('First-person view. '+text+'\n')
    manifest={'model':model,'case':str(case),'marks':m,'fps':fps,'context_frames':nctx,'prediction_frames':nf,
      'context_source_frames':src[hist].tolist(),'future_source_frames':src[future].tolist(),
      'future_timestamp_ns':future_target.tolist(),'context_policy':'uniform sparse entire prefix' if model=='Genie-Envisioner' else 'recent history; repeat earliest if short',
      'action_text':text,'action_source':'explicit --prompt' if prompt else 'read_action template / action_description only',
      'read_action':read_action,'game_mapping':{'translation_threshold_mps':.05,'mouse_clip_degrees':20,'interaction':'open/close_drawer -> use while action active'},
      'limitations':['WASD/mouse encode camera motion, not calibrated game dynamics','use is an approximate interaction; no object identity or contents encoded','fixed context budget may omit earlier memory events'],
      'future_rgb_read':False,'duration_complete':prediction_frames is None}
    modes={
      'Oasis':'head_pose_to_WASD_mouse_plus_use', 'WorldMem':'head_pose_to_WASD_mouse_plus_use_and_pose',
      'Hand2World':'hand_raster_and_camera_with_experimental_observed_KV_prefill',
      'HyDRA':'camera_plus_action_text_no_hand_branch', 'DreamDojo':'action_text_plus_neutral_GR1',
      'Ctrl-World':'heuristic_right_hand_to_DROID_plus_text', 'Genie-Envisioner':'heuristic_bimanual_end_effector_plus_text',
      'iVideoGPT':'heuristic_right_hand_to_RoboDesk', 'Matrix-Game-3.0':'WASD_mouse_plus_text_anchor_only'}
    limitations=['Fixed history budget may omit earlier memory events; source indices are recorded.']
    if model in ['Oasis','WorldMem','Matrix-Game-3.0']:limitations+=['Camera-to-game dynamics are not calibrated; keyboard/mouse is an approximation.']
    if model in ['Oasis','WorldMem']:limitations+=['Minecraft use is only an interaction proxy; the model has no text interface.']
    if model in ['Ctrl-World','Genie-Envisioner','iVideoGPT']:limitations+=['Hand-to-robot control is heuristic, without calibrated robot base or IK.']
    if model in ['Ctrl-World','Genie-Envisioner']:limitations+=['One observed view is duplicated into native multi-camera slots.']
    if model=='Hand2World':limitations+=['Observed KV prefill is experimental; native KV window remains finite.','Hand rasters are skeletons, not native MANO mesh renders.']
    if model=='DreamDojo':limitations+=['Action intent is supplied as text; GR1 controls remain neutral, not physically retargeted.']
    if model=='HyDRA':limitations+=['Camera and action text only; no explicit hand-control branch.']
    if model=='Matrix-Game-3.0':limitations+=['Only t-1 is used from external video; this is a reduced-history baseline.']
    manifest['limitations']=limitations
    manifest['action_adapter']=modes[model]
    manifest['history_used']= 'last_frame_only' if model=='Matrix-Game-3.0' else 'observed_video_subset'
    manifest['native_hand_control']=model=='Hand2World'
    manifest['action_semantics_verified']=False
    manifest['rgb_inputs']=[str(case/'input/rgb_prefix.mp4')]
    manifest['unmapped_game_action']=read_action not in ['open_drawer','close_drawer','turn_view']
    (out/'adaptation.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--benchmark-case',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',required=True);p.add_argument('--context-frames',type=int);p.add_argument('--prediction-frames',type=int);p.add_argument('--prompt')
    a=p.parse_args();prepare(a.benchmark_case,a.output,a.model,a.context_frames,a.prediction_frames,a.prompt)
