"""Bounded WMData reader: registered capture times, matching camera stream, valid pose only.
Rotation6D stores rotation COLUMNS (wmdata_hand_review.geometry.rotation_from_6d).
No robot/game action is inferred from hand landmarks. Model wrappers document substitutes.
"""
import argparse,json
from pathlib import Path
import cv2,numpy as np,pyarrow.parquet as pq

def video(path,frames,fps):
    h,w=frames[0].shape[:2]
    wr=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    if not wr.isOpened():raise RuntimeError('Video writer failed: '+str(path))
    for f in frames:wr.write(cv2.cvtColor(f,cv2.COLOR_RGB2BGR))
    wr.release()

def prepare(case,out,start):
    out.mkdir(parents=True,exist_ok=True)
    stamp={'case':str(case),'start':start,'frames':161,'fps':16,'version':2}
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text()).get('source')==stamp:return
    tables={n:pq.read_table(case/'meta'/f'{n}.parquet').to_pylist() for n in ['streams','files','extents']}
    files={x['file_id']:x for x in tables['files']}
    rgb=[s for s in tables['streams'] if s['schema_id'].startswith('rgb/')]
    if len(rgb)!=1:raise ValueError(f'Expected one RGB stream, found {len(rgb)}')
    rgb=rgb[0];cam=rgb['camera_frame_id']
    def stream(schema):
        ss=[s for s in tables['streams'] if s['schema_id']==schema and s['camera_frame_id']==cam]
        if len(ss)!=1:raise ValueError('Ambiguous/missing '+schema)
        return ss[0]
    def rows(s):
        result=[]
        for e in sorted([e for e in tables['extents'] if e['stream_id']==s['stream_id']],key=lambda x:x['logical_sample_start']):
            result+=pq.ParquetFile(case/files[e['file_id']]['relative_path']).read_row_group(e['row_group_id']).to_pylist()
        return result
    timeline=[]
    for e in sorted([e for e in tables['extents'] if e['stream_id']==rgb['stream_id']],key=lambda x:x['logical_sample_start']):
        timeline.extend((t,e['file_id'],pts) for t,pts in zip(e['video_timestamp_ns'],e['video_frame_pts']))
    times=np.array([x[0] for x in timeline],np.int64)
    target=times[0]+np.rint((start+np.arange(161)/16)*1e9).astype(np.int64)
    if target[-1]>times[-1]:raise ValueError('Case too short for 10-second sample')
    idx=np.searchsorted(times,target);idx=np.clip(idx,1,len(times)-1)
    idx-=abs(times[idx-1]-target)<abs(times[idx]-target)
    if np.any(abs(times[idx]-target)>50_000_000):raise ValueError('RGB gap exceeds 50ms at requested sample')
    chosen=[timeline[i] for i in idx]
    # Decode registered PTS, not assumed constant-rate frame indexes.
    import av
    frames=[None]*len(chosen)
    for fid in dict.fromkeys(x[1] for x in chosen):
        f=files[fid];wanted={}
        for i,(_,k,pts) in enumerate(chosen):
            if k==fid:wanted.setdefault(pts,[]).append(i)
        from fractions import Fraction
        base=Fraction(f['video_time_base_num'],f['video_time_base_den'])
        with av.open(str(case/f['relative_path'])) as c:
            vs=c.streams.video[0];c.seek(int(min(wanted)*base/vs.time_base),stream=vs,backward=True)
            for frame in c.decode(vs):
                pts=Fraction(frame.pts)*frame.time_base/base
                if pts in wanted:
                    arr=frame.to_ndarray(format='rgb24')
                    for i in wanted.pop(pts):frames[i]=arr
                if not wanted:break
        if wanted:raise ValueError('Registered RGB frames missing')
    ps=rows(stream('camera_pose/pose9@v1'));pt=np.array([p['timestamp_ns'] for p in ps])
    Ks=rows(stream('camera_calibration/pinhole@v1'));kt=np.array([k['timestamp_ns'] for k in Ks])
    poses=[];intr=[]
    for t,_,_ in chosen:
        j=int(np.clip(np.searchsorted(pt,t),0,len(pt)-1))
        if j and abs(pt[j-1]-t)<abs(pt[j]-t):j-=1
        p=ps[j]
        if abs(pt[j]-t)>50_000_000 or p['status']!='valid':raise ValueError('No valid camera pose within 50ms')
        x=np.array(p['pose9']);a=x[3:6];b=x[6:9]
        if max(abs(np.linalg.norm(a)-1),abs(np.linalg.norm(b)-1),abs(a@b))>1e-4:raise ValueError('Malformed pose9')
        M=np.eye(4);M[:3,:3]=np.column_stack((a,b,np.cross(a,b)));M[:3,3]=x[:3];poses.append(M)
        ki=np.searchsorted(kt,t,side='right')-1
        if ki<0:raise ValueError('No calibration at or before RGB timestamp')
        k=Ks[ki]
        if k['status']!='valid':raise ValueError('Invalid calibration')
        intr.append([[k['fx'],0,k['cx']],[0,k['fy'],k['cy']],[0,0,1]])
    frames=np.stack(frames);poses=np.stack(poses).astype('float32');intr=np.array(intr,dtype='float32')
    np.savez(out/'sample.npz',rgb=frames,c2w=poses,K=intr,timestamp_ns=times[idx])
    cv2.imwrite(str(out/'first.png'),cv2.cvtColor(frames[0],cv2.COLOR_RGB2BGR))
    video(out/'clip.mp4',frames,16)
    video(out/'history.mp4',frames[:77],16)
    (out/'manifest.json').write_text(json.dumps({'source':stamp,'rgb_stream_id':rgb['stream_id'],'camera_frame_id':cam,'selected_frame_indices':idx.tolist(),'timestamp_ns':times[idx].tolist(),'pose_alignment':'nearest valid <=50ms; rotation columns; optical camera-to-reference, metres','shape':list(frames.shape)},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--start',type=float,default=2);a=p.parse_args();prepare(a.case,a.output,a.start)
