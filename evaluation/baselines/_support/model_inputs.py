"""CPU-only format adapters. See each top-level model script for semantic limitations."""
import sys,json
from pathlib import Path
import cv2,numpy as np,pyarrow.parquet as pq,yaml
from scipy.spatial.transform import Rotation
from prepare_case import video
BASE=Path(__file__).resolve().parents[1];MODELS=Path('/mnt/vdc_5t/model/acwm')
mode=sys.argv[1];out=Path(sys.argv[2]);d=np.load(out/'sample.npz');rgb=d['rgb'];poses=d['c2w'];K=d['K']
if mode=='hand':
 case=Path(sys.argv[3]);manifest=json.loads((out/'manifest.json').read_text());cam=manifest['camera_frame_id']
 streams=pq.read_table(case/'meta/streams.parquet').to_pylist();files={f['file_id']:f for f in pq.read_table(case/'meta/files.parquet').to_pylist()};ext=pq.read_table(case/'meta/extents.parquet').to_pylist()
 controls=np.zeros_like(rgb);valid_counts={}
 edges=[(0,k) for k in (1,5,9,13,17)]+[(k,k+1) for k in range(1,20) if k%4!=0]
 for side,color in [('left',(0,255,0)),('right',(255,0,0))]:
  # Use only source payloads; fast-garland is a different estimator.
  candidates=[]
  for stream in streams:
   if stream.get('side')!=side or stream.get('schema_id')!='hand/landmarks21@v1':continue
   pieces=sorted([e for e in ext if e['stream_id']==stream['stream_id']],key=lambda e:e['logical_sample_start'])
   if pieces and all(files[e['file_id']]['relative_path'].startswith('data/hand/source/') for e in pieces):candidates.append(pieces)
  if len(candidates)!=1:raise ValueError(f'Expected one source {side} hand stream')
  rows=[]
  for e in candidates[0]:rows+=pq.ParquetFile(case/files[e['file_id']]['relative_path']).read_row_group(e['row_group_id']).to_pylist()
  ts=np.array([r['timestamp_ns'] for r in rows]);valid_counts[side]=0
  for i,t in enumerate(d['timestamp_ns']):
   j=int(np.clip(np.searchsorted(ts,t),0,len(ts)-1))
   if j and abs(ts[j-1]-t)<abs(ts[j]-t):j-=1
   row=rows[j]
   if abs(ts[j]-t)>50_000_000 or row['status']!='valid':continue
   xyz=np.array(row['positions_ref']);xyz=(xyz-poses[i,:3,3])@poses[i,:3,:3]
   uv=xyz@K[i].T;uv=uv[:,:2]/np.maximum(uv[:,2:],1e-8)
   ok=(xyz[:,2]>0.02)&np.isfinite(uv).all(1)&(abs(uv).max(1)<10000)
   for l,r in edges:
    if ok[l] and ok[r]:cv2.line(controls[i],tuple(uv[l].astype(int)),tuple(uv[r].astype(int)),color,5)
   valid_counts[side]+=1
 video(out/'hand_skeleton.mp4',controls,16)
 h,w=rgb.shape[1:3]
 camera={'image_width':w,'image_height':h,'frames':[{'rotation':p[:3,:3].tolist(),'translation':p[:3,3].tolist(),'intrinsics':{'fx':float(k[0,0]),'fy':float(k[1,1]),'cx':float(k[0,2]),'cy':float(k[1,2])}} for p,k in zip(poses,K)]}
 (out/'hand_camera.json').write_text(json.dumps(camera))
 (out/'hand_alignment.json').write_text(json.dumps(valid_counts))
 (out/'hand2world.json').write_text(json.dumps([{'name':'memory_office','scene_image':str(out/'first.png'),'hand_video':str(out/'hand_skeleton.mp4'),'camera_file_path':str(out/'hand_camera.json'),'text':'Egocentric view of hands interacting with objects in an office.','fps':16,'num_frames':161}],indent=2))
elif mode=='genie':
 p=out/'genie';p.mkdir(exist_ok=True)
 ix=np.arange(29)
 for cam in ['head','hand_left','hand_right']:
  q=p/(cam+'_color');q.mkdir(exist_ok=True)
  for i in range(4):cv2.imwrite(str(q/f'{i}.png'),cv2.cvtColor(rgb[i],cv2.COLOR_RGB2BGR))
  np.save(p/f'extrinsic_{cam}.npy',poses[ix]);np.save(p/f'intrinsic_{cam}.npy',K[0])
 actions=np.zeros((29,16),np.float32)
 quat=Rotation.from_matrix(poses[0,:3,:3]).as_quat()
 for offset,x in [(0,-.15),(8,.15)]:
  actions[:,offset:offset+3]=poses[0,:3,:3]@np.array([x,.2,.7])+poses[0,:3,3]
  actions[:,offset+3:offset+7]=quat
 if sys.argv[3]:actions=np.load(sys.argv[3])
 if actions.shape!=(29,16) or not np.isfinite(actions).all():raise ValueError('GE-Sim actions must be finite [29,16]')
 np.save(p/'actions.npy',actions)
 cfg=yaml.safe_load((BASE/'origin_scripts/Genie-Envisioner-V1/configs/cosmos_model/acwm_cosmos.yaml').read_text())
 for key in ['pretrained_model_name_or_path','tokenizer_pretrained_model_name_or_path']:cfg[key]=str(MODELS/'Genie-Envisioner/Cosmos-Predict2-2B-Video2World')
 cfg['diffusion_model']['model_path']=str(MODELS/'Genie-Envisioner/ge_sim_cosmos_v0.1.safetensors')
 (out/'genie.yaml').write_text(yaml.safe_dump(cfg))
