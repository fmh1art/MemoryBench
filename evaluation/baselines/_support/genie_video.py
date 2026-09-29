"""Sparse real history + approximate bimanual end-effector controls + action text."""
import sys,json,cv2,numpy as np,yaml
from pathlib import Path
from scipy.spatial.transform import Rotation
from robot_controls import hand_states
folder=Path(sys.argv[1]);out=Path(sys.argv[2]);m=json.loads((folder/'adaptation.json').read_text());d=np.load(folder/'conditions.npz');assert m['context_frames']==4
root=Path('/mnt/vdc_5t/model/acwm');p=folder/'genie';p.mkdir(exist_ok=True)
pad=(-m['prediction_frames'])%25
cameras=np.concatenate([d['c2w'],np.repeat(d['c2w'][-1:],pad,axis=0)])
# Duplicate optical views; this is explicitly a multi-view approximation.
for cam in ['head','hand_left','hand_right']:
 q=p/(cam+'_color');q.mkdir(exist_ok=True)
 for i,f in enumerate(d['context_rgb']):cv2.imwrite(str(q/f'{i}.png'),cv2.cvtColor(f,cv2.COLOR_RGB2BGR))
 np.save(p/f'extrinsic_{cam}.npy',cameras);np.save(p/f'intrinsic_{cam}.npy',d['K'][0])
actions=np.zeros((len(d['c2w']),16),np.float32)
ref=d['c2w'][3]
for side,offset,x in [('left',0,-.15),('right',8,.15)]:
 delta,opening,R=hand_states(d,side,4)
 # Synthetic robot base/neutral reach; changes are taken from actual allowed hands.
 actions[:,offset:offset+3]=(delta+np.array([x,.2,.7]))@ref[:3,:3].T+ref[:3,3]
 actions[:,offset+3:offset+7]=Rotation.from_matrix(ref[:3,:3]@R).as_quat()
 actions[:,offset+7]=opening
actions=np.concatenate([actions,np.repeat(actions[-1:],pad,axis=0)])
np.save(p/'actions.npy',actions)
cfg=yaml.safe_load(Path('configs/cosmos_model/acwm_cosmos.yaml').read_text())
for k in ['pretrained_model_name_or_path','tokenizer_pretrained_model_name_or_path']:cfg[k]=str(root/'Genie-Envisioner/Cosmos-Predict2-2B-Video2World')
cfg['diffusion_model']['model_path']=str(root/'Genie-Envisioner/ge_sim_cosmos_v0.1.safetensors')
y=folder/'genie.yaml';y.write_text(yaml.safe_dump(cfg))
import gesim_video_gen_examples.infer_gesim as native
original_save=native.save_video
# Native visualization stacks RGB/controls vertically and three cameras horizontally.
# Keep only generated head-view RGB for scoring.
def save_prediction(v,path,fps):
 return original_save(v[:,4:4+m['prediction_frames'],:v.shape[2]//2,:v.shape[3]//3],str(out),fps=m['fps'])
native.save_video=save_prediction
native.infer(str(y),str(p),str(p),str(p),str(p/'actions.npy'),(folder/'prompt.txt').read_text(),str(out.parent/'native'),default_fps=m['fps'])
