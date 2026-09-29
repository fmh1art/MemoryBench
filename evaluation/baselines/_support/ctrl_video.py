"""Six real context observations + hand-retargeted DROID states; self-generated rollout."""
import sys,json,numpy as np,torch,mediapy
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation
from robot_controls import hand_states
from config import wm_args
from scripts.rollout_replay_traj import agent
folder=Path(sys.argv[1]);out=Path(sys.argv[2]);m=json.loads((folder/'adaptation.json').read_text());d=np.load(folder/'conditions.npz');assert m['context_frames']==6
root=Path('/mnt/vdc_5t/model/acwm');a=wm_args(task_type='replay');a.svd_model_path=str(root/'shared/stable-video-diffusion-img2vid');a.clip_model_path=str(root/'shared/clip-vit-base-patch32');a.ckpt_path=str(root/'Ctrl-World/checkpoint-10000.pt');wm=agent(a)
delta,opening,R=hand_states(d,'right',6)
mid=((wm.state_p01+wm.state_p99)/2).reshape(7);states=np.repeat(mid[None],len(delta),axis=0)
# metres/radians, clipped into official DROID statistics; base offset is synthetic.
states[:,:3]+=delta;relative=R[5].T@R;states[:,3:6]+=Rotation.from_matrix(relative).as_euler('xyz');states[:,6]=opening
states=np.clip(states,wm.state_p01.reshape(7),wm.state_p99.reshape(7));np.save(folder/'droid_states.npy',states)
history=list(d['context_rgb']);generated=[]
def encode(frames):
 x=np.stack([np.array(Image.fromarray(f).resize((320,192))) for f in frames]);x=torch.from_numpy(x).permute(0,3,1,2).to(wm.device,wm.dtype)/127.5-1
 return wm.model.pipeline.vae.encode(x).latent_dist.mode()*wm.model.pipeline.vae.config.scaling_factor
with torch.inference_mode():
 for off in range(0,m['prediction_frames'],5):
  lat=encode(history[-6:]);tiles=torch.cat([lat]*3,dim=2);his=tiles[None];cur=tiles[-1:]
  idx=np.minimum(np.arange(off,off+11),len(states)-1);truth=[lat[-1:].repeat(5,1,1,1)]*3
  _,_,pred,_=wm.forward_wm(states[idx],truth,cur,his_cond=his,text=(folder/'prompt.txt').read_text())
  # Score one head-equivalent view, not a three-camera montage.
  frames=np.asarray(pred[0]);frames=(frames*255).clip(0,255).astype('uint8') if frames.dtype!=np.uint8 else frames
  keep=min(5,m['prediction_frames']-off);history+=list(frames[:keep]);generated+=list(frames[:keep])
mediapy.write_video(str(out),np.stack(generated),fps=m['fps'])
