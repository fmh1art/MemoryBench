"""Small official-agent adapter; no edits to upstream hardcoded DROID paths."""
import sys,numpy as np,torch,mediapy
from PIL import Image
from config import wm_args
from scripts.rollout_replay_traj import agent
from pathlib import Path
root=Path('/mnt/vdc_5t/model/acwm');a=wm_args(task_type='replay')
a.svd_model_path=str(root/'shared/stable-video-diffusion-img2vid');a.clip_model_path=str(root/'shared/clip-vit-base-patch32');a.ckpt_path=str(root/'Ctrl-World/checkpoint-10000.pt')
wm=agent(a)
im=np.array(Image.open(sys.argv[1]).convert('RGB').resize((320,192)))
x=torch.from_numpy(im).permute(2,0,1)[None].to(wm.device,wm.dtype)/127.5-1
with torch.inference_mode():
 latent=wm.model.pipeline.vae.encode(x).latent_dist.mode()*wm.model.pipeline.vae.config.scaling_factor
 current=torch.cat([latent]*3,dim=2)
 history=current[:,None].repeat(1,6,1,1,1)
 control=np.load(sys.argv[3]) if len(sys.argv)>3 else np.repeat((wm.state_p01+wm.state_p99)/2,11,axis=0)
 if control.shape!=(11,7):raise ValueError('Expected [11,7] DROID states')
 truth=[latent.repeat(5,1,1,1)]*3 # used ONLY by upstream visualization, never fed as future conditioning
 _,_,pred,_=wm.forward_wm(control,truth,current,his_cond=history,text='An office scene.')
 frames=np.concatenate(pred,axis=2)
 mediapy.write_video(sys.argv[2],frames,fps=5)
