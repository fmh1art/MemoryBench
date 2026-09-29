"""Official Oasis sampler with correct THWC->TCHW video loading and future-only output."""
import sys,json
from pathlib import Path
import numpy as np,torch,av
import torch.nn.functional as F
from torchvision.io import write_video
import generate

def read_prompt(path,video_offset=None,n_prompt_frames=1):
    with av.open(path) as c:frames=[f.to_ndarray(format='rgb24') for f in c.decode(video=0)]
    frames=frames[(video_offset or 0):][:n_prompt_frames]
    if len(frames)!=n_prompt_frames:raise ValueError('Insufficient context frames')
    x=torch.from_numpy(np.stack(frames)).permute(0,3,1,2).float()/255
    return F.interpolate(x,size=(360,640),mode='bilinear',align_corners=False).unsqueeze(0)

if __name__=='__main__':
    folder=Path(sys.argv[1]);out=Path(sys.argv[2]);meta=json.loads((folder/'adaptation.json').read_text())
    n=meta['context_frames'];total=n+meta['prediction_frames']
    actions=np.load(folder/'game_actions.npy')
    assert actions.shape==(total,25)
    # Native loader inserts a leading no-op; our rows denote transitions arriving at each frame.
    generate.load_actions=lambda *a,**k:torch.from_numpy(actions).unsqueeze(0)
    generate.load_prompt=read_prompt
    def save_future(path,frames,fps):write_video(str(out),frames[n:],fps=meta['fps'])
    generate.write_video=save_future
    from argparse import Namespace
    root=Path('/mnt/vdc_5t/model/acwm/Oasis')
    generate.main(Namespace(oasis_ckpt=str(root/'oasis500m.safetensors'),vae_ckpt=str(root/'vit-l-20.safetensors'),
       n_prompt_frames=n,num_frames=total,ddim_steps=10,prompt_path=str(folder/'context.mp4'),
       video_offset=None,actions_path=str(folder/'game_actions.npy'),output_path=str(out),fps=meta['fps']))
