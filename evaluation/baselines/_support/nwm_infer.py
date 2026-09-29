"""Adapted from upstream interactive_model.ipynb; local checkpoint and VAE."""
import sys,yaml,numpy as np,torch,imageio.v2 as imageio
from pathlib import Path
from PIL import Image
from diffusion import create_diffusion
from isolated_nwm_infer import model_forward_wrapper
from models import CDiT_models
from diffusers import AutoencoderKL
root=Path('/mnt/vdc_5t/model/acwm');cfg=yaml.safe_load(open('config/data_config.yaml'));cfg.update(yaml.safe_load(open('config/nwm_cdit_xl.yaml')))
device='cuda';latent_size=cfg['image_size']//8
model=CDiT_models[cfg['model']](input_size=latent_size,context_size=cfg['context_size'])
model.load_state_dict(torch.load(root/'NWM/cdit_xl_100000.pth.tar',map_location='cpu',weights_only=False)['ema'],strict=True);model.eval().to(device)
vae=AutoencoderKL.from_pretrained(root/'shared/sd-vae-ft-ema').to(device);diffusion=create_diffusion('250')
d=np.load(sys.argv[1]);ix=np.arange(8)*4
imgs=np.stack([np.array(Image.fromarray(x).resize((cfg['image_size'],cfg['image_size']))) for x in d['rgb'][ix]])
frames=torch.from_numpy(imgs).permute(0,3,1,2).float().to(device)/127.5-1
controls=[]
for i in range(3,7):
 rel=np.linalg.inv(d['c2w'][ix[i]])@d['c2w'][ix[i+1]]
 controls.append([rel[2,3],-rel[0,3],-np.arctan2(rel[0,2],rel[2,2])])
controls=np.load(sys.argv[3]) if len(sys.argv)>3 else np.array(controls,dtype=np.float32)
if controls.shape!=(4,3):raise ValueError('NWM controls must be [4,3]')
history=frames[:4];result=[]
torch.manual_seed(42)
with torch.inference_mode():
 for control in controls:
  sample=model_forward_wrapper((model,diffusion,vae),history[-cfg['context_size']:][None],torch.tensor(control,device=device,dtype=torch.float32)[None],None,latent_size,device,cfg['context_size'],num_goals=1,rel_t=torch.tensor([1./128],device=device),progress=True)
  history=torch.cat([history,sample],dim=0);result.append(((sample[0]+1)*127.5).clamp(0,255).byte().permute(1,2,0).cpu().numpy())
imageio.mimsave(sys.argv[2],result,fps=4)
