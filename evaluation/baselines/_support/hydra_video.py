"""Force single-case HyDRA mode and roll forward using generated RGB only."""
import sys,json
from pathlib import Path
import numpy as np,torch,imageio.v2 as imageio
from PIL import Image
import infer_hydra as upstream
from diffsynth import save_video
folder=Path(sys.argv[1]);out=Path(sys.argv[2]);d=np.load(folder/'conditions.npz');m=json.loads((folder/'adaptation.json').read_text())
assert m['context_frames']==77,'HyDRA native context requires 77 frames'
root=Path('/mnt/vdc_5t/model/acwm');base=root/'shared/Wan2.1-T2V-1.3B'
pipe=upstream.build_pipeline('cuda',str(root/'HyDRA/hydra.ckpt'),str(base/'diffusion_pytorch_model.safetensors'),str(base/'models_t5_umt5-xxl-enc-bf16.pth'),str(base/'Wan2.1_VAE.pth'),True)
history=[Image.fromarray(f) for f in d['context_rgb']];poses=d['c2w'];generated=[]
for offset in range(0,m['prediction_frames'],77):
    start=77+offset;ci=np.arange(start-77,start,4);ti=np.minimum(np.arange(start,start+77,4),len(poses)-1)
    ref=np.linalg.inv(poses[start]);ctx=torch.tensor((ref@poses[ci])[:,:3,:].reshape(20,12),dtype=torch.float32)
    tgt=torch.tensor((ref@poses[ti])[:,:3,:].reshape(20,12),dtype=torch.float32)
    # Reuse official normalization/crop, including for self-generated history.
    tmp=out.parent/'rollout_context.mp4';imageio.mimsave(tmp,[np.array(f) for f in history[-77:]],fps=m['fps'])
    condition=upstream.load_condition_video(str(tmp),77,480,832)
    with torch.no_grad():
        pred,_=pipe(prompt=(folder/'prompt.txt').read_text(),negative_prompt='',source_video=condition.to(device=pipe.device,dtype=pipe.torch_dtype),ref_camera=ctx.to(device=pipe.device,dtype=pipe.torch_dtype),target_camera=tgt.to(device=pipe.device,dtype=pipe.torch_dtype),cfg_scale=5,tiled=True,num_frames=77,height=480,width=832,num_inference_steps=50,seed=42+offset)
    keep=min(77,m['prediction_frames']-offset);generated+=list(pred)[:keep];history+=list(pred)[:keep]
save_video(generated,str(out),fps=m['fps'],quality=5)
