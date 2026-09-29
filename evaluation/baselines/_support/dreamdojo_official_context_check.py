"""Single-clip official inference API; avoids private paths in eval.sh."""
import os,sys,subprocess
from pathlib import Path
import numpy as np,torch,imageio.v2 as imageio
from PIL import Image
from cosmos_oss.init import init_environment,cleanup_environment
root=Path('/mnt/vdc_5t/model/acwm');ck=root/'DreamDojo/2B_GR1_post-train/iter_000050000'
if not (ck/'model_ema_bf16.pt').exists():
 subprocess.run([sys.executable,'scripts/convert_distcp_to_pt.py',str(ck/'model'),str(ck)],check=True)
# Config registration eagerly resolves unrelated 2B/14B base training weights.
# model_loader overrides checkpoint.load_path with our explicit GR1 checkpoint;
# use that same file for these unused default entries, not extra base downloads.
from cosmos_predict2._src.imaginaire.utils import checkpoint_db
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey
base_uris={MODEL_CHECKPOINTS[ModelKey(post_trained=post,size=size)].s3.uri for size in ['2B','14B'] for post in [False,True]}
resolve_checkpoint=checkpoint_db.get_checkpoint_path
def resolve_gr1(uri):
 # Upstream tokenizer revision 6787e176 returns 404; pin accessible official revision.
 if uri.rstrip('/')=='s3://bucket/cosmos_diffusion_v2/pretrain_weights/tokenizer/wan2pt1/Wan2.1_VAE.pth':
  from huggingface_hub import hf_hub_download
  return hf_hub_download('nvidia/Cosmos-Predict2.5-2B','tokenizer.pth',revision='85f8ae7bfe8f5525c8d103429524dcf12f98bf7b')
 return str(ck/'model_ema_bf16.pt') if uri in base_uris else resolve_checkpoint(uri)
checkpoint_db.get_checkpoint_path=resolve_gr1
init_environment()
from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
runner=Video2WorldInference(experiment_name='cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes_release_oss',ckpt_path=str(ck/'model_ema_bf16.pt'),s3_credential_path='',config_file='cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py',offload_text_encoder=True,experiment_opts=['~data_val'])

import json,piq
out=root/'DreamDojo/official_validation/episode_000000'
data=np.load(out/'official_sample.npz');gt=data['video'][:13];actions=data['action'][:12]
report={'context_frames':5,'conditional_latents':2,'prompt':'','guidance':0,'seed':0,'modes':{}}
for mode in ['real_actions','zero_actions']:
    video=torch.zeros((1,3,13,480,640),dtype=torch.uint8)
    video[0,:,:5]=torch.from_numpy(gt[:5].copy()).permute(3,0,1,2)
    control=actions.copy() if mode=='real_actions' else np.zeros_like(actions)
    with torch.inference_mode():
        pred=runner.generate_vid2world(prompt='',input_path=video,action=torch.from_numpy(control).float(),guidance=0,num_video_frames=13,num_latent_conditional_frames=2,resolution='480,640',seed=0,negative_prompt='',lam_video=None)
    frames=((pred[0].clamp(-1,1)+1)*127.5).byte().permute(1,2,3,0).cpu().numpy()
    imageio.mimsave(out/('context5_'+mode+'.mp4'),frames,fps=10)
    imageio.mimsave(out/('context5_'+mode+'_vs_gt.mp4'),np.concatenate([gt,frames],axis=2),fps=10)
    p=torch.from_numpy(frames[5:].copy()).permute(0,3,1,2).float()/255
    g=torch.from_numpy(gt[5:].copy()).permute(0,3,1,2).float()/255
    report['modes'][mode]={'future_frames':8,'psnr_db':float(piq.psnr(p,g,data_range=1.0)),'ssim':float(piq.ssim(p,g,data_range=1.0))}
    (out/'context5_results.json').write_text(json.dumps(report,indent=2))
    print(mode,report['modes'][mode],flush=True)
cleanup_environment()
