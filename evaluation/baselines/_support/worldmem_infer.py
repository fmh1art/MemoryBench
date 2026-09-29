"""Process-local data adapter; preserve upstream model and sampler."""
import sys,numpy as np,torch
from PIL import Image
from pathlib import Path
sample,output=sys.argv[1:3];actions_path=sys.argv[3] if len(sys.argv)>3 else None
# pytorchvideo 0.1.5 still imports the name removed by torchvision 0.17.
import torchvision.transforms._functional_tensor as functional_tensor
sys.modules.setdefault('torchvision.transforms.functional_tensor',functional_tensor)
from datasets.video.minecraft_video_dataset import MinecraftVideoDataset
import json
benchmark=Path(sample).is_dir()
if benchmark:
 folder=Path(sample);meta=json.loads((folder/'adaptation.json').read_text());d=np.load(folder/'conditions.npz')
 count=meta['context_frames'];total=count+meta['prediction_frames']
 # Future RGB is an all-zero placeholder only, never ground truth.
 raw={'rgb':np.concatenate([d['context_rgb'],np.zeros((meta['prediction_frames'],*d['context_rgb'].shape[1:]),np.uint8)]),'c2w':d['c2w']}
 ix=np.arange(total);actions_path=str(folder/'game_actions.npy')
else:
 raw=np.load(sample);ix=np.rint(np.arange(32)*16/10).astype(int);count=16;total=32
frames=np.stack([np.array(Image.fromarray(f).resize((640,360))) for f in raw['rgb'][ix]])
pose=np.linalg.inv(raw['c2w'][0])@raw['c2w'][ix]
xyz=pose[:,:3,3]*np.array([1,-1,1])
forward=pose[:,:3,2]
yaw=np.rad2deg(np.arctan2(forward[:,0],forward[:,2]))%360
pitch=np.rad2deg(np.arcsin(np.clip(forward[:,1],-1,1)))%360
poses=np.column_stack([xyz,pitch,yaw]).astype('float32')
controls=np.load(actions_path) if actions_path else np.zeros((32,25),np.float32)
if controls.shape!=(total,25):raise ValueError('WorldMem actions must be [32,25]')
class OfficeDataset(MinecraftVideoDataset):
 def __init__(self,cfg,split='validation'):
  self.cfg=cfg;self.split=split
 def __len__(self):return 1
 def __getitem__(self,index):
  return torch.from_numpy(frames.copy()).permute(0,3,1,2).float()/255,torch.from_numpy(controls.copy()).float(),poses.copy(),np.arange(total)
# Experiment registry imports its own alias; replace that exact binding.
import datasets, datasets.video
for module in [datasets,datasets.video,sys.modules['datasets.video.minecraft_video_dataset']]:
 if hasattr(module,'MinecraftVideoDataset'):module.MinecraftVideoDataset=OfficeDataset
from algorithms.worldmem import WorldMemMinecraft
# Save predictions directly: upstream gates video output on a wandb logger.
def save_predictions(self, namespace="validation"):
 import imageio.v2 as imageio
 pred=torch.cat([x[0] for x in self.validation_step_outputs],dim=1)
 frames=(pred[:,0].clamp(0,1)*255).byte().permute(0,2,3,1).numpy()
 imageio.mimsave(str(Path(output)/'prediction.mp4'),frames,fps=10)
 self.validation_step_outputs.clear()
WorldMemMinecraft.on_validation_epoch_end=save_predictions
root=Path('/mnt/vdc_5t/model/acwm')
sys.argv=['main.py','+name=memory_office','experiment.tasks=[validation]','wandb.mode=disabled','experiment.validation.batch_size=1','experiment.validation.limit_batch=1',f'hydra.run.dir={output}/hydra',f'+diffusion_model_path={root}/WorldMem/diffusion_only.ckpt',f'+vae_path={root}/WorldMem/vae_only.ckpt','+customized_load=true','+seperate_load=true','dataset.n_frames=8',f'+dataset.n_frames_valid={total}','+dataset.memory_condition_length=8','+dataset.customized_validation=true','+dataset.add_timestamp_embedding=true','+algorithm.n_tokens=8','+algorithm.memory_condition_length=8',f'algorithm.context_frames={count}','+algorithm.relative_embedding=true','+algorithm.log_video=true','+algorithm.add_timestamp_embedding=true']
import runpy
runpy.run_module('main',run_name='__main__')
