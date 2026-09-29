"""Isolated inference environments. Training/demo-only dependencies are omitted.
Run: python _support/setup_env.py iVideoGPT (or one of the other manifest names).
"""
import os,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];ROOT=Path('/mnt/vdc_5t/model/acwm')
name=sys.argv[1];envdir=ROOT/'.envs'/name
run_env=os.environ.copy();run_env.update(UV_DEFAULT_INDEX='https://pypi.org/simple',UV_INDEX_URL='https://pypi.org/simple',UV_HTTP_TIMEOUT='120',UV_LINK_MODE='copy',MAX_JOBS='4')
def run(args,cwd=None):subprocess.run([str(x) for x in args],cwd=cwd,env=run_env,check=True)
if not (envdir/'bin/python').exists():run(['uv','venv','--python','/usr/bin/python3.10',envdir])
py=envdir/'bin/python'
def pip(*args):run(['uv','pip','install','--python',py,'--index-url','https://pypi.org/simple',*args])
common=['imageio','imageio-ffmpeg','av','Pillow','numpy<2','PyYAML','packaging','setuptools<81','wheel','einops','scipy','safetensors','opencv-python-headless<4.12']
if name=='iVideoGPT':
 pip('torch==2.2.1','torchvision==0.17.1',*common,'transformers==4.38.2','diffusers==0.27.0','huggingface_hub==0.25.2','accelerate==0.29.3')
elif name=='Hand2World':
 pip('torch==2.7.0','torchvision==0.22.0',*common,'diffusers==0.36.0','transformers==4.57.3','accelerate==1.12.0','peft==0.18.1','safetensors==0.4.5','decord==0.6.0','moviepy==1.0.3','albumentations','func-timeout','omegaconf==2.3.0','sentencepiece','ninja','psutil')
 # Official AR path supports SDPA without flash-attn (static KV cache is disabled).
 # Do not build a CUDA extension with incompatible host CUDA 12.4 / torch CUDA 12.6.
elif name=='Ctrl-World':
 pip('torch==2.7.1','torchvision==0.22.1',*common,'diffusers==0.34.0','transformers==4.48.1','mediapy','wandb','swanlab','accelerate','decord','pandas','sentencepiece','timm')
elif name=='Genie-Envisioner':
 pip('torch==2.7.1','torchvision==0.22.1',*common,'diffusers==0.32.0','transformers==4.51.3','accelerate==1.0.0','matplotlib','decord','h5py','tensorboard','sentencepiece','pandas','omegaconf','timm')
elif name=='WorldMem':
 pip('torch==2.4.1','torchvision==0.19.1',*common,'lightning==2.1.4','wandb==0.17.9','hydra-core==1.3.2','omegaconf==2.3.0','torchmetrics[image]==0.11.4','wandb-osh==1.2.1','gluonts[torch]==0.13.1','pytorchvideo==0.1.5','colorama','matplotlib','moviepy==1.0.3','pandas','pyzmq','h5py','rotary_embedding_torch','diffusers==0.30.3','timm==0.9.16','gradio','spaces','lpips')
elif name=='Oasis':
 pip('torch==2.4.1','torchvision==0.19.1',*common,'diffusers==0.30.3','timm==0.9.16','av==12.3.0')
elif name=='HyDRA':
 pip('torch==2.7.1','torchvision==0.22.1',*common,'cupy-cuda12x','transformers==4.46.2','controlnet-aux==0.0.7','sentencepiece','protobuf','modelscope','ftfy','accelerate','timm')
elif name=='Matrix-Game-3.0':
 # This repository explicitly pins torch 2.10 for its streaming sampler.
 req=BASE/'origin_scripts/Matrix-Game/Matrix-Game-3/requirements.txt'
 pip('-r',str(req))
elif name=='NWM':
 pip('torch==2.7.1','torchvision==0.22.1',*common,'decord','evo','transformers==4.48.1','diffusers==0.34.0','timm','dreamsim','torcheval','lpips','matplotlib','pandas')
elif name=='DreamDojo':
 run_env['UV_PROJECT_ENVIRONMENT']=str(envdir)
 run(['uv','sync','--extra=cu128','--python','/usr/bin/python3.10'],BASE/'origin_scripts/DreamDojo')
 pip('openai','tyro','numpydantic','albumentations','tianshou','pyarrow','fastparquet','piq','lightning','moviepy==2.2.1','jsonargparse[signatures]>=4.27.7','imageio','imageio-ffmpeg')
 # Only pure-PyTorch transforms are required by GR1 input normalization.
 run_env['PYTORCH3D_NO_EXTENSION']='1'
 pip('--no-build-isolation',str(BASE/'origin_scripts/pytorch3d'))
else:raise SystemExit('Unknown model '+name)
(envdir/'setup.ok').write_text('Dependency installation completed; this does NOT mean inference passed.\n')
print(py)
