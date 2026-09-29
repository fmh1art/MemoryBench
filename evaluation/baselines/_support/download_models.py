"""Sequential, resumable model queue. Run in tmux; failed models do not block later ones."""
import os, sys, json, time, subprocess, traceback
from pathlib import Path
ROOT=Path('/mnt/vdc_5t/model/acwm'); ROOT.mkdir(parents=True,exist_ok=True)
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='90'
from huggingface_hub import snapshot_download, hf_hub_download

def hf(repo, dest, patterns=None):
    return snapshot_download(repo,local_dir=ROOT/dest,allow_patterns=patterns,max_workers=4)

def dream():
    hf('nvidia/DreamDojo','DreamDojo',['2B_GR1_post-train/*/model/*','2B_GR1_post-train/latest_checkpoint.txt','README.md','config.json'])
    # Use the same cache as the official inference checkpoint resolver.
    hf_hub_download('nvidia/Cosmos-Predict2.5-2B','tokenizer.pth',revision='85f8ae7bfe8f5525c8d103429524dcf12f98bf7b',cache_dir=ROOT/'.hf_cache/hub')

def hand():
    import gdown
    dest=ROOT/'Hand2World/checkpoints';dest.mkdir(parents=True,exist_ok=True)
    fs=gdown.download_folder(id='1J37kycnFz8jvkyN0Wkm9vrsXmHjqBms-',output=str(dest),skip_download=True,quiet=True,timeout=60)
    for f in fs:
        if not f.path.startswith('V_0.9/'): continue
        p=dest/f.path;p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists() and p.stat().st_size>0: continue
        tmp=str(p)+'.part'
        result=gdown.download(id=f.id,output=tmp,resume=True,timeout=(30,90),retries=3)
        if not result: raise RuntimeError('Google Drive download failed: '+f.path)
        os.replace(tmp,p)

def genie():
    # ModelScope metadata AND redirects must bypass every environment proxy.
    env=os.environ.copy()
    for k in ('http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY'):env.pop(k,None)
    env.update(NO_PROXY='*',no_proxy='*')
    code='''import requests, pathlib, os
s=requests.Session();s.trust_env=False
p=pathlib.Path('/mnt/vdc_5t/model/acwm/Genie-Envisioner/ge_sim_cosmos_v0.1.safetensors');p.parent.mkdir(parents=True,exist_ok=True)
if not p.exists():
 t=pathlib.Path(str(p)+'.part');n=t.stat().st_size if t.exists() else 0
 u='https://modelscope.cn/models/agibot_world/Genie-Envisioner/resolve/master/ge_sim_cosmos_v0.1.safetensors'
 with s.get(u,headers={'Range':f'bytes={n}-'} if n else {},stream=True,timeout=(30,120)) as r:
  r.raise_for_status()
  with t.open('ab' if r.status_code==206 else 'wb') as f:
   for b in r.iter_content(8*1024*1024):f.write(b)
 os.replace(t,p)
'''
    subprocess.run([sys.executable,'-c',code],env=env,check=True)
    hf('nvidia/Cosmos-Predict2-2B-Video2World','Genie-Envisioner/Cosmos-Predict2-2B-Video2World',['scheduler/*','text_encoder/*','tokenizer/*','vae/*','*.json'])

jobs=[('Hand2World',hand),('DreamDojo',dream),('Ctrl-World',lambda:(hf('yjguo/Ctrl-World','Ctrl-World'),hf('stabilityai/stable-video-diffusion-img2vid','shared/stable-video-diffusion-img2vid',['*.json','scheduler/*','feature_extractor/*','image_encoder/*','unet/*','vae/*']),hf('openai/clip-vit-base-patch32','shared/clip-vit-base-patch32',['*.json','*.txt','pytorch_model.bin']))),('Genie-Envisioner',genie),('iVideoGPT',lambda:hf('thuml/ivideogpt-vp2-robodesk-64-act-cond','iVideoGPT')),('WorldMem',lambda:hf('zeqixiao/worldmem_checkpoints','WorldMem')),('Oasis',lambda:hf('Etched/oasis-500m','Oasis',['*.safetensors','README.md','LICENSE'])),('HyDRA',lambda:(hf('H-EmbodVis/HyDRA','HyDRA'),hf('Wan-AI/Wan2.1-T2V-1.3B','shared/Wan2.1-T2V-1.3B'))),('Matrix-Game-3.0',lambda:hf('Skywork/Matrix-Game-3.0','Matrix-Game-3.0',['base_distilled_model/*','google/*','*.pth','*.json','README.md']))]
status_path=ROOT/'download_status.json'
status=json.loads(status_path.read_text()) if status_path.exists() else {}
def save():
 p=status_path.with_suffix('.tmp');p.write_text(json.dumps(status,indent=2,ensure_ascii=False));p.replace(status_path)
for name,fn in jobs:
 if len(sys.argv)>1 and name not in sys.argv[1:]:continue
 print(time.ctime(), 'START',name,flush=True)
 status[name]={'status':'downloading','started':time.strftime('%Y-%m-%d %H:%M:%S')};save()
 try:fn();status[name]['status']='downloaded'
 except Exception as e:
  traceback.print_exc();status[name].update(status='failed',error=str(e))
 status[name]['ended']=time.strftime('%Y-%m-%d %H:%M:%S');save()
 print(time.ctime(),status[name]['status'],name,flush=True)
