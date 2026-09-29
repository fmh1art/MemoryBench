"""Pinned GE-Base-slow and LTX dependencies; process-local company proxy."""
import os,json,time
from pathlib import Path
for key in ['http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY']:os.environ[key]='http://100.68.170.171:3128'
os.environ['NO_PROXY']=os.environ['no_proxy']='localhost,127.0.0.1,::1,100.68.170.171'
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='120'
from huggingface_hub import snapshot_download
root=Path('/mnt/vdc_5t/model/acwm/GE-Base');root.mkdir(exist_ok=True)
jobs=[('agibot-world/Genie-Envisioner','797c0676521df1c9e6ff7f78d7e96b1d0a2b44c5',['ge_base_slow_v0.1.safetensors'],root),('Lightricks/LTX-Video','8984fa25007f376c1a299016d0957a37a2f797bb',['model_index.json','text_encoder/*','tokenizer/*','vae/*'],root/'LTX-Video')]
(root/'download_manifest.json').write_text(json.dumps([dict(repo=r,revision=v,patterns=p,directory=str(d)) for r,v,p,d in jobs],indent=2))
for repo,rev,patterns,dst in jobs:
 for attempt in range(1,31):
  (root/'download_status.json').write_text(json.dumps(dict(status='downloading',repo=repo,attempt=attempt)))
  try:
   snapshot_download(repo,revision=rev,allow_patterns=patterns,local_dir=dst,max_workers=2)
   break
  except Exception as exc:
   state=dict(status='failed' if attempt==30 else 'retrying',repo=repo,attempt=attempt,error_type=type(exc).__name__)
   (root/'download_status.json').write_text(json.dumps(state))
   print(json.dumps(state),flush=True)
   if attempt==30:raise
   time.sleep(min(30*attempt,300))
(root/'download_status.json').write_text(json.dumps({'status':'completed'}))
(root/'download_complete.json').write_text(json.dumps({'status':'completed'}));print('DOWNLOAD_COMPLETE',flush=True)
