"""Background continuation: wait for sequential download queue, install isolated envs,
fetch auxiliary assets, run ONE case per available model, record failures separately.
No model failure is counted as successful reproduction. No retry loop on inference.
"""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];ROOT=Path('/mnt/vdc_5t/model/acwm')
JOBS=[('Hand2World','hand2world'),('DreamDojo','dreamdojo'),('Ctrl-World','ctrl_world'),('Genie-Envisioner','genie_envisioner'),('iVideoGPT','ivideogpt'),('WorldMem','worldmem'),('Oasis','oasis'),('HyDRA','hydra'),('Matrix-Game-3.0','matrix_game')]
status_path=ROOT/'smoke_status.json'
if len(sys.argv)>1: JOBS=[j for j in JOBS if j[0] in sys.argv[1:]]
status=json.loads(status_path.read_text()) if status_path.exists() else {}
for n,_ in JOBS:status.setdefault(n,{'status':'awaiting_download'})
def save():
 tmp=status_path.with_suffix('.tmp');tmp.write_text(json.dumps(status,indent=2,ensure_ascii=False));tmp.replace(status_path)
save()
# Complete the requested model download order before downloading any extra assets.
while subprocess.run(['tmux','has-session','-t','acwm-baselines-download'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:time.sleep(20)
while True:
 downloads=json.loads((ROOT/'download_status.json').read_text()) if (ROOT/'download_status.json').exists() else {}
 if all(downloads.get(n,{}).get('status') in ('downloaded','failed') for n,_ in JOBS):break
 time.sleep(30)
os.environ.setdefault('HF_TOKEN_PATH',str(Path.home()/'.cache/huggingface/token'));os.environ['HF_HOME']=str(ROOT/'.hf_cache');os.environ['TORCH_HOME']=str(ROOT/'.torch_cache');os.environ['HF_HUB_DISABLE_XET']='1'
from huggingface_hub import snapshot_download
for name,script in JOBS:
 if status.get(name,{}).get('status')=='smoke_passed':continue
 if downloads[name]['status']!='downloaded':
  status[name]={'status':'blocked_download','error':downloads[name].get('error','')};save();continue
 logfile=ROOT/f'smoke-{script}.log'
 try:
  status[name]={'status':'installing','log':str(logfile)};save()
  with logfile.open('a') as log:
   envdir=ROOT/'.envs'/name
   if not (envdir/'setup.ok').exists():
    subprocess.run([sys.executable,str(BASE/'_support/setup_env.py'),name],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7200)
   # DreamDojo resolves its own Cosmos tokenizer / Reason text encoder via the
   # official checkpoint database; HF_HOME keeps these assets on the model disk.
   status[name]['status']='waiting_gpu';save()
   # Respect other jobs: choose a currently idle GPU; never use occupied 4-7.
   while True:
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
    available=[s.split(',')[0].strip() for s in lines if int(s.split(',')[0]) in range(4) and int(s.split(',')[1])>70000 and int(s.split(',')[2])<5]
    if available:break
    time.sleep(30)
   status[name].update(status='running',gpu=available[0]);save()
   started=time.time()
   subprocess.run([str(ROOT/'.tools-env/bin/python'),str(BASE/(script+'.py')),'--python',str(envdir/'bin/python'),'--gpu',available[0]],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7200)
   # Exit zero alone is insufficient: verify a nonempty decodable result.
   outputs=list((ROOT/'baseline_runs'/name).rglob('*.mp4'))+list((ROOT/'baseline_runs'/name).rglob('*.gif'))
   outputs=[p for p in outputs if p.is_file() and 'input' not in p.relative_to(ROOT/'baseline_runs'/name).parts and p.stat().st_size>0 and p.stat().st_mtime>=started]
   if not outputs:raise RuntimeError('Command exited zero but produced no result video')
   import av
   for p in outputs:
    with av.open(str(p)) as c:
     if next(c.decode(video=0),None) is None:raise RuntimeError('Empty video: '+str(p))
   status[name].update(status='smoke_passed',outputs=[str(p) for p in outputs]);save()
 except Exception as e:
  status[name].update(status='failed',error=str(e));save()
  with logfile.open('a') as log:traceback.print_exc(file=log)
 print(name,status[name]['status'],flush=True)
