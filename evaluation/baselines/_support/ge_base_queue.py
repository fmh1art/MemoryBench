"""Wait for pinned download, then run both official GE-Base-slow examples in tmux."""
import json,time,subprocess
from pathlib import Path
root=Path('/mnt/vdc_5t/model/acwm/GE-Base');base=Path(__file__).resolve().parents[1];state=root/'queue_status.json'
def status(**kw):state.write_text(json.dumps(kw,indent=2))
status(status='waiting_for_download')
while not (root/'download_complete.json').exists():time.sleep(20)
for i in [0,1]:
 out=root/'official_validation'/f'sample_{i}';out.mkdir(parents=True,exist_ok=True)
 while True:
  rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
  free=[x.split(',')[0].strip() for x in rows if int(x.split(',')[0]) in [0,1,2,3] and int(x.split(',')[1])>70000 and int(x.split(',')[2])<5]
  if free:break
  status(status='waiting_for_gpu',sample=i);time.sleep(20)
 status(status='running',sample=i,gpu=free[0],log=str(out/'run.log'))
 with (out/'run.log').open('a') as log:r=subprocess.run(['/mnt/vdc_5t/model/acwm/.tools-env/bin/python',str(base/'ge_base.py'),'--sample',str(i),'--gpu',free[0]],stdout=log,stderr=subprocess.STDOUT)
 if r.returncode:status(status='failed',sample=i,log=str(out/'run.log'));raise SystemExit(r.returncode)
status(status='completed',samples=2)
