"""Sequential annotated-case smoke tests; independent from old smoke_status.json."""
import subprocess,sys,json,time
from pathlib import Path
import av
B=Path(__file__).resolve().parents[1];R=Path('/mnt/vdc_5t/model/acwm');case=B.parent/'evaluation_cases/eval_001_a17f6882'
jobs=[('Oasis','oasis'),('Hand2World','hand2world'),('DreamDojo','dreamdojo'),('Ctrl-World','ctrl_world'),('iVideoGPT','ivideogpt'),('WorldMem','worldmem'),('Genie-Envisioner','genie_envisioner'),('Matrix-Game-3.0','matrix_game'),('HyDRA','hydra')]
if len(sys.argv)>1:jobs=[j for j in jobs if j[0] in sys.argv[1:]]
p=R/'benchmark_adapter_status.json';status=json.loads(p.read_text()) if p.exists() else {}
for name,script in jobs:
 out=R/'benchmark_adapter_tests'/name;out.mkdir(parents=True,exist_ok=True);log=out/'test.log'
 try:
  while True:
   rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
   free=[x.split(',')[0].strip() for x in rows if int(x.split(',')[0]) in [1,3] and int(x.split(',')[1])>70000 and int(x.split(',')[2])<5]
   if free:break
   time.sleep(20)
  status[name]={'status':'running','log':str(log),'gpu':free[0]};p.write_text(json.dumps(status,indent=2))
  started=time.time()
  with log.open('a') as f:subprocess.run([str(R/'.tools-env/bin/python'),str(B/(script+'.py')),'--benchmark-case',str(case),'--prediction-frames','8','--output',str(out),'--gpu',free[0]],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=5400)
  video=out/'prediction.mp4';assert video.stat().st_mtime>=started
  with av.open(str(video)) as c:
   frames=list(c.decode(video=0));assert len(frames)==8
  status[name].update(status='short_case_passed',frames=8,output=str(video),note='Input adapter smoke only; full horizon and semantic Gate not evaluated')
 except Exception as e:status[name].update(status='failed',error=str(e))
 p.write_text(json.dumps(status,indent=2));print(name,status[name]['status'],flush=True)
