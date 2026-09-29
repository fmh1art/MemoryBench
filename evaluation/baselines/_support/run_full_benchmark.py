"""Resume all 9 models x 5 cases, full future; one process per available GPU 0..3."""
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

B = Path(__file__).resolve().parents[1]
R = Path('/mnt/vdc_5t/model/acwm')
PY = R / '.tools-env/bin/python'
JOBS = [('Oasis','oasis'), ('Hand2World','hand2world'), ('DreamDojo','dreamdojo'),
        ('Ctrl-World','ctrl_world'), ('iVideoGPT','ivideogpt'), ('WorldMem','worldmem'),
        ('Genie-Envisioner','genie_envisioner'), ('Matrix-Game-3.0','matrix_game'), ('HyDRA','hydra')]
ROOT = R / 'benchmark_full_length'
ROOT.mkdir(exist_ok=True)
lock_file = (ROOT / 'runner.lock').open('w')
fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
status_file = ROOT / 'status.json'
status = json.loads(status_file.read_text()) if status_file.exists() else {}
lock = threading.Lock()
work = queue.Queue()
for case in sorted((B.parent / 'evaluation_cases').glob('eval_*/case.json')):
    for model, script in JOBS:
        key = f'{case.parent.name}/{model}'
        out = ROOT / model / case.parent.name
        if (out / 'output_alignment.json').exists():
            report = json.loads((out / 'output_alignment.json').read_text())
            if report['status'] == 'full_length_verified' and (out / 'prediction.mp4').exists():
                status[key] = dict(status='full_length_verified', output=str(out/'prediction.mp4'),frames=report['frames'],duration_seconds=report['duration_seconds'])
                continue
        status[key] = dict(status='queued', output=str(out/'prediction.mp4'))
        work.put((key, model, script, case.parent, out))

def update(key, **values):
    with lock:
        status[key].update(values)
        temp = status_file.with_suffix('.tmp')
        temp.write_text(json.dumps(status, indent=2))
        temp.replace(status_file)
        print(key, values.get('status', ''), flush=True)

status_file.write_text(json.dumps(status, indent=2))
env = os.environ.copy()
for var in ('http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY'):
    env[var] = 'http://100.68.170.171:3128'
env['NO_PROXY'] = env['no_proxy'] = 'localhost,127.0.0.1,::1,100.68.170.171'
env['OMP_NUM_THREADS'] = '4'

def worker(gpu):
    while True:
        try:
            key, model, script, case, out = work.get_nowait()
        except queue.Empty:
            return
        out.mkdir(parents=True, exist_ok=True)
        while True:
            row = subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'], text=True).strip().split(',')
            if int(row[0]) > 70000 and int(row[1]) < 5:
                break
            update(key, status='waiting_for_gpu', gpu=gpu)
            time.sleep(20)
        started = time.time()
        update(key, status='running', gpu=gpu, started_at=started, log=str(out/'run.log'))
        try:
            with (out/'run.log').open('a') as log:
                subprocess.run([str(PY), str(B/(script+'.py')), '--benchmark-case', str(case), '--output', str(out), '--gpu', str(gpu)], stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
            report = json.loads((out/'output_alignment.json').read_text())
            update(key, status=report['status'], frames=report['frames'], duration_seconds=report['duration_seconds'], elapsed_seconds=round(time.time()-started,1))
        except Exception as error:
            update(key, status='failed', error=str(error), elapsed_seconds=round(time.time()-started,1))
        finally:
            work.task_done()

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(worker, range(4)))
print('Queue finished', flush=True)
