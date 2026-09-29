"""Resume after active workers finish; avoid concurrent status writers."""
import subprocess,time,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
for name in ['acwm-baselines-download','acwm-baselines-test']:
 while subprocess.run(['tmux','has-session','-t',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:time.sleep(20)
# Refresh download statuses after the prior worker (which may hold old statuses).
subprocess.run([sys.executable,str(BASE/'finish_and_smoke.py'),'DreamDojo','Genie-Envisioner','HyDRA','Oasis'],check=True)
