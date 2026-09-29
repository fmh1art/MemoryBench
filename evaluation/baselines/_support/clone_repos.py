from pathlib import Path
import subprocess, os
root=Path(__file__).resolve().parents[1]/'origin_scripts'
repos=['NTUYWANG103/Hand2World','NVIDIA/DreamDojo','Robert-gyj/Ctrl-World','AgibotTech/Genie-Envisioner-V1','thuml/iVideoGPT','xizaoqu/WorldMem','etched-ai/open-oasis','H-EmbodVis/HyDRA','SkyworkAI/Matrix-Game','facebookresearch/nwm']
env=os.environ.copy(); env['GIT_LFS_SKIP_SMUDGE']='1'; env['GIT_TERMINAL_PROMPT']='0'
for repo in repos:
 p=root/repo.split('/')[-1]
 if (p/'.git').exists(): continue
 print(repo,flush=True)
 r=subprocess.run(['git','clone','--depth','1','https://github.com/'+repo+'.git',str(p)],env=env)
 print('EXIT',r.returncode,flush=True)
