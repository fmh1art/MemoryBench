#!/usr/bin/env python3
"""GE-Base-slow official samples: sparse multi-view history + language -> future RGB.
This is separate from GE-Sim (genie_envisioner.py). No benchmark adaptation yet.
"""
import argparse,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--sample',type=int,choices=[0,1],default=0)
p.add_argument('--gpu',default='0')
a=p.parse_args();base=Path(__file__).resolve().parent;root=Path('/mnt/vdc_5t/model/acwm')
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=a.gpu;env['PYTHONPATH']=str(base/'origin_scripts/Genie-Envisioner-V1');env['HF_TOKEN_PATH']=str(Path.home()/'.cache/huggingface/token');env['HF_HOME']=str(root/'.hf_cache');env['OMP_NUM_THREADS']='4'
subprocess.run([str(root/'.envs/Genie-Envisioner/bin/python'),'-u',str(base/'_support/ge_base_official.py'),str(a.sample)],cwd=base/'origin_scripts/Genie-Envisioner-V1',env=env,check=True)
