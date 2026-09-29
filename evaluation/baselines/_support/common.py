"""Shared CLI and subprocess helpers; no model dependencies imported during --prepare-only."""
import argparse, json, os, shlex, subprocess, sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
MODELS=Path('/mnt/vdc_5t/model/acwm')
TOOLS=MODELS/'.tools-env/bin/python'
DEFAULT_CASE=Path('/mnt/vdc_5t/data/acwm/memory-office/QUEST-20260806-memory-office__qv-02bb3439c90d7cfc')

def setup(name, note):
    p=argparse.ArgumentParser(description=note)
    p.add_argument('--case',type=Path,default=DEFAULT_CASE)
    p.add_argument('--start',type=float,default=2.,help='Clip start in seconds, using registered RGB capture timestamps')
    p.add_argument('--output',type=Path,default=MODELS/'baseline_runs'/name)
    p.add_argument('--python',default=os.environ.get('BASELINE_PYTHON',str(MODELS/'.envs'/name/'bin/python')),help='Python from this model\'s isolated environment')
    p.add_argument('--prepare-only',action='store_true',help='Convert inputs, save command, do not load model')
    p.add_argument('--actions',type=Path,help='Optional model-native .npy control sequence; otherwise documented smoke-test controls')
    p.add_argument('--gpu',default='0')
    p.add_argument('--benchmark-case',type=Path,help='Annotated evaluation_case directory with case.json and input/')
    p.add_argument('--context-frames',type=int,help='Observed history budget; recorded in adaptation.json')
    p.add_argument('--prediction-frames',type=int,help='Optional short smoke horizon; omit for full annotated interval')
    p.add_argument('--prompt',help='Action-only text; no hidden object answer')
    a=p.parse_args()
    if a.benchmark_case:
        from _support.benchmark_entry import launch
        if a.output==MODELS/'baseline_runs'/name:a.output=MODELS/'benchmark_runs'/name/a.benchmark_case.name
        launch(a,name,BASE,MODELS,TOOLS)
        raise SystemExit(0)
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    a.case=a.case.resolve()
    subprocess.run([str(TOOLS),str(BASE/'_support/prepare_case.py'),'--case',str(a.case),'--start',str(a.start),'--output',str(a.output/'input')],check=True)
    a.input=a.output/'input'
    (a.output/'compatibility.txt').write_text(note+'\n')
    return a

def repo(name):return BASE/'origin_scripts'/name

def execute(a, command, cwd, required=()):
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=a.gpu
    env['PYTHONPATH']=str(cwd)+os.pathsep+env.get('PYTHONPATH','')
    env['HF_TOKEN_PATH']=os.environ.get('HF_TOKEN_PATH',str(Path.home()/'.cache/huggingface/token'));env['HF_HOME']=str(MODELS/'.hf_cache');env['TORCH_HOME']=str(MODELS/'.torch_cache');env['WANDB_MODE']='offline';env['TOKENIZERS_PARALLELISM']='false'
    plan={'cwd':str(cwd),'argv':[str(x) for x in command], 'gpu':a.gpu,'required_files':[str(x) for x in required]}
    (a.output/'command.json').write_text(json.dumps(plan,indent=2))
    print('cd',shlex.quote(str(cwd)),'&&',shlex.join(plan['argv']),flush=True)
    if a.prepare_only:return
    missing=[str(x) for x in required if not Path(x).is_file()]
    if missing:raise SystemExit('权重尚未就绪：\n'+'\n'.join(missing)+'\n查看 /mnt/vdc_5t/model/acwm/download_status.json')
    result=subprocess.run(plan['argv'],cwd=cwd,env=env)
    (a.output/'run_status.json').write_text(json.dumps({'exit_code':result.returncode,'command':plan},indent=2))
    if result.returncode:raise SystemExit(result.returncode)

def link(source, target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_symlink():
        if target.resolve()!=Path(source).resolve():raise FileExistsError(target)
    elif target.exists():raise FileExistsError(target)
    else:target.symlink_to(source,target_is_directory=True)
