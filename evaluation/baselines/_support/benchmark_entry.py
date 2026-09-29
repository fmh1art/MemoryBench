"""Shared benchmark input path, selected with --benchmark-case by model entrypoints."""
import json,subprocess
from pathlib import Path

def launch(a,name,base,models,tools):
    if a.actions:raise ValueError('Benchmark mode derives native controls from allowed case actions; --actions overrides are not supported in this mode')
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True);a.input=a.output/'input'
    command=[str(tools),str(base/'_support/benchmark_inputs.py'),'--benchmark-case',str(a.benchmark_case.resolve()),'--output',str(a.input),'--model',name]
    for key in ['context_frames','prediction_frames','prompt']:
        value=getattr(a,key,None)
        if value is not None:command += ['--'+key.replace('_','-'),str(value)]
    subprocess.run(command,check=True)
    drivers={'Oasis':('oasis_video.py','open-oasis'),'HyDRA':('hydra_video.py','HyDRA'),'Hand2World':('hand_video.py','Hand2World'),
             'DreamDojo':('dreamdojo_infer.py','DreamDojo'),'Ctrl-World':('ctrl_video.py','Ctrl-World'),'iVideoGPT':('ivideo_video.py','iVideoGPT'),
             'WorldMem':('worldmem_video.py','WorldMem'),'Genie-Envisioner':('genie_video.py','Genie-Envisioner-V1'),'Matrix-Game-3.0':('matrix_video.py','Matrix-Game/Matrix-Game-3')}
    script,repo=drivers[name]
    plan={'model':name,'adapter':script,'case':str(a.benchmark_case),'input':str(a.input),'output':'prediction.mp4','status':'prepared'}
    (a.output/'benchmark_run.json').write_text(json.dumps(plan,indent=2))
    (a.output/'compatibility.txt').write_text('See INPUT_ADAPTATION.md and input/adaptation.json. Benchmark input mode; generated-only output; no future RGB conditioning.\n')
    if a.prepare_only:return
    if not (base/'_support'/script).exists():raise SystemExit(f'{name}: inputs prepared; native inference integration is not yet implemented')
    from .common import execute,link
    if name=='Hand2World':
        target=base/'origin_scripts/Hand2World/checkpoints'
        if not target.exists():link(models/'Hand2World/checkpoints',target)
        required=[models/'Hand2World/checkpoints/V_0.9/bidirectional.safetensors']
    else:required=[]
    (a.output/'output_alignment.json').unlink(missing_ok=True)
    command=[a.python,base/'_support'/script,a.input,a.output/'prediction.mp4']
    execute(a,command,base/'origin_scripts'/repo,required)
    if a.prediction_frames is None:
        (a.output/'prediction_native.mp4').unlink(missing_ok=True)
        subprocess.run([str(tools),str(base/'_support/align_prediction.py'),str(a.output)],check=True)
    plan['status']='full_length_verified' if a.prediction_frames is None else 'short_prediction_generated'
    (a.output/'benchmark_run.json').write_text(json.dumps(plan,indent=2))
