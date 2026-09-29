"""Action-conditioned Matrix rollout: annotated text + deterministic WASD/mouse.
External video history is currently reduced to t-1; native memory uses own rollout.
"""
import sys,json,math,runpy
from pathlib import Path
import numpy as np,torch,imageio.v2 as imageio
from matrix_infer import sdpa
import wan.modules.model as model
model.flash_attention=sdpa
import pipeline.inference_pipeline as pipe
import utils.utils as utils
folder=Path(sys.argv[1]);out=Path(sys.argv[2]);m=json.loads((folder/'adaptation.json').read_text());g=np.load(folder/'game_actions.npy');g=g[m['context_frames']-1:]
# Official keyboard has six channels; map only documented WASD, leave remaining idle.
# Native pose integration consumes controls at row i to reach pose i+1.
def actions(n):
 idx=np.minimum(np.arange(1,n+1),len(g)-1);keys=np.zeros((n,6),np.float32);keys[:,:4]=g[idx,11:15]
 mouse=g[idx,15:17]*20/15
 np.savez(folder/'matrix_actions.npz',keyboard=keys,mouse=mouse)
 return {'keyboard_condition':torch.tensor(keys),'mouse_condition':torch.tensor(mouse)}
utils.Bench_actions_universal=actions
# Avoid controller overlays and remove the anchor from the actual scoring output.
def save_clean(frames,path,*args,**kwargs):
 future=frames[1:1+m['prediction_frames']]
 if len(future)!=m['prediction_frames']:raise ValueError('Matrix output shorter than annotated horizon')
 imageio.mimsave(out,future,fps=m['fps'],macro_block_size=1)
pipe.process_video=save_clean
iterations=max(1,1+math.ceil(max(0,m['prediction_frames']+1-57)/40))
sys.argv=['generate.py','--ckpt_dir','/mnt/vdc_5t/model/acwm/Matrix-Game-3.0','--image',str(folder/'anchor.png'),'--prompt',(folder/'prompt.txt').read_text(),'--size','704*1280','--num_iterations',str(iterations),'--num_inference_steps','3','--fa_version','0','--t5_cpu','--seed','42','--output_dir',str(out.parent),'--save_name','prediction']
runpy.run_path('generate.py',run_name='__main__')
