"""Run the original Ctrl-World replay entrypoint; change local paths and capture outputs only."""
import ast,json,os,random,sys
from pathlib import Path
import numpy as np
import torch
root=Path('/mnt/vdc_5t/model/acwm')
mode=os.environ.get('CTRL_ACTION_MODE','recorded')
assert mode in ['recorded','hold']
out=root/'Ctrl-World'/('official_validation' if mode=='recorded' else 'official_validation_hold');out.mkdir(parents=True,exist_ok=True)
repo=Path(__file__).resolve().parents[1]/'origin_scripts/Ctrl-World'
sys.path.insert(0,str(repo))
import config
config.wm_args.save_dir=str(out/'upstream')
from scripts import rollout_replay_traj as upstream
agent=upstream.agent
if mode=='hold':
    get_original=agent.get_traj_info
    def get_hold(self,*args,**kwargs):
        states,joints,videos,latents,instruction=get_original(self,*args,**kwargs)
        return np.repeat(states[:1],len(states),axis=0),joints,videos,latents,instruction
    agent.get_traj_info=get_hold
original=agent.forward_wm
count=0

def capture(self,action_cond,video_latent_true,video_latent_cond,his_cond=None,text=None):
    global count
    result=original(self,action_cond,video_latent_true,video_latent_cond,his_cond=his_cond,text=text)
    trajectory=self.args.val_id[count//self.args.interact_num]
    step=count%self.args.interact_num
    folder=out/str(trajectory);folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/f'chunk_{step:02d}.npz',prediction=result[2],decoded_gt=result[1],action=action_cond)
    count+=1
    (out/'progress.json').write_text(json.dumps({'status':'running','completed_chunks':count,'total_chunks':36,'last_trajectory':trajectory,'last_step':step},indent=2))
    return result

agent.forward_wm=capture
random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
settings={'action_mode':mode,'upstream_commit':'99fb20683fd79dfa6d0c6feb9d49c6c55eecd50d','seed':0,'trajectories':['899','18599','199'],'start_indices':[8,14,8],'interact_num':12,'chunk_frames':5,'chunk_stride':4,'history_indices':[0,0,-8,-6,-4,-2],'model_fps_parameter':7,'output_fps':4,'weights':str(root/'Ctrl-World/checkpoint-10000.pt'),'note':'Original rollout loop and agent unchanged; wrapper only sets paths, seed, and captures returned frames/actions. Official display GT is VAE reconstruction; analysis exports raw GT separately.'}
(out/'settings.json').write_text(json.dumps(settings,indent=2))
sys.argv=['scripts/rollout_replay_traj.py','--svd_model_path',str(root/'shared/stable-video-diffusion-img2vid'),'--clip_model_path',str(root/'shared/clip-vit-base-patch32'),'--ckpt_path',str(root/'Ctrl-World/checkpoint-10000.pt'),'--task_type','replay']
# Execute the unedited __main__ body in the imported module so capture wraps the same class.
source=repo/'scripts/rollout_replay_traj.py'
tree=ast.parse(source.read_text(),filename=str(source))
main=next(node for node in tree.body if isinstance(node,ast.If) and ast.unparse(node.test)=="__name__ == '__main__'")
exec(compile(ast.Module(body=main.body,type_ignores=[]),str(source),'exec'),upstream.__dict__)
assert count==36,count
(out/'finished.json').write_text(json.dumps({'status':'completed','captured_chunks':count}))

(out/'progress.json').write_text(json.dumps({'status':'completed','completed_chunks':count,'total_chunks':36},indent=2))
