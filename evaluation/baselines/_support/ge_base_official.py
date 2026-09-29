"""Unmodified GE-Base-slow inference with local paths; separate official output root."""
import sys,json,random,time
from pathlib import Path
import yaml,numpy as np,torch
base=Path(__file__).resolve().parents[1];repo=base/'origin_scripts/Genie-Envisioner-V1';sys.path.insert(0,str(repo))
root=Path('/mnt/vdc_5t/model/acwm/GE-Base');idx=int(sys.argv[1]);out=root/'official_validation'/f'sample_{idx}';out.mkdir(parents=True,exist_ok=True)
sample=repo/'video_gen_examples'/f'sample_{idx}';cfg=yaml.safe_load((repo/'configs/ltx_model/video_model_infer_slow.yaml').read_text());cfg['pretrained_model_name_or_path']=str(root/'LTX-Video');cfg['diffusion_model']['model_path']=str(root/'ge_base_slow_v0.1.safetensors')
config=out/'config.yaml';config.write_text(yaml.safe_dump(cfg));prompt=(sample/'prompt.txt').read_text().splitlines()[0].strip()
seed=42;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
from video_gen_examples import infer as native
save=native.save_video
report={'model':'GE-Base-slow v0.1','sample':idx,'status':'running','condition':'4 historical images per camera + official text; no future numeric actions','prompt':prompt,'gt_future_available':False,'seed':seed,'n_chunk':1,'num_steps':50}
(out/'status.json').write_text(json.dumps(report,indent=2))
def saved(video,path,fps):
 save(video,path,fps=fps)
 # Official pipeline prepends the input history; benchmark predictions exclude it.
 history=int(cfg['data']['train']['n_previous'])
 future=video[:,history:]
 save(future,str(out/'prediction.mp4'),fps=fps)
 w=video.shape[3]//3
 for i,cam in enumerate(['head','hand_left','hand_right']):save(future[:,:,:,i*w:(i+1)*w],str(out/f'prediction_{cam}.mp4'),fps=fps)
 report.update(frames=int(video.shape[1]),history_frames=history,prediction_frames=int(future.shape[1]),fps=fps)
native.save_video=saved
try:
 native.infer(str(config),str(sample),prompt,str(out),n_chunk=1,seed=seed)
 import av
 with av.open(str(out/'video.mp4')) as c:
  count=sum(1 for _ in c.decode(video=0))
 expected=int(cfg['data']['train']['n_previous'])+int(cfg['data']['train']['chunk'])
 assert count==expected,(count,f'expected {expected} frames including history')
 with av.open(str(out/'prediction.mp4')) as c:
  future_count=sum(1 for _ in c.decode(video=0))
 assert future_count==int(cfg['data']['train']['chunk']),future_count
 report.update(status='completed',decoded_frames=count)
except Exception as error:
 report.update(status='failed',error=str(error));raise
finally:(out/'status.json').write_text(json.dumps(report,indent=2))
