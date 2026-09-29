"""Official GE-Sim inputs and inference; only local paths, fixed RNG, and output capture."""
import sys,json,random,hashlib,time
from pathlib import Path
import numpy as np,torch,yaml
idx=int(sys.argv[1]);assert idx in [0,1,2]
base=Path(__file__).resolve().parents[1];repo=base/'origin_scripts/Genie-Envisioner-V1'
root=Path('/mnt/vdc_5t/model/acwm/Genie-Envisioner');out=root/'official_validation'/f'sample_{idx}';out.mkdir(parents=True,exist_ok=True)
sample=repo/'gesim_video_gen_examples'/f'sample_{idx}'
sys.path.insert(0,str(repo))
import gesim_video_gen_examples.infer_gesim as native
cfg=yaml.safe_load((repo/'configs/cosmos_model/acwm_cosmos.yaml').read_text())
for key in ['pretrained_model_name_or_path','tokenizer_pretrained_model_name_or_path']:cfg[key]=str(root/'Cosmos-Predict2-2B-Video2World')
cfg['diffusion_model']['model_path']=str(root/'ge_sim_cosmos_v0.1.safetensors')
config=out/'config.yaml';config.write_text(yaml.safe_dump(cfg))
seed=42;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
manifest={'sample':idx,'upstream_commit':'d41358f7548f2cd9fc4b79fcdf7d86d15a8d7b1e','seed':seed,'fps':30,'context_frames':4,'input_action_frames':300,'chunk_frames':25,'inference_steps':cfg['num_inference_step'],'prompt':'best quality, consistent and smooth motion, realistic, clear and distinct.','gt_future_available':False,'changes':['Local weight paths only; RNG explicitly fixed because upstream seed argument is unused.','Official infer and condition/memory updates unchanged; output capture only.'],'input_sha256':{str(p.relative_to(sample)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sample.rglob('*')) if p.is_file()}}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
prepare=native.prepare_model;save=native.save_video
count=0

def prepared(*args,**kwargs):
    models=prepare(*args,**kwargs);pipe=models[-1];generate=pipe.infer
    def captured(*a,**kw):
        global count
        result=generate(*a,**kw);frames=result['frames'].detach().cpu()
        # Per-chunk outputs survive even if a later block fails.
        c,v,t,h,w=frames.permute(1,0,2,3,4).shape
        tiled=frames.permute(1,2,3,0,4).reshape(c,t,h,v*w)
        folder=out/'chunks';folder.mkdir(exist_ok=True)
        save(tiled,str(folder/f'{count:02d}.mp4'),fps=30)
        count+=1
        (out/'progress.json').write_text(json.dumps({'status':'running','completed_chunks':count,'total_chunks':12,'last_condition_frames':int(kw['cond_to_concat'].shape[2])},indent=2))
        return result
    pipe.infer=captured
    return models

def outputs(video,path,fps):
    save(video,path,fps=fps)
    h=video.shape[2]//2;w=video.shape[3]//3
    save(video[:,:,:h],str(out/'rgb_with_context.mp4'),fps=fps)
    save(video[:,4:,:h],str(out/'prediction.mp4'),fps=fps)
    for i,cam in enumerate(['head','hand_left','hand_right']):
        save(video[:,4:,:h,i*w:(i+1)*w],str(out/f'prediction_{cam}.mp4'),fps=fps)
    manifest.update(status='completed',frames_including_context=int(video.shape[1]),prediction_frames=int(video.shape[1]-4),elapsed_seconds=round(time.time()-started,2))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))

native.prepare_model=prepared;native.save_video=outputs
started=time.time()
try:
    native.infer(str(config),str(sample),str(sample),str(sample),str(sample/'actions.npy'),manifest['prompt'],str(out/'upstream'),seed=seed,default_fps=30)
    (out/'progress.json').write_text(json.dumps({'status':'completed','completed_chunks':count,'total_chunks':12},indent=2))
except Exception as error:
    (out/'progress.json').write_text(json.dumps({'status':'failed','completed_chunks':count,'error':str(error)},indent=2));raise
