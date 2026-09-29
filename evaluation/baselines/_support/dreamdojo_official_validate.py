"""Single-clip official inference API; avoids private paths in eval.sh."""
import os,sys,subprocess
from pathlib import Path
import numpy as np,torch,imageio.v2 as imageio
from PIL import Image
from cosmos_oss.init import init_environment,cleanup_environment
root=Path('/mnt/vdc_5t/model/acwm');ck=root/'DreamDojo/2B_GR1_post-train/iter_000050000'
if not (ck/'model_ema_bf16.pt').exists():
 subprocess.run([sys.executable,'scripts/convert_distcp_to_pt.py',str(ck/'model'),str(ck)],check=True)
# Config registration eagerly resolves unrelated 2B/14B base training weights.
# model_loader overrides checkpoint.load_path with our explicit GR1 checkpoint;
# use that same file for these unused default entries, not extra base downloads.
from cosmos_predict2._src.imaginaire.utils import checkpoint_db
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey
base_uris={MODEL_CHECKPOINTS[ModelKey(post_trained=post,size=size)].s3.uri for size in ['2B','14B'] for post in [False,True]}
resolve_checkpoint=checkpoint_db.get_checkpoint_path
def resolve_gr1(uri):
 # Upstream tokenizer revision 6787e176 returns 404; pin accessible official revision.
 if uri.rstrip('/')=='s3://bucket/cosmos_diffusion_v2/pretrain_weights/tokenizer/wan2pt1/Wan2.1_VAE.pth':
  from huggingface_hub import hf_hub_download
  return hf_hub_download('nvidia/Cosmos-Predict2.5-2B','tokenizer.pth',revision='85f8ae7bfe8f5525c8d103429524dcf12f98bf7b')
 return str(ck/'model_ema_bf16.pt') if uri in base_uris else resolve_checkpoint(uri)
checkpoint_db.get_checkpoint_path=resolve_gr1
init_environment()
from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
runner=Video2WorldInference(experiment_name='cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes_release_oss',ckpt_path=str(ck/'model_ema_bf16.pt'),s3_credential_path='',config_file='cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py',offload_text_encoder=True,experiment_opts=['~data_val'])

# Local compatibility bootstrap above is identical to the working benchmark loader.
# Below, reuse official dataset preprocessing and official single-frame/chunk settings.
import json, time
import piq
from groot_dreams.dataloader import MultiVideoActionDataset
from cosmos_predict2.action_conditioned_config import DEFAULT_NEGATIVE_PROMPT
out=root/'DreamDojo/official_validation'
dataset_path=out/'dataset/In-lab_Eval/gr1_unified.pnp_corn_robot'
dataset=MultiVideoActionDataset(num_frames=49,dataset_path=str(dataset_path),data_split='full')
underlying=dataset.datasets[0].lerobot_dataset
report={'checkpoint':str(ck),'dataset_revision':(out/'dataset_revision.txt').read_text().strip(),
        'dataset':str(dataset_path),'settings':{'context_frames':1,'future_per_chunk':12,'guidance':0,'prompt':'','resolution':[480,640],'save_fps':10,'seeds':[0,12,24,36]},
        'notes':['Official dataset and normalization used unchanged.','Native action row 0 includes state as implemented by the official loader.','For timestamp-correct stitching, later chunks drop their first conditioning frame; upstream concatenation is saved separately.','GT RGB is used for evaluation only; lam_video is not supplied for GR1 action inference.'], 'samples':[]}

def metrics(pred, gt):
    p=torch.from_numpy(pred.copy()).permute(0,3,1,2).float()/255
    g=torch.from_numpy(gt.copy()).permute(0,3,1,2).float()/255
    return {'psnr_db':float(piq.psnr(p,g,data_range=1.0)), 'ssim':float(piq.ssim(p,g,data_range=1.0))}

def save_report():
    (out/'results.json').write_text(json.dumps(report,indent=2))

for episode in [0,1,2]:
    positions=[(i,int(step)) for i,(ep,step) in enumerate(underlying.all_steps) if int(ep)==episode]
    index,base=min(positions,key=lambda x:abs(x[1]-60))
    data=dataset[index]
    gt=data['video'].permute(1,2,3,0).cpu().numpy()
    actions=data['action'][:48].cpu().numpy().copy()
    assert gt.shape==(49,480,640,3) and actions.shape==(48,384)
    assert np.isfinite(actions).all() and np.count_nonzero(actions)>0
    assert np.count_nonzero(actions[:,29:])==0
    dst=out/f'episode_{episode:06d}';dst.mkdir(exist_ok=True)
    np.save(dst/'actions.npy',actions)
    np.savez_compressed(dst/'official_sample.npz',video=gt,action=actions)
    imageio.mimsave(dst/'gt.mp4',gt,fps=10)
    item={'episode':episode,'dataset_index':index,'base_index':base,'action_shape':list(actions.shape),'action_nonzero':int(np.count_nonzero(actions)),'modes':{}}
    report['samples'].append(item)
    for mode in ['real_actions','zero_actions']:
        chunks=[];anchor=gt[0];started=time.time()
        with torch.inference_mode():
            for off in range(0,48,12):
                video=torch.zeros((1,3,13,480,640),dtype=torch.uint8)
                video[0,:,0]=torch.from_numpy(anchor.copy()).permute(2,0,1)
                control=actions[off:off+12] if mode=='real_actions' else np.zeros_like(actions[off:off+12])
                pred=runner.generate_vid2world(prompt='',input_path=video,action=torch.from_numpy(control).float(),guidance=0,num_video_frames=13,num_latent_conditional_frames=1,resolution='480,640',seed=off,negative_prompt=DEFAULT_NEGATIVE_PROMPT,lam_video=None)
                frames=((pred[0].clamp(-1,1)+1)*127.5).byte().permute(1,2,3,0).cpu().numpy()
                assert frames.shape==gt[:13].shape
                chunks.append(frames);anchor=frames[-1]
        aligned=np.concatenate([chunks[0]]+[c[1:] for c in chunks[1:]])
        official=np.concatenate([chunks[0]]+[c[:12] for c in chunks[1:]])
        imageio.mimsave(dst/(mode+'.mp4'),aligned,fps=10)
        imageio.mimsave(dst/(mode+'_official_stitch.mp4'),official,fps=10)
        imageio.mimsave(dst/(mode+'_vs_gt.mp4'),np.concatenate([gt,aligned],axis=2),fps=10)
        item['modes'][mode]={'first_chunk':metrics(aligned[1:13],gt[1:13]),'full_rollout':metrics(aligned[1:],gt[1:]),'seconds':round(time.time()-started,2)}
        save_report();print('VALIDATED',episode,mode,item['modes'][mode],flush=True)
    item['static_anchor_baseline']=metrics(np.repeat(gt[:1],48,axis=0),gt[1:])
    save_report()
report['status']='completed';save_report()
cleanup_environment()
