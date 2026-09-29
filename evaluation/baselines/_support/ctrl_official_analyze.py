"""Check original replay outputs against raw GT (excluding clamped repeated tail frames)."""
import json
from pathlib import Path
import av,numpy as np
from PIL import Image,ImageDraw
root=Path('/mnt/vdc_5t/model/acwm/Ctrl-World')
out=root/'official_validation';held=root/'official_validation_hold'
repo=Path(__file__).resolve().parents[1]/'origin_scripts/Ctrl-World'

def read(path):
    with av.open(str(path)) as c:
        fps=float(c.streams.video[0].average_rate)
        return np.stack([f.to_ndarray(format='rgb24') for f in c.decode(video=0)]),fps

def write(path,frames,fps=5):
    with av.open(str(path),'w') as c:
        stream=c.add_stream('libx264',rate=fps);stream.height,stream.width=frames.shape[1:3];stream.pix_fmt='yuv420p';stream.options={'crf':'18','preset':'fast'}
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame,format='rgb24')):c.mux(packet)
        for packet in stream.encode():c.mux(packet)
    check,rate=read(path);assert len(check)==len(frames) and rate==fps

def psnr(a,b):
    mse=((a.astype(np.float32)-b.astype(np.float32))/255)**2
    mse=mse.mean(axis=tuple(range(1,mse.ndim)))
    return float(np.mean(-10*np.log10(np.maximum(mse,1e-12))))

def prediction(folder):
    files=[folder/f'chunk_{i:02d}.npz' for i in range(12)]
    if not all(p.exists() for p in files):return None
    chunks=[np.load(p)['prediction'] for p in files]
    assert all(c.shape==(3,5,192,320,3) for c in chunks)
    return np.concatenate([c[:,:4] for c in chunks[:-1]]+[chunks[-1]],axis=1)

results={'status':'partial','upstream_commit':'99fb20683fd79dfa6d0c6feb9d49c6c55eecd50d','comparison_layout':'top raw GT / middle recorded-state prediction / bottom held-initial-state prediction; columns camera 0,1,2','evaluation':'raw GT, excludes initial frame and repeated clamped tail','samples':[]}
for id,start in [('899',8),('18599',14),('199',8)]:
    pred=prediction(out/id);hold=prediction(held/id)
    if pred is None:continue
    anno=json.loads((repo/f'dataset_example/droid_subset/annotation/val/{id}.json').read_text())
    frames=np.minimum(start+np.arange(49),len(anno['states'])-1);valid=min(49,len(anno['states'])-start)
    videos=[]
    for v in anno['videos']:
        video,fps=read(repo/'dataset_example/droid_subset'/v['video_path']);assert fps==5
        videos.append(video[frames])
    gt=np.stack(videos)
    item={'trajectory':id,'instruction':anno['texts'][0],'start_index':start,'output_frames':49,'unique_source_frames':valid,'comparison_frames':valid,'evaluated_future_frames':valid-1,'padded_frames':49-valid,'source_fps':5,'official_save_fps':4,'aligned_save_fps':5,'modes':{}}
    for name,value in [('recorded',pred),('hold',hold),('static_anchor',np.repeat(gt[:,:1],49,axis=1))]:
        if value is None:continue
        per_view=[psnr(value[v,1:valid],gt[v,1:valid]) for v in range(3)]
        first=[psnr(value[v,1:5],gt[v,1:5]) for v in range(3)]
        item['modes'][name]={'full_psnr_db_per_camera':per_view,'full_mean_psnr_db':float(np.mean(per_view)),'first_chunk_mean_psnr_db':float(np.mean(first))}
    # Concat camera images horizontally, time is first dimension.
    tile=lambda x: np.concatenate(list(x),axis=2)
    dst=out/id
    write(dst/'gt_raw.mp4',tile(gt));write(dst/'prediction.mp4',tile(pred))
    write(dst/'prediction_vs_raw_gt.mp4',np.concatenate([tile(gt),tile(pred)],axis=1)[:valid])
    if hold is not None:
        write(dst/'held_state_prediction.mp4',tile(hold))
        write(dst/'comparison.mp4',np.concatenate([tile(gt),tile(pred),tile(hold)],axis=1)[:valid])
        canvas=Image.new('RGB',(960,630),'white');draw=ImageDraw.Draw(canvas)
        # Most visible external camera is 0; all three are present in videos.
        for row,(name,arr) in enumerate([('raw GT',gt),('recorded',pred),('held state',hold)]):
            for col,t in enumerate([1, min(12,valid-1),min(24,valid-1),valid-1]):
                canvas.paste(Image.fromarray(arr[0,t]).resize((240,180)),(col*240,row*210+25));draw.text((col*240+4,row*210+4),f'{name}, frame {t}',fill='black')
        canvas.save(dst/'preview.png')
    results['samples'].append(item)
if len(results['samples'])==3 and all('hold' in s['modes'] for s in results['samples']):results['status']='completed'
(out/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
