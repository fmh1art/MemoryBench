"""Decode GE-Sim official outputs and make previews; no GT metrics without GT future."""
import json
from pathlib import Path
import av,numpy as np
from PIL import Image,ImageDraw
root=Path('/mnt/vdc_5t/model/acwm/Genie-Envisioner/official_validation')
repo=Path(__file__).resolve().parents[1]/'origin_scripts/Genie-Envisioner-V1'
results={'status':'completed','gt_future_available':False,'quantitative_prediction_accuracy':'not_evaluated','samples':[]}
for i in range(3):
    out=root/f'sample_{i}'
    spec=json.loads((out/'manifest.json').read_text());assert spec['status']=='completed'
    log=(out/'run.log').read_text()
    assert '>>> mismatched_keys: []' in log and '>>> missing: []' in log and '>>> unexpected: []' in log
    checks=[]
    for name,expected in [('upstream/video.mp4',300),('rgb_with_context.mp4',300),('prediction.mp4',296)]+[(f'prediction_{cam}.mp4',296) for cam in ['head','hand_left','hand_right']]:
        with av.open(str(out/name)) as c:
            fps=float(c.streams.video[0].average_rate);stream=c.streams.video[0];shape=[stream.width,stream.height]
            count=sum(1 for _ in c.decode(video=0))
        assert count==expected and fps==30,(name,count,fps)
        checks.append({'file':name,'frames':count,'fps':fps,'resolution':shape})
    canvas=Image.new('RGB',(1024,636),'white');draw=ImageDraw.Draw(canvas)
    for row,cam in enumerate(['head','hand_left','hand_right']):
        context=Image.open(repo/'gesim_video_gen_examples'/f'sample_{i}'/f'{cam}_color/3.png')
        with av.open(str(out/f'prediction_{cam}.mp4')) as c:
            frames={j:f.to_image() for j,f in enumerate(c.decode(video=0)) if j in [49,149,295]}
        for col,(label,im) in enumerate([('last context',context)]+[(f'future {j}',frames[j]) for j in [49,149,295]]):
            canvas.paste(im.resize((256,192)),(col*256,row*212+20));draw.text((col*256+4,row*212+4),f'{cam}: {label}',fill='black')
    canvas.save(out/'preview.png')
    results['samples'].append({'sample':i,'status':'decoded_verified','context_frames':4,'prediction_frames':296,'prediction_seconds':296/30,'checks':checks,'weights_missing_keys':[],'weights_mismatched_keys':[],'weights_unexpected_keys':[]})
(root/'results.json').write_text(json.dumps(results,indent=2));print('All 3 official samples and 18 videos decoded and verified')
