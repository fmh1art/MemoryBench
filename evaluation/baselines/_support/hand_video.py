"""Observed-history Hand2World inference; finite native KV window remains unchanged."""
import sys,json,math
from pathlib import Path
import numpy as np
from PIL import Image
folder=Path(sys.argv[1]);out=Path(sys.argv[2]);m=json.loads((folder/'adaptation.json').read_text());d=np.load(folder/'conditions.npz')
import hand2world_model.pipeline.causal_infer as engine
from hand_context import install
install(engine,str(folder/'context.mp4'),m['context_frames'])
import predict
n=m['context_frames']+m['prediction_frames'];pad=(-(n-1))%4
poses=d['c2w'];poses=np.concatenate([poses,np.repeat(poses[-1:],pad,axis=0)])
K=np.concatenate([d['K'],np.repeat(d['K'][-1:],pad,axis=0)])
frames=[{'rotation':p[:3,:3].tolist(),'translation':p[:3,3].tolist(),'intrinsics':{'fx':float(k[0,0]),'fy':float(k[1,1]),'cx':float(k[0,2]),'cy':float(k[1,2])}} for p,k in zip(poses,K)]
h,w=d['context_rgb'].shape[1:3];camera={'image_width':w,'image_height':h,'frames':frames}
p=folder/'camera.json';p.write_text(json.dumps(camera))
model=predict.Hand2WorldAR(target_fps=m['fps'])
item={'scene_image':Image.fromarray(d['context_rgb'][0]),'hand_video':str(folder/'hands.mp4'),'camera_file_path':str(p),'text':(folder/'prompt.txt').read_text()}
result=model.predict(item,num_frames=n+pad)
model.save(result[m['context_frames']:n],str(out),fps=m['fps'])
