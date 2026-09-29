"""Validate the prepared real sample and model-specific format contracts on CPU."""
from pathlib import Path
import json,numpy as np,cv2,ast
ROOT=Path('/mnt/vdc_5t/model/acwm/baseline_runs')
NAMES=['Hand2World','DreamDojo','Ctrl-World','Genie-Envisioner','iVideoGPT','WorldMem','Oasis','HyDRA','Matrix-Game-3.0','NWM']
for name in NAMES:
 p=ROOT/name;d=np.load(p/'input/sample.npz');rgb=d['rgb'];pose=d['c2w'];K=d['K'];manifest=json.loads((p/'input/manifest.json').read_text())
 assert rgb.shape==(161,480,640,3) and rgb.dtype==np.uint8
 assert pose.shape==(161,4,4) and np.isfinite(pose).all()
 assert np.allclose(pose[:,:3,:3].transpose(0,2,1)@pose[:,:3,:3],np.eye(3),atol=1e-4)
 assert np.all(np.diff(d['timestamp_ns'])>0)
 assert len(manifest['selected_frame_indices'])==161
 cap=cv2.VideoCapture(str(p/'input/history.mp4'));assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==77;cap.release()
 command=json.loads((p/'command.json').read_text());assert Path(command['cwd']).is_dir()
 print(name,'input + command OK')
h=ROOT/'HyDRA/input';c=json.loads((h/'camera.json').read_text());d=np.load(h/'sample.npz')
assert_camera=np.array(c['video_camera'])
assert assert_camera.shape==(20,12) and np.allclose(assert_camera[0].reshape(3,4),np.eye(4)[:3],atol=1e-5)
ref=np.linalg.inv(d['c2w'][77]);assert np.allclose(c['context_camera'],(ref@d['c2w'][0:77:4])[:,:3,:].reshape(20,12),atol=1e-5)
i=np.load(ROOT/'iVideoGPT/input/ivideogpt.npz');assert i['image'].shape[0]==12 and i['action'].shape==(12,5)
g=ROOT/'Genie-Envisioner/input/genie';assert np.load(g/'actions.npy').shape==(29,16)
for cam in ['head','hand_left','hand_right']:
 assert np.load(g/f'extrinsic_{cam}.npy').shape==(29,4,4)
 assert np.load(g/f'intrinsic_{cam}.npy').shape==(3,3)
 assert len(list((g/(cam+'_color')).glob('*.png')))==4
hand=json.loads((ROOT/'Hand2World/input/hand_camera.json').read_text());assert len(hand['frames'])==161
print('Camera alignment, no future RGB in HyDRA history, RoboDesk/GE-Sim/hand schemas OK')
