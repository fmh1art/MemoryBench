"""RoboDesk wrist displacement/yaw/pinch proxy with two-frame autoregressive history."""
import sys,json
from pathlib import Path
import numpy as np,torch,imageio.v2 as imageio
from PIL import Image
from scipy.spatial.transform import Rotation
from robot_controls import hand_states
from ivideogpt.vq_model import CompressiveVQModel
from ivideogpt.transformer import HeadModelWithAction
from transformers import AutoConfig,AutoModelForCausalLM
from safetensors.torch import load_file
folder=Path(sys.argv[1]);out=Path(sys.argv[2]);m=json.loads((folder/'adaptation.json').read_text());d=np.load(folder/'conditions.npz');assert m['context_frames']==2
root='/mnt/vdc_5t/model/acwm/iVideoGPT';torch.manual_seed(42)
tokenizer=CompressiveVQModel.from_pretrained(root,subfolder='tokenizer',low_cpu_mem_usage=False).cuda().eval()
llm=AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(root,subfolder='transformer'))
model=HeadModelWithAction(llm,action_dim=5,prelude_tokens_num=(256+1)*2-1,tokens_num_per_dyna=16,context=2,segment_length=12).cuda().eval()
model.load_state_dict(load_file(root+'/transformer/model.safetensors'),strict=True)
delta,opening,R=hand_states(d,'right',2);ctrl=np.zeros((len(delta),5),np.float32)
ctrl[1:,:3]=np.diff(delta,axis=0)/.05
ctrl[1:,3]=Rotation.from_matrix(R[:-1].transpose(0,2,1)@R[1:]).as_euler('xyz')[:,2]/.2
ctrl[:,4]=opening*2-1;ctrl=np.clip(ctrl,-1,1);np.save(folder/'robodesk_actions.npy',ctrl)
history=list(d['context_rgb']);generated=[]
with torch.inference_mode():
 for off in range(0,m['prediction_frames'],10):
  frames=np.stack([np.array(Image.fromarray(f).resize((64,64))) for f in history[-2:]]+[np.zeros((64,64,3),np.uint8)]*10)
  x=torch.tensor(frames).permute(0,3,1,2).float().cuda()[None]/255
  tokens,_=tokenizer.tokenize(x,2);idx=np.minimum(np.arange(off,off+12),len(ctrl)-1)
  action=torch.tensor(ctrl[idx]).cuda()[None]
  gen=model.generate(tokens[:,:2*257],do_sample=True,temperature=1.,top_k=100,max_new_tokens=17*10-1,pad_token_id=50256,action=action)
  pred=tokenizer.detokenize(gen,2)[0,2:].clamp(0,1).permute(0,2,3,1).cpu().numpy()
  frames=(pred*255).astype('uint8');keep=min(10,m['prediction_frames']-off);history+=list(frames[:keep]);generated+=list(frames[:keep])
imageio.mimsave(out,generated,fps=m['fps'],macro_block_size=1)
