#!/usr/bin/env python3
"""Benchmark: observed video latents prefill the AR KV cache; real hand/camera controls.
Use --benchmark-case <evaluation_case>; see INPUT_ADAPTATION.md.

Legacy smoke mode without --benchmark-case:
Hand2World AR Open Version.
Compatibility: optical c2w + K -> DA3 camera JSON; timestamp-aligned source 21-joint
hands -> colored skeleton control video, NOT the original rendered MANO mesh.
This rendering-domain approximation is for smoke testing; use native rendered hand
video for faithful evaluation. The isolated environment uses the official SDPA
fallback without flash-attn; static KV cache remains disabled. First RGB is the only appearance condition.
"""
# Benchmark mode: --benchmark-case <evaluation_case>; see INPUT_ADAPTATION.md.
from _support.common import *
def main():
 a=setup('Hand2World',__doc__)
 subprocess.run([str(TOOLS),str(BASE/'_support/model_inputs.py'),'hand',str(a.input),str(a.case)],check=True)
 link(MODELS/'Hand2World/checkpoints',repo('Hand2World')/'checkpoints')
 execute(a,[a.python,'predict.py','--mode','ar','--json_path',a.input/'hand2world.json','--save_path',a.output/'prediction','--num_frames','161'],repo('Hand2World'),[MODELS/'Hand2World/checkpoints/V_0.9/bidirectional.safetensors',MODELS/'Hand2World/checkpoints/V_0.9/ar_stage3_dmd.safetensors',MODELS/'Hand2World/checkpoints/V_0.9/Wan2.2-Fun-5B-Control/Wan2.2_VAE.pth',MODELS/'Hand2World/checkpoints/V_0.9/Wan2.2-Fun-5B-Control/models_t5_umt5-xxl-enc-bf16.pth',MODELS/'Hand2World/checkpoints/V_0.9/Wan2.2-Fun-5B-Control/diffusion_pytorch_model.safetensors'])
if __name__=='__main__':main()
