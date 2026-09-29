#!/usr/bin/env python3
"""Benchmark: 77 real context frames + camera trajectory and action text; single-case API and rollout.
Use --benchmark-case <evaluation_case>; see INPUT_ADAPTATION.md.

Legacy smoke mode without --benchmark-case:
HyDRA: 77 observed frames -> 77 future frames; no future RGB as condition.
Compatibility: optical c2w (metres) -> relative 20x12 camera arrays directly,
bypassing upstream Unreal axis permutation and centimetre conversion. RGB is
center-cropped/resized by official loader to 480x832. Camera scale kept in metres.
"""
# Benchmark mode: --benchmark-case <evaluation_case>; see INPUT_ADAPTATION.md.
from _support.common import *
import numpy as np

def main():
 a=setup('HyDRA',__doc__);d=np.load(a.input/'sample.npz');poses=d['c2w'];ref=np.linalg.inv(poses[77])
 cams={k:(ref@poses[ix])[...,:3,:].reshape(20,12).tolist() for k,ix in [('context_camera',np.arange(0,77,4)),('video_camera',np.arange(77,154,4))]}
 (a.input/'camera.json').write_text(json.dumps(cams));(a.input/'caption.txt').write_text('First-person view inside an office.\n')
 shared=MODELS/'shared/Wan2.1-T2V-1.3B'
 execute(a,[a.python,'infer_hydra.py','--cond_video',a.input/'history.mp4','--cond_json',a.input/'camera.json','--caption_txt',a.input/'caption.txt','--output_path',a.output/'prediction.mp4','--ckpt_path',MODELS/'HyDRA/hydra.ckpt','--base_dit_path',shared/'diffusion_pytorch_model.safetensors','--base_text_encoder_path',shared/'models_t5_umt5-xxl-enc-bf16.pth','--base_vae_path',shared/'Wan2.1_VAE.pth'],repo('HyDRA'),[MODELS/'HyDRA/hydra.ckpt',shared/'diffusion_pytorch_model.safetensors',shared/'models_t5_umt5-xxl-enc-bf16.pth',shared/'Wan2.1_VAE.pth'])
if __name__=='__main__':main()
