#!/usr/bin/env python3
"""Benchmark: last observed frame + deterministic WASD/mouse and action text. Full external video context unsupported.
Use --benchmark-case <evaluation_case>; see INPUT_ADAPTATION.md.

Legacy smoke mode without --benchmark-case:
Matrix-Game 3.0 5B distilled, first office image -> interactive video.
Compatibility: office is out of the game training domain; the official offline
entry generates RANDOM controls with seed 42 (not replay of office camera/hand
motion). One iteration, 3 denoise steps. FA3/int8 disabled for A800 compatibility. Process-local SDPA adapter also handles
upstream hard-coded flash_attention calls, preserving lengths/scaling/masks.
"""
# Benchmark mode: --benchmark-case <evaluation_case>; see INPUT_ADAPTATION.md.
from _support.common import *
def main():
 a=setup('Matrix-Game-3.0',__doc__)
 if a.actions:raise ValueError('Official offline entry has no action-file CLI; use its interactive pipeline for native controls')
 execute(a,[a.python,BASE/'_support/matrix_infer.py','--ckpt_dir',MODELS/'Matrix-Game-3.0','--image',a.input/'first.png','--prompt','First-person view inside an office.','--size','704*1280','--num_iterations','1','--num_inference_steps','3','--fa_version','0','--t5_cpu','--seed','42','--output_dir',a.output,'--save_name','prediction'],repo('Matrix-Game')/'Matrix-Game-3',[MODELS/'Matrix-Game-3.0/base_distilled_model/diffusion_pytorch_model.safetensors',MODELS/'Matrix-Game-3.0/models_t5_umt5-xxl-enc-bf16.pth',MODELS/'Matrix-Game-3.0/MG-LightVAE_v2.pth'])
if __name__=='__main__':main()
