#!/usr/bin/env python3
"""Benchmark: two real context frames + wrist-motion/grasp RoboDesk controls; autoregressive rollout.
Use --benchmark-case <evaluation_case>; see INPUT_ADAPTATION.md.

Legacy smoke mode without --benchmark-case:
VP2 RoboDesk: RGB -> 64x64, 2 context + 10 predicted frames.
Compatibility: office has no RoboDesk 5D actions; default zeros are only an OOD
smoke test. --actions accepts native [12,5] actions; output is upstream side-by-side GIF.
"""
# Benchmark mode: --benchmark-case <evaluation_case>; see INPUT_ADAPTATION.md.
from _support.common import *
import numpy as np

def main():
 a=setup('iVideoGPT',__doc__);d=np.load(a.input/'sample.npz')
 action=np.load(a.actions) if a.actions else np.zeros((12,5),np.float32)
 if action.shape!=(12,5):raise ValueError('VP2 RoboDesk actions must be [12,5]')
 np.savez(a.input/'ivideogpt.npz',image=d['rgb'][::4][:12],action=action)
 execute(a,[a.python,'inference/predict.py','--pretrained_model_name_or_path',MODELS/'iVideoGPT','--input_path',a.input/'ivideogpt.npz','--dataset_name','memory_office','--action_conditioned','--action_dim','5','--segment_length','12','--repeat_times','1','--output_path',a.output/'prediction'],repo('iVideoGPT'),[MODELS/'iVideoGPT/transformer/model.safetensors',MODELS/'iVideoGPT/tokenizer/diffusion_pytorch_model.safetensors'])
if __name__=='__main__':main()
