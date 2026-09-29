#!/usr/bin/env python3
"""Benchmark: six real history frames + heuristic wrist/palm/grasp DROID controls and action text.
Use --benchmark-case <evaluation_case>; see INPUT_ADAPTATION.md.

Legacy smoke mode without --benchmark-case:
Ctrl-World DROID: one 5-frame rollout using the official agent.forward_wm.
Compatibility: single office image is duplicated into three 192x320 cameras;
6 history states repeat the first frame. No DROID robot state exists: default
Cartesian condition is fixed at the midpoint of official DROID normalization
bounds. --actions accepts 11x7 native Cartesian states (6 history + 5 targets).
This does not replay human-hand interactions. Output is generated-only tiled views.
"""
# Benchmark mode: --benchmark-case <evaluation_case>; see INPUT_ADAPTATION.md.
from _support.common import *
def main():
 a=setup('Ctrl-World',__doc__)
 command=[a.python,BASE/'_support/ctrl_infer.py',a.input/'first.png',a.output/'prediction.mp4']
 if a.actions:command+=[a.actions.resolve()]
 execute(a,command,repo('Ctrl-World'),[MODELS/'Ctrl-World/checkpoint-10000.pt',MODELS/'shared/stable-video-diffusion-img2vid/model_index.json',MODELS/'shared/clip-vit-base-patch32/pytorch_model.bin'])
if __name__=='__main__':main()
