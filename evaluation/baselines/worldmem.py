#!/usr/bin/env python3
"""Benchmark: real video history + pose-derived WASD/mouse and approximate interaction control.
Use --benchmark-case <evaluation_case>; see INPUT_ADAPTATION.md.

Legacy smoke mode without --benchmark-case:
WorldMem Minecraft checkpoint, 16 observed + 16 predicted frames.
Compatibility: 360x640 RGB at 10Hz; optical relative c2w mapped to approximate
Minecraft xyz/pitch/yaw, dropping roll, preserving metres (no learned game scale).
Game controls default to 25D zero; --actions accepts native [32,25]. A process-local
dataset adapter bypasses upstream's fixed 100-frame skip and retry-forever loader.
No future RGB enters context; full clip is supplied only for upstream GT comparison.
The adapter restores the renamed torchvision functional_tensor import used by
pytorchvideo 0.1.5 and saves video without a wandb account.
This office-domain smoke test is not a Minecraft benchmark reproduction.
"""
# Benchmark mode: --benchmark-case <evaluation_case>; see INPUT_ADAPTATION.md.
from _support.common import *
def main():
 a=setup('WorldMem',__doc__)
 cmd=[a.python,BASE/'_support/worldmem_infer.py',a.input/'sample.npz',a.output]
 if a.actions:cmd+=[a.actions.resolve()]
 execute(a,cmd,repo('WorldMem'),[MODELS/'WorldMem/diffusion_only.ckpt',MODELS/'WorldMem/vae_only.ckpt'])
if __name__=='__main__':main()
