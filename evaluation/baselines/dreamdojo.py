#!/usr/bin/env python3
"""Benchmark: five real context frames + action text; neutral GR1 controls; autoregressive rollout.
Use --benchmark-case <evaluation_case>; see INPUT_ADAPTATION.md.

Legacy smoke mode without --benchmark-case:
DreamDojo 2B GR1 post-train, first RGB -> 12 predicted frames.
Compatibility: office has no GR1 joint controls; default [12,384] zero controls
are a synthetic smoke test. --actions accepts the native normalized action layout.
Uses the registered GR1 OSS configuration: upstream eval.sh references an absent
'dreamdojo_2b_480_640_gr1' alias in this checkout. Converts official DCP shards
without upstream sudo/chmod. PyTorch3D v0.7.9 is installed without compiled
ops because this adapter needs only its pure-PyTorch rotation transforms.
Unused base checkpoint defaults point to the explicit GR1 checkpoint; remove
validation dataloader config for direct inference. Tokenizer revision is pinned
to 85f8ae7b because the upstream 6787e176 revision returns HTTP 404. Does not extract latent actions from future RGB.
"""
# Benchmark mode: --benchmark-case <evaluation_case>; see INPUT_ADAPTATION.md.
from _support.common import *
def main():
 a=setup('DreamDojo',__doc__)
 cmd=[a.python,BASE/'_support/dreamdojo_infer.py',a.input/'first.png',a.output/'prediction.mp4']
 if a.actions:cmd+=[a.actions.resolve()]
 execute(a,cmd,repo('DreamDojo'),[MODELS/'DreamDojo/2B_GR1_post-train/iter_000050000/model/.metadata'])
if __name__=='__main__':main()
