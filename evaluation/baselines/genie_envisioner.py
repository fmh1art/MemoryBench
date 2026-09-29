#!/usr/bin/env python3
"""Benchmark: four sparse prefix observations + bimanual end-effector proxies and action text.
Use --benchmark-case <evaluation_case>; see INPUT_ADAPTATION.md.

Legacy smoke mode without --benchmark-case:
GE-Sim Cosmos v0.1, 4 memory + 25 generated frames.
Compatibility: duplicate the single office camera into head/hand_left/hand_right;
all three have the real office c2w and K. Missing robot controls become fixed end-
effector poses relative to first camera (explicit synthetic smoke test). --actions
accepts native [29,16] = two xyz+xyzw+gripper blocks. No real wrist-camera geometry
or robot dynamics can be reconstructed from this single camera example.
Input is sampled at 16Hz; the official CLI writes output at its default 30fps
(temporal retiming for this smoke test, not calibrated robot-time replay).
"""
# Benchmark mode: --benchmark-case <evaluation_case>; see INPUT_ADAPTATION.md.
from _support.common import *
def main():
 a=setup('Genie-Envisioner',__doc__)
 subprocess.run([str(TOOLS),str(BASE/'_support/model_inputs.py'),'genie',str(a.input),str(a.actions.resolve()) if a.actions else ''],check=True)
 execute(a,[a.python,'gesim_video_gen_examples/infer_gesim.py','--config_file',a.input/'genie.yaml','--image_root',a.input/'genie','--extrinsic_root',a.input/'genie','--intrinsic_root',a.input/'genie','--action_path',a.input/'genie/actions.npy','--output_path',a.output/'prediction'],repo('Genie-Envisioner-V1'),[MODELS/'Genie-Envisioner/ge_sim_cosmos_v0.1.safetensors',MODELS/'Genie-Envisioner/Cosmos-Predict2-2B-Video2World/model_index.json'])
if __name__=='__main__':main()
