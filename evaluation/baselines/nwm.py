#!/usr/bin/env python3
"""NWM CDiT-XL, 4-frame context -> 4 autoregressive navigation predictions.
Compatibility: each RGB resized to model's square resolution; camera displacement
projected onto local horizontal plane [forward=z,left=-x,yaw], ignoring roll/height.
Office is sampled at 4Hz; each step uses the official frame-offset value 1/128.
Metres are used as native displacement units without dataset-specific scale calibration. Camera movement
is used as navigation control, not an inferred robot command. --actions: [4,3].
"""
from _support.common import *
def main():
 a=setup('NWM',__doc__)
 cmd=[a.python,BASE/'_support/nwm_infer.py',a.input/'sample.npz',a.output/'prediction.mp4']
 if a.actions:cmd+=[a.actions.resolve()]
 execute(a,cmd,repo('nwm'),[MODELS/'NWM/cdit_xl_100000.pth.tar',MODELS/'shared/sd-vae-ft-ema/config.json'])
if __name__=='__main__':main()
