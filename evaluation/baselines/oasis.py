#!/usr/bin/env python3
"""Oasis: real video history + head-motion WASD/mouse, predict future-only RGB.
--benchmark-case uses annotated prefix/t/action_end/eval_end and an approximate
use control for open/close_drawer. Minecraft use cannot guarantee drawer behavior.
Without --benchmark-case: 16 observed + 16 predicted at 16Hz, camera controls only.
Official video loader THWC bug is corrected in the process-local adapter.
"""
from _support.common import *
import numpy as np
from _support.prepare_case import video
from _support.benchmark_inputs import game_controls

def main():
 a=setup('Oasis',__doc__);d=np.load(a.input/'sample.npz');n=16;total=32
 video(a.input/'context.mp4',d['rgb'][:n],16)
 x=np.load(a.actions) if a.actions else game_controls(d['c2w'][:total],d['timestamp_ns'][:total],'camera_only',total,n)
 if x.shape!=(total,25):raise ValueError('Oasis controls must be [32,25], transition arriving at each frame')
 np.save(a.input/'game_actions.npy',x)
 (a.input/'adaptation.json').write_text(json.dumps({'context_frames':n,'prediction_frames':16,'fps':16,'future_rgb_read':False,'mode':'short_smoke'}))
 execute(a,[a.python,BASE/'_support/oasis_video.py',a.input,a.output/'prediction.mp4'],repo('open-oasis'),[MODELS/'Oasis/oasis500m.safetensors',MODELS/'Oasis/vit-l-20.safetensors'])
if __name__=='__main__':main()
