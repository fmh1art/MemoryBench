"""Heuristic hand retargeting, not calibrated robot inverse kinematics.
Reference axes: initial optical camera (+x right,+y down,+z forward).
Translation is wrist displacement relative to the last observed wrist.
"""
import numpy as np
from scipy.spatial.transform import Rotation

def hand_states(d,side,nctx):
    points=d[side+'_hand_positions_ref'];valid=d[side+'_hand_valid'].astype(bool)
    ref=d['c2w'][nctx-1];local=(points-ref[:3,3])@ref[:3,:3]
    # Sensor gaps are held, never filled using future samples.
    local=local.copy();last=None
    for i in range(len(local)):
        if valid[i] and np.isfinite(local[i]).all():last=local[i].copy()
        elif last is not None:local[i]=last
        else:raise ValueError(f'{side}: missing initial hand observation; cannot retarget')
    wrist=local[:,0];delta=wrist-wrist[nctx-1]
    # Thumb/index distance is a continuous grasp proxy, not object-state inference.
    opening=np.clip((np.linalg.norm(local[:,4]-local[:,8],axis=-1)-.015)/.065,0,1)
    # Palm basis: index to little MCP, wrist to middle MCP.
    x=local[:,5]-local[:,17];x/=np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)
    y=local[:,9]-local[:,0];y-=np.sum(y*x,axis=-1,keepdims=True)*x;y/=np.maximum(np.linalg.norm(y,axis=-1,keepdims=True),1e-8)
    z=np.cross(x,y);R=np.stack([x,y,z],axis=-1)
    if np.any(abs(np.linalg.det(R)-1)>.01):raise ValueError('Degenerate hand orientation')
    return delta,opening,R
