"""Repair demonstrated tool-pose tracking flips while the tool is held.

A grasped tool moves rigidly with the hand. TACO's object tracking can flip a
nearly symmetric tool for a few frames (Cut 040: the knife turns 2.5 rad in one
frame at 30 Hz and flips back four frames later while the wrist turns
0.2-0.6 rad/s). Such a frame is a measurement error, not motion to imitate.
Only hand-relative jumps that return to the pre-jump relative pose are repaired;
fast swings of the whole hand (a hammer blow) keep a slow hand-relative pose.
"""
import numpy as np
from scipy.spatial.transform import Rotation,Slerp


def hand_frames(joints):
    """Wrist frames from 21 hand joints (wrist, index MCP 5, pinky MCP 17)."""
    joints=np.asarray(joints);origin=joints[:,0]
    x=joints[:,5]-origin;x/=np.linalg.norm(x,axis=1,keepdims=True)
    z=np.cross(x,joints[:,17]-origin);z/=np.linalg.norm(z,axis=1,keepdims=True)
    y=np.cross(z,x)
    frames=np.repeat(np.eye(4)[None],len(joints),axis=0)
    frames[:,:3,:3]=np.stack((x,y,z),axis=2);frames[:,:3,3]=origin
    return frames


def repair_held_tool_flips(tool_poses,joints,windows,fps,max_rate=10.,return_rad=.35,max_frames=10):
    """Return repaired tool poses and a report of every replaced frame run.

    windows: [start, end) source-frame ranges in which the tool is held.
    """
    poses=np.array(tool_poses,dtype=float,copy=True);hands=hand_frames(joints)
    relative=np.linalg.inv(hands)@poses;report=[]
    def angle(a,b):
        return float(Rotation.from_matrix(a[:3,:3].T@b[:3,:3]).magnitude())
    for start,end in windows:
        t=max(start+1,1)
        while t<min(end,len(poses)):
            rate=angle(relative[t-1],relative[t])*fps
            if rate<=max_rate:t+=1;continue
            back=next((j for j in range(t+1,min(t+max_frames+1,end,len(poses)))
                       if angle(relative[t-1],relative[j])<=return_rad),None)
            if back is None:
                report.append(dict(frames=[t],hand_relative_rate_rad_s=rate,repaired=False,
                                   reason='no return to the pre-jump hand-relative pose'))
                t+=1;continue
            a,b=relative[t-1],relative[back];span=back-(t-1)
            slerp=Slerp([0,1],Rotation.from_matrix([a[:3,:3],b[:3,:3]]))
            peak=max(angle(relative[k-1],relative[k])*fps for k in range(t,back+1))
            for k in range(t,back):
                s=(k-(t-1))/span;local=np.eye(4)
                local[:3,:3]=slerp(s).as_matrix();local[:3,3]=(1-s)*a[:3,3]+s*b[:3,3]
                poses[k]=hands[k]@local
            report.append(dict(frames=list(range(t,back)),hand_relative_rate_rad_s=peak,repaired=True,
                               return_rad=angle(a,b),method='hand-rigid slerp between the bounding valid frames'))
            relative=np.linalg.inv(hands)@poses;t=back+1
    return poses,report
