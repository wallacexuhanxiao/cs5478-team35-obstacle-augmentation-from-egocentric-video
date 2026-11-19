"""Full-tracking Oracle schedule derived from TACO ground-truth hand/object poses.

No VLM and no learned segmentation: boundaries come from measured events.
Calibrated against the seven manual schedules (Pour 246, Brush 057, Hit 003,
Put-In 253/248/243/258):
  A0 (grasp)        = [right-hand/tool contact onset - 5, max(lift onset, contact onset) + 8]
  A1 (carry/use)    = [A0 end, last tool motion - 10]
  A2 (place/release, tracked to the end) = [last tool motion - 10, end]
Only the free approach [0, A0 start) is cuRobo. Thresholds follow EgoEngine
p.22 (set from the demonstrated object motion): tool rotation >= 1.0 rad ->
1.5 rad (Pour-like), >= 0.5 -> 1.2 rad (Brush/Scrape-like), else 0.9 rad
(Cut-like); position 0.12 m.
"""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--triplet',required=True);p.add_argument('--sequence',required=True)
    p.add_argument('--object-cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--contact-m',type=float,default=.01)
    p.add_argument('--context-frames',type=int,default=6)
    p.add_argument('--compare',type=Path,help='Manual schedule of the same source, for boundary differences only')
    p.add_argument('--aria-release',type=Path,help='EgoEngine release root: read its Aria human keypoints/object poses instead of TACO')
    p.add_argument('--aria-case',help='Aria case under examples/prepared, e.g. mustard_1')
    p.add_argument('--variant',choices=('trackfull','bridge'),default='trackfull',
                   help='trackfull: Anchors to the end, only the free approach is cuRobo; bridge: the carry between '
                        'the grasp and the placement is a grasped cuRobo bridge (A0 grasp, A1 place)')
    p.add_argument('--grasp-tail',type=int,default=8,help='A0 ends this many frames after max(lift onset, contact onset)')
    p.add_argument('--place-lead',type=int,default=10,help='the placement Anchor starts this many frames before the last tool motion')
    p.add_argument('--quiet-boundaries',action='store_true',
                   help='bridge: place the A0 end in [lift+2, lift+10] and the placement start in [last-12, last-3] at '
                        'the frame of least demonstrated wrist acceleration (a planned bridge must take over the '
                        'measured boundary velocity within its acceleration budget)')
    p.add_argument('--release-frames',type=int,default=4,
                   help='bridge: the source releases if its last right-hand contact is at least this many frames before the end')
    args=p.parse_args()
    import trimesh
    from scipy.spatial.transform import Rotation
    from egoengine.taco import load_sequence,mano_forward
    from egoengine.taco.loader import HandData
    from multipath.object_assets import prepare_source_meshes
    if args.aria_release:
        from multipath.aria_source import AriaSource
        seq=AriaSource(args.aria_release,args.aria_case)
    else:seq=load_sequence(args.triplet,args.sequence)
    n=seq.n_frames
    prepare_source_meshes(seq.objects,args.object_cache)
    tool=seq.objects['tool'];P=tool.poses
    mesh=trimesh.load(args.object_cache/tool.object_id/(tool.object_id+'_m.obj'),force='mesh')
    hand=seq.hands['right']
    if args.aria_release:J=np.asarray(hand.joints)
    else:
        _,J,_=mano_forward(HandData(side='right',pose=hand.pose,trans=hand.trans,betas=hand.betas,frame_ids=hand.frame_ids))
        J=np.asarray(J)
    gap=np.asarray([-np.max(trimesh.proximity.signed_distance(mesh,(J[f,[4,8,12,16,20]]-P[f][:3,3])@P[f][:3,:3]))
                    for f in range(n)])
    contact=np.flatnonzero(gap<=args.contact_m)
    if not len(contact):raise ValueError('The right hand never reaches the tool')
    displacement=np.linalg.norm(P[:,:3,3]-P[0,:3,3],axis=1)
    lift=P[:,2,3]-P[0,2,3]
    moved=np.flatnonzero(displacement>.005)
    if not len(moved):raise ValueError('The tool never moves')
    lifted=np.flatnonzero(lift>.01)
    onset=int(lifted[0]) if len(lifted) else int(moved[0])
    step=np.r_[0.,np.linalg.norm(np.diff(P[:,:3,3],axis=0),axis=1)]
    last=int(np.flatnonzero(step>.001).max())
    # The grasp Anchor ends after both contact and lift onset: a tool can
    # start moving under the palm before a fingertip is within contact_m.
    a0s=max(1,int(contact[0])-5);a0e=min(n-20,max(onset,int(contact[0]))+args.grasp_tail)
    a2s=max(a0e+10,last-args.place_lead);a2s=min(a2s,n-6)
    boundary_choice=None
    if args.variant=='bridge' and args.quiet_boundaries:
        wrist=np.asarray(J)[:,0]
        accel=np.r_[0.,np.linalg.norm(np.diff(wrist,2,axis=0),axis=1),0.]
        accel=np.convolve(accel,np.ones(3)/3,mode='same')
        base=max(onset,int(contact[0]))
        lo,hi=base+2,min(n-20,base+10);a0e=int(lo+np.argmin(accel[lo:hi+1]))
        lo,hi=max(a0e+10,last-12),min(n-6,last-3);a2s=int(lo+np.argmin(accel[lo:hi+1])) if hi>=lo else a2s
        boundary_choice=dict(rule='least demonstrated wrist acceleration (3-frame mean) in the allowed ranges',
                             grasp_end=a0e,placement_start=a2s,accel_m_per_frame2=[float(accel[a0e]),float(accel[a2s])])
    if not (a0s<a0e<a2s<n):raise ValueError(f'Degenerate events: contact {contact[0]}, lift {onset}, last motion {last}')
    c=args.context_frames
    anchors=[dict(id='anchor_0',core_start=a0s,core_end=a0e,synth_start=max(0,a0s-c),synth_end=min(n,a0e+c),
                  entry_contact='free',exit_contact='grasped'),
             dict(id='anchor_1',core_start=a0e,core_end=a2s,synth_start=max(0,a0e-c),synth_end=min(n,a2s+c),
                  entry_contact='grasped',exit_contact='grasped'),
             dict(id='anchor_2',core_start=a2s,core_end=n,synth_start=max(0,a2s-c),synth_end=n,
                  entry_contact='grasped',exit_contact='free')]
    events=dict(right_tool_contact=[int(contact[0]),int(contact[-1])],tool_lift_onset=onset,last_tool_motion=last)
    final_contact='free'
    if args.variant=='bridge':
        # Key interactions stay Anchors (grasp + lift, placement); the carry
        # between them is a grasped cuRobo bridge. The final contact follows
        # the source: a demonstrator still holding the placed object at the
        # last frame is not asked to release it.
        released=int(contact[-1])<n-args.release_frames
        final_contact='free' if released else 'grasped'
        anchors=[anchors[0],dict(anchors[2],id='anchor_1',exit_contact=final_contact)]
        events.update(source_releases=bool(released))
    if 'left' in seq.hands and 'target' in seq.objects:
        # Left-hand/target support statistics (reported only; the schedule stays right-hand).
        target=seq.objects['target'];T=target.poses
        tmesh=trimesh.load(args.object_cache/target.object_id/(target.object_id+'_m.obj'),force='mesh')
        left=seq.hands['left']
        _,L,_=mano_forward(HandData(side='left',pose=left.pose,trans=left.trans,betas=left.betas,frame_ids=left.frame_ids))
        L=np.asarray(L)
        lgap=np.asarray([-np.max(trimesh.proximity.signed_distance(tmesh,(L[f,[4,8,12,16,20]]-T[f][:3,3])@T[f][:3,:3]))
                         for f in range(n)])
        support=np.flatnonzero(lgap<=args.contact_m)
        events.update(left_target_contact_fraction=float(len(support)/n),
                      left_target_contact=[int(support[0]),int(support[-1])] if len(support) else None,
                      target_max_displacement_m=float(np.linalg.norm(T[:,:3,3]-T[0,:3,3],axis=1).max()))
    rotation=Rotation.from_matrix(P[:,:3,:3]);rot=float((rotation*rotation[0].inv()).magnitude().max())
    threshold=1.5 if rot>=1. else 1.2 if rot>=.5 else .9
    schedule=dict(source='oracle',fps=float(seq.fps),num_frames=int(n),sequence=args.sequence,triplet=args.triplet,
        instruction=f'{args.triplet}: grasp the tool, use it as demonstrated, return and release',
        segmentation_provenance=(('Derived from EgoEngine-release Aria hand keypoints/object poses' if args.aria_release else
            'Derived from TACO ground-truth hand/object poses')+' by scripts/derive_oracle_schedule.py '
            '(contact onset, lift onset, last tool motion); no VLM, no learned segmentation. '+(
            'Bridge variant: grasp and placement are Anchors, the free approach and the grasped carry are cuRobo.'
            if args.variant=='bridge' else 'Full-tracking variant: only the free approach is cuRobo.')),
        derived_events=dict(events,tool_max_rotation_rad=rot,tool_max_displacement_m=float(displacement.max())),
        task_contract=dict(kind='anchor_sequence_and_final_object_pose',position_tolerance_m=.05,rotation_tolerance_rad=.35,
                           stable_speed_m_s=.03,final_contact=final_contact),
        schedule_variant=args.variant,boundary_choice=boundary_choice,
        semantic_scope='Right hand operates the tool; no appearance/fluid/cutting model.',
        anchors=anchors,
        windows=([dict(start=0,end=a0s,mode='RELAXED',contact='free'),
                  dict(start=a0s,end=a0e,mode='ANCHOR',anchor='anchor_0'),
                  dict(start=a0e,end=a2s,mode='RELAXED',contact='grasped'),
                  dict(start=a2s,end=n,mode='ANCHOR',anchor='anchor_1')] if args.variant=='bridge' else
                 [dict(start=0,end=a0s,mode='RELAXED',contact='free'),
                  dict(start=a0s,end=a0e,mode='ANCHOR',anchor='anchor_0'),
                  dict(start=a0e,end=a2s,mode='ANCHOR',anchor='anchor_1'),
                  dict(start=a2s,end=n,mode='ANCHOR',anchor='anchor_2')]),
        object_tracking_thresholds=dict(position_m=.12,rotation_rad=threshold,
            provenance=f'EgoEngine p.22 tiers from demonstrated tool rotation {rot:.2f} rad; 0.12 m position',
            scope='C.3 reward and C.4 termination/acceptance only; grasp retention and bridge boundaries keep 5 cm / 30 deg'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(schedule,indent=2))
    report=dict(output=str(args.output),anchors=[(a['core_start'],a['core_end']) for a in anchors],events=schedule['derived_events'],
                rotation_threshold=threshold)
    if args.compare:
        manual=json.loads(args.compare.read_text())
        manual=sorted(manual['anchors'],key=lambda a:a['core_start'])
        report['manual']=[(a['core_start'],a['core_end']) for a in manual]
    print('SCHEDULE',json.dumps(report),flush=True)


if __name__=='__main__':main()
