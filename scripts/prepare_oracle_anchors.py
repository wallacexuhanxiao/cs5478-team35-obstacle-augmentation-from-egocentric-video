"""Retarget ONLY padded oracle intervals directly from human TACO observations."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from multipath.oracle_schedule import OracleSchedule


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--schedule',type=Path,required=True)
    p.add_argument('--triplet',required=True)
    p.add_argument('--sequence',required=True)
    p.add_argument('--mount-report',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--object-cache',type=Path,help='Optional existing meshes for local object-aware grasp retargeting.')
    p.add_argument('--template-artifacts',type=Path,help='Validated local acquisition library, for initialization only.')
    p.add_argument('--template-proposals',type=Path)
    p.add_argument('--retarget',choices=('raw','egoengine','egoengine_official','dexagent','egoengine_surface','egoengine_wrap','egoengine_wrap_face'),default='raw',
        help='egoengine: paper Eq. (1) Mink retarget; dexagent: contact-guided keypoints + SPIDER IK weights; '
             'egoengine_surface: Eq. (1) plus the project object-surface grasp fit on acquisition Anchors. None uses templates.')
    p.add_argument('--aria-release',type=Path,help='EgoEngine release root: Aria human keypoints/object poses instead of TACO MANO')
    p.add_argument('--aria-case',help='Aria case under examples/prepared, e.g. mustard_1')
    p.add_argument('--repair-source-tool-flips',action='store_true',
        help='Replace held-tool pose tracking flips (hand-relative jump that returns) by hand-rigid interpolation; reported in the manifest')
    args=p.parse_args()
    if args.retarget!='raw' and args.template_artifacts:
        p.error('Aligned retargeting has no template initialization')
    if args.retarget in ('dexagent','egoengine_official') and not args.object_cache:
        p.error('Object-aware retargeting needs the object meshes')
    from egoengine.taco import load_sequence, mano_forward, compute_pseudo_base
    from egoengine.taco.loader import HandData
    from egoengine.retarget import HandRetargeter
    schedule=OracleSchedule.load(args.schedule)
    if args.aria_release:
        from multipath.aria_source import AriaSource
        seq=AriaSource(args.aria_release,args.aria_case)
    else:seq=load_sequence(args.triplet,args.sequence)
    if seq.n_frames!=schedule.num_frames or abs(seq.fps-schedule.metadata.get('source_fps',schedule.fps))>1e-5:
        raise ValueError('Oracle/source clocks differ')
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    mount=json.loads(args.mount_report.read_text())
    if args.aria_release:
        # The release's Aria world is already table-aligned (z up, metres).
        # Translate hand and object together so the demonstrated object
        # workspace is centred on the simulated table (FloatingScene: 0.95 x
        # 1.0 m at x=0.57, y=0); the 32 cm carry otherwise leaves its edge.
        transform=np.eye(4)
        xy=seq.objects['tool'].poses[:,:2,3]
        transform[:2,3]=np.array([.57,0.])-(xy.min(0)+xy.max(0))/2
    else:
        transform=np.linalg.inv(compute_pseudo_base(seq))
        transform[:3,3]+=np.array(mount['scene_translation_xyz'])
    hand=seq.hands['right']
    source_repairs=None
    if args.repair_source_tool_flips:
        from multipath.source_pose_repair import repair_held_tool_flips
        all_joints=hand.joints if args.aria_release else mano_forward(hand)[1]
        held=[(a.synth_start,a.synth_end) for a in schedule.anchors.values()
              if (a.entry_contact,a.exit_contact)==('grasped','grasped')]
        repaired,source_repairs=repair_held_tool_flips(seq.objects['tool'].poses,np.asarray(all_joints),held,schedule.fps)
        seq.objects['tool'].poses=repaired
        print('SOURCE_TOOL_POSE_REPAIRS',json.dumps(source_repairs),flush=True)
    betas=np.zeros(10,np.float32) if args.aria_release else hand.betas
    flat=HandData(side='right',pose=np.zeros((1,48),np.float32),trans=np.zeros((1,3),np.float32),betas=betas,frame_ids=np.zeros(1,np.int64))
    _,calibration,_=mano_forward(flat)
    if args.aria_release:
        # Mean MANO rest hand scaled to the tracked bone lengths.
        calibration=seq.calibrate(np.asarray(calibration)[0])[None]
    provenance={'triplet':args.triplet,'sequence':args.sequence,
                'input':('EgoEngine release Aria hand-tracking keypoints and object poses (no release robot retarget)'
                         if args.aria_release else 'raw TACO MANO and object poses'),
                'source_video':seq.video_path,'schedule_sha256':hashlib.sha256(args.schedule.read_bytes()).hexdigest(),
                'full_robot_reference_used':False,'vlm_used':False,'transform_world_to_robot':transform.tolist()}
    initial={}; final={}
    object_assets={}
    if args.object_cache:
        from multipath.object_assets import prepare_source_meshes
        object_assets=prepare_source_meshes(seq.objects,args.object_cache)
    for role,obj in seq.objects.items():
        def pose(frame):
            t=transform@obj.poses[frame]
            return np.r_[t[:3,3],Rotation.from_matrix(t[:3,:3]).as_quat()[[3,0,1,2]]].tolist()
        initial[role]={'id':obj.object_id,'pose':pose(0)}
        final[role]=pose(seq.n_frames-1)
    paths={};retarget_details={};table_z=None
    library_initialization=False
    if args.template_artifacts and args.template_proposals:
        prior=json.loads((args.template_proposals/'manifest.json').read_text())
        if prior['initial_objects']['tool']['id']==seq.objects['tool'].object_id:
            from multipath.anchors.artifact import load_validated_anchor
            from multipath.anchors.schema import InteractionAnchor
            for report_path in args.template_artifacts.glob('*/report.json'):
                report=json.loads(report_path.read_text())
                if (report['entry_contact'],report['exit_contact'])!=('free','grasped'):continue
                old=InteractionAnchor(**{key:report[key] for key in InteractionAnchor.__dataclass_fields__})
                load_validated_anchor(args.template_artifacts,old,provenance=prior['provenance'])
                library_initialization=True
                break
    if args.object_cache:
        import trimesh
        table_z=min(((transform@obj.poses[0])[:3,:3]@
            trimesh.load(args.object_cache/obj.object_id/(obj.object_id+'_m.obj'),force='mesh').vertices.T).T[:,2].min()
            +(transform@obj.poses[0])[2,3] for obj in seq.objects.values())-.001
    working_face=None
    if args.retarget=='egoengine_wrap_face' and args.object_cache and 'target' in seq.objects:
        # The tool side that touches the target in the demonstration stays
        # free of the hand (rub with the eraser, not with the fingers).
        import trimesh
        from multipath.anchors.grasp_retarget import tool_working_face
        meshes={role:trimesh.load(args.object_cache/seq.objects[role].object_id/(seq.objects[role].object_id+'_m.obj'),force='mesh')
                for role in ('tool','target')}
        working_face=tool_working_face(meshes['tool'],transform[None]@seq.objects['tool'].poses,
                                       meshes['target'],transform[None]@seq.objects['target'].poses)
        print('TOOL_WORKING_FACE',json.dumps(working_face),flush=True)
    for anchor in schedule.anchors.values():
        s,e=anchor.synth_start,anchor.synth_end
        if args.aria_release:joints=np.asarray(hand.joints[s:e])
        else:
            local=HandData(side='right',pose=hand.pose[s:e],trans=hand.trans[s:e],betas=hand.betas,frame_ids=hand.frame_ids[s:e])
            _,joints,_=mano_forward(local)
        joints=joints@transform[:3,:3].T+transform[:3,3]
        tool_poses=transform[None]@seq.objects['tool'].poses[s:e]
        if args.retarget=='egoengine_official':
            # The EgoEngine release's code (show_human_new.py / rby1_kinematics.py):
            # fingertip targets within 3 cm of an object surface are snapped to
            # it over the whole interval, and the wrist position is tracked too.
            import trimesh
            from multipath.anchors.egoengine_retarget import EgoEngineRetargeter
            snapped=0;joints=np.array(joints,copy=True)
            for role,obj in seq.objects.items():
                oid=obj.object_id;mesh=trimesh.load(args.object_cache/oid/(oid+'_m.obj'),force='mesh')
                poses=transform[None]@obj.poses[s:e]
                for f in range(len(joints)):
                    tips=(joints[f,[4,8,12,16,20]]-poses[f][:3,3])@poses[f][:3,:3]
                    closest,distance,_=trimesh.proximity.closest_point(mesh,tips)
                    near=distance<=.03
                    if near.any():
                        joints[f,np.array([4,8,12,16,20])[near]]=closest[near]@poses[f][:3,:3].T+poses[f][:3,3];snapped+=int(near.sum())
            qpos,detail=EgoEngineRetargeter('right',calibration[0],wrist_position_cost=1.).retarget(joints)
            retarget_details[anchor.id]=dict(detail,official_release_terms=dict(tip_surface_snap_m=.03,snapped_tip_targets=snapped,
                wrist_position_cost=1.,source='EgoEngine-CoRL2026 visual/retargeting (b1363c2)'))
        elif args.retarget in ('egoengine','egoengine_surface','egoengine_wrap','egoengine_wrap_face'):
            from multipath.anchors.egoengine_retarget import EgoEngineRetargeter
            qpos,retarget_details[anchor.id]=EgoEngineRetargeter('right',calibration[0]).retarget(joints)
        elif args.retarget=='dexagent':
            import trimesh
            from multipath.anchors.dexagent_retarget import DexAgentRetargeter
            oid=seq.objects['tool'].object_id;mesh=trimesh.load(args.object_cache/oid/(oid+'_m.obj'),force='mesh')
            qpos,retarget_details[anchor.id]=DexAgentRetargeter('right',calibration[0]).retarget(joints,tool_poses,mesh,schedule.fps)
        else:qpos=HandRetargeter('right',calibration[0]).retarget(joints).qpos
        arrays={'source_frames':np.arange(s,e),'wrist_hand_q':qpos,'human_joints':joints}
        arrays['raw_wrist_hand_q']=np.array(qpos,copy=True)
        for role,obj in seq.objects.items():arrays[role+'_pose']=transform[None]@obj.poses[s:e]
        if args.retarget in ('egoengine','egoengine_official','dexagent'):pass
        elif args.object_cache and not library_initialization:
            import trimesh
            from egoengine.retarget.ik import SIM_URDF
            from multipath.anchors.grasp_retarget import surface_grasp_prior
            oid=initial['tool']['id'];mesh=trimesh.load(args.object_cache/oid/(oid+'_m.obj'),force='mesh')
            arrays['wrist_hand_q'],detail=surface_grasp_prior(anchor,arrays['source_frames'],arrays['wrist_hand_q'],
                joints,arrays['tool_pose'],mesh,SIM_URDF['right'],fps=schedule.fps,table_z=table_z,
                geometry_cache=args.object_cache/oid/'geometry_cache',wrap=args.retarget.startswith('egoengine_wrap'),
                working_face=working_face)
            if detail:retarget_details[anchor.id]=(dict(detail,base_retarget=retarget_details.get(anchor.id))
                                                   if args.retarget in ('egoengine_surface','egoengine_wrap','egoengine_wrap_face') else detail)
        elif library_initialization and (anchor.entry_contact,anchor.exit_contact)==('free','grasped'):
            retarget_details[anchor.id]=dict(method='validated_local_acquisition_library_initialization',
                library=str(args.template_artifacts),physical_acceptance=False,
                redundant_surface_IK_skipped=True)
        path=out/(anchor.id+'.npz');np.savez_compressed(path,**arrays)
        paths[anchor.id]=path.name
        print(anchor.id,s,e,'retargeted',flush=True)
    manifest={'provenance':provenance,'initial_objects':initial,'final_objects':final,
              'object_assets':object_assets,
              'anchors':paths,'mount_xyzw':mount['candidate_mount_quaternion_xyzw'],
              'interaction_constraints':schedule.metadata.get('interaction_constraints',{}),
              'object_tracking_thresholds':schedule.metadata.get('object_tracking_thresholds'),
              'retarget_details':retarget_details,'retarget':args.retarget,'tool_working_face':working_face,'source_tool_pose_repairs':source_repairs,
              'fps':schedule.fps,'num_frames':schedule.num_frames,
              'physical_validation':False,'kind':'local_kinematic_proposals'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'oracle_schedule.json').write_text(args.schedule.read_text())

if __name__=='__main__':main()
