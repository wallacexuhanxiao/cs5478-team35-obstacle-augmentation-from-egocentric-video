"""RB-Y1 right arm of the release's visual model as a cuRobo robot in the MultiPath simulation frame.

The visual replay (robosuite RB-Y1, torso joints locked by the release) and the
MultiPath floating hand share the XHand `right_hand_link` frame (offset measured
by `physics_to_retarget.py`, spread ~1e-7). This writes
  * a URDF: sim-frame base -> fixed torso mount -> the seven right-arm revolute
    joints (MuJoCo body/joint frames copied) -> fixed links down to the XHand
    root -> `wrist` = the floating planner's hand frame;
  * per-link collision spheres covering the arm's mesh surfaces (the hand
    itself is left to the floating planner's measured hand spheres);
  * the release's first-frame arm angles as the retract configuration.
Kinematic export only; the MultiPath physics keeps its floating hand.
"""
import argparse
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

SHIFT = np.eye(4)
SHIFT[0, 3] = -0.7
WRIST_BODY = 'gripper0_right_x_right_hand_root'


def transform(pos, quat_wxyz):
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat(np.asarray(quat_wxyz)[[1, 2, 3, 0]]).as_matrix()
    out[:3, 3] = pos
    return out


def origin(matrix):
    rpy = Rotation.from_matrix(matrix[:3, :3]).as_euler('xyz')
    return (f'<origin xyz="{" ".join(f"{v:.9f}" for v in matrix[:3, 3])}" '
            f'rpy="{" ".join(f"{v:.9f}" for v in rpy)}"/>')


def axis_frame(axis):
    """Rotation C with C @ z = axis (cuRobo joints must turn about a principal axis)."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    for k in range(3):
        if abs(abs(axis[k]) - 1) < 1e-9:
            return np.eye(3), np.eye(3)[k] * np.sign(axis[k])
    rotation, _ = Rotation.align_vectors([axis], [[0., 0., 1.]])
    return rotation.as_matrix(), np.array([0., 0., 1.])


def cover(points, cell):
    """Spheres on occupied voxels; radius reaches every point of its cell."""
    if not len(points):
        return []
    keys = np.unique(np.floor(points / cell), axis=0)
    radius = float(np.sqrt(3) * cell / 2 + 1e-3)
    return [dict(center=((k + .5) * cell).tolist(), radius=radius) for k in keys]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--visual-dir', type=Path, required=True, help='Aria case visual inputs (retarget.json or retarget_source.json)')
    p.add_argument('--manifest', type=Path, required=True, help='MultiPath proposals manifest.json (transform_world_to_robot)')
    p.add_argument('--hand-offset', type=Path, required=True, help='physics_to_retarget.json with hand_link_offset')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cell', type=float, default=.04)
    p.add_argument('--velocity', type=float, default=2.0, help='planning joint velocity limit (rad/s)')
    args = p.parse_args()
    project = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(project), str(project / 'visual')]
    original = importlib.import_module('rl_tracking_dev.replay_trajectory_seg_aria')
    source = args.visual_dir / 'retarget_source.json'
    source = source if source.exists() else args.visual_dir / 'retarget.json'
    retarget = json.loads(source.read_text())
    first = retarget['traj'][int(retarget.get('warm_up_steps', 0))]['robot_cfg']
    env = original.make_tracking_env(SimpleNamespace(data_file=str(source), cam_name='cameraview'), use_cam_obs=False)()
    try:
        sim = env.env.sim
        model, data = sim.model._model, sim.data._data
        env._set_full_config(first)
        sim.forward()
        arm = [model.joint(f'robot0_right_arm_{i}') for i in range(7)]
        chain = [int(model.jnt_bodyid[j.id]) for j in arm]
        tail, b = [], model.body(WRIST_BODY).id
        while b != chain[-1]:
            tail.append(b)
            b = int(model.body_parentid[b])
        bodies = chain + tail[::-1]
        for k in range(1, len(bodies)):
            if int(model.body_parentid[bodies[k]]) != bodies[k - 1]:
                raise ValueError(f'arm bodies are not one chain: {model.body(bodies[k]).name}')
        mount = int(model.body_parentid[bodies[0]])
        world_mount = np.eye(4)
        world_mount[:3, :3] = data.xmat[mount].reshape(3, 3)
        world_mount[:3, 3] = data.xpos[mount]
        to_sim = np.asarray(json.loads(args.manifest.read_text())['provenance']['transform_world_to_robot'], float)
        sim_mount = to_sim @ np.linalg.inv(SHIFT) @ world_mount
        offset = np.asarray(json.loads(args.hand_offset.read_text())['hand_link_offset'], float)

        # URDF link frame = MuJoCo body frame @ C (C turns a tilted joint axis onto a principal axis).
        xml = ['<robot name="rby1_right_arm"><link name="base"/>', '<link name="arm_mount"/>',
               f'<joint name="mount" type="fixed"><parent link="base"/><child link="arm_mount"/>{origin(sim_mount)}</joint>']
        parent, correction, names, limits, spheres, joint_names, entries = 'arm_mount', np.eye(4), {}, [], {}, [], []
        for b in bodies:
            name = model.body(b).name
            names[b] = name
            local = transform(model.body_pos[b], model.body_quat[b])
            joints = [j for j in range(model.njnt) if int(model.jnt_bodyid[j]) == b]
            if len(joints) > 1:
                raise ValueError(f'{name} has {len(joints)} joints')
            frame = np.eye(4)
            if joints:
                j = joints[0]
                if np.linalg.norm(model.jnt_pos[j]) > 1e-9:
                    raise ValueError(f'{model.joint(j).name}: joint not at its body origin')
                frame[:3, :3], axis = axis_frame(model.jnt_axis[j])
            joint_origin = np.linalg.inv(correction) @ local @ frame
            if joints:
                lo, hi = model.jnt_range[j]
                jn = model.joint(j).name
                joint_names.append(jn)
                limits.append([float(lo), float(hi)])
                entries.append((joint_origin, axis))
                xml.append(f'<link name="{name}"/><joint name="{jn}" type="revolute"><parent link="{parent}"/>'
                           f'<child link="{name}"/>{origin(joint_origin)}<axis xyz="{" ".join(f"{v:.0f}" for v in axis)}"/>'
                           f'<limit lower="{lo:.9f}" upper="{hi:.9f}" velocity="{args.velocity}" effort="100"/></joint>')
            else:
                entries.append((joint_origin, None))
                xml.append(f'<link name="{name}"/><joint name="fixed_{name}" type="fixed"><parent link="{parent}"/>'
                           f'<child link="{name}"/>{origin(joint_origin)}</joint>')
            parent, correction = name, frame
            if name.startswith('gripper0_right'):
                continue  # the floating planner's measured XHand spheres cover the hand
            points = []
            for g in range(model.ngeom):
                if int(model.geom_bodyid[g]) != b or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                    continue
                m = int(model.geom_dataid[g]); a = int(model.mesh_vertadr[m]); n = int(model.mesh_vertnum[m])
                pose = np.linalg.inv(frame) @ transform(model.geom_pos[g], model.geom_quat[g])
                points.append(model.mesh_vert[a:a + n].astype(float) @ pose[:3, :3].T + pose[:3, 3])
            if points:
                spheres[name] = cover(np.concatenate(points), args.cell)
        wrist_origin = np.linalg.inv(correction) @ np.linalg.inv(offset)
        xml.append(f'<link name="wrist"/><joint name="hand_link" type="fixed"><parent link="{parent}"/><child link="wrist"/>'
                   f'{origin(wrist_origin)}</joint></robot>')
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / 'rby1_right_arm.urdf').write_text('\n'.join(xml))

        # Self-check: the URDF chain as written reproduces the replay's hand link at the first frame (sim frame).
        q = [float(first[n.replace('robot0_', '')]) for n in joint_names]
        matrix, k = sim_mount.copy(), 0
        for joint_origin, axis in entries:
            matrix = matrix @ joint_origin
            if axis is not None:
                turn = np.eye(4)
                turn[:3, :3] = Rotation.from_rotvec(axis * q[k]).as_matrix()
                matrix, k = matrix @ turn, k + 1
        hand_urdf = matrix @ wrist_origin
        root = np.eye(4); root[:3, :3] = data.xmat[model.body(WRIST_BODY).id].reshape(3, 3); root[:3, 3] = data.xpos[model.body(WRIST_BODY).id]
        hand_replay = to_sim @ np.linalg.inv(SHIFT) @ root @ np.linalg.inv(offset)
        fk_error = float(np.linalg.norm(hand_urdf[:3, 3] - hand_replay[:3, 3])
                         + Rotation.from_matrix(hand_urdf[:3, :3] @ hand_replay[:3, :3].T).magnitude())
        report = dict(urdf=str(args.out / 'rby1_right_arm.urdf'), joint_names=joint_names, joint_limits=limits,
                      collision_spheres=spheres, sphere_count=sum(len(v) for v in spheres.values()),
                      retract_config=q, base_link='base', ee_link='wrist', mount_pose_sim=sim_mount.tolist(),
                      chain=[names[b] for b in bodies], fk_check_error=fk_error,
                      torso=dict((k, first[k]) for k in first if k.startswith('torso')),
                      provenance=dict(visual_model=str(source), manifest=str(args.manifest), hand_offset=str(args.hand_offset),
                                      scope='release RB-Y1 visual kinematics, torso locked; planning model only'))
        (args.out / 'arm_model.json').write_text(json.dumps(report, indent=1))
        print('RBY1_ARM', json.dumps({k: report[k] for k in ('joint_names', 'sphere_count', 'fk_check_error', 'chain')}), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
