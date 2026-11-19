"""RB-Y1 visual commands from a physically optimized floating-XHand trajectory.

The release pairs the RB-Y1 visual replay with the *original* retarget
(`traj[*].robot_cfg`); the action branch's optimized states never drive it.
For an augmented case whose motion differs from the demonstration, this writes
a new `retarget.json` whose right arm follows the optimized hand instead:

1. Hand-link offset: the action scene's `right_hand_link` (original kinematic
   reference) and the replay's XHand root (FK of the original `robot_cfg`) are
   the same rigid hand; their per-frame relative transform is averaged and its
   spread reported (the replay world is the retarget world shifted -0.7 m in X).
2. Per reference frame, the seven right-arm joints are fitted to the optimized
   hand-link pose (warm started from the previous fit; the original frame's
   torso, head and left side are kept). Right XHand joints and the tool pose
   are copied from the optimized state.
3. `init_state.obstacle` records the static obstacle pose for the replay.
Kinematic visual commands only; physics came from the action branch.

`--multipath-run` instead reads a fresh PhysX rollout of the MultiPath pipeline
(same XHand model and `right_hand_link` frame; finger joints by name): its
measured hand link, finger joints and tool poses are mapped from the
simulation frame back to the Aria world with the proposals' recorded
`transform_world_to_robot`, and its environment objects become the obstacle.

The floating hand of the physics never had an arm. Per frame the fitted arm's
mesh surfaces (right arm links and hand) are checked against the obstacle hull
(`arm_obstacle`, face-plane signed distance: negative = the rendered arm passes
through the obstacle), and with `--multipath-run` their union is written in the
simulation frame (`arm_surface_sim.npz`) so obstacle placement can keep clear of
the arm.
"""
import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

SHIFT = np.eye(4)
SHIFT[0, 3] = -0.7
WRIST_BODY = 'gripper0_right_x_right_hand_root'


def pose7_matrix(p):
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat(np.asarray(p[3:7])[[1, 2, 3, 0]]).as_matrix()
    out[:3, 3] = p[:3]
    return out


def action_poses(scene, qpos):
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    hand = model.body('right_hand_link').id
    oa = model.jnt_qposadr[model.joint('right_object_joint').id]
    names = [model.joint(j).name for j in range(model.njnt)]
    hands, tools = [], []
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        m = np.eye(4)
        m[:3, :3] = data.xmat[hand].reshape(3, 3)
        m[:3, 3] = data.xpos[hand]
        hands.append(m)
        tools.append(pose7_matrix(q[oa:oa + 7]))
    return np.asarray(hands), np.asarray(tools), names, model


def mean_transform(transforms):
    rotation = Rotation.from_matrix(transforms[:, :3, :3]).mean().as_matrix()
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = transforms[:, :3, 3].mean(0)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--case', required=True)
    p.add_argument('--source-root', type=Path, required=True, help='released prepared package')
    p.add_argument('--trajectory', type=Path, help='optimized trajectory_mjwp_mixed.npz (official action branch)')
    p.add_argument('--multipath-run', type=Path, help='MultiPath final rollout dir (actual_trace.json, pipeline_result.json)')
    p.add_argument('--multipath-manifest', type=Path, help='proposals manifest.json of that run (sim <- Aria transform)')
    p.add_argument('--source-case', help='released case of a --multipath-run package (default: augmentation.json)')
    p.add_argument('--ik-restart-above-m', type=float, default=.005,
                   help='restart the arm fit when the warm-started fit misses by more than this')
    p.add_argument('--ik-restarts', type=int, default=40)
    p.add_argument('--obstacle-mesh', type=Path, help='obstacle OBJ (m, pose frame); default: --object-cache/<id>/<id>_m.obj')
    p.add_argument('--composite-object', type=Path,
                   help='environment-object JSON (sim frame, static pose) drawn as the obstacle instead of the run\'s own '
                        '(an object composited where the rollout leaves room; it was never simulated)')
    p.add_argument('--object-cache', type=Path, required=True)
    args = p.parse_args()
    if (args.trajectory is None) == (args.multipath_run is None):
        p.error('give exactly one of --trajectory / --multipath-run')
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    sys.path.insert(0, str(project / 'visual'))
    from egoengine.data import get_case, aligned_trajectory
    case = get_case(args.case, args.package)
    augmentation_path = args.package / 'augmentation.json'
    augmentation = json.loads(augmentation_path.read_text()) if augmentation_path.exists() else {}
    source = get_case(args.source_case or augmentation['source_case'], args.source_root)
    original = importlib.import_module('rl_tracking_dev.replay_trajectory_seg_aria')

    source_retarget = case.visual / 'retarget_source.json'
    if not source_retarget.exists():
        shutil.copy2(case.visual / 'retarget.json', source_retarget)
    retarget = json.loads(source_retarget.read_text())
    frames = retarget['traj'][int(retarget.get('warm_up_steps', 0)):]
    env = original.make_tracking_env(SimpleNamespace(data_file=str(source_retarget), cam_name='cameraview'),
                                     use_cam_obs=False)()
    try:
        sim = env.env.sim
        model, data = sim.model._model, sim.data._data
        wrist = model.body(WRIST_BODY).id

        def root_pose():
            m = np.eye(4)
            m[:3, :3] = data.xmat[wrist].reshape(3, 3)
            m[:3, 3] = data.xpos[wrist]
            return m
        replay_roots = []
        for f in frames:
            env._set_full_config(f['robot_cfg'])
            sim.forward()
            replay_roots.append(root_pose())
        replay_roots = np.asarray(replay_roots)
        def body_name(g):
            return model.body(int(model.geom_bodyid[g])).name
        arm_geoms = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                     and ('right_arm' in body_name(g) or body_name(g).startswith('gripper0_right'))]
        arm_vertices = []
        for g in arm_geoms:
            m = int(model.geom_dataid[g]); a = int(model.mesh_vertadr[m]); n = int(model.mesh_vertnum[m])
            arm_vertices.append(model.mesh_vert[a:a + n][::max(1, n // 150)].astype(float))

        def arm_surface():
            return np.concatenate([v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g] for g, v in zip(arm_geoms, arm_vertices)])
        source_hands, _, _, _ = action_poses(source.scene, source.reference()['qpos'])
        if len(source_hands) != len(replay_roots):
            raise ValueError(f'{len(source_hands)} reference frames vs {len(replay_roots)} retarget frames')
        relative = np.asarray([np.linalg.inv(SHIFT @ a) @ r for a, r in zip(source_hands, replay_roots)])
        offset = mean_transform(relative)
        spread = dict(
            position_m=float(np.linalg.norm(relative[:, :3, 3] - offset[:3, 3], axis=1).max()),
            rotation_rad=float(Rotation.from_matrix(relative[:, :3, :3] @ offset[:3, :3].T).magnitude().max()))

        hand_keys = list(original.RIGHT_XHAND_JOINTS)
        obstacle = None
        if args.trajectory:
            trajectory = aligned_trajectory(case, args.trajectory)
            hands, tools, action_joints, action_model = action_poses(case.scene, trajectory['qpos'])
            missing = [k for k in hand_keys if k not in action_joints]
            if missing:
                raise ValueError(f'XHand joints without an action-scene counterpart: {missing}')
            hand_addr = [int(action_model.jnt_qposadr[action_model.joint(k).id]) for k in hand_keys]
            fingers = [dict(zip(hand_keys, q[hand_addr])) for q in trajectory['qpos']]
            if augmentation.get('obstacle'):
                obstacle = pose7_matrix(np.asarray(augmentation['obstacle']['pose_wxyz'], float))
        else:
            from multipath.xhand import RIGHT_XHAND_FINGER_JOINT_NAMES as ours
            to_sim = np.asarray(json.loads(args.multipath_manifest.read_text())['provenance']['transform_world_to_robot'], float)
            to_world = np.linalg.inv(to_sim)
            run_trace = json.loads((args.multipath_run / 'actual_trace.json').read_text())
            states = {round(s['source_frame']): s for s in run_trace}
            picked = [states[k] for k in range(len(frames))]
            hands = np.asarray([to_world @ pose7_matrix(s['wrist_xyz_wxyz']) for s in picked])
            tools = np.asarray([to_world @ pose7_matrix(s['objects']['tool']['pose']) for s in picked])
            fingers = [dict(zip(ours, s['hand_q'])) for s in picked]
            inserted = picked[0].get('environment_objects') or {}
            if args.composite_object:
                item = json.loads(args.composite_object.read_text())[0]
                obstacle = to_world @ pose7_matrix(item['pose'])
                args.obstacle_mesh = args.obstacle_mesh or Path(item['mesh'])
            elif inserted:
                name = sorted(inserted)[0]
                obstacle = to_world @ pose7_matrix(inserted[name]['pose'])
                if args.obstacle_mesh is None:
                    result = json.loads((args.multipath_run / 'pipeline_result.json').read_text())
                    ids = {o['name']: o['id'] for o in result['initial_state']['physical_properties']['environment_objects']}
                    args.obstacle_mesh = args.object_cache / ids[name] / f'{ids[name]}_m.obj'
        hull = None
        if obstacle is not None and args.obstacle_mesh is not None:
            local = np.array([l.split()[1:4] for l in args.obstacle_mesh.read_text().splitlines() if l.startswith('v ')], float)
            placed = SHIFT @ obstacle
            hull = ConvexHull(local @ placed[:3, :3].T + placed[:3, 3]).equations  # n.x + d <= 0 inside
        targets = np.asarray([SHIFT @ h @ offset for h in hands])

        cfg_keys = list(frames[0]['robot_cfg'])
        arm_keys = [k for k in cfg_keys if k.startswith('right_arm_')]
        arm_keys.sort(key=lambda k: int(k.rsplit('_', 1)[1]))
        def model_joint(key):
            hits = [model.joint(j).name for j in range(model.njnt) if model.joint(j).name.endswith(key)]
            if len(hits) != 1:
                raise ValueError(f'robot_cfg key {key} matches joints {hits}')
            return model.joint(hits[0])
        arm = [model_joint(k) for k in arm_keys]
        addresses = [int(model.jnt_qposadr[j.id]) for j in arm]
        bounds = np.asarray([model.jnt_range[j.id] for j in arm])

        new_frames, errors, clearances, surfaces = [], [], [], []
        previous = None
        for k, (f, target) in enumerate(zip(frames, targets)):
            cfg = dict(f['robot_cfg'])
            for key in hand_keys:
                cfg[key] = float(fingers[k][key])
            env._set_full_config(cfg)
            sim.forward()
            start = np.asarray([cfg[key] for key in arm_keys], float) if previous is None else previous

            def residual(q):
                data.qpos[addresses] = q
                mujoco.mj_kinematics(model, data)
                rot = Rotation.from_matrix(target[:3, :3] @ data.xmat[wrist].reshape(3, 3).T).as_rotvec()
                return np.r_[data.xpos[wrist] - target[:3, 3], .2 * rot, (q - start) * 1e-4]
            lo, hi = bounds[:, 0] + 1e-6, bounds[:, 1] - 1e-6

            def fit_from(q0):
                q = least_squares(residual, np.clip(q0, lo, hi), bounds=(lo, hi), max_nfev=300,
                                  ftol=1e-12, xtol=1e-12, gtol=1e-12).x
                r = residual(q)
                return float(np.linalg.norm(r[:3]) + .25 * np.linalg.norm(r[3:6])), q
            best = fit_from(start)
            if best[0] > args.ik_restart_above_m:
                # Warm start stuck: restart from the original frame's arm and seeded uniform draws; the
                # continuity term only breaks near-ties between branches.
                rng = np.random.default_rng(k)
                seeds = [np.asarray([f['robot_cfg'][key] for key in arm_keys], float)]
                seeds += [rng.uniform(lo, hi) for _ in range(args.ik_restarts)]
                for q0 in seeds:
                    cost, q = fit_from(q0)
                    if cost + 1e-3 * np.linalg.norm(q - start) < best[0] + 1e-3 * np.linalg.norm(best[1] - start):
                        best = (cost, q)
            previous = best[1].copy()
            residual(previous)
            actual = root_pose()
            points = arm_surface()
            surfaces.append(points)
            if hull is not None:
                clearances.append(float(((points @ hull[:, :3].T) + hull[:, 3]).max(1).min()))
            errors.append(dict(position_m=float(np.linalg.norm(actual[:3, 3] - target[:3, 3])),
                               rotation_rad=float(Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude())))
            for key, value in zip(arm_keys, previous):
                cfg[key] = float(value)
            new_frames.append(dict(f, robot_cfg=cfg, tool=tools[k].tolist()))
    finally:
        env.close()
    arm_obstacle = None
    if clearances:
        arm_obstacle = dict(min_signed_distance_m=min(clearances), frames_inside=[k for k, c in enumerate(clearances) if c < 0],
                            mesh=str(args.obstacle_mesh), geoms=len(arm_geoms),
                            scope='face-plane signed distance of rendered right-arm/hand mesh vertices to the obstacle hull')
    if args.multipath_run is not None:
        # Per source frame, voxelized (1 cm) in the simulation frame.
        to_sim_replay = to_sim @ np.linalg.inv(SHIFT)
        points, frame = [], []
        for k, surface in enumerate(surfaces):
            cells = np.unique(np.floor((surface @ to_sim_replay[:3, :3].T + to_sim_replay[:3, 3]) / .01), axis=0)
            points.append((cells + .5) * .01)
            frame.append(np.full(len(cells), k))
        np.savez_compressed(case.visual / 'arm_surface_sim.npz', points=np.concatenate(points), frame=np.concatenate(frame),
                            frames=len(surfaces), source=str(args.multipath_run))

    out = dict(retarget, traj=retarget['traj'][:int(retarget.get('warm_up_steps', 0))] + new_frames)
    out['init_state'] = dict(retarget['init_state'], **({'obstacle': obstacle.tolist()} if obstacle is not None else {}))
    out['augmentation'] = dict(
        source=('physically optimized action trajectory' if args.trajectory else 'MultiPath PhysX rollout (floating XHand)'),
        trajectory=str(args.trajectory or args.multipath_run),
        hand_link_offset=offset.tolist(), hand_link_offset_spread=spread,
        ik_worst=dict(position_m=max(e['position_m'] for e in errors), rotation_rad=max(e['rotation_rad'] for e in errors)),
        frames_ik_over_1cm=[k for k, e in enumerate(errors) if e['position_m'] > .01],
        arm_obstacle=arm_obstacle,
        scope='right arm joints fitted to the optimized XHand; torso/head/left from the original retarget '
              '(the release locks the torso joints); visual only. Frames over 1 cm are beyond the fixed-torso '
              'arm after restarts: the rendered hand there is not at the physical hand pose')
    (case.visual / 'retarget.json').write_text(json.dumps(out))
    report = dict(out['augmentation'], per_frame_ik=errors, per_frame_arm_obstacle_m=clearances, arm_keys=arm_keys,
                  hand_keys=hand_keys)
    (case.visual / 'physics_to_retarget.json').write_text(json.dumps(report, indent=1))
    print('PHYSICS_TO_RETARGET', json.dumps({k: report[k] for k in ('hand_link_offset_spread', 'ik_worst', 'frames_ik_over_1cm',
                                                                     'arm_obstacle', 'arm_keys')}),
          flush=True)


if __name__ == '__main__':
    main()
