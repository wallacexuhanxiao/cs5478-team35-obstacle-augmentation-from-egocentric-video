"""EgoEngine MPC mode: SPIDER sampling MPC (MuJoCo Warp) for one local Anchor.

Runs in the SPIDER environment, not Isaac. The window starts at the actual
reached entry (core_start) of a continuous PhysX rollout and spans
core_start..synth_end at the 120 Hz physics clock (no time stretching). The
hand reference is the local Eq. (1) retarget; the object reference is the
source object motion applied from the actual resting pose. Wrist and finger
actuation mirror the PhysX floating-XHand controller (gains, force limits,
gravity compensation) so SPIDER's controls can run there as commands.

Output: commands.npz (25-column floating commands, one per physics tick) and
external_mpc_report.json. These are proposals; PhysX Replay/PPO and fresh
padded/core rollouts decide acceptance.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROBOT, DATASET, EMBODIMENT = 'xhand', 'multipath', 'right'
DEFAULTS = dict(
    sim_dt=1/120, ref_dt=1/120, ctrl_dt=.05, knot_dt=.1, horizon=.4, trace_dt=.025,
    num_samples=1024, max_num_iterations=16, improvement_threshold=0., temperature=.1,
    first_ctrl_noise_scale=.5, last_ctrl_noise_scale=1., final_noise_scale=.1,
    joint_noise_scale=.1, pos_noise_scale=.005, rot_noise_scale=.05,
    base_pos_rew_scale=.3, base_rot_rew_scale=.1, joint_rew_scale=.01,
    pos_rew_scale=1., rot_rew_scale=.3, terminal_rew_scale=1., contact_rew_scale=0.,
    nconmax_per_env=500, njmax_per_env=1200, num_dr=1, pair_margin_range=[0., 0.], xy_offset_range=[0., 0.],
    contact_guidance=False, show_viewer=False, viewer='', save_video=False, rerun_spawn=False, save_rerun=False,
    wait_on_finish=False, sanity_check_seconds=0., use_torch_compile=False, seed=0)
BLEND_FRAMES = 4                  # actual entry -> reference offset fades over 4 source frames
HAND_TOOL_FRICTION = [1., 1., .005, .0001, .0001]


def pose_matrix(pose):
    m = np.eye(4); m[:3, 3] = pose[:3]; m[:3, :3] = Rotation.from_quat(np.asarray(pose)[[4, 5, 6, 3]]).as_matrix()
    return m


def matrix_pose(m):
    return np.r_[m[:3, 3], Rotation.from_matrix(m[:3, :3]).as_quat()[[3, 0, 1, 2]]]


def resample_poses(matrices, frames, times):
    """Linear position / slerp rotation of 4x4 poses at fractional source frames."""
    matrices = np.asarray(matrices); frames = np.asarray(frames, float)
    out = np.repeat(np.eye(4)[None], len(times), 0)
    out[:, :3, 3] = np.stack([np.interp(times, frames, matrices[:, j, 3]) for j in range(3)], 1)
    out[:, :3, :3] = Slerp(frames, Rotation.from_matrix(matrices[:, :3, :3]))(times).as_matrix()
    return out


def wrist_joints(matrix):
    """right_hand_link pose -> SPIDER virtual joints: p, R = Rz(q3) Rx(q4) Ry(-q5)."""
    a, b, c = Rotation.from_matrix(matrix[:3, :3]).as_euler('ZXY')
    return np.r_[matrix[:3, 3], a, b, -c]


def decompose(mesh_path, cache):
    """CoACD convex pieces of an object mesh, cached as OBJ files."""
    import trimesh
    cache.mkdir(parents=True, exist_ok=True)
    pieces = sorted(cache.glob('[0-9]*.obj'), key=lambda p: int(p.stem))
    if pieces:
        return pieces
    import coacd
    mesh = trimesh.load(mesh_path, force='mesh')
    parts = coacd.run_coacd(coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)), threshold=.05, max_convex_hull=16)
    for k, (v, f) in enumerate(parts):
        trimesh.Trimesh(v, f).export(cache/f'{k}.obj')
    return sorted(cache.glob('[0-9]*.obj'), key=lambda p: int(p.stem))


def build_scene(spider_dir, cfg, tool_pieces, target_pieces, tool_mass, entry, table, path):
    """SPIDER right XHand + table + static target + free tool (last joint)."""
    import mujoco, trimesh
    robot = Path(spider_dir)/'spider/assets/robots/xhand/right.xml'
    spec = mujoco.MjSpec.from_file(str(robot)); spec.meshdir = str(robot.parent/'assets')
    control = cfg['controller']; limited = mujoco.mjtLimited.mjLIMITED_TRUE
    bodies = spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY)
    for body in bodies:
        body.gravcomp = 1.
        if body.name.startswith('R_forearm'):
            body.mass = 1e-4; body.inertia = [1e-7]*3
    for joint in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_JOINT):
        if joint.name.startswith('R_forearm'):
            joint.armature = 0. if joint.type == mujoco.mjtJoint.mjJNT_SLIDE else 1e-3
        else:
            joint.armature = 5e-3       # >= finger kp * dt^2 for a stable explicit stiffness
    for actuator in spec.actuators:
        name = actuator.name
        if name.startswith('R_forearm') and any(axis in name for axis in ('_tx', '_ty', '_tz')):
            kp, kv, limit = control['wrist_kp'], control['wrist_kd'], control['wrist_max_force']
        elif name.startswith('R_forearm'):
            kp, kv, limit = control['wrist_kp_rot'], control['wrist_kd_rot'], control['wrist_max_torque']
        else:
            kp, kv, limit = control['hand_stiffness'], control['hand_damping'], control['hand_max_effort']
        actuator.gainprm = [kp]+[0.]*9; actuator.biasprm = [0., -kp, -kv]+[0.]*7
        actuator.forcerange = [-limit, limit]; actuator.forcelimited = limited
    world = spec.worldbody
    world.add_geom(name='table', type=mujoco.mjtGeom.mjGEOM_BOX, size=np.asarray(table['dims'])/2,
                   pos=table['pose'][:3], contype=0, conaffinity=0)
    target = (world.add_body(name='static_target', pos=entry['objects']['target']['pose'][:3],
                             quat=entry['objects']['target']['pose'][3:]) if target_pieces else None)
    for k, piece in enumerate(target_pieces):
        spec.add_mesh(name=f'target_{k}', file=str(Path(piece).resolve()))
        target.add_geom(name=f'static_target_{k}', type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f'target_{k}',
                        contype=0, conaffinity=0, group=3)
    volume = sum(trimesh.load(p, force='mesh').convex_hull.volume for p in tool_pieces)
    tool = world.add_body(name='right_object', pos=entry['objects']['tool']['pose'][:3],
                          quat=entry['objects']['tool']['pose'][3:])
    tool.add_freejoint(name='right_object_joint')
    for k, piece in enumerate(tool_pieces):
        spec.add_mesh(name=f'right_{k}', file=str(Path(piece).resolve()))
        tool.add_geom(name=f'right_object_{k}', type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f'right_{k}',
                      density=tool_mass/volume, contype=0, conaffinity=0, group=3)
    hand = [g.name for g in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM) if g.name.startswith('collision_hand_right_')]
    tool_geoms = [f'right_object_{k}' for k in range(len(tool_pieces))]
    environment = ['table']+[f'static_target_{k}' for k in range(len(target_pieces))]
    for h in hand:
        for g in tool_geoms:
            spec.add_pair(geomname1=h, geomname2=g, condim=4 if ('thumb' in h or 'index' in h) else 3,
                          friction=HAND_TOOL_FRICTION)
        for g in environment:
            spec.add_pair(geomname1=h, geomname2=g, condim=3, friction=HAND_TOOL_FRICTION)
    for g in tool_geoms:
        for e in environment:
            spec.add_pair(geomname1=g, geomname2=e, condim=3, friction=HAND_TOOL_FRICTION)
    model = spec.compile()
    Path(path).write_text(spec.to_xml())
    return model, hand, tool_geoms


def run_profiles(args, spec, out):
    """DexAgent physics_informed_retargeting: try the sampling profiles in order
    and keep the one whose object stays closest to the reference in SPIDER's
    own rollout (mean MuJoCo object position error; more thumb-opposition
    frames break ties). No profile fits every object: refine helped a hammer
    handle and hurt a bowl rim (2026-10-24). PhysX still decides acceptance."""
    import shutil
    rows = []
    # Profiles run one after another: concurrently on one GPU each 612-tick
    # knife profile took ~1060 s instead of ~446 s (2026-10-25), slower overall.
    launched = []
    for k, profile in enumerate(spec['profiles']):
        name = profile.get('name', f'profile_{k}')
        settings = {key: value for key, value in profile.items() if key != 'name'}
        folder = out/f'profile_{name}'; folder.mkdir(parents=True, exist_ok=True)
        (folder/'overrides.json').write_text(json.dumps(settings, indent=2))
        command = [sys.executable, str(Path(__file__).resolve()), '--schedule', str(args.schedule), '--proposals', str(args.proposals),
                   '--entry-run', str(args.entry_run), '--object-cache', str(args.object_cache), '--spider-dir', str(args.spider_dir),
                   '--anchor-id', args.anchor_id, '--floating-config', str(args.floating_config), '--device', args.device,
                   '--timeout-s', str(args.timeout_s), '--decomposition-cache', str(args.decomposition_cache),
                   '--overrides', str(folder/'overrides.json'), '--output-dir', str(folder)]
        with (folder/'profile.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        launched.append((name, folder))
    for name, folder in launched:
        try:
            result = json.loads((folder/'result.json').read_text())
        except (OSError, ValueError) as error:
            result = dict(ok=False, error=f'{type(error).__name__}: {error}')
        row = dict(name=name, ok=bool(result.get('ok')), error=result.get('error'),
                   object_position_error_m=result.get('mujoco_object_position_error_m'),
                   opposition_frames=result.get('mujoco_opposition_frames'), max_tool_lift_m=result.get('mujoco_max_tool_lift_m'))
        rows.append(row)
        print('SPIDER_PROFILE', json.dumps(row), flush=True)
    passed = [r for r in rows if r['ok']]
    selected = min(passed, key=lambda r: (r['object_position_error_m']['mean'], -(r['opposition_frames'] or 0))) if passed else None
    if selected is None:
        result = dict(ok=False, error='every SPIDER profile failed', method='SPIDER_mjwp_sampling_MPC', profiles=rows)
    else:
        folder = out/f'profile_{selected["name"]}'
        shutil.copy(folder/'commands.npz', out/'commands.npz')
        result = dict(json.loads((folder/'result.json').read_text()), selected_profile=selected['name'], profiles=rows,
                      profile_selection='lowest mean MuJoCo object position error (DexAgent physics_informed_retargeting)')
    (out/'result.json').write_text(json.dumps(result, indent=2, default=str))
    (out/'external_mpc_report.json').write_text(json.dumps({k: v for k, v in result.items() if k != 'ok'}, indent=2, default=str))
    print('SPIDER_MPC_RESULT', result['ok'], result.get('error'), result.get('selected_profile'), flush=True)


def main():
    p = argparse.ArgumentParser()
    for name in ('schedule', 'proposals', 'entry-run', 'object-cache', 'spider-dir', 'output-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--anchor-id', required=True)
    p.add_argument('--floating-config', type=Path, default=Path(__file__).resolve().parents[1]/'configs/floating_xhand.json')
    p.add_argument('--overrides', default='{}', help='JSON SPIDER settings on top of DEFAULTS, or a path to a JSON file.')
    p.add_argument('--device', default='cuda:0'); p.add_argument('--timeout-s', type=int, default=1800)
    p.add_argument('--prepare-only', action='store_true', help='Write the SPIDER scene/reference and stop (CPU check).')
    p.add_argument('--decomposition-cache', type=Path,
                   default=Path(__file__).resolve().parents[1]/'outputs/spider_decomposition_cache')
    args = p.parse_args(); out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    def overrides_text(value):
        try:
            return Path(value).read_text() if Path(value).is_file() else value
        except OSError:              # an inline JSON string longer than a file name
            return value
    requested = overrides_text(args.overrides)
    if not args.prepare_only and 'profiles' in json.loads(requested):
        return run_profiles(args, json.loads(requested), out)
    result = dict(ok=False, method='SPIDER_mjwp_sampling_MPC', anchor_id=args.anchor_id, physical_acceptance=False)
    started = time.monotonic()
    try:
        import mujoco
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
        from multipath.oracle_schedule import OracleSchedule
        schedule = OracleSchedule.load(args.schedule); anchor = schedule.anchors[args.anchor_id]
        manifest = json.loads((args.proposals/'manifest.json').read_text())
        predecessor = json.loads((args.entry_run/'pipeline_result.json').read_text())
        if predecessor['source_provenance'] != manifest['provenance']:
            raise ValueError('Entry run belongs to a different source')
        trace = json.loads((args.entry_run/'actual_trace.json').read_text())
        entry = next(s for s in trace if np.isclose(s['source_frame'], anchor.core_start))
        if not entry.get('physically_acquired_state') or predecessor.get('state_resets_between_segments') != 0:
            raise ValueError('SPIDER MPC requires a continuously reached physical entry')
        data = dict(np.load(args.proposals/manifest['anchors'][anchor.id], allow_pickle=False))
        # PBS -v cannot carry commas; a JSON file path is the portable form.
        settings = dict(DEFAULTS, device=args.device, **json.loads(requested))
        # Optional: the EgoEngine release's mixed runner (Replay -> MPC -> 1024-env
        # residual PPO per chunk) on the same prepared scene, instead of SPIDER MPC.
        runner = settings.pop('runner', 'spider'); mixed = settings.pop('mixed', {})
        if runner not in ('spider', 'egoengine_mixed'):
            raise ValueError(f'Unknown proposal runner {runner}')
        dt = float(settings['sim_dt']); fps = schedule.fps; tpf = round(1/(fps*dt))
        if not np.isclose(tpf*dt*fps, 1.) or not np.isclose(settings['ref_dt'], dt):
            raise ValueError('SPIDER must run on the source-aligned physics clock')
        ticks = (anchor.synth_end-anchor.core_start)*tpf
        times = anchor.core_start+np.arange(ticks+1)/tpf
        frames = np.asarray(data['source_frames'], float)
        frames_ext = np.r_[frames, frames[-1]+1]                           # hold the last frame to synth_end
        reference = np.asarray(data['wrist_hand_q']); reference = np.vstack((reference, reference[-1]))
        wrists = resample_poses([pose_matrix(r[:7]) for r in reference], frames_ext, times)
        fingers = np.stack([np.interp(times, frames_ext, reference[:, 7+j]) for j in range(12)], 1)
        tools = np.asarray(data['tool_pose']); tools = np.concatenate((tools, tools[-1:]))
        source_tool = resample_poses(tools, frames_ext, times)
        actual_tool = pose_matrix(entry['objects']['tool']['pose'])
        tool_ref = source_tool@np.linalg.inv(source_tool[0])@actual_tool  # source motion from the actual rest pose
        # Start exactly at the actual entry; the offset to the reference fades out.
        fade = np.clip((times-anchor.core_start)/BLEND_FRAMES, 0, 1); fade = 1-fade**2*(3-2*fade)
        actual_wrist = pose_matrix(entry['wrist_xyz_wxyz'])
        offset_p = actual_wrist[:3, 3]-wrists[0, :3, 3]
        offset_R = Rotation.from_matrix(actual_wrist[:3, :3]@wrists[0, :3, :3].T).as_rotvec()
        for k in range(len(times)):
            wrists[k, :3, 3] += fade[k]*offset_p
            wrists[k, :3, :3] = Rotation.from_rotvec(fade[k]*offset_R).as_matrix()@wrists[k, :3, :3]
        fingers += fade[:, None]*(np.asarray(entry['hand_q'])-fingers[0])
        # Scene: the same table as FloatingScene and the actual fixed target.
        initial = manifest['initial_objects']; meshes = {}
        import trimesh
        for role, obj in initial.items():
            meshes[role] = args.object_cache/obj['id']/(obj['id']+'_m.obj')
        z = min((pose_matrix(o['pose'])[:3, :3]@trimesh.load(meshes[r], force='mesh').vertices.T).T[:, 2].min()+o['pose'][2]
                for r, o in initial.items())-.001
        table = dict(dims=[.95, 1., .04], pose=[.57, 0, float(z-.02), 1, 0, 0, 0])
        pieces = {role: decompose(meshes[role], args.decomposition_cache/initial[role]['id']) for role in ('tool', 'target')
                  if role in initial}
        pieces.setdefault('target', [])
        cfg = json.loads(args.floating_config.read_text())
        tool_mass = float(predecessor['initial_state']['physical_properties']['tool_mass_kg'])
        task = f'mp_{manifest["provenance"]["sequence"]}_{anchor.id}'
        task_dir = out/'dataset'/'processed'/DATASET/ROBOT/EMBODIMENT/task; trial = task_dir/'0'; trial.mkdir(parents=True, exist_ok=True)
        model, hand_geoms, tool_geoms = build_scene(args.spider_dir, cfg, pieces['tool'], pieces['target'], tool_mass,
                                                    entry, table, task_dir/'scene.xml')
        if (model.nq, model.nv, model.nu) != (25, 24, 18):
            raise ValueError(f'Unexpected SPIDER model size {(model.nq, model.nv, model.nu)}')
        mj = mujoco.MjData(model); link = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_hand_link')
        def wrist_pose(q6):
            mj.qpos[:6] = q6; mujoco.mj_kinematics(model, mj)
            return np.r_[mj.xpos[link], mj.xquat[link]]
        qpos = np.zeros((len(times), model.nq))
        qpos[:, :6] = [wrist_joints(m) for m in wrists]; qpos[:, 3:6] = np.unwrap(qpos[:, 3:6], axis=0)
        qpos[:, 6:18] = fingers; qpos[:, 18:] = [matrix_pose(m) for m in tool_ref]
        check = np.abs(wrist_pose(qpos[0, :6])[:3]-actual_wrist[:3, 3]).max()
        turn = Rotation.from_quat(wrist_pose(qpos[0, :6])[[4, 5, 6, 3]]).inv()*Rotation.from_matrix(actual_wrist[:3, :3])
        if check > 1e-6 or turn.magnitude() > 1e-6:
            raise ValueError('SPIDER wrist joint convention does not reproduce right_hand_link')
        qvel = np.zeros((len(times), model.nv))
        for k in range(1, len(times)):
            mujoco.mj_differentiatePos(model, qvel[k], dt, qpos[k-1], qpos[k])
        # Actual entry velocity: wrist twist (world) -> virtual joint rates; finger and tool velocities.
        twist = np.asarray(entry['wrist_twist'], float); ahead = actual_wrist.copy()
        ahead[:3, 3] += dt*twist[:3]; ahead[:3, :3] = Rotation.from_rotvec(dt*twist[3:]).as_matrix()@ahead[:3, :3]
        q_ahead = wrist_joints(ahead); q_ahead[3:] = qpos[0, 3:6]+np.angle(np.exp(1j*(q_ahead[3:]-qpos[0, 3:6])))
        qvel[0, :6] = (q_ahead-qpos[0, :6])/dt; qvel[0, 6:18] = entry['hand_dq']
        velocity = np.asarray(entry['objects']['tool'].get('velocity', np.zeros(6)), float)
        qvel[0, 18:21] = velocity[:3]; qvel[0, 21:24] = actual_tool[:3, :3].T@velocity[3:]
        np.savez(trial/'trajectory_kinematic.npz', qpos=qpos, qvel=qvel, contact=np.zeros((len(qpos), 10)),
                 contact_pos=np.zeros((len(qpos), 10, 3)), frequency=1/dt)
        (task_dir/'task_info.json').write_text(json.dumps(dict(ref_dt=dt, source='multipath local Anchor', anchor=anchor.id)))
        if runner == 'egoengine_mixed':
            # The mixed runner never executes its last horizon + one control
            # step; hold the final reference so the Anchor window is covered.
            horizon = float(mixed.get('args', {}).get('horizon', 1.6)); ctrl_dt = float(mixed.get('args', {}).get('ctrl_dt', .05))
            pad = int(np.ceil((horizon+ctrl_dt)/dt))+round(ctrl_dt/dt)
            np.savez(trial/'trajectory_kinematic.npz', qpos=np.vstack((qpos, np.repeat(qpos[-1:], pad, 0))),
                     qvel=np.vstack((qvel, np.zeros((pad, qvel.shape[1])))), contact=np.zeros((len(qpos)+pad, 10)),
                     contact_pos=np.zeros((len(qpos)+pad, 10, 3)), frequency=1/dt)
            thresholds = manifest.get('object_tracking_thresholds') or {}
            cli = dict(dataset_dir=out/'dataset', dataset_name=DATASET, robot_type=ROBOT, embodiment_type=EMBODIMENT,
                       active_hand='right', task=task, data_id=0, device=args.device, sim_dt=dt, ctrl_dt=ctrl_dt,
                       object_mass_scale=1.0, object_pos_threshold=float(thresholds.get('position_m', .05)),
                       object_rot_threshold=float(thresholds.get('rotation_rad', 1.0)), methods='replay,mpc,rl',
                       output_name='trajectory_mjwp_mixed.npz', debug_output_dir=out/'mixed')
            cli.update({k: v for k, v in mixed.get('args', {}).items()})
            command = [mixed.get('python', os.environ.get('PY_EGO_ACTION', 'python')),
                       str(Path(__file__).resolve().parent/'egoengine_mixed_launcher.py'),
                       str(Path(mixed.get('repo', os.environ.get('EGOENGINE_DIR', '.')))/'action/examples/run_mjwp_mixed.py')]
            for key, v in cli.items():
                command += ['--'+key.replace('_', '-'), (('True' if v else 'False') if isinstance(v, bool) else str(v))]
            env = dict(os.environ, PYTHONPATH=str(mixed.get('repo', os.environ.get('EGOENGINE_DIR', '.'))),
                       MUJOCO_GL='egl', WARP_CACHE_PATH=mixed.get('warp_cache', os.environ.get('WARP_CACHE_PATH', '.warp_cache')))
            solve = time.monotonic()
            with (out/'mixed.log').open('w') as log:
                code = subprocess.run(command, cwd=mixed.get('repo', os.environ.get('EGOENGINE_DIR', '.')),
                                      env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout_s).returncode
            found = sorted((out/'mixed').rglob('trajectory_mjwp_mixed.npz')) + sorted(task_dir.rglob('trajectory_mjwp_mixed.npz'))
            if code != 0 or not found:
                tail = (out/'mixed.log').read_text().splitlines()[-25:]
                raise RuntimeError(f'EgoEngine mixed runner failed (exit {code}): '+' | '.join(tail))
            z = np.load(found[0]); C = np.asarray(z['ctrl']).reshape(-1, model.nu); Q = np.asarray(z['qpos']).reshape(-1, model.nq)
            statistics = json.loads(str(z['statistics_json'])) if 'statistics_json' in z.files else {}
            modes = dict(zip(*np.unique(np.asarray(z['window_method_id']), return_counts=True))) if 'window_method_id' in z.files else {}
            if len(C) < ticks:
                raise RuntimeError(f'EgoEngine mixed runner committed {len(C)} of {ticks} ticks (MuJoCo success={bool(z["success"])})')
            C, Q = C[:ticks], Q[:ticks]
            poses = np.vstack([entry['command_wrist']]+[wrist_pose(c[:6]) for c in C])
            hands = np.vstack([entry['command_hand_q'], C[:, 6:18]])
            rotation = Rotation.from_quat(poses[:, [4, 5, 6, 3]])
            commands = np.c_[poses[1:], hands[1:], np.diff(poses[:, :3], axis=0)/dt, (rotation[1:]*rotation[:-1].inv()).as_rotvec()/dt]
            np.savez_compressed(out/'commands.npz', commands=commands, spider_qpos=Q, spider_ctrl=C, reference_qpos=qpos)
            reference_tool = qpos[1:ticks+1, 18:]
            position_error = np.linalg.norm(Q[:, 18:21]-reference_tool[:, :3], axis=1)
            result.update(ok=True, ticks=int(ticks), window=[anchor.core_start, anchor.synth_end], spider_seconds=time.monotonic()-solve,
                method='EgoEngine_release_mixed_replay_mpc_parallel_rl', runner_args={k: str(v) for k, v in cli.items()},
                mujoco_success=bool(z['success']), window_modes={int(k): int(v) for k, v in modes.items()},
                runner_statistics=statistics, mujoco_object_position_error_m=dict(mean=float(position_error.mean()),
                    max=float(position_error.max()), end=float(position_error[-1])),
                tool_mass_kg=tool_mass, simulator='MuJoCo Warp (EgoEngine release runner); PhysX acceptance pending',
                source_time_stretch=False)
            return
        if args.prepare_only:
            result.update(prepared=str(task_dir), ticks=int(ticks), tool_mass_kg=tool_mass,
                          tool_convex_pieces=len(pieces['tool']), target_convex_pieces=len(pieces['target']),
                          reference_tool_lift_m=float(qpos[:, 20].max()-qpos[0, 20]))
            return
        # SPIDER run_mjwp through hydra; keys absent from its default.yaml need '+'.
        from omegaconf import OmegaConf
        known = set(OmegaConf.load(args.spider_dir/'examples/config/default.yaml').keys())
        def value(v):
            if isinstance(v, bool): return 'true' if v else 'false'
            if isinstance(v, str): return f"'{v}'"
            if isinstance(v, (list, tuple)): return '['+','.join(repr(float(x)) for x in v)+']'
            return repr(v)
        overrides = [f'dataset_dir={out/"dataset"}', f'dataset_name={DATASET}', f'robot_type={ROBOT}',
                     f'embodiment_type={EMBODIMENT}', f'task={task}', 'data_id=0']
        overrides += [f'{"" if k in known else "+"}{k}={value(v)}' for k, v in settings.items()]
        overrides += [f'hydra.run.dir={out/"hydra"}', 'hydra.output_subdir=null']
        env = dict(os.environ, PYTHONPATH=str(args.spider_dir), MUJOCO_GL='egl',
                   WARP_CACHE_PATH=os.environ.get('WARP_CACHE_PATH', str(out.parent/'warp_cache')))
        solve = time.monotonic()
        with (out/'spider.log').open('w') as log:
            code = subprocess.run([sys.executable, str(args.spider_dir/'examples/run_mjwp.py'), *overrides],
                                  cwd=args.spider_dir, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout_s).returncode
        output = trial/'trajectory_mjwp.npz'
        if code != 0 or not output.exists():
            tail = (out/'spider.log').read_text().splitlines()[-25:]
            raise RuntimeError(f'SPIDER run_mjwp failed (exit {code}): '+' | '.join(tail))
        z = np.load(output); Q = np.asarray(z['qpos']).reshape(-1, model.nq); C = np.asarray(z['ctrl']).reshape(-1, model.nu)
        if len(C) < ticks:
            raise RuntimeError(f'SPIDER returned {len(C)} controls for {ticks} ticks')
        Q, C = Q[:ticks], C[:ticks]
        # Commands: SPIDER's actuator targets, as PhysX drive targets of the same controller law.
        poses = np.vstack([entry['command_wrist']]+[wrist_pose(c[:6]) for c in C])
        hands = np.vstack([entry['command_hand_q'], C[:, 6:18]])
        rotation = Rotation.from_quat(poses[:, [4, 5, 6, 3]])
        commands = np.c_[poses[1:], hands[1:], np.diff(poses[:, :3], axis=0)/dt, (rotation[1:]*rotation[:-1].inv()).as_rotvec()/dt]
        np.savez_compressed(out/'commands.npz', commands=commands, spider_qpos=Q, spider_ctrl=C, reference_qpos=qpos)
        # MuJoCo diagnostics of SPIDER's own executed rollout (not PhysX evidence).
        names = {g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(model.ngeom)}
        thumb, other = np.zeros(ticks, bool), np.zeros(ticks, bool)
        for k, q in enumerate(Q):
            mj.qpos[:] = q; mujoco.mj_forward(model, mj)
            for c in mj.contact[:mj.ncon]:
                pair = {names[c.geom1], names[c.geom2]}
                if c.dist < 1e-3 and pair & set(tool_geoms):
                    h = next((n for n in pair if n in hand_geoms), None)
                    if h is not None:
                        if 'thumb' in h: thumb[k] = True
                        elif 'palm' not in h: other[k] = True
        reference_tool = qpos[1:ticks+1, 18:]
        position_error = np.linalg.norm(Q[:, 18:21]-reference_tool[:, :3], axis=1)
        rotation_error = (Rotation.from_quat(Q[:, [22, 23, 24, 21]]).inv()*Rotation.from_quat(reference_tool[:, [4, 5, 6, 3]])).magnitude()
        opposition = thumb & other
        result.update(ok=True, ticks=int(ticks), window=[anchor.core_start, anchor.synth_end], spider_seconds=time.monotonic()-solve,
            mujoco_object_position_error_m=dict(mean=float(position_error.mean()), max=float(position_error.max()), end=float(position_error[-1])),
            mujoco_object_rotation_error_rad=dict(mean=float(rotation_error.mean()), max=float(rotation_error.max()), end=float(rotation_error[-1])),
            mujoco_opposition_frames=float(opposition.sum()/tpf), mujoco_first_opposition_frame=(
                float(anchor.core_start+(np.argmax(opposition)+1)/tpf) if opposition.any() else None),
            mujoco_max_tool_lift_m=float((Q[:, 20]-qpos[0, 20]).max()),
            tool_mass_kg=tool_mass, tool_convex_pieces=len(pieces['tool']), target_convex_pieces=len(pieces['target']),
            settings=settings, actuation='PhysX floating-XHand controller gains/limits mirrored; gravcomp; explicit pairs',
            entry=dict(source_frame=anchor.core_start, origin=str(args.entry_run/'actual_trace.json')),
            reference='Eq. (1) local retarget, actual-entry offset faded over %d frames; tool = source motion from actual rest' % BLEND_FRAMES,
            source_time_stretch=False, simulator='MuJoCo Warp (SPIDER); PhysX acceptance pending')
    except Exception as error:
        import traceback; traceback.print_exc()
        result.update(ok=False, error=f'{type(error).__name__}: {error}')
    finally:
        result['elapsed_s'] = time.monotonic()-started
        report = {k: v for k, v in result.items() if k != 'ok'}
        (out/'external_mpc_report.json').write_text(json.dumps(report, indent=2, default=str))
        (out/'result.json').write_text(json.dumps(result, indent=2, default=str))
        print('SPIDER_MPC_RESULT', result['ok'], result.get('error'), flush=True)


if __name__ == '__main__':
    main()
