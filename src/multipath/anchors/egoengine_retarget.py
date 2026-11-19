"""EgoEngine Eq. (1) Mink retargeting (paper Sec. 3.2.1, pp. 3-4).

q*_t = argmin L_tip + lambda_w L_wrist over joint limits and self-collision.
L_tip aligns robot fingertip positions AND orientations with the human
fingertips; L_wrist aligns only the wrist orientation (no wrist position
term). There are no object, contact or surface terms: contact is left to
Replay/MPC/RL. The paper publishes no weights; the costs below are recorded
engineering choices, not paper values.
"""
from __future__ import annotations

import numpy as np

TIP_POSITION_COST = 1.0
# Per rad. Measured on Pour 246 / Hit 003 anchor_0: 0.03 cuts the distal
# direction error ~40% vs 0 for +0.3-0.4 cm tip error; 0.1 costs ~2 cm.
TIP_ORIENTATION_COST = .03
WRIST_ORIENTATION_COST = .5    # lambda_w, kept from the previous reproduction
WRIST_POSITION_COST = 0.       # Eq. (1) has no wrist position term
SELF_COLLISION_MIN_DISTANCE_M = 0.
SELF_COLLISION_DETECTION_M = .01
MANO_DISTAL = {'thumb': (3, 4), 'index': (7, 8), 'middle': (11, 12), 'ring': (15, 16), 'pinky': (19, 20)}
FINGER_BODY_KEYS = {'thumb': 'thumb', 'index': 'index', 'middle': 'mid', 'ring': 'ring', 'pinky': 'pinky'}


def _unit(v):
    return v/(np.linalg.norm(v)+1e-12)


def human_tip_directions(joints21):
    """Distal-phalanx pointing directions (DIP -> tip) from MANO joints.

    Twenty-one keypoints observe the distal direction but not its roll; the
    XHand fingers cannot roll either. Orientation is aligned on this axis.
    """
    return {finger: _unit(joints21[tip]-joints21[dip]) for finger, (dip, tip) in MANO_DISTAL.items()}


def _min_rotation(a, b):
    """Smallest rotation matrix taking unit vector a onto unit vector b."""
    v, c = np.cross(a, b), float(np.clip(a@b, -1, 1))
    if c < -1+1e-9:
        axis = _unit(np.cross(a, [1., 0, 0] if abs(a[0]) < .9 else [0, 1., 0]))
        return 2*np.outer(axis, axis)-np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3)+k+k@k/(1+c)


def finger_self_collision_pairs(model, data):
    """Geom pairs between different fingers that are separated at the calibration pose."""
    import mujoco
    geoms = {finger: [] for finger in FINGER_BODY_KEYS}
    for g in range(model.ngeom):
        if not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ''
        owner = next((f for f, key in FINGER_BODY_KEYS.items() if key in body), None)
        if owner is not None:
            geoms[owner].append(g)
    names = list(geoms)
    pairs, excluded, fromto = [], [], np.zeros(6)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            keep_a, keep_b = [], []
            for ga in geoms[a]:
                for gb in geoms[b]:
                    distance = mujoco.mj_geomDistance(model, data, ga, gb, SELF_COLLISION_DETECTION_M, fromto)
                    if distance < 0:
                        excluded.append([a, b, int(ga), int(gb), float(distance)])
                        continue
                    keep_a.append(ga); keep_b.append(gb)
            if keep_a:
                pairs.append((sorted(set(keep_a)), sorted(set(keep_b))))
    return pairs, excluded


class EgoEngineRetargeter:
    """Per-frame Mink IK of Eq. (1) onto the SPIDER floating-wrist XHand."""

    def __init__(self, side, calib_joints21, *, tip_orientation_cost=TIP_ORIENTATION_COST,
                 wrist_orientation_cost=WRIST_ORIENTATION_COST, self_collision=True, wrist_position_cost=WRIST_POSITION_COST):
        import mink, mujoco
        from egoengine.retarget.ik import (build_hand_model, palm_link, finger_joint_names, palm_frame,
                                           FINGER_NAMES, MANO_TIP_IDX, MANO_WRIST, LM_DAMPING, GAIN)
        self.mink, self.mujoco, self.side = mink, mujoco, side
        self.finger_names, self.tip_index, self.wrist_index = FINGER_NAMES, MANO_TIP_IDX, MANO_WRIST
        self.model = build_hand_model(side); self.config = mink.Configuration(self.model)
        self.palm = palm_link(side); self.joint_names = finger_joint_names(side)
        self._finger_qpos_adr = np.asarray([self.model.jnt_qposadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in self.joint_names], dtype=np.int64)
        data = mujoco.MjData(self.model); data.qpos[:] = self.model.qpos0; mujoco.mj_forward(self.model, data)
        body = lambda name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        site = lambda name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        palm_R = data.xmat[body(self.palm)].reshape(3, 3).copy(); wrist = data.xpos[body(self.palm)].copy()
        mcp = {k: data.xpos[body(f'{side}_hand_{name}')].copy()
               for k, name in (('index', 'index_rota_link1'), ('middle', 'mid_link1'), ('pinky', 'pinky_link1'))}
        y = _unit(mcp['middle']-wrist); z = _unit(np.cross(y, mcp['index']-mcp['pinky']))
        robot_palm = np.stack([np.cross(y, z), y, z], axis=1)
        human_palm = palm_frame(calib_joints21); human_dirs = human_tip_directions(calib_joints21)
        self.wrist_ori_const = robot_palm.T@palm_R
        # Calibration: the flat human distal direction maps to the robot zero
        # pose distal direction, each expressed in its own anatomical palm frame.
        self.palm_R0 = palm_R; self.tip_R0, self.tip_axis_local, self.tip_dir_const = {}, {}, {}
        for f in FINGER_NAMES:
            s = site(f'{f}_tip'); distal = _unit(data.site_xpos[s]-data.xpos[self.model.site_bodyid[s]])
            self.tip_R0[f] = data.site_xmat[s].reshape(3, 3).copy()
            self.tip_axis_local[f] = self.tip_R0[f].T@distal
            self.tip_dir_const[f] = _min_rotation(human_palm.T@human_dirs[f], robot_palm.T@distal)
        self.scale = float(np.linalg.norm(data.site_xpos[site('middle_tip')]-wrist)/
                           np.linalg.norm(calib_joints21[MANO_TIP_IDX[2]]-calib_joints21[MANO_WRIST]))
        self.tip_tasks = {f: mink.FrameTask(frame_name=f'{f}_tip', frame_type='site', position_cost=TIP_POSITION_COST,
                                            orientation_cost=tip_orientation_cost, lm_damping=LM_DAMPING, gain=GAIN)
                          for f in FINGER_NAMES}
        self.wrist_task = mink.FrameTask(frame_name=self.palm, frame_type='body', position_cost=wrist_position_cost,
                                         orientation_cost=wrist_orientation_cost, lm_damping=LM_DAMPING, gain=GAIN)
        self.tasks = list(self.tip_tasks.values())+[self.wrist_task]
        self.limits = [mink.ConfigurationLimit(self.model)]
        self.collision_pairs, self.excluded_pairs = finger_self_collision_pairs(self.model, data) if self_collision else ([], [])
        if self.collision_pairs:
            self.limits.append(mink.CollisionAvoidanceLimit(self.model, self.collision_pairs,
                minimum_distance_from_collisions=SELF_COLLISION_MIN_DISTANCE_M,
                collision_detection_distance=SELF_COLLISION_DETECTION_M))
        self.settings = dict(method='egoengine_eq1_mink_fingertip_pose_wrist_orientation',
            tip_orientation='distal phalanx direction (roll unobservable from keypoints)',
            tip_position_cost=TIP_POSITION_COST, tip_orientation_cost=tip_orientation_cost,
            wrist_orientation_cost=wrist_orientation_cost, wrist_position_cost=wrist_position_cost,
            self_collision=bool(self.collision_pairs), self_collision_pairs=len(self.collision_pairs),
            excluded_initially_touching_geom_pairs=self.excluded_pairs, scale=self.scale,
            object_or_contact_terms=False, paper='EgoEngine Sec. 3.2.1 Eq. (1); weights are engineering choices')

    def _targets(self, joints21):
        """Tip position and distal direction targets.

        The unobservable roll is predicted from the target wrist orientation
        applied to the zero-pose tip frame, so it never depends on the solver's
        current (possibly unconverged) configuration.
        """
        from egoengine.retarget.ik import palm_frame
        wrist = joints21[self.wrist_index]; human = human_tip_directions(joints21); palm = palm_frame(joints21)
        wrist_target = np.eye(4); wrist_target[:3, :3] = palm@self.wrist_ori_const; wrist_target[:3, 3] = wrist
        turn = wrist_target[:3, :3]@self.palm_R0.T
        targets, directions = {}, {}
        for f, index in zip(self.finger_names, self.tip_index):
            predicted = turn@self.tip_R0[f]
            directions[f] = palm@self.tip_dir_const[f]@palm.T@human[f]
            target = np.eye(4); target[:3, :3] = _min_rotation(predicted@self.tip_axis_local[f], directions[f])@predicted
            target[:3, 3] = wrist+self.scale*(joints21[index]-wrist); targets[f] = target
        return targets, wrist_target, directions

    def retarget(self, joints_traj, dt=.05, iters_first=500, iters=100):
        mink, mujoco = self.mink, self.mujoco
        n = len(joints_traj); qpos = np.zeros((n, 19))
        tip_error_m = np.zeros((n, 5)); tip_rotation_rad = np.zeros((n, 5)); wrist_rotation_rad = np.zeros(n)
        for frame in range(n):
            targets, wrist_target, directions = self._targets(joints_traj[frame])
            for f in self.finger_names:
                self.tip_tasks[f].set_target(mink.SE3.from_matrix(targets[f]))
            self.wrist_task.set_target(mink.SE3.from_matrix(wrist_target))
            for _ in range(iters_first if frame == 0 else iters):
                velocity = mink.solve_ik(self.config, self.tasks, dt=dt, solver='daqp', limits=self.limits, damping=1e-4)
                self.config.integrate_inplace(velocity, dt)
            palm = self.config.get_transform_frame_to_world(self.palm, 'body')
            R = palm.rotation().as_matrix(); quat = np.empty(4); mujoco.mju_mat2Quat(quat, R.reshape(-1))
            qpos[frame, :3] = palm.translation(); qpos[frame, 3:7] = quat
            qpos[frame, 7:] = self.config.q[self._finger_qpos_adr]
            for i, f in enumerate(self.finger_names):
                tip = self.config.get_transform_frame_to_world(f'{f}_tip', 'site')
                tip_error_m[frame, i] = np.linalg.norm(tip.translation()-targets[f][:3, 3])
                axis = tip.rotation().as_matrix()@self.tip_axis_local[f]
                tip_rotation_rad[frame, i] = np.arccos(np.clip(axis@directions[f], -1, 1))
            delta = wrist_target[:3, :3].T@R
            wrist_rotation_rad[frame] = np.arccos(np.clip((np.trace(delta)-1)/2, -1, 1))
        return qpos, dict(self.settings, tip_position_error_m=dict(mean=float(tip_error_m.mean()), max=float(tip_error_m.max())),
            tip_direction_error_rad=dict(mean=float(tip_rotation_rad.mean()), max=float(tip_rotation_rad.max())),
            per_finger_mean_tip_direction_error_rad=dict(zip(self.finger_names, tip_rotation_rad.mean(0).round(4).tolist())),
            wrist_rotation_error_rad=dict(mean=float(wrist_rotation_rad.mean()), max=float(wrist_rotation_rad.max())),
            per_finger_mean_tip_error_m=dict(zip(self.finger_names, tip_error_m.mean(0).round(5).tolist())))
