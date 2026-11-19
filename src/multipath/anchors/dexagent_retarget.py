"""DexAgent-style contact-guided retargeting (DexAgent spider_bridge + SPIDER IK).

Mirrors DexAgent's human-hand path (tool_library/common/spider_bridge.py:
human_keypoints, contact_window, prep_keypoints) followed by SPIDER's keypoint
IK weights (spider/preprocess/ik_fast.py):

1. The robot palm follows the human anatomical frame; fingertips are scaled
   about the knuckle centre by the robot/human mean finger-length ratio.
2. Inside the object's interaction window (0.6 s before it first moves until
   0.4 s after it last moves), every fingertip within 2 cm of the object
   surface is placed ON it: nearest surface point plus one pad radius along
   the outward normal. Where a non-thumb finger touches, the thumb is placed
   at the antipode of the finger-contact centroid through the object centre.
3. Mink IK: fingertip positions (10), wrist orientation (3), weak wrist
   position (0.3), posture (1e-2); collisions off, as in SPIDER IK; then a
   3-frame moving average.

Targets live in the object frame during contact, so the reference hand moves
with the reference object. This is kinematic only; physics decides.
"""
from __future__ import annotations

import numpy as np

CONTACT_M = .02
PAD_M = .008                    # XHand distal pad: tip site to pad surface
MOVE_TOL_M, LEAD_S, LAG_S = .005, .6, .4
TIP_COST, WRIST_ORI_COST, WRIST_POS_COST, POSTURE_COST = 10., 3., .3, 1e-2
MANO_MCP = [1, 5, 9, 13, 17]
MANO_CHAINS = {'thumb': [1, 2, 3, 4], 'index': [5, 6, 7, 8], 'middle': [9, 10, 11, 12], 'ring': [13, 14, 15, 16], 'pinky': [17, 18, 19, 20]}
ROBOT_CHAINS = {'thumb': ['thumb_bend_link', 'thumb_rota_link1', 'thumb_rota_link2'], 'index': ['index_bend_link', 'index_rota_link1', 'index_rota_link2'],
                'middle': ['mid_link1', 'mid_link2'], 'ring': ['ring_link1', 'ring_link2'], 'pinky': ['pinky_link1', 'pinky_link2']}


def contact_window(positions, fps):
    """DexAgent contact_window: [first, last] interaction frames, None if the object never moves."""
    p = np.asarray(positions, float)
    off = np.linalg.norm(p-p[0], axis=1) > MOVE_TOL_M
    if not off.any():
        return None
    step = np.r_[0., np.linalg.norm(np.diff(p, axis=0), axis=1)]
    moving = np.where(step > MOVE_TOL_M/10)[0]
    a = int(np.argmax(off)); b = int(moving.max()) if len(moving) else len(p)-1
    return [max(0, a-int(LEAD_S*fps)), min(len(p)-1, b+int(LAG_S*fps))]


class ObjectSurface:
    """Dense surface samples (object frame) with outward normals and a centre."""

    def __init__(self, mesh, n=6000, seed=0):
        import trimesh
        from scipy.spatial import cKDTree
        points, faces = trimesh.sample.sample_surface_even(mesh, n, seed=seed)
        self.points, self.normals = np.asarray(points), np.asarray(mesh.face_normals[faces])
        self.centre = np.asarray(mesh.center_mass if mesh.is_watertight else mesh.centroid)
        self.tree = cKDTree(self.points)

    def nearest(self, local):
        distance, index = self.tree.query(local)
        return distance, index

    def antipode(self, contact_centroid):
        u = self.centre-contact_centroid
        if np.linalg.norm(u) < 1e-6:
            return None
        u /= np.linalg.norm(u); rel = self.points-self.centre; along = rel@u
        perp = np.linalg.norm(rel-along[:, None]*u, axis=1); candidates = np.where(along > 0)[0]
        return int(candidates[np.argmin(perp[candidates])]) if len(candidates) else None


def prep_contacts(tips, tool_poses, surface, fps, window_mode='motion', opposition_contact=True):
    """Project fingertip targets onto the object surface inside the interaction window.

    window_mode='motion' is DexAgent's (object-motion window, thumb antipode).
    'proximity' applies the same per-fingertip CONTACT_M rule on every frame,
    for a supporting hand on an object that barely moves; opposition_contact
    False keeps a steadying contact from being turned into a pinch.
    """
    tips = np.array(tips, float, copy=True); T = len(tips)
    window = contact_window(tool_poses[:, :3, 3], fps) if window_mode == 'motion' else [0, T-1]
    contact = np.zeros((T, 5), bool); opposition = 0
    if window is None:
        return tips, contact, dict(window=None, contact_frames=0, opposition_frames=0)
    for t in range(window[0], window[1]+1):
        R, p = tool_poses[t, :3, :3], tool_poses[t, :3, 3]
        local = (tips[t]-p)@R
        distance, index = surface.nearest(local)
        touching = {i for i in range(5) if distance[i] < CONTACT_M}
        if not any(i > 0 for i in touching) and opposition_contact:
            continue
        for i in touching:
            if i == 0 and opposition_contact:
                continue
            contact[t, i] = True
            tips[t, i] = p+R@(surface.points[index[i]]+PAD_M*surface.normals[index[i]])
        if not opposition_contact or not any(i > 0 for i in touching):
            continue
        k = surface.antipode(np.mean([surface.points[index[i]] for i in touching if i > 0], axis=0))
        if k is not None:
            tips[t, 0] = p+R@(surface.points[k]+PAD_M*surface.normals[k]); contact[t, 0] = True; opposition += 1
    return tips, contact, dict(window=window, contact_frames=int(contact[:, 1:].any(1).sum()), opposition_frames=opposition,
                               per_finger=dict(zip(['thumb', 'index', 'middle', 'ring', 'pinky'], contact.sum(0).tolist())))


class DexAgentRetargeter:
    def __init__(self, side, calib_joints21):
        import mink, mujoco
        from egoengine.retarget.ik import build_hand_model, palm_link, finger_joint_names, palm_frame, FINGER_NAMES, LM_DAMPING, GAIN
        self.mink, self.mujoco, self.side, self.finger_names = mink, mujoco, side, FINGER_NAMES
        self.model = build_hand_model(side); self.config = mink.Configuration(self.model)
        self.palm = palm_link(side); self.joint_names = finger_joint_names(side)
        self._finger_qpos_adr = np.asarray([self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                                            for n in self.joint_names], dtype=np.int64)
        data = mujoco.MjData(self.model); data.qpos[:] = self.model.qpos0; mujoco.mj_forward(self.model, data)
        body = lambda name: data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f'{side}_hand_{name}')].copy()
        site = lambda name: data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)].copy()
        palm_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.palm)
        palm_R, wrist = data.xmat[palm_id].reshape(3, 3).copy(), data.xpos[palm_id].copy()
        mcp = {k: body(n) for k, n in (('index', 'index_rota_link1'), ('middle', 'mid_link1'), ('pinky', 'pinky_link1'))}
        y = (mcp['middle']-wrist)/np.linalg.norm(mcp['middle']-wrist); z = np.cross(y, mcp['index']-mcp['pinky']); z /= np.linalg.norm(z)
        robot_palm = np.stack([np.cross(y, z), y, z], axis=1)
        self.wrist_ori_const = robot_palm.T@palm_R
        knuckles = np.mean([body(n) for n in ('index_rota_link1', 'mid_link1', 'ring_link1', 'pinky_link1')], axis=0)
        self.knuckle_in_palm = palm_R.T@(knuckles-wrist)
        lengths = []
        for f in FINGER_NAMES:
            points = [body(n) for n in ROBOT_CHAINS[f]]+[site(f'{f}_tip')]
            lengths.append(sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        self.robot_length = float(np.mean(lengths)); self.palm_frame = palm_frame
        self.tip_tasks = {f: mink.FrameTask(frame_name=f'{f}_tip', frame_type='site', position_cost=TIP_COST, orientation_cost=0.,
                                            lm_damping=LM_DAMPING, gain=GAIN) for f in FINGER_NAMES}
        self.wrist_task = mink.FrameTask(frame_name=self.palm, frame_type='body', position_cost=WRIST_POS_COST,
                                         orientation_cost=WRIST_ORI_COST, lm_damping=LM_DAMPING, gain=GAIN)
        self.posture = mink.PostureTask(self.model, cost=POSTURE_COST)
        self.limits = [mink.ConfigurationLimit(self.model)]

    def human_targets(self, joints21):
        """Anatomical palm match and knuckle-centred tip scaling (DexAgent human_keypoints)."""
        R = self.palm_frame(joints21)@self.wrist_ori_const
        centre = joints21[[5, 9, 13, 17]].mean(0)
        wrist = centre-R@self.knuckle_in_palm
        human_length = np.mean([np.linalg.norm(np.diff(joints21[c], axis=0), axis=1).sum() for c in MANO_CHAINS.values()])
        scale = self.robot_length/max(human_length, 1e-6)
        tips = np.stack([centre+scale*(joints21[c[-1]]-centre) for c in MANO_CHAINS.values()])
        return R, wrist, tips

    def retarget(self, joints_traj, tool_poses, tool_mesh, fps, dt=.05, iters_first=500, iters=100,
                 window_mode='motion', opposition_contact=True):
        mink, mujoco = self.mink, self.mujoco
        n = len(joints_traj); targets = [self.human_targets(j) for j in joints_traj]
        tips = np.stack([t[2] for t in targets])
        surface = ObjectSurface(tool_mesh)
        tips, contact, prep = prep_contacts(tips, np.asarray(tool_poses), surface, fps, window_mode, opposition_contact)
        prep.update(window_mode=window_mode, opposition_contact=opposition_contact)
        qpos = np.zeros((n, 19)); tip_error = np.zeros((n, 5))
        self.posture.set_target(self.model.qpos0)
        for frame in range(n):
            R, wrist, _ = targets[frame]
            for i, f in enumerate(self.finger_names):
                target = np.eye(4); target[:3, 3] = tips[frame, i]
                self.tip_tasks[f].set_target(mink.SE3.from_matrix(target))
            w = np.eye(4); w[:3, :3] = R; w[:3, 3] = wrist; self.wrist_task.set_target(mink.SE3.from_matrix(w))
            for _ in range(iters_first if frame == 0 else iters):
                velocity = mink.solve_ik(self.config, [*self.tip_tasks.values(), self.wrist_task, self.posture], dt=dt,
                                         solver='daqp', limits=self.limits, damping=1e-4)
                self.config.integrate_inplace(velocity, dt)
            palm = self.config.get_transform_frame_to_world(self.palm, 'body')
            quat = np.empty(4); mujoco.mju_mat2Quat(quat, palm.rotation().as_matrix().reshape(-1))
            qpos[frame, :3] = palm.translation(); qpos[frame, 3:7] = quat; qpos[frame, 7:] = self.config.q[self._finger_qpos_adr]
            for i, f in enumerate(self.finger_names):
                tip_error[frame, i] = np.linalg.norm(self.config.get_transform_frame_to_world(f'{f}_tip', 'site').translation()-tips[frame, i])
        smooth = qpos.copy()                       # SPIDER IK: 3-frame moving average (positions and joints)
        for k in range(1, n-1):
            smooth[k, :3] = qpos[k-1:k+2, :3].mean(0); smooth[k, 7:] = qpos[k-1:k+2, 7:].mean(0)
        return smooth, dict(method='dexagent_contact_guided_spider_ik', contact_m=CONTACT_M, pad_m=PAD_M,
            weights=dict(tip=TIP_COST, wrist_orientation=WRIST_ORI_COST, wrist_position=WRIST_POS_COST, posture=POSTURE_COST),
            contact_prep=prep, tip_position_error_m=dict(mean=float(tip_error.mean()), max=float(tip_error.max())),
            contact_tip_error_m=float(tip_error[contact].mean()) if contact.any() else None,
            object_or_contact_terms=True, collisions_in_ik=False,
            source='DexAgent spider_bridge human_keypoints/prep_keypoints + SPIDER ik_fast weights')
