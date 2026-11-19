"""Per-frame Mink IK retargeting: TACO MANO hands -> SPIDER XHand.

The public SPIDER 6-DoF URDF is used for IK. Its six Cartesian/Euler wrist
joints are an optimization device only: exported trajectories use one floating
wrist pose ``[x,y,z,qw,qx,qy,qz]`` followed by the twelve named XHand finger
joints expected by the Isaac scene.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from os.path import abspath, dirname, join

import mink
import mujoco
import numpy as np

ASSETS = abspath(join(dirname(__file__), "..", "..", "..", "assets"))
URDF_DIR = join(ASSETS, "xhand", "spider")
IK_URDF = {
    side: join(URDF_DIR, f"xhand_{side}_6dof.urdf")
    for side in ("left", "right")
}
SIM_URDF = {
    side: join(URDF_DIR, f"xhand_{side}.urdf")
    for side in ("left", "right")
}

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
MANO_TIP_IDX = [4, 8, 12, 16, 20]
MANO_WRIST, MANO_INDEX_MCP, MANO_MIDDLE_MCP, MANO_PINKY_MCP = 0, 5, 9, 17


def palm_link(side: str) -> str:
    return f"{side}_hand_link"


def tip_link_names(side: str) -> list[str]:
    prefix = f"{side}_hand_"
    return [
        prefix + "thumb_rota_tip",
        prefix + "index_rota_tip",
        prefix + "mid_tip",
        prefix + "ring_tip",
        prefix + "pinky_tip",
    ]


def finger_joint_names(side: str) -> list[str]:
    prefix = f"{side}_hand_"
    return [
        prefix + "thumb_bend_joint",
        prefix + "thumb_rota_joint1",
        prefix + "thumb_rota_joint2",
        prefix + "index_bend_joint",
        prefix + "index_joint1",
        prefix + "index_joint2",
        prefix + "mid_joint1",
        prefix + "mid_joint2",
        prefix + "ring_joint1",
        prefix + "ring_joint2",
        prefix + "pinky_joint1",
        prefix + "pinky_joint2",
    ]


# Compatibility export for callers that only need the semantic finger order.
TIP_LINKS = FINGER_NAMES

POS_COST_TIP = 1.0
# SPIDER's XHand retarget config specifies origin-to-fingertip targets, not a
# synthetic fingertip orientation. Position-only tips are essential for the
# underactuated two-flexion-joint fingers to close around the demonstrated tool.
ORI_COST_TIP = 0.0
LAMBDA_WRIST = 0.5
POS_COST_WRIST = 0.3
LM_DAMPING = 1.0
GAIN = 0.85


def build_hand_model(side: str) -> mujoco.MjModel:
    """Compile the official SPIDER floating-wrist XHand URDF for IK."""
    if side not in IK_URDF:
        raise ValueError(f"unknown hand side: {side}")
    spec = mujoco.MjSpec.from_file(IK_URDF[side])
    # SPIDER stores the files in ``assets`` while its URDF references
    # ``meshes/...``. Our source note documents the direct folder rename.
    spec.meshdir = join(URDF_DIR, "meshes")
    for finger, link in zip(FINGER_NAMES, tip_link_names(side)):
        tip = spec.body(link)
        tip.parent.add_site(name=f"{finger}_tip", pos=tip.pos, quat=tip.quat)
    return spec.compile()


def build_floating_hand_model(side: str) -> mujoco.MjModel:
    """Compile the official 12-DoF hand with a MuJoCo free wrist for QA."""
    if side not in SIM_URDF:
        raise ValueError(f"unknown hand side: {side}")
    spec = mujoco.MjSpec.from_file(SIM_URDF[side])
    spec.meshdir = join(URDF_DIR, "meshes")
    palm = spec.body(palm_link(side))
    palm.add_freejoint(name=f"{side}_wrist_free")
    for finger, link in zip(FINGER_NAMES, tip_link_names(side)):
        tip = spec.body(link)
        tip.parent.add_site(name=f"{finger}_tip", pos=tip.pos, quat=tip.quat)
    return spec.compile()


def _norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def palm_frame(joints21: np.ndarray) -> np.ndarray:
    """3x3 rotation of a MANO palm frame built from joint positions only."""
    y = _norm(joints21[MANO_MIDDLE_MCP] - joints21[MANO_WRIST])
    knuckle = joints21[MANO_INDEX_MCP] - joints21[MANO_PINKY_MCP]
    z = _norm(np.cross(y, knuckle))
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=1)


@dataclass
class RetargetResult:
    side: str
    qpos: np.ndarray
    tip_err_cm: np.ndarray
    wrist_ori_err_deg: np.ndarray
    tip_target: np.ndarray
    tip_actual: np.ndarray
    scale: float
    joint_names: list[str] = field(default_factory=list)


class HandRetargeter:
    """Retarget one MANO trajectory onto the official 12-DoF XHand."""

    def __init__(
        self,
        side: str,
        calib_joints21: np.ndarray,
        pos_cost: float = POS_COST_TIP,
        ori_cost: float = ORI_COST_TIP,
        lambda_wrist: float = LAMBDA_WRIST,
        pos_cost_wrist: float = POS_COST_WRIST,
        anatomical_alignment: bool = True,
        articulated_tip_orientation: bool = False,
    ):
        if not anatomical_alignment:
            raise ValueError("XHand retarget requires anatomical wrist alignment")
        if articulated_tip_orientation:
            raise ValueError("XHand faithful path does not use synthetic tip orientations")
        self.side = side
        self.model = build_hand_model(side)
        self.config = mink.Configuration(self.model)
        self.palm = palm_link(side)
        self.joint_names = finger_joint_names(side)
        self._finger_qpos_adr = np.asarray(
            [
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in self.joint_names
            ],
            dtype=np.int64,
        )

        data = mujoco.MjData(self.model)
        data.qpos[:] = self.model.qpos0
        mujoco.mj_forward(self.model, data)

        def _body_pos(name: str) -> np.ndarray:
            return data.xpos[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            ].copy()

        def _site_mat(name: str) -> np.ndarray:
            return data.site_xmat[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            ].reshape(3, 3).copy()

        def _site_pos(name: str) -> np.ndarray:
            return data.site_xpos[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            ].copy()

        palm_i = data.xmat[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.palm)
        ].reshape(3, 3).copy()
        tip_i = {f: _site_mat(f"{f}_tip") for f in FINGER_NAMES}
        tip_i_pos = {f: _site_pos(f"{f}_tip") for f in FINGER_NAMES}

        wrist_i = _body_pos(self.palm)
        mcp_i = {
            "index": _body_pos(f"{side}_hand_index_rota_link1"),
            "middle": _body_pos(f"{side}_hand_mid_link1"),
            "pinky": _body_pos(f"{side}_hand_pinky_link1"),
        }
        y_i = _norm(mcp_i["middle"] - wrist_i)
        z_i = _norm(np.cross(y_i, mcp_i["index"] - mcp_i["pinky"]))
        x_i = np.cross(y_i, z_i)
        robot_palm_frame = np.stack([x_i, y_i, z_i], axis=1)

        # Map the moving MANO anatomical palm frame to the XHand palm/link
        # frame. This avoids the old reproduction's orientation cancellation.
        self.wrist_ori_const = robot_palm_frame.T @ palm_i
        self.tip_ori_const = {
            f: robot_palm_frame.T @ tip_i[f] for f in FINGER_NAMES
        }

        len_m = np.linalg.norm(
            calib_joints21[MANO_TIP_IDX[2]] - calib_joints21[MANO_WRIST]
        )
        len_robot = np.linalg.norm(tip_i_pos["middle"] - wrist_i)
        self.scale = float(len_robot / len_m)

        self.tip_tasks = {
            f: mink.FrameTask(
                frame_name=f"{f}_tip",
                frame_type="site",
                position_cost=pos_cost,
                orientation_cost=ori_cost,
                lm_damping=LM_DAMPING,
                gain=GAIN,
            )
            for f in FINGER_NAMES
        }
        self.wrist_task = mink.FrameTask(
            frame_name=self.palm,
            frame_type="body",
            position_cost=pos_cost_wrist,
            orientation_cost=lambda_wrist,
            lm_damping=LM_DAMPING,
            gain=GAIN,
        )
        self.tasks = list(self.tip_tasks.values()) + [self.wrist_task]
        self.limits = [mink.ConfigurationLimit(self.model)]

    def _targets(self, joints21: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
        p_m = palm_frame(joints21)
        wrist = joints21[MANO_WRIST]
        targets = {}
        for finger, mano_idx in zip(FINGER_NAMES, MANO_TIP_IDX):
            target = np.eye(4)
            target[:3, :3] = p_m @ self.tip_ori_const[finger]
            target[:3, 3] = wrist + self.scale * (joints21[mano_idx] - wrist)
            targets[finger] = target
        wrist_target = np.eye(4)
        wrist_target[:3, :3] = p_m @ self.wrist_ori_const
        wrist_target[:3, 3] = wrist
        return targets, wrist_target

    def retarget(
        self,
        joints_traj: np.ndarray,
        dt: float = 0.05,
        iters_first: int = 500,
        iters: int = 100,
    ) -> RetargetResult:
        n = joints_traj.shape[0]
        qpos = np.zeros((n, 19))
        tip_err = np.zeros((n, 5))
        wrist_err = np.zeros(n)
        tip_tgt = np.zeros((n, 5, 3))
        tip_actual = np.zeros((n, 5, 3))

        for frame in range(n):
            targets, wrist_target = self._targets(joints_traj[frame])
            for finger in FINGER_NAMES:
                self.tip_tasks[finger].set_target(
                    mink.SE3.from_matrix(targets[finger])
                )
            self.wrist_task.set_target(mink.SE3.from_matrix(wrist_target))
            for _ in range(iters_first if frame == 0 else iters):
                velocity = mink.solve_ik(
                    self.config,
                    self.tasks,
                    dt=dt,
                    solver="daqp",
                    limits=self.limits,
                    damping=1e-4,
                )
                self.config.integrate_inplace(velocity, dt)

            palm_tf = self.config.get_transform_frame_to_world(self.palm, "body")
            palm_R = palm_tf.rotation().as_matrix()
            palm_q = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(palm_q, palm_R.reshape(-1))
            qpos[frame, :3] = palm_tf.translation()
            qpos[frame, 3:7] = palm_q
            qpos[frame, 7:] = self.config.q[self._finger_qpos_adr]
            tip_tgt[frame] = [targets[f][:3, 3] for f in FINGER_NAMES]
            for i, finger in enumerate(FINGER_NAMES):
                cur = self.config.get_transform_frame_to_world(f"{finger}_tip", "site")
                tip_actual[frame, i] = cur.translation()
                tip_err[frame, i] = (
                    np.linalg.norm(cur.translation() - targets[finger][:3, 3]) * 100.0
                )
            r_delta = wrist_target[:3, :3].T @ palm_R
            cosine = np.clip((np.trace(r_delta) - 1.0) / 2.0, -1.0, 1.0)
            wrist_err[frame] = np.degrees(np.arccos(cosine))

        return RetargetResult(
            side=self.side,
            qpos=qpos,
            tip_err_cm=tip_err,
            wrist_ori_err_deg=wrist_err,
            tip_target=tip_tgt,
            tip_actual=tip_actual,
            scale=self.scale,
            joint_names=self.joint_names,
        )
