"""IsaacLab scene assembly for one TACO sequence with SPIDER XHand.

Phase-1 default behavior stays close to the pinned nonofficial reproduction.
Retained divergences are limited to named-DOF order correctness, runtime path
portability, actual-state access, and optional read-only contact reporting.
Unproven mimic/solver/drive/startup-state/controller compensations are not part
of the default path.
"""

from __future__ import annotations

import json
import os
from os.path import abspath, dirname, isfile, join

import numpy as np
from .joint_order import to_sim_order, to_retarget_order

ROOT = abspath(join(dirname(__file__), "..", "..", ".."))
URDF_DIR = join(ROOT, "assets", "xhand", "spider")
HAND_URDF = {
    side: join(URDF_DIR, f"xhand_{side}.urdf")
    for side in ("left", "right")
}
RETARGET_DIR = join(ROOT, "outputs", "retarget")
RETARGET_DIR = os.environ.get("EGOENGINE_RETARGET_DIR", RETARGET_DIR)
USD_CACHE = join(ROOT, "outputs", "sim_usd")

MASS_BY_NAME = {"brush": 0.06, "bowl": 0.25, "knife": 0.08, "plate": 0.35}
MASS_DEFAULT = 0.15
TABLE_MARGIN_M = 0.003


def prepare_replay(
    triplet: str,
    seq_id: str,
    traj_path: str | None = None,
    retarget_dir: str | None = None,
) -> dict:
    import sys
    import trimesh

    sys.path.insert(0, join(ROOT, "src"))
    from egoengine.taco import load_sequence, compute_pseudo_base

    out_dir = join(retarget_dir or RETARGET_DIR, triplet, seq_id)
    rt = np.load(traj_path if traj_path else join(out_dir, "trajectory_kinematic.npz"))
    report = json.load(open(join(out_dir, "retarget_report.json")))

    seq = load_sequence(triplet, seq_id)
    T_base = compute_pseudo_base(seq)
    T_bi = np.linalg.inv(T_base)

    objects = {}
    z_min = np.inf
    for role, obj in seq.objects.items():
        gt = T_bi[None] @ obj.poses
        mesh = trimesh.load(obj.mesh_path, force="mesh")
        verts = np.asarray(mesh.vertices, dtype=np.float64) * 0.01
        posed = verts @ gt[0][:3, :3].T + gt[0][:3, 3]
        z_min = min(z_min, float(posed[:, 2].min()))
        objects[role] = {
            "object_id": obj.object_id,
            "name": obj.name,
            "mesh_path": obj.mesh_path,
            "gt": gt,
            "mass": MASS_BY_NAME.get(obj.name, MASS_DEFAULT),
        }

    center_xy = np.mean(
        [objects[r]["gt"][:, :3, 3].mean(axis=0)[:2] for r in objects], axis=0
    )
    return {
        "triplet": triplet,
        "seq_id": seq_id,
        "video_path": seq.video_path,
        "n_frames": int(rt["qpos_left"].shape[0]),
        "qpos": {
            s: rt[f"qpos_{s}"].astype(np.float64) for s in ("left", "right")
        },
        "joint_names": {
            side: report["hands"][side]["joint_names"]
            for side in ("left", "right")
        },
        "embodiment": report.get("embodiment", "SPIDER XHand"),
        "T_base": T_base,
        "objects": objects,
        "table_z": z_min - TABLE_MARGIN_M,
        "scene_center_xy": tuple(center_xy),
    }


def apply_tool_reference_override(prep: dict, reference_path: str) -> dict:
    """Apply an explicit derived tool reference without touching source data.

    The override artifact must contain a full-length ``object_tool_gt_T``
    array.  Only the manipulated tool reference is replaced; wrist retargeting,
    target-object annotations, physics, and controllers remain unchanged.
    """
    path = os.path.abspath(os.fspath(reference_path))
    data = np.load(path, allow_pickle=False)
    if "object_tool_gt_T" not in data.files:
        raise KeyError(f"derived reference missing object_tool_gt_T: {path}")
    candidate = np.asarray(data["object_tool_gt_T"], dtype=np.float64)
    expected = np.asarray(prep["objects"]["tool"]["gt"])
    if candidate.shape != expected.shape:
        raise ValueError(
            f"derived tool reference shape {candidate.shape} != {expected.shape}"
        )
    if not np.isfinite(candidate).all():
        raise ValueError(f"derived tool reference is non-finite: {path}")
    prep["objects"]["tool"]["gt"] = candidate.copy()
    prep["tool_reference_override"] = path
    prep["tool_reference_corrected_frames"] = (
        np.asarray(data["corrected_frames"], dtype=int).tolist()
        if "corrected_frames" in data.files
        else []
    )
    return prep


def convert_object_usd(mesh_path_cm: str, object_id: str, mass: float) -> str:
    out_dir = join(USD_CACHE, object_id)
    usd_path = join(out_dir, f"{object_id}.usd")
    if isfile(usd_path):
        return usd_path
    os.makedirs(out_dir, exist_ok=True)

    import trimesh
    from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
    from isaaclab.sim.schemas import schemas_cfg

    scaled = join(out_dir, f"{object_id}_m.obj")
    m = trimesh.load(mesh_path_cm, force="mesh")
    m.apply_scale(0.01)
    m.export(scaled)
    cfg = MeshConverterCfg(
        asset_path=scaled,
        usd_dir=out_dir,
        usd_file_name=f"{object_id}.usd",
        mass_props=schemas_cfg.MassPropertiesCfg(mass=mass),
        rigid_props=schemas_cfg.RigidBodyPropertiesCfg(),
        collision_props=schemas_cfg.CollisionPropertiesCfg(),
        mesh_collision_props=schemas_cfg.ConvexDecompositionPropertiesCfg(),
        scale=(1.0, 1.0, 1.0),
        force_usd_conversion=True,
        make_instanceable=False,
    )
    MeshConverter(cfg)
    return usd_path


def mat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(R).as_quat()


def build_scene(
    prep: dict,
    dt: float = 1.0 / 120.0,
    device: str = "cuda:0",
    friction: float = 1.0,
):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.sim import SimulationContext, SimulationCfg
    from isaaclab.sim.schemas import schemas_cfg

    sim_cfg = SimulationCfg(dt=dt, device=device)
    sim_cfg.physics_material.static_friction = friction
    sim_cfg.physics_material.dynamic_friction = friction
    sim = SimulationContext(sim_cfg)
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())

    cx, cy = prep["scene_center_xy"]
    table_cfg = sim_utils.CuboidCfg(
        size=(1.4, 1.0, 0.04),
        collision_props=schemas_cfg.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.4, 0.25)),
    )
    table_cfg.func(
        "/World/Table",
        table_cfg,
        translation=(cx, cy, prep["table_z"] - 0.02),
    )

    objects = {}
    for role, o in prep["objects"].items():
        usd = convert_object_usd(o["mesh_path"], o["object_id"], o["mass"])
        T0 = o["gt"][0]
        cfg = RigidObjectCfg(
            prim_path=f"/World/Object_{role}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd,
                rigid_props=schemas_cfg.RigidBodyPropertiesCfg(),
                collision_props=schemas_cfg.CollisionPropertiesCfg(),
                mass_props=schemas_cfg.MassPropertiesCfg(mass=o["mass"]),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=tuple(T0[:3, 3]),
                rot=tuple(mat_to_quat_xyzw(T0[:3, :3])[[3, 0, 1, 2]]),
            ),
        )
        objects[role] = RigidObject(cfg)

    hands = {}
    for side in ("left", "right"):
        q0 = prep["qpos"][side][0]
        cfg = ArticulationCfg(
            prim_path=f"/World/Hand_{side}",
            spawn=sim_utils.UrdfFileCfg(
                asset_path=HAND_URDF[side],
                fix_base=False,
                self_collision=False,
                joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
                    target_type="position",
                    gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                        stiffness=1000.0, damping=200.0
                    ),
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(q0[:3]), rot=tuple(q0[3:7])
            ),
            actuators={
                "fingers": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=1000.0,
                    damping=200.0,
                    effort_limit=100.0,
                    velocity_limit=100.0,
                )
            },
        )
        hands[side] = Articulation(cfg)

    sim.reset()

    import omni.physics.tensors
    import omni.physx
    import torch
    import warp as wp
    from isaaclab.sim.utils.stage import get_current_stage_id

    physx = omni.physx.get_physx_simulation_interface()
    view = omni.physics.tensors.create_simulation_view(
        "warp", stage_id=get_current_stage_id()
    )
    view.set_subspace_roots("/")

    art_views = {}
    for side in ("left", "right"):
        for pat in (
            f"/World/Hand_{side}/{side}_hand_link",
            f"/World/Hand_{side}/root_joint",
            f"/World/Hand_{side}",
        ):
            try:
                art_views[side] = view.create_articulation_view(pat)
                print(
                    f"[scene] articulation view {side}: {pat} "
                    f"(count={art_views[side].count}, max_dofs={art_views[side].max_dofs})"
                )
                break
            except Exception:
                continue
        else:
            raise RuntimeError(f"no articulation view pattern worked for {side}")

    rb_views = {
        role: view.create_rigid_body_view(f"/World/Object_{role}")
        for role in prep["objects"]
    }
    sim_joint_names = {
        s: list(hands[s].data.joint_names) for s in ("left", "right")
    }
    joint_idx = {
        s: torch.tensor(
            [sim_joint_names[s].index(n) for n in prep["joint_names"][s]],
            dtype=torch.long,
            device=device,
        )
        for s in ("left", "right")
    }

    def wa(t):
        return wp.from_torch(t.contiguous())

    handles = {
        "physx": physx,
        "view": view,
        "art": art_views,
        "rb": rb_views,
        "joint_idx": joint_idx,
        "sim_joint_names": sim_joint_names,
        "idx1": wp.from_torch(torch.zeros(1, dtype=torch.int32, device=device)),
        "wa": wa,
        "device": device,
        "dt": dt,
    }
    return sim, hands, objects, handles


def set_hand_state(handles: dict, side: str, qpos19: np.ndarray):
    import torch

    idx = handles["joint_idx"][side]
    root = torch.tensor(
        [
            qpos19[0],
            qpos19[1],
            qpos19[2],
            qpos19[4],
            qpos19[5],
            qpos19[6],
            qpos19[3],
        ],
        dtype=torch.float32,
        device=handles["device"],
    )[None]
    q12 = torch.tensor(
        qpos19[7:], dtype=torch.float32, device=handles["device"]
    )[None]
    q12 = to_sim_order(q12, idx)
    handles["art"][side].set_root_transforms(handles["wa"](root), handles["idx1"])
    handles["art"][side].set_dof_positions(handles["wa"](q12), handles["idx1"])


def get_object_pose(handles: dict, role: str) -> np.ndarray:
    import warp as wp
    from scipy.spatial.transform import Rotation

    t = wp.to_torch(handles["rb"][role].get_transforms())[0].cpu().numpy()
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(t[3:7]).as_matrix()
    T[:3, 3] = t[:3]
    return T


def pose_error(T_sim: np.ndarray, T_gt: np.ndarray) -> tuple[float, float]:
    dp = float(np.linalg.norm(T_sim[:3, 3] - T_gt[:3, 3]))
    Rd = T_gt[:3, :3].T @ T_sim[:3, :3]
    cos = np.clip((np.trace(Rd) - 1.0) / 2.0, -1.0, 1.0)
    return dp, float(np.arccos(cos))


def interp_qpos(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    q = (1.0 - a) * q0 + a * q1
    quat = q0[3:7].copy()
    if np.dot(q0[3:7], q1[3:7]) < 0:
        quat = -quat
    quat = (1.0 - a) * quat + a * q1[3:7]
    q[3:7] = quat / (np.linalg.norm(quat) + 1e-12)
    return q


DRIVE = {
    "finger_stiffness": 5.0,
    "finger_damping": 0.5,
    "finger_max_force": 3.0,
    "wrist_kp": 300.0,
    "wrist_kd": 12.0,
    # SPIDER XHand has substantially more rotational inertia than the former
    # hand model.  The original gains leave the floating wrist about 35 deg
    # behind its commanded orientation during open-loop execution.
    "wrist_kp_rot": 2.0,
    "wrist_kd_rot": 0.12,
    "wrist_max_force": 30.0,
    "wrist_max_torque": 2.0,
}


def build_scene_batched(
    prep: dict,
    num_envs: int,
    dt: float = 1.0 / 120.0,
    device: str = "cuda:0",
    friction: float = 1.0,
    spacing: float = 2.0,
    enable_contact_reports: bool = False,
):
    """Paper/upstream-style batched scene with optional read-only contact reports."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.sim import SimulationContext, SimulationCfg
    from isaaclab.sim.schemas import schemas_cfg

    sim_cfg = SimulationCfg(dt=dt, device=device)
    sim_cfg.physics_material.static_friction = friction
    sim_cfg.physics_material.dynamic_friction = friction
    sim = SimulationContext(sim_cfg)
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())

    for i in range(num_envs):
        sim_utils.create_prim(
            f"/World/envs/env_{i}", "Xform", translation=(i * spacing, 0.0, 0.0)
        )

    cx, cy = prep["scene_center_xy"]
    table_cfg = sim_utils.CuboidCfg(
        size=(1.4, 1.0, 0.04),
        collision_props=schemas_cfg.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.4, 0.25)),
    )
    for i in range(num_envs):
        table_cfg.func(
            f"/World/envs/env_{i}/Table",
            table_cfg,
            translation=(cx, cy, prep["table_z"] - 0.02),
        )

    objects = {}
    for role, o in prep["objects"].items():
        usd = convert_object_usd(o["mesh_path"], o["object_id"], o["mass"])
        T0 = o["gt"][0]
        cfg = RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/Object_{role}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd,
                rigid_props=schemas_cfg.RigidBodyPropertiesCfg(),
                collision_props=schemas_cfg.CollisionPropertiesCfg(),
                mass_props=schemas_cfg.MassPropertiesCfg(mass=o["mass"]),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=tuple(T0[:3, 3]),
                rot=tuple(mat_to_quat_xyzw(T0[:3, :3])[[3, 0, 1, 2]]),
            ),
        )
        objects[role] = RigidObject(cfg)

    hands = {}
    for side in ("left", "right"):
        q0 = prep["qpos"][side][0]
        cfg = ArticulationCfg(
            prim_path=f"/World/envs/env_.*/Hand_{side}",
            spawn=sim_utils.UrdfFileCfg(
                asset_path=HAND_URDF[side],
                fix_base=False,
                self_collision=False,
                joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
                    target_type="position",
                    gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                        stiffness=DRIVE["finger_stiffness"],
                        damping=DRIVE["finger_damping"],
                    ),
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(q0[:3]), rot=tuple(q0[3:7])
            ),
            actuators={
                "fingers": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=DRIVE["finger_stiffness"],
                    damping=DRIVE["finger_damping"],
                    effort_limit=DRIVE["finger_max_force"],
                    velocity_limit=100.0,
                )
            },
        )
        hands[side] = Articulation(cfg)

    if enable_contact_reports:
        import omni.usd
        from pxr import PhysxSchema, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)

    sim.reset()

    import omni.physics.tensors
    import omni.physx
    import torch
    import warp as wp
    from isaaclab.sim.utils.stage import get_current_stage_id

    physx = omni.physx.get_physx_simulation_interface()
    view = omni.physics.tensors.create_simulation_view(
        "warp", stage_id=get_current_stage_id()
    )
    view.set_subspace_roots("/")

    art_views = {}
    for side in ("left", "right"):
        for pat in (
            f"/World/envs/env_*/Hand_{side}/{side}_hand_link",
            f"/World/envs/env_*/Hand_{side}/root_joint",
            f"/World/envs/env_*/Hand_{side}",
        ):
            try:
                art_views[side] = view.create_articulation_view(pat)
                print(
                    f"[scene] batched articulation view {side}: {pat} "
                    f"(count={art_views[side].count}, max_dofs={art_views[side].max_dofs})"
                )
                break
            except Exception:
                continue
        else:
            raise RuntimeError(f"no articulation view pattern worked for {side}")

    rb_views = {
        role: view.create_rigid_body_view(f"/World/envs/env_*/Object_{role}")
        for role in prep["objects"]
    }
    sim_joint_names = {
        s: list(hands[s].data.joint_names) for s in ("left", "right")
    }
    joint_idx = {
        s: torch.tensor(
            [sim_joint_names[s].index(n) for n in prep["joint_names"][s]],
            dtype=torch.long,
            device=device,
        )
        for s in ("left", "right")
    }

    tip_links = {}
    for side in ("left", "right"):
        names = list(hands[side].data.body_names)
        # Isaac merges SPIDER's fixed, massless ``*_tip`` links into the
        # preceding distal bodies. Use those physical distal bodies for the
        # geometric contact proxy instead of searching for removed prim names.
        thumb = [i for i, n in enumerate(names) if n.endswith("thumb_rota_link2")]
        fingers = [
            i
            for i, n in enumerate(names)
            if any(
                n.endswith(suffix)
                for suffix in (
                    "index_rota_link2",
                    "mid_link2",
                    "ring_link2",
                    "pinky_link2",
                )
            )
        ]
        tip_links[side] = {
            "thumb": torch.tensor(thumb, dtype=torch.long, device=device),
            "fingers": torch.tensor(fingers, dtype=torch.long, device=device),
        }
        print(f"[scene] tip links {side}: thumb={thumb} fingers={fingers}")

    def wa(t):
        return wp.from_torch(t.contiguous())

    handles = {
        "physx": physx,
        "view": view,
        "art": art_views,
        "rb": rb_views,
        "joint_idx": joint_idx,
        "sim_joint_names": sim_joint_names,
        "tip_links": tip_links,
        "idx_w": wp.from_torch(
            torch.arange(num_envs, dtype=torch.int32, device=device)
        ),
        "num_envs": num_envs,
        "spacing": spacing,
        "wa": wa,
        "device": device,
        "dt": dt,
    }

    for side in ("left", "right"):
        av = art_views[side]
        k = torch.full(
            (av.count, av.max_dofs), DRIVE["finger_stiffness"], dtype=torch.float32
        )
        d = torch.full_like(k, DRIVE["finger_damping"])
        f = torch.full_like(k, DRIVE["finger_max_force"])
        idx_cpu = wp.from_torch(torch.arange(av.count, dtype=torch.int32))
        av.set_dof_stiffnesses(wp.from_torch(k), idx_cpu)
        av.set_dof_dampings(wp.from_torch(d), idx_cpu)
        av.set_dof_max_forces(wp.from_torch(f), idx_cpu)

    return sim, hands, objects, handles


def _quat_wxyz_to_xyzw_t(q):
    import torch

    return torch.stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]], dim=-1)


def reset_envs_state(handles: dict, qpos: dict, obj_T: dict):
    import torch

    dev = handles["device"]
    n = handles["num_envs"]
    origins = torch.arange(n, dtype=torch.float32, device=dev)[:, None] * torch.tensor(
        [handles["spacing"], 0.0, 0.0], device=dev
    )

    for side in ("left", "right"):
        q = torch.tensor(qpos[side], dtype=torch.float32, device=dev)
        root = torch.zeros(n, 7, device=dev)
        root[:, :3] = q[:3] + origins
        root[:, 3:] = _quat_wxyz_to_xyzw_t(q[3:7])
        q12 = to_sim_order(q[7:].expand(n, 12), handles["joint_idx"][side])
        av = handles["art"][side]
        av.set_root_transforms(handles["wa"](root), handles["idx_w"])
        av.set_root_velocities(
            handles["wa"](torch.zeros(n, 6, device=dev)), handles["idx_w"]
        )
        av.set_dof_positions(handles["wa"](q12), handles["idx_w"])
        av.set_dof_velocities(
            handles["wa"](torch.zeros(n, 12, device=dev)), handles["idx_w"]
        )
        av.set_dof_position_targets(handles["wa"](q12), handles["idx_w"])

    from scipy.spatial.transform import Rotation

    for role, T in obj_T.items():
        pose = torch.zeros(n, 7, device=dev)
        pose[:, :3] = torch.tensor(T[:3, 3], dtype=torch.float32, device=dev) + origins
        pose[:, 3:] = torch.tensor(
            Rotation.from_matrix(T[:3, :3]).as_quat(),
            dtype=torch.float32,
            device=dev,
        )
        handles["rb"][role].set_transforms(handles["wa"](pose), handles["idx_w"])
        handles["rb"][role].set_velocities(
            handles["wa"](torch.zeros(n, 6, device=dev)), handles["idx_w"]
        )


def get_hand_qpos19(handles: dict, side: str) -> np.ndarray:
    import warp as wp

    av = handles["art"][side]
    root = wp.to_torch(av.get_root_transforms())[0].cpu().numpy()
    q12 = wp.to_torch(av.get_dof_positions())[0]
    q12 = to_retarget_order(q12, handles["joint_idx"][side]).cpu().numpy()
    return np.concatenate([root[:3], [root[6], root[3], root[4], root[5]], q12])


def apply_hand_action(
    handles: dict,
    side: str,
    wrist_p,
    wrist_qwxyz,
    wrist_v,
    q12_targets,
):
    """Upstream wrist/finger controller with one reference-point correctness fix."""
    import torch
    import warp as wp

    dev = handles["device"]
    av = handles["art"][side]
    n = handles["num_envs"]

    q12 = to_sim_order(q12_targets, handles["joint_idx"][side])
    av.set_dof_position_targets(handles["wa"](q12), handles["idx_w"])

    lt = wp.to_torch(av.get_link_transforms()).to(dev)
    lv = wp.to_torch(av.get_link_velocities()).to(dev)
    p, q = lt[:, 0, :3], lt[:, 0, 3:7]
    v_com, w = lv[:, 0, :3], lv[:, 0, 3:6]

    # Retained correctness fix: PhysX linear velocity is at the link COM while
    # wrist_p/p refer to the link origin. Shift velocity to the same point before
    # evaluating the PD error. This changes no target and adds no stabilizing cap.
    from .control_math import origin_velocity

    local_com = wp.to_torch(av.get_coms()).to(dev)[:, 0, :3]
    root_rotation = _quat_to_rotmat_xyzw(q)
    world_com = torch.einsum("nij,nj->ni", root_rotation, local_com) + p
    stack = lambda values: torch.stack(values, dim=-1)
    v = origin_velocity(v_com, w, p - world_com, stack)

    origins = torch.arange(n, dtype=torch.float32, device=dev)[:, None] * torch.tensor(
        [handles["spacing"], 0.0, 0.0], device=dev
    )
    p_ref = wrist_p + origins
    q_ref = _quat_wxyz_to_xyzw_t(wrist_qwxyz)

    F = DRIVE["wrist_kp"] * (p_ref - p) + DRIVE["wrist_kd"] * (wrist_v - v)
    F = F.clamp(-DRIVE["wrist_max_force"], DRIVE["wrist_max_force"])

    # Keep the pinned-upstream gravity feedforward, not the later distributed
    # gravity-torque compensation experiment.
    masses = wp.to_torch(av.get_masses()).to(dev)
    F = F + torch.tensor([0.0, 0.0, 9.81], device=dev) * masses.sum(
        dim=1, keepdim=True
    )

    qc = torch.stack([q[..., 0], q[..., 1], q[..., 2], -q[..., 3]], dim=-1)
    e = _quat_mul_xyzw(q_ref, qc)
    e = torch.where(e[..., 3:4] < 0, -e, e)
    tau = DRIVE["wrist_kp_rot"] * 2.0 * e[..., :3] - DRIVE["wrist_kd_rot"] * w
    tau = tau.clamp(-DRIVE["wrist_max_torque"], DRIVE["wrist_max_torque"])

    L = av.max_links
    Ffull = torch.zeros(n, L, 3, device=dev)
    Tfull = torch.zeros(n, L, 3, device=dev)
    Ffull[:, 0], Tfull[:, 0] = F, tau
    av.apply_forces_and_torques_at_position(
        handles["wa"](Ffull),
        handles["wa"](Tfull),
        None,
        handles["idx_w"],
        True,
    )


def _quat_mul_xyzw(a, b):
    import torch

    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dim=-1,
    )


def get_object_poses_batched(handles: dict, role: str):
    import torch
    import warp as wp

    dev = handles["device"]
    n = handles["num_envs"]
    t = wp.to_torch(handles["rb"][role].get_transforms()).clone().to(dev)
    origins = torch.arange(n, dtype=torch.float32, device=dev)[:, None] * torch.tensor(
        [handles["spacing"], 0.0, 0.0], device=dev
    )
    t[:, :3] -= origins
    return t


def pose_error_batched(t_sim, gt_T, quat_wxyz: bool = False):
    import torch
    from scipy.spatial.transform import Rotation

    dev = t_sim.device
    p_gt = torch.tensor(gt_T[:3, 3], dtype=torch.float32, device=dev)
    q_gt_xyzw = torch.tensor(
        Rotation.from_matrix(gt_T[:3, :3]).as_quat(),
        dtype=torch.float32,
        device=dev,
    )
    dp = (t_sim[:, :3] - p_gt).norm(dim=-1)
    dot = (t_sim[:, 3:7] * q_gt_xyzw).sum(-1).abs().clamp(max=1.0)
    dR = 2.0 * torch.acos(dot)
    return dp, dR


def tool_contact_proxy(
    handles: dict, tool_verts_obj, thresh: float = 0.015, side: str | None = None
):
    import torch
    import warp as wp

    dev = handles["device"]
    ts = get_object_poses_batched(handles, "tool")
    q = ts[:, 3:7]
    R = _quat_to_rotmat_xyzw(q)
    verts = torch.einsum("nij,vj->nvi", R, tool_verts_obj) + ts[:, None, :3]
    tips = []
    contact_sides = (side,) if side is not None else ("left", "right")
    for contact_side in contact_sides:
        lt = (
            wp.to_torch(handles["art"][contact_side].get_link_transforms())
            .clone()
            .to(dev)
        )
        n = handles["num_envs"]
        origins = torch.arange(n, dtype=torch.float32, device=dev)[:, None] * torch.tensor(
            [handles["spacing"], 0.0, 0.0], device=dev
        )
        lt[:, :, :3] -= origins[:, None, :]
        tl = handles["tip_links"][contact_side]
        tips.append((lt[:, tl["thumb"], :3], True))
        tips.append((lt[:, tl["fingers"], :3], False))
    thumb = torch.cat([t for t, is_t in tips if is_t], dim=1)
    fing = torch.cat([t for t, is_t in tips if not is_t], dim=1)
    d_thumb = (thumb[:, :, None, :] - verts[:, None, :, :]).norm(dim=-1).amin((1, 2))
    d_fing = (fing[:, :, None, :] - verts[:, None, :, :]).norm(dim=-1).amin((1, 2))
    return d_thumb < thresh, d_fing < thresh


def _quat_to_rotmat_xyzw(q):
    import torch

    x, y, z, w = q.unbind(-1)
    return torch.stack(
        [
            torch.stack(
                [
                    1 - 2 * (y * y + z * z),
                    2 * (x * y - z * w),
                    2 * (x * z + y * w),
                ],
                -1,
            ),
            torch.stack(
                [
                    2 * (x * y + z * w),
                    1 - 2 * (x * x + z * z),
                    2 * (y * z - x * w),
                ],
                -1,
            ),
            torch.stack(
                [
                    2 * (x * z - y * w),
                    2 * (y * z + x * w),
                    1 - 2 * (x * x + y * y),
                ],
                -1,
            ),
        ],
        dim=-2,
    )
