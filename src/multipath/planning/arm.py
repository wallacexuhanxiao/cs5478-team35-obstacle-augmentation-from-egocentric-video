"""Mounted-arm planning model for the floating hand (planning only).

The physics keeps its floating XHand. When `planning.arm_model` names an arm
export (e.g. the release's RB-Y1 right arm, `tools/egoengine_aug/export_rby1_arm.py`),
cuRobo plans each RELAXED bridge in that arm's joint space: its URDF ends at the
same `wrist` (hand link) frame and carries the floating planner's measured hand
and carried-object spheres, so the arm links avoid the world and the path stays
in reach. Every fitted command pose is then re-validated by a seeded,
collision-checked arm IK. Visual and planning consistency only; no arm drives
exist in PhysX.
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


class ArmModel:
    def __init__(self, model_path, live, wrist_spheres, tensor, dt, out, tolerance=(.015, .1)):
        from curobo.geom.sdf.world import CollisionCheckerType
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
        self.spec = json.loads(Path(model_path).read_text())
        self.tensor, self.out, self.tolerance = tensor, Path(out), tuple(float(v) for v in tolerance)
        self.names = list(self.spec['joint_names'])
        dof = len(self.names)
        spheres = {k: list(v) for k, v in self.spec['collision_spheres'].items()}
        spheres['wrist'] = list(wrist_spheres)
        kin = {'urdf_path': self.spec['urdf'], 'asset_root_path': str(Path(self.spec['urdf']).parent),
               'base_link': self.spec.get('base_link', 'base'), 'ee_link': self.spec.get('ee_link', 'wrist'),
               'collision_link_names': list(spheres), 'collision_spheres': spheres, 'collision_sphere_buffer': 0.,
               'self_collision_ignore': {}, 'self_collision_buffer': {}, 'use_global_cumul': True,
               'mesh_link_names': [], 'lock_joints': {},
               'cspace': {'joint_names': self.names, 'retract_config': self.spec['retract_config'],
                          'null_space_weight': [1.] * dof, 'cspace_distance_weight': [1.] * dof,
                          'max_acceleration': [10.] * dof, 'max_jerk': 500., 'velocity_scale': 1.,
                          'acceleration_scale': 1., 'jerk_scale': 1.}}
        self.motion_gen = MotionGen(MotionGenConfig.load_from_robot_config(
            {'kinematics': kin}, live, tensor_args=tensor, interpolation_dt=dt,
            self_collision_check=False, self_collision_opt=False,
            collision_checker_type=CollisionCheckerType.MESH if live.get('mesh') else CollisionCheckerType.PRIMITIVE,
            use_cuda_graph=False))
        self.retract = np.asarray(self.spec['retract_config'], float)
        self.last = None
        self.sphere_count = sum(len(v) for v in spheres.values())

    def js(self, q):
        from curobo.types.robot import JointState
        return JointState.from_position(self.tensor.to_device(np.asarray(q, float).reshape(-1, len(self.names)).copy()),
                                        joint_names=self.names)

    def fk(self, q):
        state = self.motion_gen.kinematics.get_state(self.tensor.to_device(np.asarray(q, float).reshape(-1, len(self.names)).copy()))
        return np.c_[state.ee_position.detach().cpu().numpy(), state.ee_quaternion.detach().cpu().numpy()].astype(float)

    def ik(self, poses, seeds):
        """Seeded IK; returns joints, position/rotation errors (own FK) and collision/limit feasibility."""
        from curobo.types.math import Pose
        poses = np.asarray(poses, float)
        goal = Pose(position=self.tensor.to_device(poses[:, :3].copy()), quaternion=self.tensor.to_device(poses[:, 3:7].copy()))
        seed = self.tensor.to_device(np.asarray(seeds, float).reshape(len(poses), -1, len(self.names)).copy())
        result = self.motion_gen.ik_solver.solve_batch(goal, seed_config=seed)
        q = result.solution.detach().cpu().numpy().reshape(len(poses), -1, len(self.names))[:, 0].astype(float)
        reached = self.fk(q)
        position = np.linalg.norm(reached[:, :3] - poses[:, :3], axis=1)
        rotation = (Rotation.from_quat(reached[:, [4, 5, 6, 3]]) * Rotation.from_quat(poses[:, [4, 5, 6, 3]]).inv()).magnitude()
        metrics = self.motion_gen.check_constraints(self.js(q))
        feasible = metrics.feasible.detach().cpu().numpy().reshape(len(poses), -1).all(1)
        return q, position, rotation, feasible

    def geometric_path(self, start_pose, goal_pose):
        """cuRobo arm-joint path between the arm IK of the two hand poses, as hand poses."""
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
        seed = self.retract if self.last is None else self.last
        q, position, rotation, feasible = self.ik(np.stack([start_pose, goal_pose]), [seed, seed])
        boundary = dict(position_error_m=position.tolist(), rotation_error_rad=rotation.tolist(), feasible=feasible.tolist())
        (self.out / 'arm_boundary_ik.json').write_text(json.dumps(boundary, indent=2))
        bad = [i for i in range(2) if position[i] > self.tolerance[0] or rotation[i] > self.tolerance[1] or not feasible[i]]
        if bad:
            raise RuntimeError(f'arm: {"start" if bad[0] == 0 else "goal"} hand pose has no reachable collision-free arm IK {boundary}')
        result = self.motion_gen.plan_single_js(self.js(q[0]), self.js(q[1]),
                                                MotionGenPlanConfig(max_attempts=6, enable_graph_attempt=2))
        if not bool(result.success.item()):
            raise RuntimeError(f'arm cuRobo failed: {result.status}')
        path = result.get_interpolated_plan().position.detach().cpu().numpy().reshape(-1, len(self.names)).astype(float)
        return self.fk(path), path, float(result.interpolation_dt)

    def validate(self, poses, arm_path):
        """Arm IK of every fitted hand pose, seeded along the planned arm path.

        Without an arm path (a floating proposal) the IK runs along the path,
        each tick seeded by the previous solution, so the arm stays on one
        continuous branch instead of independent batch solutions. Only the arm
        links are checked here: the hand and carried object were already
        validated by the floating planner with its path-refined covers (the
        unrefined copies on `wrist` falsely collided along a valid carry).
        """
        spheres = self.motion_gen.robot_cfg.kinematics.kinematics_config
        spheres.disable_link_spheres('wrist')
        try:
            return self._validate(poses, arm_path)
        finally:
            spheres.enable_link_spheres('wrist')

    def _validate(self, poses, arm_path):
        if arm_path is None:
            seed = self.retract if self.last is None else self.last
            rows = []
            for pose in poses:
                rows.append(self.ik(pose[None], [seed]))
                seed = rows[-1][0][0]
            q, position, rotation, feasible = (np.concatenate([r[i] for r in rows]) for i in range(4))
        else:
            index = np.linspace(0, len(arm_path) - 1, len(poses))
            seeds = np.stack([np.interp(index, np.arange(len(arm_path)), arm_path[:, j]) for j in range(len(self.names))], 1)
            q, position, rotation, feasible = self.ik(poses, seeds)
        ok = (position <= self.tolerance[0]) & (rotation <= self.tolerance[1]) & feasible
        report = dict(ticks=len(poses), worst_position_error_m=float(position.max()), worst_rotation_error_rad=float(rotation.max()),
                      collision_or_limit_ticks=[int(k) for k in np.flatnonzero(~feasible)],
                      out_of_tolerance_ticks=[int(k) for k in np.flatnonzero((position > self.tolerance[0]) | (rotation > self.tolerance[1]))],
                      tolerance=list(self.tolerance), max_joint_step_rad=float(np.abs(np.diff(q, axis=0)).max()) if len(q) > 1 else 0.,
                      joints=q.tolist())
        (self.out / 'arm_validation.json').write_text(json.dumps(report))
        if not ok.all():
            self.rejections = getattr(self, 'rejections', 0) + 1
            (self.out / f'arm_validation_rejected_{self.rejections:02d}.json').write_text(
                json.dumps(dict(report, seeded_along='planned arm path' if arm_path is not None else 'previous tick')))
            k = int(np.flatnonzero(~ok)[0])
            reason = 'collision/limit' if not feasible[k] else f'IK error {position[k]:.4f} m/{rotation[k]:.3f} rad'
            raise RuntimeError(f'arm {reason} at fitted tick {k}')
        self.last = q[-1]
        return {k: v for k, v in report.items() if k != 'joints'}
