"""Reference-quality and motion-progress diagnostics for chunk refinement.

The reference gate is intentionally independent from pose feasibility.  A
stable grasp implies that the reference hand-to-tool transform should not make
an abrupt single-frame jump.  Such a jump marks the reference interval as
invalid and must not be sent to residual PPO.

Motion diagnostics are read-only in v0.  They describe whether an executed
tool trajectory actually advances through a trustworthy, non-trivial
reference path; they do not yet participate in solver selection.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


REFERENCE_RELATIVE_TRANSLATION_STEP_M = 0.02
REFERENCE_RELATIVE_ROTATION_STEP_RAD = 0.35
REFERENCE_MOTION_PATH_MIN_M = 0.02
MOTION_PATH_RATIO_MIN = 0.50
MOTION_PHASE_PROGRESS_MIN = 0.25
MOTION_TANGENT_ALIGNMENT_MIN = 0.20
CUT_LONG_AXIS_PROGRESS_MIN = 0.40


def _geodesic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    rel = np.einsum("...ji,...jk->...ik", a, b)
    cosine = np.clip(
        (np.trace(rel, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0
    )
    return np.arccos(cosine)


def qpos19_to_transform(qpos: np.ndarray) -> np.ndarray:
    """Convert ``[xyz, quat_wxyz, fingers]`` XHand qpos to SE(3)."""
    qpos = np.asarray(qpos, dtype=np.float64)
    out = np.broadcast_to(np.eye(4), qpos.shape[:-1] + (4, 4)).copy()
    quat_xyzw = np.concatenate([qpos[..., 4:7], qpos[..., 3:4]], axis=-1)
    out[..., :3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    out[..., :3, 3] = qpos[..., :3]
    return out


def pose_xyzw_to_transform(pose: np.ndarray) -> np.ndarray:
    """Convert ``[xyz, quat_xyzw]`` recorded PhysX poses to SE(3)."""
    pose = np.asarray(pose, dtype=np.float64)
    out = np.broadcast_to(np.eye(4), pose.shape[:-1] + (4, 4)).copy()
    out[..., :3, :3] = Rotation.from_quat(pose[..., 3:7]).as_matrix()
    out[..., :3, 3] = pose[..., :3]
    return out


def reference_relative_pose_steps(
    hand_qpos: np.ndarray, tool_reference: np.ndarray
) -> dict[str, np.ndarray]:
    """Per-frame steps of ``inverse(T_hand_ref) @ T_tool_ref``.

    Returned step index ``i`` describes the transition from reference frame
    ``i`` to ``i + 1``.
    """
    hand = qpos19_to_transform(hand_qpos)
    tool = np.asarray(tool_reference, dtype=np.float64)
    relative = np.linalg.inv(hand) @ tool
    translation = np.linalg.norm(
        np.diff(relative[:, :3, 3], axis=0), axis=1
    )
    rotation = _geodesic(
        relative[:-1, :3, :3], relative[1:, :3, :3]
    )
    return {
        "relative_transform": relative,
        "translation_step_m": translation,
        "rotation_step_rad": rotation,
    }


def reference_validity_gate(
    hand_qpos: np.ndarray,
    tool_reference: np.ndarray,
    *,
    frame_start: int = 0,
    stable_grasp: bool,
    grasp_mask: np.ndarray | None = None,
    translation_step_limit_m: float = REFERENCE_RELATIVE_TRANSLATION_STEP_M,
    rotation_step_limit_rad: float = REFERENCE_RELATIVE_ROTATION_STEP_RAD,
) -> dict:
    """Flag abrupt hand-relative tool-reference jumps during a stable grasp."""
    steps = reference_relative_pose_steps(hand_qpos, tool_reference)
    bad = (
        (steps["translation_step_m"] > translation_step_limit_m)
        | (steps["rotation_step_rad"] > rotation_step_limit_rad)
    )
    if grasp_mask is not None:
        grasp_mask = np.asarray(grasp_mask, dtype=bool)
        if len(grasp_mask) != len(hand_qpos):
            raise ValueError("grasp_mask must have one value per reference frame")
        stable_steps = grasp_mask[:-1] & grasp_mask[1:]
        bad &= stable_steps
        active = bool(stable_steps.any())
    else:
        active = bool(stable_grasp)
    local = np.flatnonzero(bad) + 1
    suspect = (local + int(frame_start)).astype(int)
    return {
        "active": active,
        "valid": bool(not active or suspect.size == 0),
        "status": "REFERENCE_INVALID" if active and suspect.size else "REFERENCE_VALID",
        "suspect_frames": suspect.tolist() if active else [],
        "translation_step_limit_m": float(translation_step_limit_m),
        "rotation_step_limit_rad": float(rotation_step_limit_rad),
        "max_relative_translation_step_m": float(
            steps["translation_step_m"].max(initial=0.0)
        ),
        "max_relative_rotation_step_rad": float(
            steps["rotation_step_rad"].max(initial=0.0)
        ),
    }


def export_reference_review_frames(
    video_path: str,
    suspect_frames: list[int],
    out_dir,
    *,
    context: int = 1,
) -> list[str]:
    """Export suspect ego frames and immediate context for manual review."""
    import cv2
    from pathlib import Path

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    wanted = sorted({
        max(0, frame + offset)
        for frame in suspect_frames
        for offset in range(-context, context + 1)
    })
    cap = cv2.VideoCapture(str(video_path))
    written = []
    for frame_index in wanted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        path = destination / f"frame_{frame_index:04d}.jpg"
        cv2.putText(
            frame,
            f"reference review frame {frame_index}",
            (24, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(path), frame)
        written.append(str(path))
    cap.release()
    return written


def _greedy_monotonic_vertex_phase(
    actual_xyz: np.ndarray, reference_xyz: np.ndarray, max_advance: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Project actual samples to a monotonic sequence of reference vertices.

    The start is anchored near the beginning of the chunk.  Each subsequent
    sample may remain at its phase or advance by at most ``max_advance``
    vertices.  This avoids crediting a stationary trajectory merely because a
    back-and-forth reference curve later revisits the same point.
    """
    actual_xyz = np.asarray(actual_xyz, dtype=np.float64)
    reference_xyz = np.asarray(reference_xyz, dtype=np.float64)
    n_ref = len(reference_xyz)
    if n_ref <= 1:
        return np.zeros(len(actual_xyz), dtype=int), np.zeros(len(actual_xyz))
    start_cap = min(n_ref, max(2, int(np.ceil(0.1 * n_ref))))
    indices = np.zeros(len(actual_xyz), dtype=int)
    indices[0] = int(np.argmin(
        np.linalg.norm(reference_xyz[:start_cap] - actual_xyz[0], axis=1)
    ))
    for i in range(1, len(actual_xyz)):
        lo = indices[i - 1]
        hi = min(n_ref, lo + max_advance + 1)
        distances = np.linalg.norm(reference_xyz[lo:hi] - actual_xyz[i], axis=1)
        indices[i] = lo + int(np.argmin(distances))
    phase = indices.astype(np.float64) / float(n_ref - 1)
    return indices, phase


def _axis_motion(T: np.ndarray, axis: np.ndarray) -> float:
    directions = np.einsum("nij,j->ni", T[:, :3, :3], axis)
    dots = np.sum(directions[:-1] * directions[1:], axis=1)
    return float(np.arccos(np.clip(dots, -1.0, 1.0)).sum())


def motion_acceptance_diagnostic(
    actual_tool_pose_xyzw: np.ndarray,
    tool_reference: np.ndarray,
    *,
    task: str | None = None,
    path_min_m: float = REFERENCE_MOTION_PATH_MIN_M,
) -> dict:
    """Compute read-only motion execution metrics for one refinement window."""
    actual = pose_xyzw_to_transform(actual_tool_pose_xyzw)
    reference = np.asarray(tool_reference, dtype=np.float64)
    actual_xyz = actual[:, :3, 3]
    reference_xyz = reference[:, :3, 3]
    da = np.diff(actual_xyz, axis=0)
    dr = np.diff(reference_xyz, axis=0)
    actual_step = np.linalg.norm(da, axis=1)
    reference_step = np.linalg.norm(dr, axis=1)
    reference_path = float(reference_step.sum())
    actual_path = float(actual_step.sum())

    indices, phase = _greedy_monotonic_vertex_phase(
        actual_xyz, reference_xyz
    )
    tangent_cosines = []
    tangent_weights = []
    for i in range(len(da)):
        if actual_step[i] <= 1e-6:
            continue
        j = min(indices[i], len(dr) - 1)
        if reference_step[j] <= 1e-6:
            continue
        tangent_cosines.append(
            float(np.dot(da[i], dr[j]) / (actual_step[i] * reference_step[j]))
        )
        tangent_weights.append(float(reference_step[j]))
    tangent_alignment = (
        float(np.average(tangent_cosines, weights=tangent_weights))
        if tangent_cosines else None
    )

    reference_rotation = float(_geodesic(
        reference[:-1, :3, :3], reference[1:, :3, :3]
    ).sum())
    actual_rotation = float(_geodesic(
        actual[:-1, :3, :3], actual[1:, :3, :3]
    ).sum())
    result = {
        "reference_motion_active": bool(reference_path > path_min_m),
        "reference_path_m": reference_path,
        "actual_path_m": actual_path,
        "path_ratio": actual_path / reference_path if reference_path else None,
        "projected_phase_start": float(phase[0]),
        "projected_phase_end": float(phase[-1]),
        "projected_phase_progress": float(phase[-1] - phase[0]),
        "local_tangent_alignment": tangent_alignment,
        "reference_rotation_path_rad": reference_rotation,
        "actual_rotation_path_rad": actual_rotation,
        "rotation_progress_ratio": (
            actual_rotation / reference_rotation if reference_rotation else None
        ),
    }
    if task == "cut":
        # Knife object frame: x is the long/cutting direction; z is the thin
        # blade normal.  Report them independently so a noisy long-axis twist
        # is not hidden inside one SO(3) scalar.
        for label, axis in (
            ("long_axis", np.array([1.0, 0.0, 0.0])),
            ("blade_normal", np.array([0.0, 0.0, 1.0])),
        ):
            ref_motion = _axis_motion(reference, axis)
            act_motion = _axis_motion(actual, axis)
            result[f"reference_{label}_motion_rad"] = ref_motion
            result[f"actual_{label}_motion_rad"] = act_motion
            result[f"{label}_progress_ratio"] = (
                act_motion / ref_motion if ref_motion else None
            )
    severe_translation_under_motion = bool(
        result["reference_motion_active"]
        and result["path_ratio"] < MOTION_PATH_RATIO_MIN
        and result["projected_phase_progress"] < MOTION_PHASE_PROGRESS_MIN
        and (
            result["local_tangent_alignment"] is None
            or result["local_tangent_alignment"] < MOTION_TANGENT_ALIGNMENT_MIN
        )
    )
    cut_orientation_under_motion = bool(
        task == "cut"
        and result.get("reference_long_axis_motion_rad", 0.0) > 0.15
        and result.get("long_axis_progress_ratio", 1.0)
        < CUT_LONG_AXIS_PROGRESS_MIN
    )
    result["proposed_rule"] = {
        "read_only": True,
        "severe_translation_under_motion": severe_translation_under_motion,
        "cut_orientation_under_motion": cut_orientation_under_motion,
        "would_accept": not (
            severe_translation_under_motion or cut_orientation_under_motion
        ),
        "thresholds": {
            "path_ratio_min": MOTION_PATH_RATIO_MIN,
            "phase_progress_min": MOTION_PHASE_PROGRESS_MIN,
            "tangent_alignment_min": MOTION_TANGENT_ALIGNMENT_MIN,
            "cut_long_axis_progress_min": CUT_LONG_AXIS_PROGRESS_MIN,
        },
        "logic": (
            "reject severe translation only when path ratio, projected phase, "
            "and tangent alignment are all low; for Cut also reject inadequate "
            "long-axis/cutting-direction progress"
        ),
    }
    return result
