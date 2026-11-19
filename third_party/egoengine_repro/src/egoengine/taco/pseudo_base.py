"""Pseudo robot-base calibration for TACO (EgoEngine paper App. A.1).

The paper has no AprilTag for TACO, so it defines a heuristic robot base:
"we define the robot base from the scene geometry by taking the center of
the tool and target objects and applying a fixed 0.6 m offset. ... We set
the table height to 0.72 m".

TACO poses live in a mocap world frame with Z up (objects rest at
z ~ 0.5-0.7 m; camera centers ~1.3-1.6 m). Our instantiation:

- scene center c = mean over frames of (tool_t + target_t) / 2 (world, m)
- standing direction d = horizontal unit vector from c toward the mean
  camera position (the human/robot stands in front of the table facing the
  objects); fallback: world -Y if the camera is degenerate
- base origin = (c - 0.6 * d) in XY, and z = c_z - 0.72 so the table
  surface (approximated by the object center height) sits 0.72 m above the
  base frame — matching the real RB-Y1 workcell
- base orientation: Z up, X pointing from the base toward the scene center

These are heuristic choices for the unspecified offset direction; they are
recorded in docs/TACO_LOADER.md and can be revisited once the sim scene
is built.
"""

from __future__ import annotations

import numpy as np


def compute_pseudo_base(sequence, offset: float = 0.6, table_height: float = 0.72) -> np.ndarray:
    """Robot base frame (4x4, base-to-world) for a TacoSequence.

    Args:
        sequence: TacoSequence with tool + target objects loaded.
        offset: horizontal distance (m) from scene center to the base.
        table_height: table surface height above the base frame (m).
    """
    tool_t = sequence.objects["tool"].poses[:, :3, 3]
    target_t = sequence.objects["target"].poses[:, :3, 3]
    center = 0.5 * (tool_t + target_t)
    c = center.mean(axis=0)

    cam_centers = np.stack([sequence.camera_position(i) for i in range(sequence.n_frames)])
    cam_mean = cam_centers.mean(axis=0)

    d = cam_mean - c
    d[2] = 0.0
    norm = np.linalg.norm(d)
    d = d / norm if norm > 1e-6 else np.array([0.0, -1.0, 0.0])

    origin = np.array([c[0] - offset * d[0], c[1] - offset * d[1], c[2] - table_height])

    x_axis = -d  # base X points toward the scene center
    z_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    T = np.eye(4, dtype=np.float64)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis
    T[:3, 3] = origin
    return T
