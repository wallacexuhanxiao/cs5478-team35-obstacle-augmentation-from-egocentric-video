"""Box-level certificate of a planning-only box after its rollout passed PhysX.

The box is never simulated, so the rollout passing does not by itself keep anything out of it. This measures, over
every frame of the actual rollout, how far the box stays from: the measured hand and carried object (tracking error
can enter it), the scene objects (target, and the tool at rest before pick-up and after release) and the rendered
RB-Y1 arm (`--keep-surface`; it is not in PhysX). With clearance >= m from all of them, any object that fits inside
the box volume is >= m clear of everything at every frame, without a per-object check. JSON on stdout.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from place_planning_box import box_distance, rollout_surfaces, voxel_centres


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--box', type=Path, required=True)
    p.add_argument('--keep-surface', type=Path, nargs='*', default=[], help='rendered arm npz (points)')
    args = p.parse_args()
    box = json.loads(args.box.read_text())[0]
    bp, pl = np.asarray(box['pose'], float), box['placement']
    half = np.array([pl.get('length_m', pl['size_m']), pl.get('width_m', pl['size_m']), pl['height_m']]) / 2
    yaw = Rotation.from_quat(bp[[4, 5, 6, 3]]).as_euler('xyz')[2]
    frames = len(np.load(args.run_dir / 'visual_actual.npz')['transforms'])
    out = dict(box=args.box.name, size_m=(2 * half).round(3).tolist())
    groups = {'hand': lambda x: 'right_hand' in x or 'left_hand' in x, 'tool': lambda x: x.startswith('/World/tool'),
              'target': lambda x: x.startswith('/World/target')}
    for name, select in groups.items():
        pts = voxel_centres(rollout_surfaces(args.run_dir, select, range(frames), points_per_mesh=1500), .005)
        if len(pts):
            d = box_distance(pts, bp[:3], yaw, half) - .0043  # voxel half-diagonal-ish
            out[f'{name}_clearance_m'] = float(d.min())
    for f in args.keep_surface:
        pts = voxel_centres([np.load(f)['points'].astype(float)], .005)
        out['rendered_arm_clearance_m'] = float((box_distance(pts, bp[:3], yaw, half) - .0043).min())
    out['min_clearance_m'] = min(v for k, v in out.items() if k.endswith('_clearance_m'))
    print(json.dumps(out))


if __name__ == '__main__':
    main()
