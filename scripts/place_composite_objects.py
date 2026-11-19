"""Visible objects composited inside the footprint of a planning-only box.

The rollout was planned around the virtual box (`place_planning_box.py`) and
validated in PhysX without any inserted object. Each library object is placed
upright on the real table, inside the box volume (footprint and height), in the first-person view,
and `--margin` outside every measured surface of that rollout at every frame
(both hands, tool, target) and of its rendered RB-Y1 arms (`--keep-surface`),
so the PhysX result is unchanged by drawing it there. Among those poses the one
that stands most in the way of the no-distractor approach (`--nominal-run`) is
kept, so the object sits where the demonstrated path would have gone. One
environment-object JSON per object (each a separate edited video of the same
rollout); `static` poses, never simulated.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from place_planning_box import box_distance, rollout_surfaces, voxel_centres


def load_obj(path):
    return np.array([l.split()[1:4] for l in Path(path).read_text().splitlines() if l.startswith('v ')], float)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True, help='final rollout planned around the box')
    p.add_argument('--box', type=Path, required=True, help='planning-only box JSON of that rollout')
    p.add_argument('--nominal-run', type=Path, required=True, help='no-distractor final rollout (the demonstrated approach)')
    p.add_argument('--schedule', type=Path, required=True)
    p.add_argument('--window', type=int, default=0)
    p.add_argument('--table-polygon', type=Path)
    p.add_argument('--table-xy', type=float, nargs=4, help='tabletop rectangle instead of a polygon')
    p.add_argument('--taco-camera', type=Path, help='TACO camera dir (view check; skipped without it)')
    p.add_argument('--keep-surface', type=Path, nargs='*', default=[], help='npz `points` kept clear at every frame (rendered arms)')
    p.add_argument('--object', action='append', required=True,
                   help='name=mesh_m.obj=w,x,y,z=yaws (rest orientation wxyz; yaw samples)')
    p.add_argument('--margin', type=float, default=.01)
    p.add_argument('--inset', type=float, default=.005, help='object footprint inside the box footprint by this')
    p.add_argument('--min-view', type=float, default=.9)
    p.add_argument('--step', type=float, default=.01)
    p.add_argument('--stride', type=int, default=8)
    p.add_argument('--certificate', type=Path, help='box_clearance.py output: with min_clearance_m >= --margin every '
                   'object inside the box volume is clear by construction and the per-object check is skipped')
    p.add_argument('--variants', type=int, default=1, help='distinct poses per object (most in the way first)')
    p.add_argument('--separation', type=float, default=.04, help='minimum xy distance between the variants of one object')
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args()

    box = json.loads(args.box.read_text())[0]
    bp = np.asarray(box['pose'], float)
    yaw_box = Rotation.from_quat(bp[[4, 5, 6, 3]]).as_euler('xyz')[2]
    pl = box['placement']
    half = np.array([pl.get('length_m', pl['size_m']) / 2, pl.get('width_m', pl['size_m']) / 2, pl['height_m'] / 2])
    result = json.loads((args.run_dir / 'pipeline_result.json').read_text())
    table_z = float(result['table_z'])

    frames = len(np.load(args.run_dir / 'visual_actual.npz')['transforms'])

    def surfaces(run, ks, match=None):
        select = lambda x: not x.startswith('/World/env_') and (match is None or any(m in x for m in match))
        return voxel_centres(rollout_surfaces(run, select, ks, points_per_mesh=1500), .005)
    keep = surfaces(args.run_dir, range(frames))
    keep = np.concatenate([keep] + [voxel_centres([np.load(f)['points'].astype(float)], .005) for f in args.keep_surface])
    w = json.loads(args.schedule.read_text())['windows'][args.window]
    nominal = surfaces(args.nominal_run, range(w['start'] + 1, w['end']), ['right_hand'])
    if args.table_polygon:
        poly = np.asarray(json.loads(args.table_polygon.read_text())['real_table_polygon_sim'], float)
    else:
        x0, x1, y0, y1 = args.table_xy
        poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    edges = ConvexHull(poly).equations
    if args.taco_camera:
        to_taco = np.linalg.inv(np.asarray(result['source_provenance']['transform_world_to_robot'], float))
        K = np.loadtxt(args.taco_camera / 'egocentric_intrinsic.txt')
        E = np.load(args.taco_camera / 'egocentric_frame_extrinsic.npy')[:frames]

    def in_view(point):
        if not args.taco_camera:
            return 1.
        q = to_taco[:3, :3] @ point + to_taco[:3, 3]
        c = E[:, :3, :3] @ q + E[:, :3, 3]
        z = np.maximum(c[:, 2], 1e-6)
        u, v = K[0, 0] * c[:, 0] / z + K[0, 2], K[1, 1] * c[:, 1] / z + K[1, 2]
        return float(((c[:, 2] > .05) & (u > .03 * 2 * K[0, 2]) & (u < .97 * 2 * K[0, 2]) &
                      (v > .03 * 2 * K[1, 2]) & (v < .97 * 2 * K[1, 2])).mean())

    certified = None
    if args.certificate:
        cert = json.loads(args.certificate.read_text())
        certified = cert['min_clearance_m'] if cert['min_clearance_m'] >= args.margin else None
        print('BOX_CERTIFICATE', json.dumps(dict(cert, certified=certified is not None)), flush=True)
    near_box = np.abs(keep[:, :2] - bp[:2]).max(1) < 2 * half[:2].max() + .2
    keep = keep[near_box]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # A box placed for a set of objects (placement.pocket.holds, occupancy names) holds only those; the ketchup
    # bottle is the mustard shape.
    holds = (pl.get('pocket') or {}).get('holds')
    for spec in args.object:
        name, mesh_path, quat, yaws = spec.split('=')
        if holds is not None and {'ketchup': 'mustard'}.get(name, name) not in holds:
            print('COMPOSITE_NONE', name, json.dumps(dict(reason='box not placed to hold it', holds=holds)), flush=True)
            continue
        rest = Rotation.from_quat(np.asarray(quat.split(','), float)[[1, 2, 3, 0]]).as_matrix()
        up = load_obj(mesh_path) @ rest.T
        up = up[ConvexHull(up).vertices]
        lift = table_z - up[:, 2].min() + 1e-3
        poses, tried = [], 0
        if np.ptp(up[:, 2]) + 1e-3 > 2 * half[2] - args.inset:
            # The object must fit inside the box volume: avoiding the box then implies avoiding the object.
            print('COMPOSITE_NONE', name, json.dumps(dict(object_height_m=float(np.ptp(up[:, 2])), box_height_m=float(2 * half[2]))), flush=True)
            continue
        for yaw in np.arange(int(yaws)) * 2 * np.pi / int(yaws):
            R = Rotation.from_euler('z', yaw).as_matrix()
            body = up @ R.T
            hull = ConvexHull(body)
            foot = body[body[:, 2] < body[:, 2].min() + .01, :2]
            radius = np.linalg.norm(body[:, :2], axis=1).max()
            reach = float(np.hypot(half[0], half[1]))  # the box may be rotated: scan its bounding disc
            for x in np.arange(bp[0] - reach, bp[0] + reach + 1e-9, args.step):
                for y in np.arange(bp[1] - reach, bp[1] + reach + 1e-9, args.step):
                    ring = body[:, :2] + [x, y]
                    flat = np.c_[ring, np.full(len(ring), bp[2])]
                    if box_distance(flat, bp[:3], yaw_box, half * [1, 1, 9]).max() > -args.inset:
                        continue  # outside the box footprint
                    if (-(foot + [x, y]) @ edges[:, :2].T - edges[:, 2]).min() < .005:
                        continue  # off the real table
                    tried += 1
                    offset = np.array([x, y, lift])
                    planes = hull.equations.copy()
                    planes[:, 3] -= planes[:, :3] @ offset
                    if certified is not None:
                        clear = certified  # inside the certified box volume: at least the box's own clearance
                    else:
                        close = keep[np.linalg.norm(keep[:, :2] - [x, y], axis=1) < radius + args.margin + .02]
                        clear = float((close @ planes[:, :3].T + planes[:, 3]).max(1).min()) - .0043 if len(close) else 1.
                        if clear < args.margin:
                            continue
                    seen = in_view(np.array([x, y, table_z + .5 * np.ptp(body[:, 2])]))
                    if seen < args.min_view:
                        continue
                    close = nominal[np.linalg.norm(nominal[:, :2] - [x, y], axis=1) < radius + .1]
                    # How far the demonstrated (no-distractor) approach would have run into the object.
                    cut = float(-(close @ planes[:, :3].T + planes[:, 3]).max(1).min()) if len(close) else -1.
                    poses.append(dict(x=float(x), y=float(y), yaw=float(yaw), clearance_m=clear, in_view=seen, nominal_cut_m=cut))
        if not poses:
            print('COMPOSITE_NONE', name, json.dumps(dict(tried=tried)), flush=True)
            continue
        # Most in the way of the demonstrated approach first; further variants stand `separation` apart.
        chosen = []
        for pose in sorted(poses, key=lambda q: -q['nominal_cut_m']):
            if all(np.hypot(pose['x'] - q['x'], pose['y'] - q['y']) >= args.separation for q in chosen):
                chosen.append(pose)
            if len(chosen) == args.variants:
                break
        for v, best in enumerate(chosen):
            rotation = Rotation.from_euler('z', best['yaw']).as_matrix() @ rest
            quat = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
            item = dict(name=name, id=name, mesh=mesh_path, pose=[best['x'], best['y'], float(lift), *quat.tolist()], static=True,
                        placement=dict(best, variant=v, clearance_source='box certificate' if certified is not None else 'per-object check', feasible_poses=len(poses), margin_m=args.margin, box=str(args.box), rollout=str(args.run_dir),
                                       scope='composited only: outside every measured surface and rendered arm at every frame; '
                                             'the rollout was planned around the planning-only box and simulated without it'))
            out = args.out_dir / (f'{name}.json' if v == 0 else f'{name}_{v + 1}.json')
            out.write_text(json.dumps([item], indent=1))
            print('COMPOSITE', str(out), json.dumps(best), flush=True)


if __name__ == '__main__':
    main()
