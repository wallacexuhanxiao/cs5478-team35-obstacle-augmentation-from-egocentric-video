"""Virtual box across the measured free approach, for planning only.

The box stands on the real (video) table, cuts at least `--depth` into the
right-hand surfaces of the measured approach of a no-distractor rollout, and
keeps the fixed start and grasp-entry poses `--end-clearance` outside, so cuRobo
re-plans the approach around it. It is never physical and never drawn: smaller
visible objects are later composited inside its footprint wherever the actual
rollout (every frame, both hands, tool, target, rendered arms) leaves room
(`place_composite_objects.py`). The box may overlap where the hands go after the
approach, because it only exists while that approach is planned. With
`--pocket-object` a box is kept only if it also contains a pocket: a pose of a
library object (upright, on the real table, in view) that stays `--pocket-margin`
clear of every measured surface outside the approach and of the rendered arms
(`--keep-surface`) at every frame, so a visible object can stand inside the
box; the box may extend from the approach corridor into such free space.
Geometry only; cuRobo plans the detour and PhysX decides.

Detour strategy (`--side`): relative to the measured corridor (per-frame centroid of the re-planned surfaces, local
normal; "inner" = the side of `--inner-point`, e.g. the demonstrator/camera), a box on the outer side asks for a
detour to the inner side and vice versa; `center` keeps the box centred on the corridor and spanning `--span` past
it on both sides, so that (with a low `--box-height`) going over is the short way. `--plate` keeps the footprint
clear of a plate item (policy_plate.py).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation


def box_distance(points, centre, yaw, half):
    """Signed distance (approximate outside, exact inside) of points to an upright box."""
    c, s = np.cos(yaw), np.sin(yaw)
    local = (points - centre) @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    q = np.abs(local) - half
    outside = np.linalg.norm(np.maximum(q, 0.), axis=1)
    return outside + np.minimum(q.max(1), 0.)


def voxel_centres(point_sets, voxel=.01):
    """Centres of the voxels hit by any point of the given point sets (1-D keys: fast union over many frames)."""
    keys = []
    for pts in point_sets:
        q = np.floor(np.asarray(pts) / voxel).astype(np.int64) + 100000
        keys.append(np.unique(q[:, 0] * (1 << 40) + q[:, 1] * (1 << 20) + q[:, 2]))
    k = np.unique(np.concatenate(keys)) if keys else np.zeros(0, np.int64)
    xyz = np.stack([k >> 40, (k >> 20) & ((1 << 20) - 1), k & ((1 << 20) - 1)], 1) - 100000
    return (xyz + .5) * voxel


def rollout_surfaces(run, select, frames, points_per_mesh=600):
    """Per-frame measured surfaces (sim frame) of the meshes whose path passes `select`, subsampled per mesh."""
    meshes = np.load(run / 'visual_meshes.npz')
    paths = json.loads((run / 'visual_mesh_paths.json').read_text())
    poses = np.load(run / 'visual_actual.npz')['transforms'].astype(float)
    ids = [i for i, x in enumerate(paths) if select(x)]
    local = [meshes[f'v{i}'][::max(1, len(meshes[f'v{i}']) // points_per_mesh)].astype(float) for i in ids]
    for k in frames:
        if k < len(poses):
            yield (np.concatenate([v @ poses[k, i, :3, :3].T + poses[k, i, :3, 3] for v, i in zip(local, ids)]) if ids
                   else np.zeros((0, 3)))  # e.g. no second object or left hand in the scene


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True, help='no-distractor final rollout dir')
    p.add_argument('--schedule', type=Path, required=True)
    p.add_argument('--window', type=int, default=0, help='RELAXED window of the approach')
    p.add_argument('--block-match', nargs='*', default=['right_hand'], help='meshes of the re-planned hand')
    p.add_argument('--table-polygon', type=Path, help='JSON with real_table_polygon_sim')
    p.add_argument('--table-xy', type=float, nargs=4, help='xmin xmax ymin ymax tabletop rectangle instead of a polygon')
    p.add_argument('--taco-camera', type=Path, help='TACO camera dir (view check; skipped without it)')
    p.add_argument('--sizes', nargs='+', default=['0.14x0.14', '0.12x0.12', '0.18x0.18', '0.10x0.10', '0.22x0.10', '0.26x0.10', '0.30x0.10', '0.34x0.12', '0.30x0.16', '0.40x0.14'],
                   help='footprints LxW (m), preferred first; elongated boxes can reach from the approach to free space')
    p.add_argument('--depth', type=float, default=.015, help='minimum cut into the measured approach')
    p.add_argument('--max-depth', type=float, default=1., help='maximum cut (a shallow box keeps the detour, and the carried grasp, small)')
    p.add_argument('--end-clearance', type=float, default=.025, help='start and entry poses outside the box by this')
    p.add_argument('--entry-frames', type=int, default=3, help='last approach frames kept clear (planner goal)')
    p.add_argument('--height-margin', type=float, default=.03, help='box top above the highest blocked hand point')
    p.add_argument('--box-height', type=float, help='fixed box height (m): a low, wide box makes going over cheaper than around')
    p.add_argument('--min-view', type=float, default=.9)
    p.add_argument('--step', type=float, default=.015)
    p.add_argument('--yaws', type=int, default=8, help='samples over half a turn (rectangles) or a quarter (squares)')
    p.add_argument('--count', type=int, default=1)
    p.add_argument('--min-separation', type=float, default=.08)
    p.add_argument('--stride', type=int, default=6)
    p.add_argument('--occupancy', type=Path, help='occupancy_ceiling.py output dir: the box must contain a free cell of an object')
    p.add_argument('--pocket-objects', nargs='*', default=['mustard', 'mug', 'wine_bottle'],
                   help='objects this box will hold (composited later): the box is at least as tall as the tallest one '
                        '(+ --contain-margin) and must have a free pose inside for every one of them')
    p.add_argument('--contain-margin', type=float, default=.01, help='box top above the tallest held object (m)')
    p.add_argument('--side', choices=('any', 'outer', 'inner', 'center'), default='any',
                   help='box side of the corridor: outer -> detour to the inner side; inner -> to the outer side; center -> over')
    p.add_argument('--inner-point', type=float, nargs=2, help='xy on the inner side (demonstrator/camera)')
    p.add_argument('--min-offset', type=float, default=.03, help='outer/inner: box centre at least this far off the corridor centre')
    p.add_argument('--center-tolerance', type=float, default=.04)
    p.add_argument('--span', type=float, default=.10, help='center: footprint reaches this far past the corridor centre on both sides')
    p.add_argument('--clear-frames', nargs='*', default=[], help='a:b frame ranges (fixed states: start, Anchors) the box must also '
                   'stay --end-clearance clear of; the box may then cut other re-planned windows (e.g. the approach) too')
    p.add_argument('--plate', type=Path, help='plate item: footprint kept --plate-margin clear of it')
    p.add_argument('--plate-margin', type=float, default=.01)
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args()

    window = json.loads(args.schedule.read_text())['windows'][args.window]
    result = json.loads((args.run_dir / 'pipeline_result.json').read_text())
    table_z = float(result['table_z'])
    meshes = np.load(args.run_dir / 'visual_meshes.npz')
    paths = json.loads((args.run_dir / 'visual_mesh_paths.json').read_text())
    poses = np.load(args.run_dir / 'visual_actual.npz')['transforms'].astype(float)
    ids = [i for i, x in enumerate(paths) if any(m in x for m in args.block_match)]
    local = [meshes[f'v{i}'][::args.stride].astype(float) for i in ids]

    def surface(k):
        return np.concatenate([v @ poses[k, i, :3, :3].T + poses[k, i, :3, 3] for v, i in zip(local, ids)])
    a, b = window['start'], window['end']
    def voxels(points):  # 1 cm grid (centres off the surface by at most half a diagonal; charged below)
        return (np.unique(np.floor(points / .01), axis=0) + .5) * .01
    blocked = voxels(np.concatenate([surface(k) for k in range(a + 1, b - args.entry_frames)]))
    fixed = [a, a + 1] + list(range(b - args.entry_frames, b + 1))
    for spec in args.clear_frames:
        f0, f1 = map(int, spec.split(':'))
        fixed += [k for k in range(f0, f1) if k < len(poses)]
    ends = voxels(np.concatenate([surface(k) for k in sorted(set(fixed))]))
    blocked_tree, ends_tree = cKDTree(blocked[:, :2]), cKDTree(ends[:, :2])
    # Corridor centre line and outward normal (per frame of the re-planned window).
    cents = np.array([surface(k)[:, :2].mean(0) for k in range(a + 1, b - args.entry_frames)])
    tang = np.gradient(cents, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    normal = np.c_[-tang[:, 1], tang[:, 0]]
    if args.inner_point is not None:
        mid = len(cents) // 2
        if (cents[mid] - np.asarray(args.inner_point)) @ normal[mid] < 0:
            normal = -normal  # normal now points to the outer side
    elif args.side != 'any':
        raise ValueError('--side needs --inner-point')
    cent_tree = cKDTree(cents)
    plate = json.loads(args.plate.read_text()) if args.plate else None

    def corridor_offset(corners_xy, centre_xy):
        k = int(cent_tree.query(centre_xy)[1])
        return float((centre_xy - cents[k]) @ normal[k]), (corners_xy - cents[k]) @ normal[k], k / max(1, len(cents) - 1)
    if args.table_polygon:
        poly = np.asarray(json.loads(args.table_polygon.read_text())['real_table_polygon_sim'], float)
    else:
        x0, x1, y0, y1 = args.table_xy
        poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    edges = ConvexHull(poly).equations
    if args.taco_camera:
        to_taco = np.linalg.inv(np.asarray(result['source_provenance']['transform_world_to_robot'], float))
        K = np.loadtxt(args.taco_camera / 'egocentric_intrinsic.txt')
        E = np.load(args.taco_camera / 'egocentric_frame_extrinsic.npy')[:len(poses)]

    def in_view(point):
        if not args.taco_camera:
            return 1.
        w = to_taco[:3, :3] @ point + to_taco[:3, 3]
        c = E[:, :3, :3] @ w + E[:, :3, 3]
        z = np.maximum(c[:, 2], 1e-6)
        u, v = K[0, 0] * c[:, 0] / z + K[0, 2], K[1, 1] * c[:, 1] / z + K[1, 2]
        return float(((c[:, 2] > .05) & (u > .03 * 2 * K[0, 2]) & (u < .97 * 2 * K[0, 2]) &
                      (v > .03 * 2 * K[1, 2]) & (v < .97 * 2 * K[1, 2])).mean())

    pockets = None  # free object cells from `occupancy_ceiling.py`: (name, height, radius, xy[N,2], gap to approach[N])
    if args.occupancy:
        grid = np.load(args.occupancy / 'occupancy.npz')
        info = json.loads((args.occupancy / 'occupancy.json').read_text())
        origin, cell = grid['origin'], float(grid['cell'])
        pockets = []
        for name in args.pocket_objects:
            ij = np.argwhere(grid[f'fits_{name}'])
            if not len(ij):
                continue
            xy = origin + ij * cell
            gap = cKDTree(blocked[:, :2]).query(xy)[0]
            pockets.append((name, info[name]['height_m'], info[name]['radius_m'], xy, gap))
        print('POCKETS', json.dumps({q[0]: len(q[3]) for q in pockets}), flush=True)
        if not pockets:
            raise RuntimeError('No library object fits anywhere clear of the replayed motion')

    reach = .1 if args.occupancy is None else .18  # elongated boxes may centre beside the approach
    lo, hi = blocked[:, :2].min(0) - reach, blocked[:, :2].max(0) + reach
    rows, rejected = [], dict(off_table=0, no_approach=0, shallow=0, near_ends=0, out_of_view=0)
    deepest = {}
    for spec in args.sizes:
        length, width = map(float, spec.split('x'))
        size = max(length, width)
        turn = np.pi / 2 if length == width else np.pi
        for yaw in np.arange(args.yaws) * turn / args.yaws:
            c, s = np.cos(yaw), np.sin(yaw)
            corners = np.array([[sx, sy] for sx in (-1, 1) for sy in (-1, 1)]) * [length / 2, width / 2] @ np.array([[c, s], [-s, c]])
            for x in np.arange(lo[0], hi[0], args.step):
                for y in np.arange(lo[1], hi[1], args.step):
                    if args.occupancy is None and (-(corners + [x, y]) @ edges[:, :2].T - edges[:, 2]).min() < 0:
                        rejected['off_table'] += 1
                        continue  # footprint off the real table
                    off, corner_off, along = corridor_offset(corners + [x, y], np.array([x, y]))
                    if args.side == 'outer' and off < args.min_offset or args.side == 'inner' and off > -args.min_offset or \
                            args.side == 'center' and (abs(off) > args.center_tolerance or corner_off.min() > -args.span or corner_off.max() < args.span):
                        rejected['side'] = rejected.get('side', 0) + 1
                        continue
                    if plate is not None:
                        rim = np.array([corners[i] + t * (corners[[0, 1, 3, 2]][(j + 1) % 4] - corners[i]) for j, i in enumerate([0, 1, 3, 2])
                                        for t in np.linspace(0, 1, 6)]) + [x, y]
                        inside_plate = (np.abs((np.asarray(plate['pose'][:2]) - [x, y]) @ np.array([[c, -s], [s, c]])) <= [length / 2, width / 2]).all()
                        if inside_plate or np.linalg.norm(rim - plate['pose'][:2], axis=1).min() < plate['placement']['radius_m'] + args.plate_margin:
                            rejected['plate'] = rejected.get('plate', 0) + 1
                            continue
                    near = blocked[blocked_tree.query_ball_point([x, y], .71 * size + .01)]
                    if not len(near):
                        rejected['no_approach'] += 1
                        continue
                    inside = near[box_distance(near, np.array([x, y, 0.]), yaw, np.array([length / 2, width / 2, 9.])) < 0]
                    if not len(inside):
                        rejected['no_approach'] += 1
                        continue
                    height = args.box_height or max(.08, float(inside[:, 2].max() - table_z + args.height_margin))
                    if pockets is not None:
                        # The box stands for every object it will hold: planning around it must also clear the tallest.
                        height = max(height, max(q[1] for q in pockets) + args.contain_margin)
                    centre = np.array([x, y, table_z + .001 + height / 2])
                    half = np.array([length / 2, width / 2, height / 2])
                    depth = float(-box_distance(near, centre, yaw, half).min()) - .0087
                    if depth < args.depth:
                        rejected['shallow'] += 1
                        continue
                    if depth > args.max_depth:
                        rejected['deep'] = rejected.get('deep', 0) + 1
                        continue
                    close = ends[ends_tree.query_ball_point([x, y], .71 * size + args.end_clearance + .02)]
                    clear = float(box_distance(close, centre, yaw, half).min()) - .0087 if len(close) else 1.
                    deepest[clear] = max(deepest.get(clear, -1), depth)
                    if clear < args.end_clearance:
                        rejected['near_ends'] += 1
                        continue
                    view = in_view(centre)
                    if args.occupancy is None and view < args.min_view:
                        rejected['out_of_view'] += 1
                        continue
                    pocket = None
                    if pockets is not None:
                        # Every held object needs a free pose inside the box (its height is already below the top).
                        held = {}
                        for name, h, r, xy, gap in pockets:
                            local = (xy - [x, y]) @ np.array([[c, -s], [s, c]])  # world -> box frame
                            inside = (np.abs(local) <= np.array([length / 2, width / 2]) - r - .005).all(1)
                            if inside.any():
                                j = np.flatnonzero(inside)[np.argmin(gap[inside])]
                                held[name] = dict(object=name, x=float(xy[j, 0]), y=float(xy[j, 1]), gap_to_approach_m=float(gap[j]))
                        if len(held) < len(pockets):
                            rejected['no_pocket'] = rejected.get('no_pocket', 0) + 1
                            continue
                        pocket = min(held.values(), key=lambda q: q['gap_to_approach_m'])
                        pocket = dict(pocket, holds=sorted(held))
                    rows.append(dict(x=float(x), y=float(y), yaw=float(yaw), size_m=spec, length_m=length, width_m=width, height_m=height,
                                     depth_m=depth, end_clearance_m=clear, in_view=view, side=args.side,
                                     corridor_offset_m=off, corridor_along=along, pocket=pocket))
    preference = {s: i for i, s in enumerate(args.sizes)}
    # With pockets: the visible object nearest the demonstrated approach first (the detour looks caused by it).
    rows.sort(key=lambda r: (r['pocket']['gap_to_approach_m'] if r.get('pocket') else 0., preference[r['size_m']], -r['depth_m']))
    picked = []
    for r in rows:
        if len(picked) == args.count:
            break
        if all(np.hypot(r['x'] - q['x'], r['y'] - q['y']) >= args.min_separation for q in picked):
            picked.append(r)
    best_clear = max(deepest) if deepest else None
    print('BOX_SEARCH', json.dumps(dict(feasible=len(rows), picked=len(picked), window=[a, b], rejected=rejected,
                                        best_end_clearance_with_depth_m=best_clear)), flush=True)
    if not picked:
        raise RuntimeError('No box cuts the approach while keeping the start and entry clear on the real table in view')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for k, r in enumerate(picked):
        mesh_path = args.out_dir / f'box_{k}.obj'
        trimesh.creation.box(extents=[r['length_m'], r['width_m'], r['height_m']]).export(mesh_path)
        quat = Rotation.from_euler('z', r['yaw']).as_quat()[[3, 0, 1, 2]]
        item = dict(name='box', id='planning_box', mesh=str(mesh_path), planning_only=True,
                    pose=[r['x'], r['y'], table_z + .001 + r['height_m'] / 2, *quat.tolist()],
                    placement=dict(r, depth_required_m=args.depth, end_clearance_required_m=args.end_clearance,
                                   source_run=str(args.run_dir),
                                   scope='planning-only virtual box: shapes the approach plan; never physical or drawn'))
        out = args.out_dir / f'box_{k}.json'
        out.write_text(json.dumps([item], indent=1))
        print('BOX', str(out), json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
