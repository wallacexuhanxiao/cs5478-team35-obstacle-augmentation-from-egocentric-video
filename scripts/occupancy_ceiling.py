"""Where can an object of a given height stand without touching the replayed motion?

Set analysis on a 1 cm tabletop grid (sim frame) of one validated rollout:
  * ceiling(c):  lowest height above the table at which anything that is not
    re-planned (both hands outside the re-planned window, the replayed hand's
    whole Anchors, tool, target) or the rendered RB-Y1 arms passes over cell c
    during the clip;
  * corridor(c): lowest height of the re-planned hand over c inside that window
    (the demonstrated free approach), +inf where it never passes.
An upright object of height h whose footprint (disc of radius r) covers cells D
fits when min ceiling(D) > h + margin; it also *blocks* the approach when
min corridor(D) < h. Free and blocking maps for each object are drawn over the
first video frame; `occupancy.json` holds the grids for placement tools.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import minimum_filter
from scipy.spatial import ConvexHull

from place_planning_box import rollout_surfaces


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--schedule', type=Path, required=True)
    p.add_argument('--window', type=int, default=0, help='re-planned RELAXED window')
    p.add_argument('--extra-free', type=int, nargs=2, help='[a, b) more re-planned frames of that hand (e.g. a retreat)')
    p.add_argument('--block-match', default='right_hand')
    p.add_argument('--free-tool', action='store_true', help='the tool is carried (re-planned) in the free frames')
    p.add_argument('--keep-surface', type=Path, nargs='*', default=[], help='rendered arm npz (points, frame)')
    p.add_argument('--table-polygon', type=Path)
    p.add_argument('--table-xy', type=float, nargs=4, help='tabletop rectangle instead of a polygon')
    p.add_argument('--taco-root', type=Path, help='TACO root (view check and overlay; skipped without it)')
    p.add_argument('--triplet')
    p.add_argument('--seq')
    p.add_argument('--objects', nargs='+', default=['mug:0.09:0.123:0.093', 'mustard:0.19:0.10:0.061', 'wine_bottle:0.30:0.075:0.075'],
                   help='name:height_m:footprint_length_m:footprint_width_m (rectangle, tried at --yaws orientations)')
    p.add_argument('--yaws', type=int, default=8, help='footprint orientations over half a turn')
    p.add_argument('--margin', type=float, default=.01, help='clearance to the replayed motion, sideways and above')
    p.add_argument('--depth', type=float, default=.01, help='the approach passes this far inside the footprint')
    p.add_argument('--min-view', type=float, default=.9)
    p.add_argument('--cell', type=float, default=.01)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()

    result = json.loads((args.run_dir / 'pipeline_result.json').read_text())
    table_z = float(result['table_z'])
    frames = len(np.load(args.run_dir / 'visual_actual.npz')['transforms'])
    w = json.loads(args.schedule.read_text())['windows'][args.window]
    free = set(range(w['start'] + 1, w['end']))
    if args.extra_free:
        free |= set(range(*args.extra_free))
    kept = [k for k in range(frames) if k not in free]
    # With --free-tool the carried tool is re-planned with the hand (a grasped RELAXED window): it is part of the
    # corridor there and kept only outside the free frames.
    hand = lambda x: args.block_match in x or (args.free_tool and x.startswith('/World/tool'))
    other = lambda x: not x.startswith('/World/env_') and not hand(x)
    static = np.concatenate([*rollout_surfaces(args.run_dir, other, range(frames), 400),
                             *rollout_surfaces(args.run_dir, hand, kept, 400)] +
                            [np.load(f)['points'][np.isin(np.load(f)['frame'], kept)].astype(float) for f in args.keep_surface])
    corridor_pts = np.concatenate(list(rollout_surfaces(args.run_dir, hand, sorted(free), 400)))

    if args.table_polygon:
        poly = np.asarray(json.loads(args.table_polygon.read_text())['real_table_polygon_sim'], float)
    else:
        x0, x1, y0, y1 = args.table_xy
        poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    lo = np.minimum(poly.min(0), corridor_pts[:, :2].min(0)) - .05
    hi = np.maximum(poly.max(0), corridor_pts[:, :2].max(0)) + .05
    shape = tuple(np.ceil((hi - lo) / args.cell).astype(int) + 1)

    def lowest(points):
        grid = np.full(shape, np.inf)
        ij = np.floor((points[:, :2] - lo) / args.cell).astype(int)
        ok = (ij >= 0).all(1) & (ij < shape).all(1)
        np.minimum.at(grid, (ij[ok, 0], ij[ok, 1]), points[ok, 2] - table_z)
        return grid
    ceiling, corridor = lowest(static), lowest(corridor_pts)
    xs, ys = lo[0] + np.arange(shape[0]) * args.cell, lo[1] + np.arange(shape[1]) * args.cell
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    eq = ConvexHull(poly).equations
    on_table = (-(np.stack([X, Y], -1) @ eq[:, :2].T) - eq[:, 2]).min(-1) > 0

    if args.taco_root:
        to_taco = np.linalg.inv(np.asarray(result['source_provenance']['transform_world_to_robot'], float))
        seq_dir = args.taco_root / 'Egocentric_Camera_Parameters' / args.triplet / args.seq
        K = np.loadtxt(seq_dir / 'egocentric_intrinsic.txt')
        E_all = np.load(seq_dir / 'egocentric_frame_extrinsic.npy')[:frames]

    def view_fraction(height):
        if not args.taco_root:
            return np.ones(X.shape)
        pts = np.c_[X.ravel(), Y.ravel(), np.full(X.size, table_z + height / 2)] @ to_taco[:3, :3].T + to_taco[:3, 3]
        c = np.einsum('fij,nj->fni', E_all[:, :3, :3], pts) + E_all[:, None, :3, 3]
        z = np.maximum(c[..., 2], 1e-6)
        u, v = K[0, 0] * c[..., 0] / z + K[0, 2], K[1, 1] * c[..., 1] / z + K[1, 2]
        inside = (c[..., 2] > .05) & (u > .06 * K[0, 2]) & (u < 1.94 * K[0, 2]) & (v > .06 * K[1, 2]) & (v < 1.94 * K[1, 2])
        return inside.mean(0).reshape(X.shape)

    def rectangle(length, width, yaw, grow):
        """Grid mask of an upright rectangle footprint (grown by `grow` on every side) centred on a cell."""
        half = np.array([length / 2 + grow, width / 2 + grow])
        n = int(np.ceil(np.hypot(*half) / args.cell))
        u, v = np.mgrid[-n:n + 1, -n:n + 1] * args.cell
        c, s = np.cos(yaw), np.sin(yaw)
        return (np.abs(c * u + s * v) <= half[0] + .5 * args.cell) & (np.abs(-s * u + c * v) <= half[1] + .5 * args.cell)

    report, maps = {}, {}
    for spec in args.objects:
        name, height, length, width = spec.split(':'); height, length, width = float(height), float(length), float(width)
        view = view_fraction(height) >= args.min_view
        fits = np.zeros(X.shape, bool); blocks = np.zeros(X.shape, bool); best_yaw = np.full(X.shape, np.nan)
        for yaw in np.arange(args.yaws) * np.pi / args.yaws:
            # Sideways and above, the replayed motion stays `margin` (+ one cell of sampling) away from the object.
            roof = minimum_filter(ceiling, footprint=rectangle(length, width, yaw, args.margin + args.cell), mode='constant', cval=np.inf)
            on = minimum_filter(on_table.astype(np.uint8), footprint=rectangle(length, width, yaw, .005), mode='constant', cval=0) > 0
            floor = minimum_filter(corridor, footprint=rectangle(max(length - 2 * args.depth, args.cell), max(width - 2 * args.depth, args.cell),
                                                                  yaw, 0.), mode='constant', cval=np.inf)
            ok = (roof > height + args.margin + args.cell) & on & view
            blocking = ok & (floor < height - args.depth)
            best_yaw[ok & ~fits] = yaw
            best_yaw[blocking & ~blocks] = yaw
            fits |= ok; blocks |= blocking
        maps[name] = dict(fits=fits, blocks=blocks, yaw=best_yaw)
        report[name] = dict(height_m=height, length_m=length, width_m=width, radius_m=width / 2, fit_cells=int(fits.sum()),
                            blocking_cells=int(blocks.sum()),
                            blocking=[dict(x=float(X[i, j]), y=float(Y[i, j]), yaw=float(best_yaw[i, j])) for i, j in zip(*np.nonzero(blocks))][:200])
    approach = np.isfinite(corridor) & on_table
    report['_grid'] = dict(cell_m=args.cell, origin_xy=lo.tolist(), shape=[int(s) for s in shape], table_z=table_z,
                           approach_cells=int(approach.sum()), table_cells=int(on_table.sum()),
                           free_frames=[int(k) for k in sorted(free)[:1] + sorted(free)[-1:]])

    if args.taco_root:
        # Overlay on the first frame: table outline, approach corridor, fit / blocking cells per object.
        from egoengine.taco import load_sequence
        seq = load_sequence(args.triplet, args.seq)
        image = seq.read_frame(0)[:, :, ::-1].copy()
        E = seq.extrinsics[0].astype(float)

        def project(xy):
            pts = np.c_[xy, np.full(len(xy), table_z)] @ to_taco[:3, :3].T + to_taco[:3, 3]
            c = pts @ E[:3, :3].T + E[:3, 3]
            return np.c_[K[0, 0] * c[:, 0] / c[:, 2] + K[0, 2], K[1, 1] * c[:, 1] / c[:, 2] + K[1, 2]].astype(int)
        tiles = []
        for name in maps:
            layer = image.copy()
            for mask, colour in ((approach, (255, 160, 0)), (maps[name]['fits'], (0, 200, 0)), (maps[name]['blocks'], (0, 0, 255))):
                for u, v in project(np.stack([X[mask], Y[mask]], 1)):
                    cv2.circle(layer, (int(u), int(v)), 4, colour, -1)
            tile = cv2.addWeighted(layer, .55, image, .45, 0)
            cv2.putText(tile, f'{name} h={report[name]["height_m"]:.2f} fit={report[name]["fit_cells"]} block={report[name]["blocking_cells"]}',
                        (20, 50), 0, 1.4, (255, 255, 255), 3)
            tiles.append(cv2.resize(tile, None, fx=.5, fy=.5))
    args.out.mkdir(parents=True, exist_ok=True)
    if args.taco_root:
        cv2.imwrite(str(args.out / 'occupancy.jpg'), np.hstack(tiles) if len(tiles) < 3 else np.vstack(
            [np.hstack(tiles[i:i + 2]) if i + 1 < len(tiles) else np.hstack([tiles[i], np.zeros_like(tiles[i])]) for i in range(0, len(tiles), 2)]))
    np.savez_compressed(args.out / 'occupancy.npz', ceiling=ceiling, corridor=corridor, on_table=on_table, origin=lo, cell=args.cell,
                        **{f'fits_{n}': m['fits'] for n, m in maps.items()}, **{f'yaw_{n}': m['yaw'] for n, m in maps.items()})
    (args.out / 'occupancy.json').write_text(json.dumps(report, indent=1))
    print('OCCUPANCY', json.dumps({k: {kk: v[kk] for kk in ('fit_cells', 'blocking_cells')} for k, v in report.items() if k != '_grid'}),
          json.dumps(report['_grid']), flush=True)


if __name__ == '__main__':
    main()
