"""Remove the demonstrated object from the prepared demonstrator-free background.

The released `removed_w_mask_5.mp4` removes the demonstrator only; the object
stays as real pixels at its demonstrated poses. An augmented case renders the
object at its new (physically optimized) poses, so the source object is
inpainted out first: per frame, the object mesh at the source `traj[*].tool`
pose is projected with the recorded camera (`camera_to_world`, OpenCV) and the
dilated convex hull is filled by LaMa. Frame count, size and rate are kept;
the released background is kept as `removed_w_mask_5_source.mp4`.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--visual-dir', type=Path, required=True, help='case visual_inputs dir (augmented package)')
    p.add_argument('--source-retarget', type=Path, required=True, help='retarget.json with the demonstrated tool poses')
    p.add_argument('--mesh', type=Path, required=True)
    p.add_argument('--lama-root', type=Path, required=True)
    p.add_argument('--helpers', type=Path, required=True, help='dir with render_edited_video.py (load_lama, inpaint)')
    p.add_argument('--dilate', type=int, default=41)
    p.add_argument('--temporal', type=int, default=2, help='union with the masks of this many neighbouring frames (motion blur, tracking lag)')
    args = p.parse_args()
    sys.path.insert(0, str(args.helpers))
    from render_edited_video import load_lama, inpaint

    retarget = json.loads(args.source_retarget.read_text())
    cameras = [np.asarray(c, float) for c in retarget['camera_traj']['target2camera']]
    K = np.asarray(retarget['camera_traj']['cam_K'], float)
    tools = [np.asarray(f['tool'], float).reshape(4, 4) for f in retarget['traj']]
    vertices = np.array([l.split()[1:4] for l in args.mesh.read_text().splitlines() if l.startswith('v ')], float)
    vertices = vertices[::max(1, len(vertices) // 4000)]

    released = args.visual_dir / 'removed_w_mask_5.mp4'
    backup = args.visual_dir / 'removed_w_mask_5_source.mp4'
    if not backup.exists():
        shutil.copy2(released, backup)
    cap = cv2.VideoCapture(str(backup))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) != len(tools):
        raise ValueError(f'{len(frames)} background frames vs {len(tools)} retarget frames')
    h, w = frames[0].shape[:2]
    lama = load_lama(args.lama_root)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.dilate, args.dilate))
    out = cv2.VideoWriter(str(released), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    masks = []
    for k in range(len(frames)):
        world = vertices @ tools[k][:3, :3].T + tools[k][:3, 3]
        cam = (world - cameras[k][:3, 3]) @ cameras[k][:3, :3]  # inverse(camera_to_world)
        front = cam[:, 2] > .05
        mask = np.zeros((h, w), np.uint8)
        if front.any():
            uv = cam[front, :2] / cam[front, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
            cv2.fillConvexPoly(mask, cv2.convexHull(np.round(uv).astype(np.int32)), 1)
        masks.append(mask)
    areas = []
    preview = {}
    for k, frame in enumerate(frames):
        mask = np.max(masks[max(0, k - args.temporal):k + args.temporal + 1], axis=0)
        mask = cv2.dilate(mask, kernel) > 0
        areas.append(int(mask.sum()))
        result = inpaint(lama, frame[:, :, ::-1].copy(), mask)[:, :, ::-1]
        out.write(np.ascontiguousarray(result))
        if k in (0, 58, 66, len(frames) - 1):
            overlay = frame.copy()
            overlay[mask] = (0.5 * overlay[mask] + [0, 0, 127]).astype(np.uint8)
            preview[k] = np.hstack([overlay, result])
    out.release()
    sheet = np.vstack([cv2.resize(v, (v.shape[1] // 3, v.shape[0] // 3)) for v in preview.values()])
    cv2.imwrite(str(args.visual_dir / 'source_tool_removal_preview.jpg'), sheet)
    report = dict(released_background=str(backup.name), mask='projected source tool mesh, convex hull, union over '
                  f'+-{args.temporal} frames, dilated {args.dilate} px', inpainting='LaMa big-lama', frames=len(frames), mask_area_px=dict(
                      min=min(areas), max=max(areas)))
    (args.visual_dir / 'source_tool_removal.json').write_text(json.dumps(report, indent=1))
    print('SOURCE_TOOL_REMOVED', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
