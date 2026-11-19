"""Edited egocentric video rendered from the measured digital twin.

Background: the source video with the demonstrator's hands removed (prepared
SAM2 + LaMa frames), then the originally manipulated objects removed with LaMa
inside their dilated ground-truth projections (the twin moves them on its own).
Foreground: the measured PhysX state of every frame (robot hands, tool, target
and environment objects such as an inserted obstacle) rendered at the TACO
egocentric camera, with planar contact shadows. Nothing comes from a reference
animation. Camera registration is the dataset's (about 20 px residual, not
refined here).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_lama(root, device='cuda'):
    import sys
    import torch
    from omegaconf import OmegaConf
    sys.path.insert(0, str(root / 'third_party' / 'lama'))
    from saicinpainting.training.modules.ffc import FFCResNetGenerator
    ckpt = root / 'data' / 'models' / 'lama' / 'big-lama'
    g = OmegaConf.load(ckpt / 'config.yaml').generator
    model = FFCResNetGenerator(
        g.input_nc, g.output_nc, ngf=g.ngf, n_downsampling=g.n_downsampling, n_blocks=g.n_blocks,
        add_out_act=g.add_out_act,
        init_conv_kwargs=OmegaConf.to_container(g.init_conv_kwargs, resolve=True),
        downsample_conv_kwargs=OmegaConf.to_container(g.downsample_conv_kwargs, resolve=True),
        resnet_conv_kwargs=OmegaConf.to_container(g.resnet_conv_kwargs, resolve=True))
    state = torch.load(ckpt / 'models' / 'best.ckpt', map_location='cpu', weights_only=False)['state_dict']
    model.load_state_dict({k[len('generator.'):]: v for k, v in state.items() if k.startswith('generator.')})
    return model.eval().to(device)


def inpaint(model, rgb, mask, margin=.5):
    import torch
    if not mask.any():
        return rgb
    h, w = rgb.shape[:2]
    ys, xs = np.nonzero(mask)
    my, mx = int((np.ptp(ys) + 1) * margin), int((np.ptp(xs) + 1) * margin)
    y0, y1, x0, x1 = max(ys.min() - my, 0), min(ys.max() + my + 1, h), max(xs.min() - mx, 0), min(xs.max() + mx + 1, w)
    img = rgb[y0:y1, x0:x1].astype(np.float32) / 255.
    m = mask[y0:y1, x0:x1].astype(np.float32)
    ph, pw = (8 - img.shape[0] % 8) % 8, (8 - img.shape[1] % 8) % 8
    img, m = np.pad(img, ((0, ph), (0, pw), (0, 0))), np.pad(m, ((0, ph), (0, pw)))
    with torch.inference_mode():
        t_img = torch.from_numpy(img).permute(2, 0, 1)[None].cuda()
        t_m = torch.from_numpy(m)[None, None].cuda()
        out = model(torch.cat([t_img * (1 - t_m), t_m], 1))[0].permute(1, 2, 0).cpu().numpy()
    out = (np.clip(out[:y1 - y0, :x1 - x0], 0, 1) * 255).astype(np.uint8)
    result = rgb.copy()
    region = result[y0:y1, x0:x1]
    region[mask[y0:y1, x0:x1]] = out[mask[y0:y1, x0:x1]]
    return result


def load_obj(path, scale):
    vertices, faces = [], []
    for line in open(path):
        if line.startswith('v '):
            vertices.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('f '):
            ids = [int(t.split('/')[0]) - 1 for t in line.split()[1:]]
            faces += [[ids[0], ids[k], ids[k + 1]] for k in range(1, len(ids) - 1)]
    return np.asarray(vertices) * scale, np.asarray(faces)


def silhouette(points, faces, w2c, K, shape):
    cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    uv = cam[:, :2] / np.maximum(z, 1e-6)[:, None] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    keep = (z[faces] > .05).all(1)
    mask = np.zeros(shape, np.uint8)
    for tri in np.round(uv[faces[keep]] * 4).astype(np.int32):
        cv2.fillConvexPoly(mask, tri, 1, shift=2)
    return mask.astype(bool)


def transform(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--triplet', required=True)
    p.add_argument('--sequence', required=True)
    p.add_argument('--run-dir', type=Path, required=True, help='final rollout dir (visual_actual.npz, actual_trace.json)')
    p.add_argument('--background-dir', type=Path, required=True, help='hand-removed frames f%%04d_inpainted.png')
    p.add_argument('--environment-objects', type=Path, help='the JSON given to the rollout (mesh paths and names)')
    p.add_argument('--lama-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--object-dilate', type=int, default=31)
    p.add_argument('--fps', type=float, default=30., help='playback rate; the executed rate (source fps / time scale) shows the robot at its real speed')
    args = p.parse_args()
    out = args.out
    (out / 'frames').mkdir(parents=True, exist_ok=True)

    import pyrender
    import trimesh
    from egoengine.taco import load_sequence
    seq = load_sequence(args.triplet, args.sequence)
    K = seq.intrinsic.astype(float)
    E = seq.extrinsics.astype(float)
    result = json.loads((args.run_dir / 'pipeline_result.json').read_text())
    robot_to_world = np.linalg.inv(np.asarray(result['source_provenance']['transform_world_to_robot'], float))
    table_z = float(result['table_z'])
    meshes = np.load(args.run_dir / 'visual_meshes.npz')
    paths = json.loads((args.run_dir / 'visual_mesh_paths.json').read_text())
    poses = np.load(args.run_dir / 'visual_actual.npz')['transforms'].astype(float)
    trace = json.loads((args.run_dir / 'actual_trace.json').read_text())
    n = min(seq.n_frames, len(poses))

    cap = cv2.VideoCapture(seq.video_path)
    originals = []
    while len(originals) < n:
        ok, frame = cap.read()
        if not ok:
            break
        originals.append(frame)
    cap.release()
    shape = originals[0].shape[:2]

    # Appearance: manipulated objects keep their colour measured in frame 0.
    gt = {}
    for role, obj in seq.objects.items():
        v, f = load_obj(obj.mesh_path, .01)
        gt[role] = (v, f[::max(1, len(f) // 30000)], obj.poses.astype(float))
    colours = {}
    for role, (v, f, P) in gt.items():
        full = silhouette(transform(v, P[0]), f, E[0], K, shape).astype(np.uint8)
        full = cv2.morphologyEx(full, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        mask = cv2.erode(full, np.ones((15, 15), np.uint8)) > 0
        colours[role] = (np.median(originals[0][mask][:, ::-1], axis=0) / 255.).tolist() + [1.] if mask.any() else [.6, .6, .6, 1.]
    print('OBJECT_COLOURS', json.dumps(colours), flush=True)

    groups = {}
    for i, path in enumerate(paths):
        name = ('tool' if path.startswith('/World/tool') else 'target' if path.startswith('/World/target')
                else path.split('/')[2] if path.startswith('/World/env_') else 'left_hand' if 'xhand_left' in path
                else 'right_hand' if 'xhand_right' in path else None)
        if name:
            groups.setdefault(name, []).append(i)
    environment = json.loads(args.environment_objects.read_text()) if args.environment_objects else []
    env_meshes = {}
    for item in environment:
        if 'env_' + item['name'] in groups:
            continue
        mesh = trimesh.load(item['mesh'], force='mesh')
        env_meshes[item['name']] = (np.asarray(mesh.vertices), np.asarray(mesh.faces), mesh)
    material = {
        'right_hand': [.22, .22, .24, 1.], 'left_hand': [.22, .22, .24, 1.],
        'tool': colours.get('tool', [.8, .6, .2, 1.]), 'target': colours.get('target', [.3, .3, .3, 1.])}
    env_colour = [.93, .72, .12, 1.]
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[.45, .45, .45])
    nodes = {}
    for name, ids in groups.items():
        colour = material.get(name, env_colour)
        for i in ids:
            mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(meshes[f'v{i}'], meshes[f'f{i}'], process=False),
                material=pyrender.MetallicRoughnessMaterial(baseColorFactor=colour, metallicFactor=.1, roughnessFactor=.7))
            nodes[i] = scene.add(mesh)
    env_nodes = {}
    for name, (v, f, source) in env_meshes.items():
        # Use the asset's own texture when it has one (scanned objects).
        mesh = (pyrender.Mesh.from_trimesh(source) if getattr(source.visual, 'kind', None) == 'texture' else
                pyrender.Mesh.from_trimesh(trimesh.Trimesh(v, f, process=False),
                    material=pyrender.MetallicRoughnessMaterial(baseColorFactor=env_colour, metallicFactor=.05, roughnessFactor=.6)))
        env_nodes[name] = scene.add(mesh)
    camera = scene.add(pyrender.IntrinsicsCamera(K[0, 0], K[1, 1], K[0, 2], K[1, 2], znear=.03, zfar=10.))
    light = scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.5))
    renderer = pyrender.OffscreenRenderer(shape[1], shape[0])
    lama = load_lama(args.lama_root)
    flip = np.diag([1., -1., -1., 1.])

    def env_pose(state, name):
        s = state['environment_objects'][name]['pose']
        from scipy.spatial.transform import Rotation
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(np.asarray(s[3:])[[1, 2, 3, 0]]).as_matrix()
        matrix[:3, 3] = s[:3]
        return matrix

    writer = cv2.VideoWriter(str(out / 'edited_mp4v.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, shape[::-1])
    pair = cv2.VideoWriter(str(out / 'side_by_side_mp4v.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (shape[1], shape[0] // 2))
    for k in range(n):
        background = cv2.imread(str(args.background_dir / f'f{k:04d}_inpainted.png'))
        if background is None:
            raise FileNotFoundError(f'missing hand-removed frame {k}')
        removal = np.zeros(shape, bool)
        for role, (v, f, P) in gt.items():
            removal |= silhouette(transform(v, P[k]), f, E[k], K, shape)
        removal = cv2.dilate(removal.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.object_dilate,) * 2)) > 0
        background = inpaint(lama, background[:, :, ::-1].copy(), removal)[:, :, ::-1]

        world = {}
        for name, ids in groups.items():
            for i in ids:
                matrix = robot_to_world @ poses[k, i]
                scene.set_pose(nodes[i], matrix)
                world[i] = (name, matrix)
        for name in env_nodes:
            matrix = robot_to_world @ env_pose(trace[k], name)
            scene.set_pose(env_nodes[name], matrix)
            world['env_' + name] = (name, matrix)
        c2w = np.linalg.inv(E[k]) @ flip
        scene.set_pose(camera, c2w)
        scene.set_pose(light, c2w)
        colour, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        alpha = (depth > 0).astype(np.float32)
        alpha = cv2.GaussianBlur(alpha, (3, 3), .7)[..., None]

        # Planar contact shadows: every body projected straight down onto
        # the twin table plane, fading with its height above the table.
        shadow = np.zeros(shape, np.float32)
        for key, (name, matrix) in world.items():
            if isinstance(key, str):
                v = env_meshes[key[4:]][0]
            else:
                v = meshes[f'v{key}']
            v = v[::max(1, len(v) // 1500)]
            sim = transform(v, np.linalg.inv(robot_to_world) @ matrix)
            weight = np.exp(-np.maximum(sim[:, 2] - table_z, 0) / .06)
            flat = sim.copy()
            flat[:, 2] = table_z
            cam = transform(transform(flat, robot_to_world), E[k])
            uv = np.round(cam[:, :2] / cam[:, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]).astype(int)
            ok = (cam[:, 2] > .05) & (uv[:, 0] >= 0) & (uv[:, 0] < shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < shape[0])
            np.add.at(shadow, (uv[ok, 1], uv[ok, 0]), weight[ok])
        shadow = cv2.GaussianBlur(shadow, (0, 0), 9)
        shadow = np.clip(shadow / max(1e-6, np.percentile(shadow[shadow > 0], 95) if (shadow > 0).any() else 1), 0, 1)
        base = background.astype(np.float32) * (1 - .45 * shadow[..., None])
        edited = base * (1 - alpha) + colour[:, :, 2::-1].astype(np.float32) * alpha
        edited = np.clip(edited, 0, 255).astype(np.uint8)
        writer.write(edited)
        top = np.hstack((cv2.resize(originals[k], (shape[1] // 2, shape[0] // 2)), cv2.resize(edited, (shape[1] // 2, shape[0] // 2))))
        cv2.putText(top, 'source video', (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(top, 'edited: measured PhysX twin', (shape[1] // 2 + 12, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2, cv2.LINE_AA)
        pair.write(top)
        if k % 12 == 0 or k == n - 1:
            cv2.imwrite(str(out / 'frames' / f'f{k:04d}_edited.jpg'), edited, [cv2.IMWRITE_JPEG_QUALITY, 90])
            cv2.imwrite(str(out / 'frames' / f'f{k:04d}_pair.jpg'), top, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if k % 20 == 0:
            print('FRAME', k, 'removed_px', int(removal.sum()), flush=True)
    writer.release()
    pair.release()
    (out / 'render_report.json').write_text(json.dumps(dict(
        triplet=args.triplet, sequence=args.sequence, run_dir=str(args.run_dir), frames=n,
        background=str(args.background_dir), object_colours=colours,
        environment_objects=[i['name'] for i in environment],
        scope='foreground = measured PhysX twin; background = hand-removed source with manipulated objects inpainted; '
              'camera from TACO (no refinement); planar shadows'), indent=1))
    print('RENDER_DONE', out, flush=True)


if __name__ == '__main__':
    main()
