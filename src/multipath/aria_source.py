"""EgoEngine's released Aria demonstrations as a raw human source.

Only the human data shipped with the release are read: per-frame Aria hand
tracking keypoints in the camera frame (`all_pose.json`, `right_camera`),
per-frame world->camera extrinsics (`upd_egocentric_frame_extrinsic.npy`) and
the object trajectory in the same world (`retarget.json` `traj[*].tool`, equal
to the object of the prepared MuJoCo scene). The release's XHand retarget
(`trajectory_kinematic.npz`, `robot_cfg`) is never used.

Frame convention, checked on mustard_1 frames 44-62: world = inv(w2c) @ camera
point, no extra rotation; fingertips sit 2.3-2.5 cm inside the bottle mesh and
the wrist lies 2.0 cm from the release retarget. The video is 30 fps; the
release's 20 Hz playback of the same frames is not the source clock.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Aria landmarks: 0-4 tips (thumb..pinky), 5 wrist, 6/7 thumb MCP/IP,
# 8-10 index MCP/PIP/DIP, 11-13 middle, 14-16 ring, 17-19 pinky, 20 palm.
# MANO order: 0 wrist, then (MCP/CMC, PIP/MCP, DIP/IP, tip) per finger.
ARIA_TO_MANO = [5, None, 6, 7, 0, 8, 9, 10, 1, 11, 12, 13, 2, 14, 15, 16, 3, 17, 18, 19, 4]
MIDDLE_CHAIN = [0, 9, 10, 11, 12]


@dataclass
class AriaObject:
    role: str
    object_id: str
    name: str
    poses: np.ndarray
    mesh_path: str
    mesh_scale: float = 1.


@dataclass
class AriaHand:
    side: str
    joints: np.ndarray
    filled_frames: list = field(default_factory=list)
    calibration: np.ndarray = None


class AriaSource:
    """TACO-loader-like view: n_frames, fps, objects, hands, video and camera."""

    def __init__(self, release_root, case, fps=30.):
        root = Path(release_root) / 'examples' / 'prepared'
        visual = root / 'visual_inputs' / 'Aria' / case
        frames = json.loads((visual / 'all_pose.json').read_text())['frames']
        w2c = np.load(visual / 'upd_egocentric_frame_extrinsic.npy').astype(float)
        traj = json.loads((visual / 'retarget.json').read_text())['traj']
        n = min(len(frames), len(w2c), len(traj))
        raw, filled = [], []
        for f in range(n):
            points = np.asarray(frames[f].get('right_camera') or [], float)
            if points.shape != (21, 3) or not np.isfinite(points).all():
                raw.append(None)
                filled.append(f)
                continue
            c2w = np.linalg.inv(w2c[f])
            raw.append(points @ c2w[:3, :3].T + c2w[:3, 3])
        valid = [f for f in range(n) if raw[f] is not None]
        if not valid:
            raise ValueError(f'No valid right-hand keypoints in {visual}')
        for f in filled:
            # Hold the nearest observed frame, as the release's own bridge does.
            raw[f] = raw[min(valid, key=lambda v: (abs(v - f), v))]
        aria = np.stack(raw)
        mesh = self._scene_mesh(root / 'processed' / 'aria' / 'xhand' / 'right' / case / 'scene.xml')
        self.case, self.n_frames, self.fps = case, n, float(fps)
        self.video_path = str(visual / 'cropped_video.mp4')
        self.intrinsic = np.loadtxt(visual / 'egocentric_intrinsic.txt').astype(float)
        self.extrinsics = w2c[:n]
        name = Path(mesh).stem.split('_', 1)[-1]
        self.objects = {'tool': AriaObject('tool', 'aria_' + Path(mesh).stem, name,
                                           np.stack([np.asarray(t['tool'], float) for t in traj[:n]]), mesh)}
        self.hands = {'right': AriaHand('right', self._mano_order(aria), filled)}

    @staticmethod
    def _scene_mesh(scene_xml):
        import xml.etree.ElementTree as ET
        for mesh in ET.parse(scene_xml).getroot().iter('mesh'):
            if 'visual' in mesh.get('name', '') and mesh.get('scale') is None:
                return str((scene_xml.parent / mesh.get('file')).resolve())
        raise FileNotFoundError(f'No unscaled visual object mesh in {scene_xml}')

    @staticmethod
    def _mano_order(aria):
        joints = np.zeros((len(aria), 21, 3))
        for mano, index in enumerate(ARIA_TO_MANO):
            if index is not None:
                joints[:, mano] = aria[:, index]
        return joints

    def calibrate(self, mano_rest):
        """Fit the MANO rest hand to this hand's measured bone lengths.

        Aria has no thumb CMC: place it on the wrist->thumb-MCP segment at the
        MANO rest-pose fraction. Scale the rest hand about the wrist by the
        ratio of middle-finger chain lengths (pose invariant).
        """
        rest = np.asarray(mano_rest, float)
        joints = self.hands['right'].joints
        axis = rest[2] - rest[0]
        fraction = float(np.dot(rest[1] - rest[0], axis) / np.dot(axis, axis))
        joints[:, 1] = joints[:, 0] + fraction * (joints[:, 2] - joints[:, 0])

        def chain(points):
            return np.linalg.norm(np.diff(points[..., MIDDLE_CHAIN, :], axis=-2), axis=-1).sum(-1)
        ratio = float(np.median(chain(joints)) / chain(rest))
        self.hands['right'].calibration = rest[0] + ratio * (rest - rest[0])
        self.hands['right'].thumb_cmc_fraction = fraction
        self.hands['right'].scale_to_mano = ratio
        return self.hands['right'].calibration
